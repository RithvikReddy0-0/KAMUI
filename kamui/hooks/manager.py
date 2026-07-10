"""HookManager — context-managed activation capture for any nn.Module.

This is the most important file in the hooks subpackage.  Every
interpretability tool in ``kamui.mechinterp`` depends on it.

Responsibilities:
    - ``HookManager``:
        A context manager that:
        1. Registers PyTorch forward hooks on named model submodules
        2. Caches the captured tensors in a dictionary keyed by hook name
        3. Removes all hooks on exit (including on exceptions)
        4. Guarantees that attaching hooks does not alter model output

        Interface:

            with HookManager(model) as hooks:
                hooks.attach("blocks.3.attn", point="output")
                hooks.attach("blocks.3.attn", point="weights")
                hooks.attach("embed",         point="output")

                logits = model(token_ids)

                attn_out  = hooks.get("blocks.3.attn.output")  # (B, S, D)
                attn_wts  = hooks.get("blocks.3.attn.weights") # (B, H, S, S)
                embedding = hooks.get("embed.output")          # (B, S, D)

        ``hooks.get_all() -> dict[str, Tensor]``:
            Return all cached activations as a dictionary.

        ``hooks.clear()``:
            Clear the activation cache without removing hooks.

Hook point naming convention:
    "<module_path>.<io>"
    module_path: the PyTorch named_modules() key, e.g. "blocks.3.attn"
    io:          "output" | "input" | "weights" | "mid"

    Examples:
        "blocks.0.attn.output"       — attention output, shape (B, S, D)
        "blocks.0.attn.weights"      — attention weights, shape (B, H, S, S)
        "blocks.3.ffn.mid"           — FFN hidden activations, shape (B, S, F)
        "blocks.3.ffn.output"        — FFN output, shape (B, S, D)
        "embed.output"               — combined embeddings, shape (B, S, D)
        "unembed.input"              — residual stream before unembedding

How the internal ``weights`` point is captured without touching the model:
    Attention weights are an internal functional computation, not a module
    output, so a plain forward hook cannot reach them.  For a ``weights``
    point the manager temporarily replaces the attention module's ``forward``
    with a wrapper that calls the real forward with ``return_weights=True``,
    caches the weights, and returns the identical output tensor.  The wrapper
    is removed on exit, so the model itself remains completely hook-agnostic
    and its output is unchanged.

Why context manager (not explicit attach/detach)?
    PyTorch's register_forward_hook returns a handle that MUST be removed
    after use.  Forgotten handles accumulate across forward passes,
    corrupting the cache and leaking memory.  The context manager makes
    correct usage the only possible usage: hooks are always removed on
    exit, even if an exception is raised inside the block.

Implemented in: Phase 3.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from kamui.hooks.registry import HookRegistry
from kamui.hooks.types import ActivationCache


class HookManager:
    """Context-managed activation capture over a model's named submodules."""

    def __init__(self, model: nn.Module) -> None:
        """Create a hook manager for ``model``.

        Args:
            model: The model to attach hooks to.  Must expose a ``config``
                attribute (a ``ModelConfig``) so hook points can be validated.
        """
        self.model = model
        self._cache: ActivationCache = {}
        self._handles: list[Any] = []
        self._wrapped: list[nn.Module] = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.remove()
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Attaching
    # ------------------------------------------------------------------

    def attach(self, module_path: str, point: str) -> "HookManager":
        """Attach a capture hook at ``"<module_path>.<point>"``.

        Args:
            module_path: A ``named_modules()`` key, e.g. ``"blocks.3.attn"``.
            point:       One of ``"output"``, ``"input"``, ``"weights"``, ``"mid"``.

        Returns:
            ``self`` (so calls can be chained).

        Raises:
            ValueError: If the resulting hook point is not valid for this model.
        """
        hook_point = f"{module_path}.{point}"
        if not HookRegistry.validate(hook_point, self.model.config):
            valid = HookRegistry.all_points(self.model.config)
            raise ValueError(
                f"'{hook_point}' is not a valid hook point. "
                f"Valid points: {valid}"
            )

        if point == "output":
            module = self._resolve(module_path)
            handle = module.register_forward_hook(self._output_saver(hook_point))
            self._handles.append(handle)
        elif point == "input":
            module = self._resolve(module_path)
            handle = module.register_forward_pre_hook(self._input_saver(hook_point))
            self._handles.append(handle)
        elif point == "mid":
            # FFN hidden activation = output of the GELU submodule.
            activation = self._resolve(f"{module_path}.activation")
            handle = activation.register_forward_hook(self._output_saver(hook_point))
            self._handles.append(handle)
        else:  # point == "weights" (registry guarantees no other value)
            module = self._resolve(module_path)
            self._wrap_for_weights(module, hook_point)

        return self

    # ------------------------------------------------------------------
    # Reading the cache
    # ------------------------------------------------------------------

    def get(self, hook_point: str) -> Tensor:
        """Return a cached activation.

        Args:
            hook_point: The full hook-point string, e.g. ``"embed.output"``.

        Returns:
            The captured tensor.

        Raises:
            KeyError: If nothing is cached at ``hook_point`` (not attached, or
                no forward pass has run since the last ``clear``).
        """
        if hook_point not in self._cache:
            raise KeyError(
                f"'{hook_point}' is not in the cache. Did you attach it and run "
                f"a forward pass? Cached points: {list(self._cache)}"
            )
        return self._cache[hook_point]

    def get_all(self) -> ActivationCache:
        """Return a shallow copy of all cached activations."""
        return dict(self._cache)

    def clear(self) -> None:
        """Clear the activation cache without removing hooks."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove all hooks and restore any wrapped forwards.

        The activation cache is left intact so captured tensors remain
        accessible after the ``with`` block exits.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module in self._wrapped:
            # Delete the instance-level override so the class ``forward`` returns.
            del module.forward
        self._wrapped.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, module_path: str) -> nn.Module:
        # attach() validates the hook point against the registry first, so the
        # module path is always present here; a direct lookup suffices.
        return dict(self.model.named_modules())[module_path]

    def _output_saver(self, hook_point: str):
        # Every hookable point (embed, attn output, ffn output, ffn.activation)
        # yields a plain tensor: attention returns a bare tensor in the model's
        # forward, and the ``weights`` path is captured by the forward wrapper.
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
            self._cache[hook_point] = output.detach()

        return hook

    def _input_saver(self, hook_point: str):
        def pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            self._cache[hook_point] = inputs[0].detach()

        return pre_hook

    def _wrap_for_weights(self, module: nn.Module, hook_point: str) -> None:
        if module in self._wrapped:
            return
        original_forward = module.forward

        def wrapper(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            output, weights = original_forward(x, return_weights=True)
            self._cache[hook_point] = weights.detach()
            return output

        module.forward = wrapper  # type: ignore[method-assign]
        self._wrapped.append(module)
