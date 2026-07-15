"""Activation patching: causal interventions to localise model behaviours.

Activation patching is the core methodology of mechanistic interpretability.
It answers: which components are *causally responsible* for a behaviour, not
merely correlated with it?

Technique:
    1. Run the model on a *clean* input and cache an activation.
    2. Run the model on a *corrupted* input, but replace (patch) that
       activation with the cached clean one.
    3. Measure how far the output moves back toward the clean output.
       A large recovery → that component is on the causal path.

Metric — logit-difference recovery:
    Let ``answer`` be the clean run's top prediction and ``corrupt`` the
    corrupted run's top prediction (both at the final position).  Define
    ``metric(logits) = logits[answer] - logits[corrupt]``.  Then::

        recovery = (metric(patched) - metric(corrupt))
                   / (metric(clean) - metric(corrupt))

    0.0 = patching had no effect (component not causally relevant);
    1.0 = patching fully restored the clean behaviour (component is critical).

Responsibilities:
    - ``ActivationPatcher.patch_single(clean_ids, corrupted_ids, hook_point)``:
        Patch one output hook point; return the recovery as a float.
    - ``ActivationPatcher.patch_all_layers(clean_ids, corrupted_ids, component)``:
        Patch each layer's ``attn`` or ``ffn`` output; return a ``PatchingResult``.
    - ``ActivationPatcher.patch_all_heads(clean_ids, corrupted_ids)``:
        Patch each (layer, head) pair; return a ``HeadPatchingResult``.

References:
    Meng, K. et al. (2022). Locating and Editing Factual Associations in GPT
    (ROME). NeurIPS 2022. https://arxiv.org/abs/2202.05262
    Wang, K. et al. (2022). Interpretability in the Wild (IOI).
    https://arxiv.org/abs/2211.00593

Implemented in: Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from kamui.hooks.manager import HookManager

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Denominator below which the clean/corrupt metrics are treated as equal.
_EPS: float = 1e-8


@dataclass
class PatchingResult:
    """Per-layer causal effects from ``patch_all_layers``.

    Attributes:
        effects:   ``(n_layers,)`` recovery per layer.
        component: ``"attn"`` or ``"ffn"``.
    """

    effects: Tensor
    component: str

    def best_layer(self) -> int:
        """Return the layer with the largest recovery."""
        return int(self.effects.argmax().item())

    def plot(self, figsize: tuple[float, float] | None = None) -> Figure:
        """Bar chart of causal effect by layer."""
        import matplotlib.pyplot as plt

        effects = self.effects.cpu().numpy()
        fig, ax = plt.subplots(figsize=figsize or (max(6, len(effects) * 0.6), 4))
        ax.bar(range(len(effects)), effects, color="steelblue")
        ax.set_xlabel("layer")
        ax.set_ylabel("logit-diff recovery")
        ax.set_title(f"Activation patching — {self.component} output by layer")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_xticks(range(len(effects)))
        fig.tight_layout()
        return fig


@dataclass
class HeadPatchingResult:
    """Per-(layer, head) causal effects from ``patch_all_heads``.

    Attributes:
        effects: ``(n_layers, n_heads)`` recovery per attention head.
    """

    effects: Tensor

    def best_head(self) -> tuple[int, int]:
        """Return the ``(layer, head)`` with the largest recovery."""
        flat = int(self.effects.argmax().item())
        n_heads = self.effects.shape[1]
        return flat // n_heads, flat % n_heads

    def plot(self, figsize: tuple[float, float] | None = None) -> Figure:
        """Heatmap of causal effect by (layer, head)."""
        import matplotlib.pyplot as plt

        effects = self.effects.cpu().numpy()
        n_layers, n_heads = effects.shape
        fig, ax = plt.subplots(figsize=figsize or (max(5, n_heads), max(4, n_layers)))
        im = ax.imshow(effects, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title("Activation patching — per-head recovery")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        fig.colorbar(im, ax=ax, label="logit-diff recovery")
        fig.tight_layout()
        return fig


class ActivationPatcher:
    """Causal localisation via activation patching.

    Attributes:
        model: A trained ``KAMUITransformer``.
    """

    def __init__(self, model: nn.Module) -> None:
        """Create a patcher over ``model``.

        Args:
            model: A ``KAMUITransformer``.
        """
        self.model = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_batch(ids: Tensor) -> Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(
                f"ids must be a single sequence (S,) or (1, S), got shape {tuple(ids.shape)}"
            )
        return ids

    def _resolve(self, module_path: str) -> nn.Module:
        return dict(self.model.named_modules())[module_path]

    @staticmethod
    def _metric(logits: Tensor, answer: int, corrupt: int) -> float:
        return (logits[0, -1, answer] - logits[0, -1, corrupt]).item()

    def _run_patched(self, ids: Tensor, module: nn.Module, replacement: Tensor) -> Tensor:
        """Run the model with ``module``'s output replaced by ``replacement``."""

        def hook(_m: nn.Module, _inp: tuple, _out: Tensor) -> Tensor:
            return replacement

        handle = module.register_forward_hook(hook)
        try:
            return self.model(ids)
        finally:
            handle.remove()

    def _recovery(
        self, patched_logits: Tensor, answer: int, corrupt: int, l_clean: float, l_corrupt: float
    ) -> float:
        denom = l_clean - l_corrupt
        if abs(denom) < _EPS:
            return 0.0
        return (self._metric(patched_logits, answer, corrupt) - l_corrupt) / denom

    @staticmethod
    def _answers(
        clean_logits: Tensor,
        corrupt_logits: Tensor,
        answer_token: int | None,
        corrupt_token: int | None,
    ) -> tuple[int, int]:
        answer = answer_token if answer_token is not None else int(clean_logits[0, -1].argmax())
        corrupt = (
            corrupt_token if corrupt_token is not None else int(corrupt_logits[0, -1].argmax())
        )
        return answer, corrupt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def patch_single(
        self,
        clean_ids: Tensor,
        corrupted_ids: Tensor,
        hook_point: str,
        answer_token: int | None = None,
        corrupt_token: int | None = None,
    ) -> float:
        """Patch a single output hook point and return the recovery.

        Args:
            clean_ids:     Clean token IDs ``(S,)`` or ``(1, S)``.
            corrupted_ids: Corrupted token IDs of the same shape.
            hook_point:    A patchable output point: ``"embed.output"``,
                ``"blocks.{i}.attn.output"``, or ``"blocks.{i}.ffn.output"``.
            answer_token:  Optional metric answer token (defaults to the clean
                run's top prediction).
            corrupt_token: Optional metric corrupt token (defaults to the
                corrupted run's top prediction).

        Returns:
            The logit-difference recovery in roughly ``[0, 1]``.

        Raises:
            ValueError: If shapes differ or ``hook_point`` is not patchable.
        """
        clean_ids = self._as_batch(clean_ids)
        corrupted_ids = self._as_batch(corrupted_ids)
        if clean_ids.shape != corrupted_ids.shape:
            raise ValueError("clean_ids and corrupted_ids must have the same shape")
        if not self._is_patchable(hook_point):
            raise ValueError(
                f"'{hook_point}' is not a patchable output point "
                f"(use embed.output / blocks.i.attn.output / blocks.i.ffn.output)"
            )

        self.model.eval()
        module_path = hook_point.rsplit(".", 1)[0]

        with HookManager(self.model) as hooks:
            hooks.attach(module_path, "output")
            clean_logits = self.model(clean_ids)
            clean_act = hooks.get(hook_point)
        corrupt_logits = self.model(corrupted_ids)

        answer, corrupt = self._answers(clean_logits, corrupt_logits, answer_token, corrupt_token)
        l_clean = self._metric(clean_logits, answer, corrupt)
        l_corrupt = self._metric(corrupt_logits, answer, corrupt)

        patched = self._run_patched(corrupted_ids, self._resolve(module_path), clean_act)
        return self._recovery(patched, answer, corrupt, l_clean, l_corrupt)

    @torch.no_grad()
    def patch_all_layers(
        self,
        clean_ids: Tensor,
        corrupted_ids: Tensor,
        component: str = "attn",
        answer_token: int | None = None,
        corrupt_token: int | None = None,
    ) -> PatchingResult:
        """Patch each layer's ``attn`` or ``ffn`` output, one at a time.

        Args:
            clean_ids:     Clean token IDs.
            corrupted_ids: Corrupted token IDs of the same shape.
            component:     ``"attn"`` or ``"ffn"``.
            answer_token:  Optional metric answer token.
            corrupt_token: Optional metric corrupt token.

        Returns:
            A ``PatchingResult`` with one recovery per layer.

        Raises:
            ValueError: If ``component`` is not ``"attn"``/``"ffn"`` or shapes differ.
        """
        if component not in ("attn", "ffn"):
            raise ValueError(f"component must be 'attn' or 'ffn', got '{component}'")
        clean_ids = self._as_batch(clean_ids)
        corrupted_ids = self._as_batch(corrupted_ids)
        if clean_ids.shape != corrupted_ids.shape:
            raise ValueError("clean_ids and corrupted_ids must have the same shape")

        self.model.eval()
        n_layers = self.model.config.n_layers
        module_paths = [f"blocks.{i}.{component}" for i in range(n_layers)]

        with HookManager(self.model) as hooks:
            for path in module_paths:
                hooks.attach(path, "output")
            clean_logits = self.model(clean_ids)
            clean_acts = {path: hooks.get(f"{path}.output") for path in module_paths}
        corrupt_logits = self.model(corrupted_ids)

        answer, corrupt = self._answers(clean_logits, corrupt_logits, answer_token, corrupt_token)
        l_clean = self._metric(clean_logits, answer, corrupt)
        l_corrupt = self._metric(corrupt_logits, answer, corrupt)

        effects = []
        for path in module_paths:
            patched = self._run_patched(corrupted_ids, self._resolve(path), clean_acts[path])
            effects.append(self._recovery(patched, answer, corrupt, l_clean, l_corrupt))
        return PatchingResult(torch.tensor(effects), component)

    @torch.no_grad()
    def patch_all_heads(
        self,
        clean_ids: Tensor,
        corrupted_ids: Tensor,
        answer_token: int | None = None,
        corrupt_token: int | None = None,
    ) -> HeadPatchingResult:
        """Patch each attention head's contribution, one at a time.

        A head is patched by replacing its slice of the ``out_proj`` input (the
        concatenated per-head outputs) with the clean run's slice.

        Args:
            clean_ids:     Clean token IDs.
            corrupted_ids: Corrupted token IDs of the same shape.
            answer_token:  Optional metric answer token.
            corrupt_token: Optional metric corrupt token.

        Returns:
            A ``HeadPatchingResult`` with an ``(n_layers, n_heads)`` effect matrix.

        Raises:
            ValueError: If shapes differ.
        """
        clean_ids = self._as_batch(clean_ids)
        corrupted_ids = self._as_batch(corrupted_ids)
        if clean_ids.shape != corrupted_ids.shape:
            raise ValueError("clean_ids and corrupted_ids must have the same shape")

        self.model.eval()
        n_layers = self.model.config.n_layers
        n_heads = self.model.config.n_heads
        d_head = self.model.config.d_head

        # Capture the clean out_proj inputs (concatenated head outputs) per layer.
        clean_head_inputs: dict[int, Tensor] = {}
        handles = []
        for i in range(n_layers):
            out_proj = self._resolve(f"blocks.{i}.attn.out_proj")

            def _capture(layer: int) -> object:
                def pre_hook(_m: nn.Module, inp: tuple) -> None:
                    clean_head_inputs[layer] = inp[0].detach()

                return pre_hook

            handles.append(out_proj.register_forward_pre_hook(_capture(i)))
        clean_logits = self.model(clean_ids)
        for handle in handles:
            handle.remove()

        corrupt_logits = self.model(corrupted_ids)
        answer, corrupt = self._answers(clean_logits, corrupt_logits, answer_token, corrupt_token)
        l_clean = self._metric(clean_logits, answer, corrupt)
        l_corrupt = self._metric(corrupt_logits, answer, corrupt)

        effects = torch.zeros(n_layers, n_heads)
        for i in range(n_layers):
            out_proj = self._resolve(f"blocks.{i}.attn.out_proj")
            clean_in = clean_head_inputs[i]
            for head in range(n_heads):
                patched = self._run_head_patched(
                    corrupted_ids, out_proj, clean_in, head, n_heads, d_head
                )
                effects[i, head] = self._recovery(patched, answer, corrupt, l_clean, l_corrupt)
        return HeadPatchingResult(effects)

    def _run_head_patched(
        self,
        ids: Tensor,
        out_proj: nn.Module,
        clean_in: Tensor,
        head: int,
        n_heads: int,
        d_head: int,
    ) -> Tensor:
        """Run the model, replacing one head's slice of ``out_proj``'s input."""

        def pre_hook(_m: nn.Module, inp: tuple) -> tuple:
            current = inp[0]
            b, s, _ = current.shape
            patched = current.clone().view(b, s, n_heads, d_head)
            clean_reshaped = clean_in.view(b, s, n_heads, d_head)
            patched[:, :, head, :] = clean_reshaped[:, :, head, :]
            return (patched.view(b, s, n_heads * d_head),)

        handle = out_proj.register_forward_pre_hook(pre_hook)
        try:
            return self.model(ids)
        finally:
            handle.remove()

    @staticmethod
    def _is_patchable(hook_point: str) -> bool:
        return (
            hook_point == "embed.output"
            or hook_point.endswith(".attn.output")
            or hook_point.endswith(".ffn.output")
        )
