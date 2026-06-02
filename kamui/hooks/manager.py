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
            Useful for caching activations across multiple forward passes
            without re-registering hooks each time.

Hook point naming convention:
    "<module_path>.<io>"
    module_path: the PyTorch named_modules() key, e.g. "blocks.3.attn"
    io:          "output" | "input" | "weights" | "pre_softmax" | "mid"

    Examples:
        "blocks.0.attn.output"       — attention output, shape (B, S, D)
        "blocks.0.attn.weights"      — attention weights, shape (B, H, S, S)
        "blocks.0.attn.pre_softmax"  — pre-softmax scores, shape (B, H, S, S)
        "blocks.3.ffn.mid"           — FFN hidden activations, shape (B, S, F)
        "blocks.3.ffn.output"        — FFN output, shape (B, S, D)
        "embed.output"               — combined embeddings, shape (B, S, D)
        "unembed.input"              — residual stream before unembedding

Why context manager (not explicit attach/detach)?
    PyTorch's register_forward_hook returns a handle that MUST be removed
    after use.  Forgotten handles accumulate across forward passes,
    corrupting the cache and leaking memory.  The context manager makes
    correct usage the only possible usage: hooks are always removed on
    exit, even if an exception is raised inside the block.

    This design is inspired by TransformerLens's ``run_with_cache`` but
    is more Pythonic: it uses the standard context manager protocol
    instead of a custom method.

Implemented in: Phase 3, Week 13
"""

# Implementation begins in Phase 3.
