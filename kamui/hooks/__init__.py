"""Hook system subpackage for KAMUI.

The hook system is the bridge between the model and the interpretability
tools.  It allows any activation or gradient in the model's forward pass
to be captured, inspected, and (for activation patching) modified — without
changing a single line of model code.

Responsibilities:
    - Provide a context-managed interface for attaching and detaching hooks
    - Cache named activations during a forward pass
    - Guarantee that hooks are removed even if an exception occurs
    - Ensure that attaching hooks does not change model output

Design principle — hooks are external, not internal:
    The model (``kamui.model``) has no knowledge of the hook system.
    Hooks are attached from outside, via ``HookManager``, by referencing
    module names as strings.  This means:
    1. The model can be used without hooks (clean inference)
    2. The hook system can be upgraded without touching the model
    3. Multiple independent hook managers can be used simultaneously

Module layout:
    manager.py    — HookManager: the primary public interface
    registry.py   — named hook points for all KAMUI model components
    types.py      — type definitions for hook functions and cached values

Public API (available after Phase 3):
    HookManager   — context-managed activation capture

Usage pattern:
    with HookManager(model) as hooks:
        hooks.attach("blocks.3.attn", point="output")
        logits = model(token_ids)
        attn_out = hooks.get("blocks.3.attn.output")   # (B, S, D)

Implemented in: Phase 3, Weeks 13–14
"""

# from kamui.hooks.manager import HookManager

__all__: list[str] = []
