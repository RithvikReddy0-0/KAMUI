"""Unit tests for the hook system.

The most critical property of the hook system is transparency:
attaching hooks must never change model output.  These tests enforce that.
"""

import pytest


@pytest.mark.xfail(reason="HookManager not yet implemented — Phase 3")
def test_hooks_do_not_alter_model_output() -> None:
    """Attaching hooks and running a forward pass must produce byte-identical
    output to a hook-free forward pass.

    This is the fundamental correctness guarantee of the hook system.
    """
    # import torch
    # from kamui.model import KAMUITransformer, ModelConfig
    # from kamui.hooks import HookManager
    # model = KAMUITransformer(ModelConfig(...))
    # ids = torch.randint(0, 100, (1, 16))
    # baseline = model(ids).detach()
    # with HookManager(model) as hooks:
    #     hooks.attach("blocks.1.attn", point="output")
    #     hooks.attach("blocks.1.ffn", point="output")
    #     hooked = model(ids).detach()
    # assert baseline.allclose(hooked), "Hook system altered model output"
    pass


@pytest.mark.xfail(reason="HookManager not yet implemented — Phase 3")
def test_hooks_are_removed_after_context_exit() -> None:
    """All hooks must be removed when the context manager exits normally."""
    pass


@pytest.mark.xfail(reason="HookManager not yet implemented — Phase 3")
def test_hooks_are_removed_after_exception() -> None:
    """All hooks must be removed even if an exception is raised inside the block.

    This prevents hook accumulation that would corrupt subsequent runs.
    """
    # try:
    #     with HookManager(model) as hooks:
    #         hooks.attach("blocks.0.attn", point="output")
    #         raise RuntimeError("simulated error")
    # except RuntimeError:
    #     pass
    # assert len(model._forward_hooks) == 0, "Hooks not cleaned up after exception"
    pass


@pytest.mark.xfail(reason="HookManager not yet implemented — Phase 3")
def test_activation_cache_correct_shape() -> None:
    """Cached activations must have the shapes specified in the hook registry."""
    pass


@pytest.mark.xfail(reason="HookManager not yet implemented — Phase 3")
def test_invalid_hook_point_raises_error() -> None:
    """Attempting to attach a hook to a nonexistent module must raise a clear error,
    not silently fail.
    """
    pass


@pytest.mark.xfail(reason="HookManager not yet implemented — Phase 3")
def test_activation_patching_changes_output() -> None:
    """Replacing an activation with a different tensor must change the model output.

    This validates that the hook's return-value mechanism (which is used by
    ActivationPatcher) actually modifies the computation graph.
    """
    pass
