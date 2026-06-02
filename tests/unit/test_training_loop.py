"""Unit tests for the training loop.

These tests verify training mechanics without requiring full convergence.
They run quickly (< 5 seconds each) and are suitable for CI.
"""

import pytest


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_loss_decreases_in_first_10_steps() -> None:
    """Loss must decrease over the first 10 training steps on a fixed batch.

    This is the minimal sanity check for a training loop.  If loss does
    not decrease on a single fixed batch (which the model can overfit),
    the training loop has a fundamental bug.
    """
    pass


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_gradient_accumulation_matches_large_batch() -> None:
    """2 accumulation steps of batch_size=8 must produce the same weight
    update as 1 step of batch_size=16 (same data, same seed).

    This verifies that gradient accumulation is implemented correctly.
    """
    pass


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_lr_warmup_ramps_correctly() -> None:
    """LR must increase linearly from 0 to max_lr over warmup_steps."""
    pass


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_lr_cosine_decay_reaches_min_lr() -> None:
    """LR must reach min_lr at max_steps (not go below it)."""
    pass


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_checkpoint_resume_is_exact() -> None:
    """Resuming from a checkpoint must produce byte-identical loss curves.

    Steps 0–100, save checkpoint, resume, run steps 100–200.
    Must match a continuous run of 200 steps.
    """
    pass


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_weight_decay_not_applied_to_biases() -> None:
    """Bias parameters must have weight_decay=0.0 in the optimiser."""
    pass


@pytest.mark.xfail(reason="Trainer not yet implemented — Phase 2")
def test_weight_decay_not_applied_to_layernorm() -> None:
    """LayerNorm gamma and beta must have weight_decay=0.0."""
    pass
