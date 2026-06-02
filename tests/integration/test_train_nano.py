"""Integration test: train nano config to convergence on TinyShakespeare.

This is the primary end-to-end test.  It verifies that the full pipeline
(tokeniser → model → training loop → checkpoint) works correctly.

Marked as slow — excluded from the fast CI unit test run.
Run with: pytest tests/integration/ -m slow
"""

import pytest


@pytest.mark.slow
@pytest.mark.xfail(reason="Training pipeline not yet implemented — Phase 2")
def test_nano_model_loss_decreases_30_percent_in_200_steps() -> None:
    """Nano model must reduce loss by at least 30% in 200 training steps.

    Rationale: a randomly initialised model has loss ≈ log(vocab_size).
    After 200 steps on a fixed dataset with correct training mechanics,
    loss must drop significantly.  A 30% reduction is a conservative
    threshold — a correct implementation typically achieves 50%+.

    If this test fails, the most likely causes are:
    1. Incorrect causal mask (model leaks future tokens → too-easy training)
    2. Gradient not flowing (loss decreases too slowly)
    3. LR too high (loss spikes or diverges)
    4. Tokeniser producing wrong token IDs
    """
    pass


@pytest.mark.slow
@pytest.mark.xfail(reason="Training pipeline not yet implemented — Phase 2")
def test_nano_model_generates_coherent_text_after_training() -> None:
    """After training, generated text must have the statistical properties
    of the training data (TinyShakespeare).

    Coherence is measured by:
    1. No repeated 5-gram sequences in a 100-token generation
    2. Vocabulary coverage (at least 50 unique tokens in 200-token generation)
    3. No <UNK> tokens in the generation
    """
    pass


@pytest.mark.slow
@pytest.mark.xfail(reason="Training pipeline not yet implemented — Phase 2")
def test_checkpoint_loads_and_resumes_training() -> None:
    """Save a checkpoint at step 100, reload it, train to step 200.
    The loss at step 200 must match a continuous run (no checkpoint).
    """
    pass
