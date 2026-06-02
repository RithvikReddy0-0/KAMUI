"""Integration test: end-to-end interpretability pipeline.

Verifies that the full pipeline — train → hook → interpret — works
without errors and produces outputs of the correct type and shape.

These tests do not verify research findings (those are in research/experiments/).
They verify that the code runs correctly end-to-end.
"""

import pytest


@pytest.mark.slow
@pytest.mark.xfail(reason="Interpretability tools not yet implemented — Phase 4")
def test_logit_lens_runs_without_error() -> None:
    """LogitLens.run() must complete without error and return a LogitLensResult
    with the correct shape: probs of shape (n_layers+1, S, V).
    """
    pass


@pytest.mark.slow
@pytest.mark.xfail(reason="Interpretability tools not yet implemented — Phase 4")
def test_activation_patcher_effect_in_range() -> None:
    """ActivationPatcher effect scores must be in [0, 1] for all layers."""
    pass


@pytest.mark.slow
@pytest.mark.xfail(reason="Interpretability tools not yet implemented — Phase 4")
def test_induction_head_scores_in_range() -> None:
    """InductionHeadDetector scores must be in [0, 1] for all (layer, head) pairs."""
    pass


@pytest.mark.slow
@pytest.mark.xfail(reason="Interpretability tools not yet implemented — Phase 4")
def test_circuit_ablation_degrades_performance() -> None:
    """Ablating all heads in the model must result in significantly higher loss
    than the baseline (unablated) model.
    """
    pass
