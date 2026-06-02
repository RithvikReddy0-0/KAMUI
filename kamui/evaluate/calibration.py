"""Calibration metrics: does the model's confidence match its accuracy?

A model is well-calibrated if, among all predictions assigned probability p,
approximately fraction p are correct.  Miscalibrated models are overconfident
(high probability, often wrong) or underconfident (low probability, often right).

Responsibilities:
    - ``expected_calibration_error(probs, labels, n_bins=15) -> float``:
        Compute ECE: partition predictions into probability bins, measure
        the average gap between mean confidence and mean accuracy per bin.

    - ``reliability_diagram(probs, labels, n_bins=15) -> Figure``:
        Plot confidence (x-axis) vs. accuracy (y-axis) with a diagonal
        reference line.  Deviations from the diagonal indicate miscalibration.

    - ``temperature_scaling(model, val_dataloader) -> float``:
        Post-hoc calibration: find the temperature T that minimises NLL
        on the validation set when logits are divided by T.
        Returns the optimal temperature.  T > 1 = model was overconfident.

Why calibration matters for interpretability:
    Interpretability experiments often compare model confidence before and
    after an intervention (ablation, patching).  If the model is poorly
    calibrated, raw probability changes are hard to interpret.  Calibration
    analysis establishes a baseline understanding of what the model's
    confidence actually means.

Implemented in: Phase 4 (after training is complete)
"""

# Implementation begins in Phase 4.
