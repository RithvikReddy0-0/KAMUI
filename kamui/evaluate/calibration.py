"""Calibration metrics: does the model's confidence match its accuracy?

A model is well-calibrated if, among all predictions assigned probability p,
approximately fraction p are correct.  Miscalibrated models are overconfident
(high probability, often wrong) or underconfident (low probability, often
right).

Responsibilities:
    - ``expected_calibration_error(probs, labels, n_bins)``:
        ECE — partition top-1 predictions into confidence bins and measure
        the confidence-weighted gap between mean confidence and accuracy.

    - ``reliability_diagram(probs, labels, n_bins)``:
        Confidence vs. accuracy bar plot with the diagonal as the perfect
        reference; deviations indicate miscalibration.

    - ``temperature_scaling(model, val_dataloader)``:
        Post-hoc calibration: find the temperature T minimising NLL on a
        validation set when logits are divided by T.  T > 1 means the model
        was overconfident.

Why calibration matters for interpretability:
    Interpretability experiments compare model confidence before and after
    interventions.  If the model is poorly calibrated, raw probability
    changes are hard to interpret — calibration establishes what the model's
    confidence actually means.

Implemented in: Phase 4.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from kamui.model.transformer import KAMUITransformer

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _validate_probs_labels(probs: Tensor, labels: Tensor, n_bins: int) -> None:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if probs.dim() != 2:
        raise ValueError(f"probs must be 2-D (N, C), got shape {tuple(probs.shape)}")
    if labels.dim() != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError(
            f"labels must be 1-D with length {probs.shape[0]}, got shape {tuple(labels.shape)}"
        )


def _bin_stats(probs: Tensor, labels: Tensor, n_bins: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-bin (confidence, accuracy, count) for top-1 predictions."""
    confidence, prediction = probs.max(dim=-1)
    correct = (prediction == labels).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    conf_mean = torch.zeros(n_bins)
    acc_mean = torch.zeros(n_bins)
    counts = torch.zeros(n_bins)
    for b in range(n_bins):
        # Left-open bins except the first, so p=1.0 lands in the last bin.
        lo, hi = edges[b], edges[b + 1]
        mask = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence <= hi)
        counts[b] = mask.sum()
        if counts[b] > 0:
            conf_mean[b] = confidence[mask].mean()
            acc_mean[b] = correct[mask].mean()
    return conf_mean, acc_mean, counts


def expected_calibration_error(probs: Tensor, labels: Tensor, n_bins: int = 15) -> float:
    """Compute the expected calibration error of top-1 predictions.

    Args:
        probs:  ``(N, C)`` probability distributions.
        labels: ``(N,)`` integer ground-truth labels.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The ECE in ``[0, 1]`` (0 = perfectly calibrated).

    Raises:
        ValueError: If shapes are inconsistent or ``n_bins < 1``.
    """
    _validate_probs_labels(probs, labels, n_bins)
    conf_mean, acc_mean, counts = _bin_stats(probs, labels, n_bins)
    total = counts.sum()
    return float((counts / total * (conf_mean - acc_mean).abs()).sum().item())


def reliability_diagram(probs: Tensor, labels: Tensor, n_bins: int = 15) -> Figure:
    """Plot per-bin confidence vs. accuracy with a perfect-calibration diagonal.

    Args:
        probs:  ``(N, C)`` probability distributions.
        labels: ``(N,)`` integer ground-truth labels.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If shapes are inconsistent or ``n_bins < 1``.
    """
    import matplotlib.pyplot as plt

    _validate_probs_labels(probs, labels, n_bins)
    conf_mean, acc_mean, counts = _bin_stats(probs, labels, n_bins)
    centers = (torch.arange(n_bins) + 0.5) / n_bins
    occupied = counts > 0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.bar(
        centers[occupied].numpy(),
        acc_mean[occupied].numpy(),
        width=1.0 / n_bins,
        edgecolor="black",
        color="steelblue",
        label="accuracy",
    )
    ece = expected_calibration_error(probs, labels, n_bins)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


@torch.no_grad()
def temperature_scaling(
    model: KAMUITransformer,
    val_dataloader: Iterable[tuple[Tensor, Tensor] | Tensor],
    temperatures: Tensor | None = None,
) -> float:
    """Find the temperature that minimises validation NLL.

    Logits are divided by each candidate temperature; the argmin of the
    resulting NLL over the validation set is returned.  ``T > 1`` means the
    model was overconfident, ``T < 1`` underconfident.

    Args:
        model:          A ``KAMUITransformer``.
        val_dataloader: Iterable of ``(inputs, targets)`` pairs (both
            ``(B, S)``) or plain ``(B, S)`` token tensors (shifted internally).
        temperatures:   Candidate grid (defaults to 0.05…5.0).

    Returns:
        The optimal temperature as a float.

    Raises:
        ValueError: If the dataloader yields no tokens.
    """
    if temperatures is None:
        temperatures = torch.linspace(0.05, 5.0, steps=100)

    was_training = model.training
    model.eval()

    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []
    for batch in val_dataloader:
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        all_logits.append(logits.reshape(-1, logits.shape[-1]))
        all_targets.append(targets.reshape(-1))
    if was_training:
        model.train()
    if not all_logits:
        raise ValueError("val_dataloader produced no tokens")

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    best_t = 1.0
    best_nll = float("inf")
    for t in temperatures.tolist():
        nll = F.cross_entropy(logits / t, targets).item()
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t
