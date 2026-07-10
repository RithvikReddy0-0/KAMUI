"""AdamW optimiser configuration with correct weight decay separation.

Responsibilities:
    - ``build_optimizer(model, ...) -> torch.optim.AdamW``:
        Construct an AdamW optimiser with two parameter groups:

        Group 1 — weight decay applied:
            All weight matrices (linear layers, embedding tables); ndim >= 2.

        Group 2 — weight decay NOT applied:
            All biases, LayerNorm γ and β parameters; ndim < 2.

Why separate parameter groups?
    AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the
    gradient update.  Applying weight decay to biases and normalisation
    parameters is incorrect: biases should be unconstrained, and LayerNorm
    scale/shift parameters should not be shrunk toward zero.  The separation
    is subtle but measurably improves training stability and final perplexity.
    This is the approach used by nanoGPT.

References:
    Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay
    Regularization. ICLR 2019. https://arxiv.org/abs/1711.05101

Implemented in: Phase 5.
"""

from __future__ import annotations

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build an AdamW optimiser with decayed / non-decayed parameter groups.

    Parameters with ``ndim >= 2`` (weight matrices, embedding tables) receive
    weight decay; parameters with ``ndim < 2`` (biases, LayerNorm scale/shift)
    do not.  Only trainable parameters are included, and each is placed in
    exactly one group (weight-tied parameters are de-duplicated by identity).

    Args:
        model:        The model to optimise.
        lr:           Initial learning rate (> 0).  Overridden per step by the
            scheduler in the training loop.
        weight_decay: Weight decay for the decayed group (>= 0).
        betas:        AdamW betas.
        eps:          AdamW epsilon.

    Returns:
        A configured ``torch.optim.AdamW``.

    Raises:
        ValueError: If ``lr`` or ``weight_decay`` is out of range.
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

    # ``parameters()`` already yields each shared (weight-tied) parameter once.
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
