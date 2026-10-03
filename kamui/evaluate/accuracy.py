"""Next-token accuracy: how often the model's top guesses contain the right token.

Perplexity summarises the whole predicted distribution; accuracy answers the
simpler question people usually ask first — *did the model get it right?*

    top-k accuracy = fraction of positions whose target token is among the
                     model's k highest-scoring predictions

Top-1 accuracy is plain "argmax is correct"; larger k is more forgiving.  For
any fixed data, accuracy can only rise (or stay flat) as k grows.

Responsibilities:
    - ``compute_accuracy(model, dataloader, k)``:
        Top-k next-token accuracy over a dataset, accepting the same batch
        formats as ``compute_perplexity``.

Implemented in: v0.4.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from kamui.evaluate.perplexity import _split_batch
from kamui.model.transformer import KAMUITransformer


@torch.no_grad()
def compute_accuracy(
    model: KAMUITransformer,
    dataloader: Iterable[tuple[Tensor, Tensor] | Tensor],
    k: int = 1,
) -> float:
    """Compute top-k next-token accuracy over a dataset.

    Args:
        model:      A ``KAMUITransformer``.
        dataloader: An iterable of batches.  Each batch is either a
            ``(inputs, targets)`` pair (both ``(B, S)``) or a single ``(B, S)``
            token tensor (split into next-token inputs/targets internally).
        k:          Count a position as correct if the target is among the top
            ``k`` predictions (clamped to the vocabulary size).

    Returns:
        The fraction of positions predicted correctly, in ``[0, 1]``.

    Raises:
        ValueError: If ``k < 1`` or the dataloader yields no tokens.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    for batch in dataloader:
        inputs, targets = _split_batch(batch)
        logits = model(inputs)  # (B, S, V)
        top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices  # (B, S, k)
        correct += int((top == targets.unsqueeze(-1)).any(dim=-1).sum().item())
        total += targets.numel()

    if was_training:
        model.train()
    if total == 0:
        raise ValueError("dataloader produced no tokens")
    return correct / total
