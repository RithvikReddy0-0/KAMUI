"""Perplexity computation: token-level and sequence-level.

Perplexity is the primary metric for evaluating language models.  It
measures how surprised the model is by the held-out data:

    Perplexity = exp(mean cross-entropy loss over all tokens)

Lower perplexity = better model.  A perplexity of N means the model is
as confused as if it were choosing uniformly among N options at each step.

Responsibilities:
    - ``compute_perplexity(model, dataloader) -> float``:
        Compute token-level perplexity over a full dataset.
        Runs without gradient computation.

    - ``compute_token_loss(model, token_ids) -> Tensor``:
        Return per-token cross-entropy loss of shape (S-1,).
        Useful for identifying which tokens are hardest to predict.

    - ``compute_sequence_perplexity(model, token_ids) -> float``:
        Compute perplexity for a single sequence, using a sliding window for
        sequences longer than ``context_length`` (Press et al., 2021) so each
        counted token is scored with as much context as the model allows.

Implemented in: Phase 4.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _as_sequence(token_ids: Tensor) -> Tensor:
    """Return a 1-D tensor of token IDs from a (S,) or (1, S) input."""
    if not isinstance(token_ids, Tensor):
        raise TypeError(f"token_ids must be a torch.Tensor, got {type(token_ids)}")
    if token_ids.dim() == 2 and token_ids.shape[0] == 1:
        token_ids = token_ids[0]
    if token_ids.dim() != 1:
        raise ValueError(
            f"token_ids must be 1-D (S,) or (1, S), got shape {tuple(token_ids.shape)}"
        )
    if token_ids.shape[0] < 2:
        raise ValueError("need at least 2 tokens to compute a next-token loss")
    return token_ids


@torch.no_grad()
def compute_token_loss(model: nn.Module, token_ids: Tensor) -> Tensor:
    """Return the per-token next-token cross-entropy loss for one sequence.

    Args:
        model:     A ``KAMUITransformer`` (or compatible) returning logits.
        token_ids: Token IDs of shape ``(S,)`` or ``(1, S)`` with ``S >= 2``
            and ``S - 1 <= context_length``.

    Returns:
        A tensor of shape ``(S-1,)`` — the loss of predicting each token from
        its predecessors.

    Raises:
        TypeError:  If ``token_ids`` is not a tensor.
        ValueError: If ``token_ids`` has the wrong rank or fewer than 2 tokens.
    """
    ids = _as_sequence(token_ids)
    was_training = model.training
    model.eval()
    inputs = ids[:-1].unsqueeze(0)     # (1, S-1)
    targets = ids[1:].unsqueeze(0)     # (1, S-1)
    logits = model(inputs)             # (1, S-1, V)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    if was_training:
        model.train()
    return loss


@torch.no_grad()
def compute_sequence_perplexity(
    model: nn.Module, token_ids: Tensor, stride: int | None = None
) -> float:
    """Compute perplexity for a single sequence.

    For sequences whose prediction length fits in the context window this is
    ``exp(mean per-token loss)``.  Longer sequences use a sliding window so
    each target token is counted exactly once, scored with up to
    ``context_length`` tokens of preceding context.

    Args:
        model:     A ``KAMUITransformer``.
        token_ids: Token IDs of shape ``(S,)`` or ``(1, S)``.
        stride:    Sliding-window step (defaults to ``context_length // 2``).

    Returns:
        The perplexity as a float.
    """
    ids = _as_sequence(token_ids)
    length = ids.shape[0]
    ctx = model.config.context_length

    if length - 1 <= ctx:
        return math.exp(compute_token_loss(model, ids).mean().item())

    if stride is None:
        stride = max(1, ctx // 2)

    vocab = model.config.vocab_size
    was_training = model.training
    model.eval()

    nll_sum = 0.0
    n_counted = 0
    prev_end = 0
    for begin in range(0, length, stride):
        end = min(begin + ctx, length)
        window = ids[begin:end]
        if window.shape[0] < 2:
            break
        inputs = window[:-1].unsqueeze(0)
        targets = window[1:].clone().unsqueeze(0)
        n_targets = targets.shape[1]
        # Count only the tokens that are new since the previous window.
        count = min(end - prev_end, n_targets)
        targets[:, : n_targets - count] = -100  # ignore already-counted tokens
        logits = model(inputs)
        nll_sum += F.cross_entropy(
            logits.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        ).item()
        n_counted += count
        prev_end = end
        if end == length:
            break

    if was_training:
        model.train()
    return math.exp(nll_sum / n_counted)


def _split_batch(batch: object) -> tuple[Tensor, Tensor]:
    """Return ``(inputs, targets)`` from a batch.

    A ``(inputs, targets)`` pair is used directly; a single ``(B, S)`` tensor
    is split into next-token ``inputs``/``targets`` by an internal shift.
    """
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    if isinstance(batch, Tensor):
        return batch[:, :-1], batch[:, 1:]
    raise TypeError(
        f"batch must be a (inputs, targets) pair or a tensor, got {type(batch)}"
    )


@torch.no_grad()
def compute_perplexity(model: nn.Module, dataloader: object) -> float:
    """Compute token-level perplexity over a dataset.

    Args:
        model:      A ``KAMUITransformer``.
        dataloader: An iterable of batches.  Each batch is either a
            ``(inputs, targets)`` pair (both ``(B, S)``) or a single ``(B, S)``
            token tensor (split into next-token inputs/targets internally).

    Returns:
        The corpus perplexity as a float.

    Raises:
        ValueError: If the dataloader yields no tokens.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        inputs, targets = _split_batch(batch)
        logits = model(inputs)
        total_loss += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        ).item()
        total_tokens += targets.numel()

    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("dataloader produced no tokens")
    return math.exp(total_loss / total_tokens)
