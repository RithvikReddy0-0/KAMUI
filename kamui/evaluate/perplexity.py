"""Perplexity computation: token-level and sequence-level.

Perplexity is the primary metric for evaluating language models.  It
measures how surprised the model is by the held-out data:

    Perplexity = exp(mean cross-entropy loss over all tokens)

Lower perplexity = better model.  A perplexity of N means the model is
as confused as if it were choosing uniformly among N options at each step.

Responsibilities:
    - ``compute_perplexity(model, dataloader) -> float``:
        Compute token-level perplexity over a full dataset.
        Runs without gradient computation (torch.no_grad()).

    - ``compute_token_loss(model, token_ids) -> Tensor``:
        Return per-token cross-entropy loss of shape (S,).
        Useful for identifying which tokens are hardest to predict.

    - ``compute_sequence_perplexity(model, token_ids) -> float``:
        Compute perplexity for a single sequence.
        Used in interpretability experiments to measure the effect of
        component ablations on specific sequences.

Implementation note:
    For sequences longer than ``context_length``, perplexity must be
    computed with a sliding window: the model is run on overlapping
    chunks and only the loss on the new tokens in each chunk is counted.
    This is the standard approach (Press et al., 2021) and avoids the
    artificially low perplexity that results from always having the
    full context available for the first chunk's tokens.

Implemented in: Phase 2, Week 11 (used during training evaluation)
"""

# Implementation begins in Phase 2.
