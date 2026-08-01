"""Multi-head causal self-attention, implemented from scratch.

This is the most important module in KAMUI.  Every line is written to be
readable, not optimal.  The goal is that a student can read this file and
understand exactly what multi-head attention does, without consulting any
other source.

Responsibilities:
    - ``scaled_dot_product_attention``:
        The core attention function.  Takes Q, K, V tensors and an optional
        causal mask.  Returns the attention output and the attention weight
        matrix.  The weight matrix is always returned — it is needed by the
        interpretability tools in ``kamui.mechinterp``.

    - ``MultiHeadAttention``:
        Projects the residual stream into Q, K, V for each head, runs
        scaled dot-product attention in parallel across heads, concatenates
        the results, and projects back to ``d_model``.

        Crucially, this module does NOT apply layer normalisation or the
        residual connection — those are the responsibility of ``block.py``.
        Separation of concerns is enforced strictly.

Key equations:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
    where head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)

Implementation notes:
    - Causal mask is applied as -inf (not 0) before softmax.
      Using 0 is a common bug: it does not prevent attention to future
      tokens because exp(0) = 1 != 0.
    - Scaling by 1/sqrt(d_k) prevents dot products from growing large
      in magnitude as d_k increases, which would push softmax into
      regions of near-zero gradient.
    - einops is used for all reshape operations:
        rearrange(x, "b s (h dh) -> b h s dh", h=n_heads)
      Never use .view().transpose() — it is unreadable and error-prone.

Tensor shapes:
    Input (residual stream):  x            (B, S, D)
    Q, K, V (projected):      q/k/v        (B, H, S, Dh)
    Attention scores:         scores       (B, H, S, S)   pre-softmax
    Attention weights:        weights      (B, H, S, S)   post-softmax
    Attention output:         out          (B, S, D)

Mask convention:
    ``mask`` is a boolean tensor that is ``True`` at positions that must NOT
    be attended to (those scores are set to -inf before softmax).

Hook points (attached externally by kamui.hooks):
    "attn.weights"   — attention probabilities (B, H, S, S)
    "attn.output"    — projected output (B, S, D), pre-residual add

References:
    Vaswani et al. (2017). Attention Is All You Need.
    https://arxiv.org/abs/1706.03762

Implemented in: Phase 2E.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import Tensor, nn

from kamui.model.config import ModelConfig
from kamui.model.embedding import RotaryPositionalEncoding


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Implements ``softmax(Q K^T / sqrt(d_k)) V`` with an optional additive
    mask applied (as -inf) to the pre-softmax scores.

    Args:
        q:    Query tensor of shape ``(..., S_q, d_k)``.
        k:    Key tensor of shape ``(..., S_k, d_k)``.
        v:    Value tensor of shape ``(..., S_k, d_v)``.
        mask: Optional boolean tensor broadcastable to ``(..., S_q, S_k)``.
            ``True`` marks positions that must NOT be attended to; their
            scores are set to ``-inf`` before the softmax so they receive
            exactly zero weight.

    Returns:
        A tuple ``(output, weights)`` where:
            - ``output`` has shape ``(..., S_q, d_v)``.
            - ``weights`` has shape ``(..., S_q, S_k)`` and is the post-softmax
              attention probability matrix (always returned, for interpretability).
    """
    d_k = q.shape[-1]
    # (..., S_q, d_k) @ (..., d_k, S_k) -> (..., S_q, S_k)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # -inf so that exp(score) = 0 for masked positions (0 would NOT work).
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention over the residual stream.

    Projects the input into per-head queries, keys and values, runs scaled
    dot-product attention with a causal mask (each position attends only to
    itself and earlier positions), concatenates the heads, and projects back
    to ``d_model``.

    No normalisation and no residual connection are applied here — both are
    owned by the enclosing transformer block (Pre-LN convention).

    Attributes:
        n_heads:    Number of attention heads.
        d_head:     Per-head dimension (``d_model // n_heads``).
        q_proj:     Query projection ``d_model -> d_model``.
        k_proj:     Key projection ``d_model -> d_model``.
        v_proj:     Value projection ``d_model -> d_model``.
        out_proj:   Output projection ``d_model -> d_model`` (W_O).
        dropout:    Dropout applied to the projected output.
        rope:       A ``RotaryPositionalEncoding`` applied to Q/K when
            ``config.positional_encoding == "rope"``, else ``None``.
        causal_mask: Boolean ``(context_length, context_length)`` buffer,
            ``True`` above the diagonal (future positions).
    """

    causal_mask: Tensor

    def __init__(self, config: ModelConfig) -> None:
        """Build multi-head attention from a model configuration.

        Args:
            config: The model configuration.  ``d_model`` and ``n_heads`` (with
                the invariant ``d_model % n_heads == 0`` enforced by
                ``ModelConfig``) determine the head geometry; ``context_length``
                sizes the causal mask; ``dropout`` sets the output dropout.
        """
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_head  # d_model // n_heads (validated by ModelConfig)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Rotary positional encoding (RoPE) is applied to Q/K when selected;
        # otherwise positions come from the additive embedding (learned/sinusoidal).
        self.rope: RotaryPositionalEncoding | None
        if config.positional_encoding == "rope":
            self.rope = RotaryPositionalEncoding(self.d_head, config.context_length)
        else:
            self.rope = None

        # Precompute the causal mask once: True above the diagonal means a
        # query at position i may not attend to key positions j > i.
        causal = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal)

    def forward(self, x: Tensor, return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Apply multi-head causal self-attention.

        Args:
            x:              Residual-stream tensor of shape ``(B, S, d_model)``.
            return_weights: If True, also return the attention probability
                matrix of shape ``(B, n_heads, S, S)``.

        Returns:
            ``out`` of shape ``(B, S, d_model)`` if ``return_weights`` is False,
            otherwise ``(out, weights)``.

        Raises:
            TypeError:  If ``x`` is not a tensor.
            ValueError: If ``x`` is not 3-D, its last dimension is not
                ``d_model``, or ``S`` exceeds ``context_length``.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"x must be 3-D (B, S, d_model), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"last dimension of x ({x.shape[-1]}) does not match "
                f"d_model ({self.config.d_model})"
            )

        seq_len = x.shape[1]
        if seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length ({seq_len}) exceeds context_length "
                f"({self.config.context_length})"
            )

        # Project, then split d_model into (heads, head_dim): (B, H, S, Dh).
        q = rearrange(self.q_proj(x), "b s (h dh) -> b h s dh", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b s (h dh) -> b h s dh", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b s (h dh) -> b h s dh", h=self.n_heads)

        # Rotary encoding rotates Q and K by their position (V is left untouched).
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Slice the causal mask to the current sequence length; broadcasts over
        # batch and heads inside scaled_dot_product_attention.
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_out, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # Concatenate heads back into d_model and project.
        attn_out = rearrange(attn_out, "b h s dh -> b s (h dh)")
        out = self.dropout(self.out_proj(attn_out))

        if return_weights:
            return out, weights
        return out

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(d_model={self.config.d_model}, "
            f"n_heads={self.n_heads}, d_head={self.d_head}, "
            f"dropout={self.config.dropout})"
        )
