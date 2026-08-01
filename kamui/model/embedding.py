"""Token and positional embedding layers.

Responsibilities:
    - ``TokenEmbedding``:
        A learnable lookup table mapping integer token IDs to dense vectors
        of shape (B, S, D).  Implemented as a raw ``nn.Parameter`` matrix
        rather than ``nn.Embedding`` so the weight matrix is explicitly
        visible and its initialisation is fully controlled.

    - ``SinusoidalPositionalEncoding``:
        Fixed (non-learnable) positional encodings from Vaswani et al.
        (2017).  Encodes position as a sum of sine and cosine waves at
        geometrically spaced frequencies.  Allows extrapolation beyond the
        training context length in principle.

    - ``LearnedPositionalEncoding``:
        A learnable lookup table over positions, identical in structure to
        ``TokenEmbedding`` but indexed by position rather than token ID.
        Used by GPT-2 and all its descendants.  Performs marginally better
        than sinusoidal in practice but cannot extrapolate.

    - ``Embedding`` (combined):
        The public-facing module used by ``KAMUITransformer``.  Adds token
        and positional embeddings, applies dropout (if configured), and
        returns the combined embedding tensor of shape (B, S, D).
        The choice between sinusoidal and learned positional encoding is
        controlled by ``ModelConfig.positional_encoding``.

Design notes:
    Both positional encoding variants are implemented so that students can
    switch between them via config and observe the difference in attention
    patterns and perplexity.

    Output shape invariant: regardless of which positional encoding is
    used, the output is always (B, S, D) — the residual stream shape.

Key equations:
    PE(pos, 2i)   = sin(pos / 10000^(2i/D))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/D))

    combined = token_embedding(ids) + positional_encoding(positions)

Tensor shapes:
    Input:  token_ids      (B, S)        int64
    Output: embeddings     (B, S, D)     float32

Hook point:
    "embed.output" — the combined embedding before any transformer block

Implemented in: Phase 2A.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from kamui.model.config import ModelConfig

#: Default standard deviation for normal-distribution weight init (GPT-2 convention).
_DEFAULT_INIT_STD: float = 0.02

#: Base wavelength constant for sinusoidal positional encoding (Vaswani et al. 2017).
_SINUSOID_BASE: float = 10000.0

#: Integer tensor dtypes accepted as token IDs.
_INT_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


class TokenEmbedding(nn.Module):
    """Learnable token-ID → vector lookup table.

    Backed by a raw ``nn.Parameter`` of shape ``(vocab_size, d_model)`` instead
    of ``nn.Embedding`` so the weight matrix is directly inspectable and its
    initialisation is under explicit control.  The forward pass is a single
    advanced-indexing operation: ``weight[token_ids]``.

    Attributes:
        vocab_size: Number of rows (one per token).
        d_model:    Embedding dimension (residual stream width).
        weight:     The ``(vocab_size, d_model)`` embedding matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        init_std: float = _DEFAULT_INIT_STD,
    ) -> None:
        """Create a token embedding table.

        Args:
            vocab_size: Number of tokens.  Must be > 0.
            d_model:    Embedding dimension.  Must be > 0.
            init_std:   Standard deviation of the ``N(0, init_std)`` weight init.

        Raises:
            ValueError: If ``vocab_size`` or ``d_model`` is not positive.
        """
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Look up the embedding vector for each token ID.

        Args:
            token_ids: Integer tensor of token IDs, any shape ``(...)``.

        Returns:
            Float tensor of shape ``(..., d_model)``.

        Raises:
            TypeError:  If ``token_ids`` is not a tensor, or not an integer dtype.
            ValueError: If any ID is outside ``[0, vocab_size)``.
        """
        if not isinstance(token_ids, Tensor):
            raise TypeError(f"token_ids must be a torch.Tensor, got {type(token_ids)}")
        if token_ids.dtype not in _INT_DTYPES:
            raise TypeError(f"token_ids must have an integer dtype, got {token_ids.dtype}")
        if ((token_ids < 0) | (token_ids >= self.vocab_size)).any():
            raise ValueError(f"token_ids contains values outside [0, {self.vocab_size})")
        return self.weight[token_ids]

    def __repr__(self) -> str:
        return f"TokenEmbedding(vocab_size={self.vocab_size}, d_model={self.d_model})"


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    The encoding for position ``pos`` and dimension ``i`` is::

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Values are precomputed once up to ``context_length`` and stored in a
    non-learnable buffer, so they carry no gradients but still move with the
    module across devices and serialise in the ``state_dict``.

    Attributes:
        context_length: Number of precomputed positions.
        d_model:        Encoding dimension.
        pe:             ``(context_length, d_model)`` buffer of encodings.
    """

    pe: Tensor

    def __init__(self, context_length: int, d_model: int) -> None:
        """Precompute the positional-encoding table.

        Args:
            context_length: Maximum sequence length.  Must be > 0.
            d_model:        Encoding dimension.  Must be > 0.

        Raises:
            ValueError: If ``context_length`` or ``d_model`` is not positive.
        """
        super().__init__()
        if context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {context_length}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        self.context_length = context_length
        self.d_model = d_model

        position = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
        # Frequencies for the even dimensions: 10000^(-2i/d_model).
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(_SINUSOID_BASE) / d_model)
        )
        pe = torch.zeros(context_length, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        # When d_model is odd there is one fewer cosine column than sine column.
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> Tensor:
        """Return the positional encodings for the first ``seq_len`` positions.

        Args:
            seq_len: Number of positions to return.  Must be in
                ``[1, context_length]``.

        Returns:
            Float tensor of shape ``(seq_len, d_model)``.

        Raises:
            ValueError: If ``seq_len`` is not in ``[1, context_length]``.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if seq_len > self.context_length:
            raise ValueError(f"seq_len ({seq_len}) exceeds context_length ({self.context_length})")
        return self.pe[:seq_len]

    def __repr__(self) -> str:
        return (
            f"SinusoidalPositionalEncoding("
            f"context_length={self.context_length}, d_model={self.d_model})"
        )


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding (GPT-2 style).

    Structurally identical to ``TokenEmbedding`` but indexed by position.  A
    learnable ``(context_length, d_model)`` matrix; row ``p`` is the encoding
    for position ``p``.  Cannot represent positions beyond ``context_length``.

    Attributes:
        context_length: Number of learnable positions.
        d_model:        Encoding dimension.
        weight:         ``(context_length, d_model)`` position matrix.
    """

    def __init__(
        self,
        context_length: int,
        d_model: int,
        init_std: float = _DEFAULT_INIT_STD,
    ) -> None:
        """Create a learned positional-encoding table.

        Args:
            context_length: Maximum sequence length.  Must be > 0.
            d_model:        Encoding dimension.  Must be > 0.
            init_std:       Standard deviation of the ``N(0, init_std)`` init.

        Raises:
            ValueError: If ``context_length`` or ``d_model`` is not positive.
        """
        super().__init__()
        if context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {context_length}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        self.context_length = context_length
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(context_length, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, seq_len: int) -> Tensor:
        """Return the learned encodings for the first ``seq_len`` positions.

        Args:
            seq_len: Number of positions to return.  Must be in
                ``[1, context_length]``.

        Returns:
            Float tensor of shape ``(seq_len, d_model)``.

        Raises:
            ValueError: If ``seq_len`` is not in ``[1, context_length]``.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if seq_len > self.context_length:
            raise ValueError(f"seq_len ({seq_len}) exceeds context_length ({self.context_length})")
        return self.weight[:seq_len]

    def __repr__(self) -> str:
        return (
            f"LearnedPositionalEncoding("
            f"context_length={self.context_length}, d_model={self.d_model})"
        )


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE; Su et al. 2021), applied to Q/K.

    Instead of *adding* a positional vector to the embedding, RoPE *rotates*
    each query/key vector by an angle proportional to its position.  The dot
    product between a rotated query at position ``m`` and a rotated key at
    position ``n`` then depends only on the relative offset ``m - n``, giving
    the model translation-equivariant positional information that extrapolates
    better than learned embeddings.  This is the scheme used by LLaMA / Mistral.

    Operates on tensors of shape ``(..., S, d_head)`` with an even ``d_head``.
    The cos/sin tables are precomputed once up to ``context_length`` and stored
    as non-learnable buffers.

    Key equations (half-split layout):
        θ_i        = base^(-2i / d_head),  i = 0 .. d_head/2 - 1
        RoPE(x)_p  = x_p · cos(p·θ) + rotate_half(x_p) · sin(p·θ)
        rotate_half([x1, x2]) = [-x2, x1]   (x1, x2 the two halves of d_head)

    Attributes:
        d_head:         Per-head dimension (must be even).
        context_length: Number of precomputed positions.
        cos, sin:       ``(context_length, d_head)`` angle buffers.

    References:
        Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary
        Position Embedding. https://arxiv.org/abs/2104.09864
    """

    cos: Tensor
    sin: Tensor

    def __init__(self, d_head: int, context_length: int, base: float = _SINUSOID_BASE) -> None:
        """Precompute the rotary angle tables.

        Args:
            d_head:         Per-head dimension.  Must be > 0 and even.
            context_length: Maximum sequence length.  Must be > 0.
            base:           Rotary base wavelength (default 10000).

        Raises:
            ValueError: If ``d_head`` is not a positive even integer, or
                ``context_length`` is not positive.
        """
        super().__init__()
        if d_head <= 0 or d_head % 2 != 0:
            raise ValueError(f"d_head must be a positive even integer, got {d_head}")
        if context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {context_length}")

        self.d_head = d_head
        self.context_length = context_length

        # One inverse frequency per rotation plane (there are d_head / 2 planes).
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )  # (d_head / 2,)
        positions = torch.arange(context_length, dtype=torch.float32)  # (S,)
        angles = torch.outer(positions, inv_freq)  # (S, d_head / 2)
        emb = torch.cat([angles, angles], dim=-1)  # (S, d_head) — half-split layout
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Return ``[-x2, x1]`` where ``x1, x2`` are the two halves of the last dim."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Rotate ``x`` by its per-position angle.

        Args:
            x: Tensor of shape ``(..., S, d_head)`` (typically ``(B, H, S, Dh)``).

        Returns:
            The rotated tensor, same shape as ``x``.

        Raises:
            ValueError: If the last dim is not ``d_head`` or ``S`` exceeds
                ``context_length``.
        """
        if x.shape[-1] != self.d_head:
            raise ValueError(
                f"last dimension of x ({x.shape[-1]}) does not match d_head ({self.d_head})"
            )
        seq_len = x.shape[-2]
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length ({seq_len}) exceeds context_length ({self.context_length})"
            )
        cos = self.cos[:seq_len]  # (S, d_head), broadcasts over leading dims
        sin = self.sin[:seq_len]
        return x * cos + self._rotate_half(x) * sin

    def __repr__(self) -> str:
        return (
            f"RotaryPositionalEncoding(d_head={self.d_head}, "
            f"context_length={self.context_length})"
        )


class Embedding(nn.Module):
    """Combined token + positional embedding for the transformer input.

    Adds a token embedding and a positional encoding, then applies dropout.
    The positional-encoding variant is selected by
    ``ModelConfig.positional_encoding``:

    - ``"learned"`` / ``"sinusoidal"``: a positional vector is added here.
    - ``"rope"``: no positional vector is added — rotary encoding is applied to
      Q/K inside attention instead, so this module only embeds tokens.

    Output is the residual-stream tensor of shape ``(B, S, D)``.

    Attributes:
        token_embedding:     The ``TokenEmbedding`` lookup table.
        positional_encoding: A ``LearnedPositionalEncoding`` /
                             ``SinusoidalPositionalEncoding`` module, or ``None``
                             for RoPE.
        dropout:             ``nn.Dropout`` applied to the (summed) embeddings.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the combined embedding from a model configuration.

        Args:
            config: The model configuration.  ``positional_encoding`` selects
                the encoding variant; ``vocab_size``, ``d_model``,
                ``context_length`` and ``dropout`` size the layers.
        """
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)

        self.positional_encoding: nn.Module | None
        if config.positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(
                config.context_length, config.d_model
            )
        elif config.positional_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                config.context_length, config.d_model
            )
        else:  # "rope" — positions are injected in attention, not added here.
            self.positional_encoding = None

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed a batch of token-ID sequences.

        Args:
            token_ids: Integer tensor of shape ``(B, S)``.

        Returns:
            Float tensor of shape ``(B, S, d_model)`` — the residual stream.

        Raises:
            TypeError:  If ``token_ids`` is not a tensor.
            ValueError: If ``token_ids`` is not 2-D, or ``S`` exceeds the
                configured ``context_length``.
        """
        if not isinstance(token_ids, Tensor):
            raise TypeError(f"token_ids must be a torch.Tensor, got {type(token_ids)}")
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2-D (B, S), got shape {tuple(token_ids.shape)}")

        seq_len = token_ids.shape[1]
        if seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length ({seq_len}) exceeds context_length "
                f"({self.config.context_length})"
            )

        tok = self.token_embedding(token_ids)  # (B, S, D)
        if self.positional_encoding is None:
            # RoPE: positions are injected in attention, so nothing is added here.
            return self.dropout(tok)
        # (B, S, D) token vectors plus (S, D) position vectors, broadcast over batch.
        pos = self.positional_encoding(seq_len)
        return self.dropout(tok + pos)

    def __repr__(self) -> str:
        return (
            f"Embedding(vocab_size={self.config.vocab_size}, "
            f"d_model={self.config.d_model}, "
            f"context_length={self.config.context_length}, "
            f"positional_encoding='{self.config.positional_encoding}')"
        )
