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

Implemented in: Phase 1, Week 5
"""

# Implementation begins in Phase 1.
