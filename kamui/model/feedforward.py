"""Position-wise feed-forward network (FFN), implemented from scratch.

The FFN is applied independently to each token position.  It is the
"computation" counterpart to attention's "communication": attention moves
information between positions; the FFN processes each position's
representation in isolation.

Responsibilities:
    - ``FeedForward``:
        Two linear transformations with a GELU activation in between:

            FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2

        Where:
            W_1 ∈ R^(D × F),  b_1 ∈ R^F     (expansion layer)
            W_2 ∈ R^(F × D),  b_2 ∈ R^D     (projection layer)
            F = 4 * D  (standard expansion factor)

Design notes:
    Why GELU over ReLU?
        GELU (Gaussian Error Linear Unit) is smooth and has a small
        negative region, providing a soft learned gating effect.
        All modern transformers (GPT-2 onwards, LLaMA, Mistral) use GELU
        or its fast approximation.  ReLU is simpler but performs
        measurably worse on language modelling at equivalent scale.

    Why 4× expansion?
        The 4× factor is empirical and has been stable since the original
        transformer paper.  It gives the FFN sufficient capacity to act as
        a key-value memory (Geva et al., 2021) while keeping parameter
        count manageable.

    This module does NOT apply layer normalisation or the residual
    connection — those are the responsibility of ``block.py``.

Tensor shapes:
    Input:   x        (B, S, D)     residual stream
    Hidden:  h        (B, S, F)     after W_1 + GELU
    Output:  out      (B, S, D)     after W_2

Hook points exposed:
    "blocks.{L}.ffn.mid"     — hidden activation (B, S, F), post-GELU
    "blocks.{L}.ffn.output"  — FFN output (B, S, D), pre-residual add

References:
    Geva, M. et al. (2021). Transformer Feed-Forward Layers Are Key-Value
    Memories. EMNLP 2021. https://arxiv.org/abs/2012.14913

Implemented in: Phase 1, Week 7
"""

# Implementation begins in Phase 1.
