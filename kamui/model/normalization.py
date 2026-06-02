"""Layer normalisation and RMS normalisation, implemented from scratch.

Responsibilities:
    - ``LayerNorm``:
        Normalises activations across the feature dimension (D) for each
        token position independently.

            LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

        Where μ and σ are computed over the D-dimensional feature vector
        at each (batch, position) independently.  γ (scale) and β (bias)
        are learnable parameters of shape (D,).

    - ``RMSNorm``:
        A simplified variant that omits the mean-centering step.
        Used by LLaMA and Mistral because it is faster and performs
        equivalently in practice.

            RMSNorm(x) = γ · x / RMS(x),   RMS(x) = sqrt(mean(x²) + ε)

        Provided as an alternative for research comparisons; not used by
        default in KAMUI v0.1.

Design notes:
    KAMUI uses Pre-LN (layer norm applied before the attention and FFN
    sublayers, not after).  This is the modern standard since Xiong et al.
    (2020) showed it enables more stable training at higher learning rates.

    Pre-LN:   x = x + Sublayer(LayerNorm(x))
    Post-LN:  x = LayerNorm(x + Sublayer(x))    ← original Vaswani et al.

    Why implement LayerNorm from scratch rather than using nn.LayerNorm?
        nn.LayerNorm is correct and fast.  KAMUI implements it from scratch
        so students see the formula explicitly and understand that it is
        simply a normalisation + affine transform with no architectural
        mystery.

Tensor shapes:
    Input:   x      (B, S, D)
    Output:  out    (B, S, D)   — same shape, values normalised

References:
    Ba, J. L. et al. (2016). Layer Normalization.
    https://arxiv.org/abs/1607.06450

    Xiong, R. et al. (2020). On Layer Normalization in the Transformer
    Architecture. ICML 2020. https://arxiv.org/abs/2002.04745

    Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization.
    NeurIPS 2019. https://arxiv.org/abs/1910.07467

Implemented in: Phase 1, Week 7
"""

# Implementation begins in Phase 1.
