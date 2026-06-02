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
      tokens because exp(0) = 1 ≠ 0.
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

Hook points exposed:
    "blocks.{L}.attn.pre_softmax"   — attention scores before masking+softmax
    "blocks.{L}.attn.weights"       — attention probabilities
    "blocks.{L}.attn.output"        — projected output before residual add

References:
    Vaswani et al. (2017). Attention Is All You Need.
    https://arxiv.org/abs/1706.03762

Implemented in: Phase 1, Week 6
"""

# Implementation begins in Phase 1.
