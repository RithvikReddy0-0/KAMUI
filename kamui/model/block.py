"""Single transformer block: Pre-LN attention + Pre-LN FFN with residuals.

The transformer block is the repeating unit of the architecture.  It
combines the attention and FFN sublayers with layer normalisation and
residual connections.  KAMUI uses the Pre-LN variant.

Responsibilities:
    - ``TransformerBlock``:
        One complete transformer block:

            # Attention sublayer
            x = x + Attention(LayerNorm(x))

            # FFN sublayer
            x = x + FFN(LayerNorm(x))

        Inputs and outputs are both (B, S, D) — the residual stream shape.
        The block reads from the stream and writes back to it via addition.

Design notes:
    Why is residual connection here and not in attention.py / feedforward.py?
        The residual connection conceptually belongs to the block, not the
        sublayer.  The sublayer computes a *delta* to add to the stream;
        the block applies the addition.  This separation makes it trivial
        to ablate a sublayer (replace its output with zeros) without
        modifying the sublayer's own code.

    Why Pre-LN?
        Post-LN (original Vaswani et al. formulation) places LayerNorm
        *after* the residual addition.  This means the residual path passes
        through LayerNorm, which constrains the gradient flow and requires
        careful warmup.  Pre-LN removes LayerNorm from the residual path,
        resulting in more stable training and allowing larger learning rates.
        See Xiong et al. (2020).

    Named submodules:
        self.ln1    — LayerNorm before attention
        self.attn   — MultiHeadAttention
        self.ln2    — LayerNorm before FFN
        self.ffn    — FeedForward

        These names are used by ``kamui.hooks.registry`` to construct hook
        point strings: "blocks.{L}.attn.output", "blocks.{L}.ffn.output".
        Do not rename them without updating the hook registry.

Tensor shapes:
    Input:   x        (B, S, D)   residual stream
    Output:  x        (B, S, D)   residual stream (updated)

Implemented in: Phase 1, Week 7
"""

# Implementation begins in Phase 1.
