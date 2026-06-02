# Transformer Architecture

## Overview

KAMUI implements a decoder-only transformer language model — the same
architecture family as GPT-2, LLaMA, and Mistral. This document explains
every design decision, with references to the code that implements it.

---

## The Residual Stream

The single most important concept for understanding transformers is the
**residual stream**.

The residual stream is a tensor of shape `(B, S, D)` that flows through the
entire model. Every layer — attention and FFN — reads from it and writes
back to it via addition. Nothing else happens.

```
residual = embed(token_ids)               # initialise the stream

for block in blocks:
    residual = residual + block.attn(block.ln1(residual))   # attention writes
    residual = residual + block.ffn(block.ln2(residual))    # FFN writes

logits = unembed(final_ln(residual))
```

This framing, formalised by Elhage et al. (2021), has a powerful
implication: **every component can be understood independently** as a
function that reads a vector from the stream and adds a delta back.
Components do not compose sequentially in the traditional sense — they all
operate on the same shared channel.

This is why activation patching works: you can replace the stream value
at any point and measure the downstream effect cleanly.

---

## Component Reference

### Token Embedding

`kamui/model/embedding.py` → `TokenEmbedding`

A learned lookup table of shape `(V, D)`. Each token ID maps to a
`D`-dimensional vector. At initialisation: `N(0, 0.02)`.

### Positional Encoding

`kamui/model/embedding.py` → `LearnedPositionalEncoding` or `SinusoidalPositionalEncoding`

KAMUI supports both. Default: learned (matches GPT-2).

**Sinusoidal** (Vaswani et al., 2017):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/D))
PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
```
Fixed, non-trainable. Can extrapolate beyond training length in principle.

**Learned**: a trainable `(context_length, D)` matrix. Better empirical
performance but cannot extrapolate.

### Pre-Layer Normalisation

`kamui/model/normalization.py` → `LayerNorm`

Applied **before** each sublayer (Pre-LN), not after (Post-LN).

```
x = x + Attention(LayerNorm(x))     # ← Pre-LN: norm before sublayer
x = x + FFN(LayerNorm(x))
```

Why Pre-LN? Post-LN places LayerNorm inside the residual path, constraining
gradient flow and requiring careful warmup. Pre-LN removes LayerNorm from
the residual path, enabling more stable training at higher learning rates.
See Xiong et al. (2020).

### Multi-Head Self-Attention

`kamui/model/attention.py` → `MultiHeadAttention`

The core attention equation:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

With `H` heads running in parallel:
```
head_i = Attention(x W_Qi, x W_Ki, x W_Vi)
output = Concat(head_1, ..., head_H) W_O
```

**Causal mask**: future positions are set to `-inf` before softmax
(not `0` — that is a common bug). This ensures `exp(-inf) = 0`, so
future tokens receive zero attention weight.

**KAMUI always returns attention weights**: the forward pass returns
`(output, weights)`, not just `output`. This is a deliberate design
choice — interpretability tools need the weights without hooks.

### Feed-Forward Network

`kamui/model/feedforward.py` → `FeedForward`

```
FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
```

Where `W_1 ∈ R^(D × 4D)` and `W_2 ∈ R^(4D × D)`. The 4× expansion factor
is standard since Vaswani et al. (2017).

GELU activation: smooth, non-monotonic, provides a learned soft gate.
All modern transformers use GELU or SwiGLU. KAMUI uses GELU for simplicity.

### Final LayerNorm + Unembedding

After all blocks, a final LayerNorm is applied to the residual stream,
then a linear projection maps `D → V` to produce logits.

**Weight tying**: the unembedding matrix is the **transpose** of the token
embedding matrix. This halves the parameter count of these two layers and
empirically improves perplexity. Used by GPT-2 and all descendants.

---

## Weight Initialisation

`kamui/model/init_weights.py`

Standard init: `N(0, 0.02)` for all weight matrices and embeddings.

**Scaled residual init**: the output projections of attention (`W_O`) and
FFN (`W_2`) are initialised with reduced scale:

```
std = 0.02 / sqrt(2 * n_layers)
```

This prevents residual stream variance from growing with depth. Without it,
a 12-layer model has activation variance ~12× larger at the output than
the embedding, causing instability.

---

## Hyperparameter Reference (small config)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_layers` | 6 | Number of transformer blocks |
| `d_model` | 256 | Residual stream dimension |
| `n_heads` | 8 | Attention heads per layer |
| `d_head` | 32 | `d_model / n_heads` |
| `d_ff` | 1024 | FFN hidden dimension (`4 × d_model`) |
| `vocab_size` | 8192 | BPE vocabulary |
| `context_length` | 256 | Maximum sequence length |
| Parameters | ~12M | Total trainable parameters |

---

## References

- Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Xiong et al. (2020). On Layer Normalization in the Transformer Architecture. https://arxiv.org/abs/2002.04745
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. https://transformer-circuits.pub/2021/framework/
