# Changelog

All notable changes to KAMUI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase 2F — transformer block (`kamui.model.block`): the `TransformerBlock`
  repeating unit assembling the Pre-LN pattern
  `x = x + attn(ln1(x)); x = x + ffn(ln2(x))` from `LayerNorm`,
  `MultiHeadAttention`, and `FeedForward`. Residual additions live in the
  block (not the sublayers) so a sublayer can be ablated without touching its
  code; named submodules `ln1/attn/ln2/ffn` are the hook-registry contract.
  Preserves `(B, S, d_model)`, exposes optional `return_weights`, and its
  parameter count matches attention + FFN + two LayerNorms. 16-test suite at
  100% coverage on `block.py`.
- Phase 2E — multi-head causal self-attention (`kamui.model.attention`):
  the standalone `scaled_dot_product_attention` function (`softmax(QKᵀ/√d_k)V`,
  optional boolean mask applied as −∞, always returns the post-softmax weights
  for interpretability) and the `MultiHeadAttention` module (separate Q/K/V/O
  linear projections, `einops` head reshaping, precomputed causal-mask buffer,
  optional `return_weights`). No normalisation or residual (owned by the block,
  Pre-LN). Parameter count matches `ModelConfig.attention_parameters`. 29-test
  suite at 100% coverage on `attention.py`, including an autograd gradient check.
- Phase 2D — position-wise feed-forward network (`kamui.model.feedforward`):
  the `FeedForward` sublayer — a two-layer MLP (expand `d_model → d_ff`, GELU,
  hidden dropout, project `d_ff → d_model`) applied independently per token,
  preserving the `(B, S, d_model)` residual-stream shape. Contains no
  normalisation or residual connection (owned by the block, Pre-LN). Parameter
  count matches `ModelConfig.feedforward_parameters`. 19-test suite at 100%
  coverage on `feedforward.py`.
- Phase 2C — normalisation layers (`kamui.model.normalization`):
  `LayerNorm` implemented from scratch (not wrapping `torch.nn.LayerNorm`)
  with learnable γ/β `nn.Parameter`s and epsilon numerical stability, plus
  `RMSNorm` (LLaMA-style, no mean-centering, scale-only) as a research
  alternative. Both normalise over the feature dimension per token and
  preserve `(..., D)` shape. KAMUI uses Pre-LN. 37-test suite at 100%
  coverage on `normalization.py`, verified against `F.layer_norm`.
- Phase 2A — token & positional embeddings (`kamui.model.embedding`):
  `TokenEmbedding` (raw `nn.Parameter` lookup table, weight-tying ready),
  `SinusoidalPositionalEncoding` (fixed, non-learnable buffer),
  `LearnedPositionalEncoding` (GPT-2 style), and the combined `Embedding`
  module that adds token + positional vectors and applies dropout to produce
  the `(B, S, D)` residual stream. Variant selected via
  `ModelConfig.positional_encoding`. 53-test suite at 100% coverage on
  `embedding.py`.
- Phase 1C — BPE tokeniser (`kamui.tokenizer`): byte-level byte-pair encoding
  implemented from scratch with no external tokenizer dependencies. Includes
  `text_to_bytes` / `bytes_to_text` / `get_stats` / `merge_pair` utilities,
  the `BPETokenizer` class (train, encode, decode, save/load, special-token
  handling), and a 196-test suite at ~94% coverage.
- Phase 1B — vocabulary management (`kamui.tokenizer.vocab`): the `Vocabulary`
  class providing a bidirectional token↔ID mapping with reserved special
  tokens, deterministic/stable ID assignment, duplicate prevention, and
  JSON save/load. Importable for direct vocabulary inspection (~93% coverage).
- Phase 1A — model configuration (`kamui.model.config`): the `ModelConfig`
  dataclass as the single source of truth for transformer hyperparameters,
  with validation, the derived `d_head` dimension, parameter-count estimators
  (attention / feed-forward / embedding / total), and YAML load/save
  (100% coverage).
- Phase 0 scaffold: complete repository structure, all placeholder modules,
  documentation pages, CI pipeline, pre-commit hooks, and architecture diagrams

---

## [0.1.0] — Planned

### Will include
- BPE tokeniser from scratch (`kamui.tokenizer`)
- Decoder-only transformer with Pre-LN and weight tying (`kamui.model`)
- Explicit training loop with warmup + cosine decay (`kamui.training`)
- Context-managed hook system (`kamui.hooks`)
- Logit lens (`kamui.mechinterp.LogitLens`)
- Attention visualisation (`kamui.mechinterp.AttentionVisualizer`)
- Linear probing (`kamui.mechinterp.LinearProbe`)
- Activation patching (`kamui.mechinterp.ActivationPatcher`)
- Induction head detection (`kamui.mechinterp.InductionHeadDetector`)
- Circuit ablation (`kamui.mechinterp.CircuitAblator`)
- Perplexity evaluation and text generation (`kamui.evaluate`)
- 7 educational Jupyter notebooks
- Full documentation site (mkdocs-material)

---

## [0.2.0] — Future

### Will include
- Sparse autoencoders for feature decomposition
  (see `research/future/sae_design.md`)
- Gradient-based attribution methods
- RoPE positional encoding
- Multi-GPU training support

---

## How to read this file

**[Unreleased]** — changes committed but not yet in a PyPI release.

**[x.y.z]** — a tagged release. Download from PyPI (`pip install kamui==x.y.z`)
or from the [GitHub releases](https://github.com/RithvikReddy0-0/kamui/releases) page.

### Change categories

| Category | Meaning |
|----------|---------|
| **Added** | New features |
| **Changed** | Changes to existing functionality |
| **Deprecated** | Features that will be removed in a future version |
| **Removed** | Features that were removed |
| **Fixed** | Bug fixes |
| **Security** | Security patches |
