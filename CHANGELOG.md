# Changelog

All notable changes to KAMUI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- Quality pass across all phases: the full `kamui` package and test suite now
  pass `ruff` (E/F/W/I/N/UP/ANN/B/SIM/C90) and `black` cleanly — PEP 604 union
  syntax, sorted imports, `OSError` over the `IOError` alias, complete return-type
  annotations, and `N812` allowed for the conventional `torch.nn.functional as F`.
  No behaviour changes; all 507 tests pass at 98% coverage.

### Added
- Mechinterp: activation patching (`kamui.mechinterp.ActivationPatcher`): causal
  localisation by caching a clean activation and patching it into a corrupted
  run, scored by logit-difference recovery (0 = no effect, 1 = fully restores
  clean behaviour). `patch_single` (any embed/attn/ffn output point),
  `patch_all_layers` → `PatchingResult`, and `patch_all_heads` →
  `HeadPatchingResult` (per-head via `out_proj`-input swap), each with a
  `.plot()`. Patching uses the hook system's output-replacement path, so the
  model stays untouched. A deterministic test anchors the method: patching
  `embed.output` yields exactly 1.0 recovery. 22-test suite at 100% coverage.
- Mechinterp: logit lens (`kamui.mechinterp.LogitLens`): projects the residual
  stream to vocabulary at every layer via `final_ln + unembed`, revealing at
  which depth each prediction emerges. Reconstructs per-layer residual streams
  purely from existing hook points (`embed.output` + running sum of each
  block's `attn.output`/`ffn.output`), so it stays decoupled from model
  internals; the final layer provably reproduces the model's own logits.
  `LogitLensResult` exposes per-layer probabilities, top-k tokens, and
  `plot()` / `plot_position()` heatmaps. 18-test suite at 100% coverage.
- Phase 5 — training pipeline (`kamui.training`): an explicit, from-scratch
  training loop with no framework magic. `CosineWithWarmup` (linear warmup +
  cosine decay), `build_optimizer` (AdamW with weight-decay separated so
  biases/LayerNorm are undecayed), `TextDataset`/`DataLoader` (packed
  next-token batches) plus `tokenise_corpus`/`train_val_split`, full-state
  `save_checkpoint`/`load_checkpoint`/`load_model_only` (with read-back
  verification), and `Trainer`/`TrainingConfig` with explicit gradient
  accumulation, gradient clipping + norm logging, scheduled LR, and
  `evaluate()`. A regression test confirms the loop reduces loss on a small
  corpus. 61-test suite at 100% coverage across all five modules. The KAMUI
  model can now be trained end-to-end.
- Phase 4 — evaluation (`kamui.evaluate`): `compute_perplexity` (corpus-level,
  accepting `(inputs, targets)` pairs or plain token tensors),
  `compute_token_loss` (per-token loss), and `compute_sequence_perplexity`
  (single sequence with a sliding window for long inputs). Plus `generate`
  with four from-scratch sampling strategies — greedy, top-k, nucleus (top-p),
  and temperature — with context cropping and reproducible seeding, and
  `generate_with_probs` returning per-step distributions. 34-test suite at
  100% coverage on both modules.
- Phase 3 — hook system (`kamui.hooks`): `HookManager`, a context manager that
  captures named activations during a forward pass and always removes its hooks
  on exit (even on exceptions), guaranteeing model output is unchanged. Supports
  `embed.output`, `blocks.{i}.attn.output`, `blocks.{i}.attn.weights`,
  `blocks.{i}.ffn.mid`, `blocks.{i}.ffn.output`, and `unembed.input`. Attention
  weights are captured via a reversible forward-wrapper (the model stays fully
  hook-agnostic). `HookRegistry` provides the canonical valid-point list so
  typos fail loudly. 22-test suite at 100% coverage on the hooks subpackage.
- Phase 2G — full model + weight init (`kamui.model.transformer`,
  `kamui.model.init_weights`): `KAMUITransformer` assembles embedding →
  `n_layers` × `TransformerBlock` → final `LayerNorm` → weight-tied linear
  unembedding, returning `(B, S, V)` logits or a scalar cross-entropy loss
  when targets are given. GPT-2-style scaled initialisation (`N(0, 0.02)`,
  residual projections scaled by `1/√(2·n_layers)`). `from_config` /
  `from_yaml` constructors and `num_parameters()`. Actual parameter count
  matches `ModelConfig.estimated_total_parameters`. The public API
  (`kamui.ModelConfig`, `kamui.KAMUITransformer`, `kamui.BPETokenizer`,
  `kamui.Vocabulary`) is now exported. 28-test suite at 100% coverage on both
  modules.
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
