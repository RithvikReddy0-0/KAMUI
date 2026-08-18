# Changelog

All notable changes to KAMUI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- SAE feature interpretation (`kamui.mechinterp.interpret_features` +
  `FeatureProfile`): closes the loop on sparse autoencoders by revealing *what
  each learned feature detects*. For every feature it reports the
  strongest-activating tokens (pair `top_token_ids` with a tokenizer's
  `decode` to read the meaning), the feature's firing **density**, and its mean
  activation when active — the token-level form of Anthropic's "what does this
  feature detect?" analysis, and the natural next step after `train_sae`.
  Correctness is anchored exactly (via an identity SAE whose encode is `ReLU`):
  the reported top tokens, ordering, density, and mean activation all match
  hand-computed values, and dead features report empty. 8-test suite at 100%
  coverage.

---

## [0.3.0] — 2026-08-15

The scaling release: from-scratch distributed data-parallel (multi-GPU)
training, with the core gradient-averaging guarantee proven by two real
`gloo` processes on CPU — shipped at 100% coverage on the new module, with
`ruff`, `black`, `mypy`, and a strict docs build all clean.

### Added
- Distributed data-parallel (multi-GPU) training (`kamui.training.distributed`)
  — the from-scratch DDP layer for the v0.3 roadmap. A thin, readable wrapper
  over `torch.distributed`: process-group lifecycle (`init_process_group` /
  `destroy_process_group` / `spawn_workers`), rank-aware queries
  (`get_rank`, `get_world_size`, `is_main_process`, `barrier`), model wrapping
  (`wrap_ddp`, `unwrap_model`), metric reduction (`all_reduce_mean`), and a
  sharding data pipeline (`shard_indices`, `DistributedDataLoader`) that gives
  every rank a disjoint, equal-sized slice of each epoch. Because `wrap_ddp`
  preserves `model.parameters()` and the forward signature, the existing
  `Trainer` and `build_optimizer` work on a wrapped model unchanged. The
  defining DDP guarantee — that the all-reduce-averaged gradient equals the
  single-process full-batch gradient — is proven by a test that spawns **two
  real `gloo` processes on CPU** (max abs difference ~1e-8), so the code path
  is genuinely exercised rather than GPU-gated. 34-test suite at 100% coverage.

---

## [0.2.0] — 2026-08-11

The interpretability release: three new analysis capabilities — sparse
autoencoders, gradient attribution, and rotary positional encoding — each
shipped with a full test suite at 100% coverage on its modules, and `ruff`,
`black`, `mypy`, and a strict docs build all clean. Multi-GPU training remains
deferred (see below).

### Added
- Sparse autoencoders for superposition analysis (`kamui.mechinterp.superposition`)
  — the flagship v0.2 feature (see `research/future/sae_design.md`).
  `SparseAutoencoder` learns an overcomplete, unit-norm feature dictionary
  (Anthropic "Towards Monosemanticity" architecture: decoder pre-bias, ReLU
  encoder, MSE + L1 objective). Plus `collect_activations` (cache any hook
  point's activations via `HookManager`, one row per token), `train_sae` (Adam
  with per-step decoder renormalisation), and `sae_feature_metrics`
  (reconstruction MSE, dead-feature fraction, mean L0/L1). Correctness anchored
  by tests: exact dictionary-atom decoding, the loss decomposition, and
  reconstruction dropping from 2.19 → 0.05 on structured data. 24-test suite at
  100% coverage.
- Gradient-based input attribution (`kamui.mechinterp.GradientAttribution`):
  attributes a target-token prediction to each input token in a single backward
  pass (vs. activation patching's many forward passes). Supports **input×gradient**
  (Simonyan et al. 2014) and **integrated gradients** (Sundararajan et al. 2017)
  against a zero-embedding baseline; `AttributionResult` exposes the input and
  baseline metrics so the IG **completeness axiom** (Σ attributions =
  f(input) − f(baseline)) is directly checkable — and is asserted by a test.
  `.plot()` renders a diverging token heatmap. 17-test suite at 100% coverage.
- Rotary positional encoding (RoPE; Su et al. 2021) as a third
  `positional_encoding` option (`"rope"`, alongside `"learned"` /
  `"sinusoidal"`) — the scheme used by LLaMA / Mistral. `RotaryPositionalEncoding`
  (in `kamui.model.embedding`) rotates Q/K by their position inside attention
  rather than adding a positional vector, so the query·key dot product depends
  only on the *relative* offset. It adds zero parameters, requires an even
  `d_head` (validated by `ModelConfig`), and leaves the embedding purely
  token-level. Full model, causality, norm-preservation, and relative-position
  invariance are covered by tests at 100% on the touched modules.

---

## [0.1.0] — 2026-07-18

The first release: the complete from-scratch transformer, training pipeline,
hook system, and all six interpretability tools, at ~99% test coverage with
`ruff`, `black`, and `mypy` all clean.

### Fixed
- BPE training now stops when the most frequent remaining pair occurs fewer
  than twice.  Merging count-1 pairs merely memorises the corpus — on a
  degenerate (periodic) corpus it snowballed into a single token spanning the
  entire text.  On such corpora the vocabulary may now be smaller than
  requested; normal corpora are unaffected.
- `BPETokenizer.encode` now applies merge rules in one pass per rule in
  training order (equivalent output, since a pair learned at step k can only
  involve tokens created earlier) instead of rescanning the whole sequence
  after every single replacement — encoding long texts is now O(merges × S).

### Added
- Type-checked codebase: `mypy` passes on the entire `kamui` package under a
  strong configuration (untyped defs disallowed; only the rules that conflict
  with PyTorch's `Any`-typed stubs are relaxed).  Interpretability tools and
  evaluation functions are now typed against `KAMUITransformer`, generation
  accepts any `TokenizerLike` protocol, and CI enforces the check.
- Working CLI entry points: `kamui-train` and `kamui-eval` (backed by the new
  `kamui.scripts` package) plus `scripts/tokenize_corpus.py` and
  `scripts/inspect_checkpoint.py` — the console scripts declared in
  `pyproject.toml` now exist and are exercised by an integration test.
- Real integration suite: `tests/integration/` now trains a small model
  end-to-end on CPU (< 1 min) and verifies ≥30% loss reduction, checkpoint
  round-trip logit equality, generation, the `kamui-train` CLI, and all six
  interpretability tools on a trained model.  The former placeholder stubs in
  `test_model_shapes.py` / `test_training_loop.py` are real tests (including
  gradient-accumulation ≡ large-batch equivalence), and GPT-2 parity stubs are
  explicit skips pending v0.2 weight loading.
- All 7 educational notebooks (`notebooks/00`–`06`) are now valid, runnable
  `.ipynb` files — previously malformed JSON — covering BPE, attention
  mechanics, training dynamics, logit lens, activation patching, induction
  heads, and circuit analysis; each is smoke-executed against the current API.
- Shipped configs (`configs/*.yaml`) are validated by a unit test;
  `pytest-timeout` added to dev dependencies (CI requires it).

### Changed
- Quality pass across all phases: the full `kamui` package and test suite now
  pass `ruff` (E/F/W/I/N/UP/ANN/B/SIM/C90) and `black` cleanly — PEP 604 union
  syntax, sorted imports, `OSError` over the `IOError` alias, complete return-type
  annotations, and `N812` allowed for the conventional `torch.nn.functional as F`.
  No behaviour changes.
- CI: unit and integration jobs install the `viz` extra (tests import
  matplotlib/plotly); the integration job no longer applies the unit-suite
  coverage gate.

### Added
- v0.1 feature-complete — the remaining four interpretability tools, the
  calibration metrics, and the shared utilities:
  - `AttentionVisualizer` (`kamui.mechinterp.attention_viz`): captures every
    head's attention pattern in one hooked forward pass; matplotlib single-head
    and grid heatmaps, an interactive plotly view with a layer/head selector,
    and `head_summary_stats` (entropy, self-attention and previous-token
    fractions) for identifying head types.
  - `LinearProbe` (`kamui.mechinterp.probing`): from-scratch logistic
    regression on cached last-position activations at any hook point, plus
    `probe_all_layers` over the reconstructed per-depth residual stream with
    an accuracy-by-depth plot. Verified on a linearly separable task.
  - `InductionHeadDetector` (`kamui.mechinterp.induction`): Olsson-et-al.
    repeated-sequence induction scores for every (layer, head), a score
    heatmap, and `ablate_and_measure` — zero-ablating chosen heads and
    measuring the rise in second-half in-context loss.
  - `CircuitAblator` + `find_minimal_circuit` (`kamui.mechinterp.circuits`):
    zero- and mean-ablation of attn/ffn components with a metric-delta
    result, and a greedy minimal-circuit search. Anchored by an exact test:
    ablating every component reduces the model to embed → final_ln → unembed.
  - Calibration (`kamui.evaluate.calibration`): `expected_calibration_error`
    (known-value tested), `reliability_diagram`, and grid-search
    `temperature_scaling`.
  - Utilities (`kamui.utils`): seeding/determinism/device helpers, the
    structured `TrainingLogger` with parseable `key=value` lines, and shared
    heatmap/layer/token plotting functions with `save_figure`.
  All six v0.1 interpretability tools are now implemented; 100% line coverage
  on every new module (81 new tests; suite total 611).
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

## [0.4.0] — Future

### Will include
- To be determined

---

## How to read this file

**[Unreleased]** — changes committed but not yet in a PyPI release.

**[x.y.z]** — a tagged release. Download from PyPI (`pip install kamui==x.y.z`)
or from the [GitHub releases](https://github.com/RithvikReddy0-0/KAMUI/releases) page.

### Change categories

| Category | Meaning |
|----------|---------|
| **Added** | New features |
| **Changed** | Changes to existing functionality |
| **Deprecated** | Features that will be removed in a future version |
| **Removed** | Features that were removed |
| **Fixed** | Bug fixes |
| **Security** | Security patches |
