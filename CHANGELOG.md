# Changelog

All notable changes to KAMUI are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
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
