<div align="center">

# KAMUI

### Knowledge Activation Mapping & Understanding Interface

**A Transformer Interpretability Framework Built From Scratch**

*"To understand a model, you must first see what it sees."*

[![Tests](https://github.com/RithvikReddy0-0/KAMUI/actions/workflows/ci.yml/badge.svg)](https://github.com/RithvikReddy0-0/KAMUI/actions/workflows/ci.yml)
[![Docs](https://github.com/RithvikReddy0-0/KAMUI/actions/workflows/docs.yml/badge.svg)](https://rithvikreddy0-0.github.io/KAMUI)
[![Version 0.3.0](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/RithvikReddy0-0/KAMUI/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://rithvikreddy0-0.github.io/KAMUI) •
[Quickstart](#quickstart) •
[Notebooks](notebooks/) •
[Research](research/) •
[Contributing](CONTRIBUTING.md)

</div>

---

## What is KAMUI?

KAMUI is a decoder-only transformer language model and mechanistic
interpretability framework **built entirely from scratch** in PyTorch.

No HuggingFace Trainer. No PyTorch Lightning. No black boxes.

Every weight matrix, every attention pattern, every residual stream
activation is exposed, documented, and inspectable by design.

KAMUI is for researchers and students who want to understand how language
models actually work — not just use them.

---

## Development Status

KAMUI is currently being built in public.

**Current Progress:**

```
Repository Foundation    ██████████  ✅ complete
ModelConfig System       ██████████  ✅ complete
Vocabulary System        ██████████  ✅ complete
BPE Tokenizer            ██████████  ✅ complete
Embeddings               ██████████  ✅ complete
LayerNorm                ██████████  ✅ complete
FeedForward Network      ██████████  ✅ complete
Attention Mechanism      ██████████  ✅ complete
Transformer Block        ██████████  ✅ complete
Transformer Architecture ██████████  ✅ complete
Training Pipeline        ██████████  ✅ complete
Hook System              ██████████  ✅ complete
Evaluation (ppl + gen)   ██████████  ✅ complete
Calibration Metrics      ██████████  ✅ complete
Attention Visualizer     ██████████  ✅ complete
Logit Lens               ██████████  ✅ complete
Linear Probing           ██████████  ✅ complete
Activation Patching      ██████████  ✅ complete
Induction Head Detector  ██████████  ✅ complete
Circuit Ablation         ██████████  ✅ complete
Shared Utilities         ██████████  ✅ complete
── v0.2 ─────────────────────────────────────────
RoPE Positional Encoding ██████████  ✅ complete
Gradient Attribution     ██████████  ✅ complete
Sparse Autoencoders      ██████████  ✅ complete
── v0.3 ─────────────────────────────────────────
Multi-GPU Training (DDP) ██████████  ✅ complete
── v0.4 (unreleased) ────────────────────────────
SAE Feature Analysis     ██████████  ✅ complete
Steering & Ablation      ██████████  ✅ complete
RMSNorm Option           ██████████  ✅ complete
Weight EMA               ██████████  ✅ complete
Decoding Controls        ██████████  ✅ complete
Accuracy Metric          ██████████  ✅ complete
```

**v0.1 – v0.3 are released and the v0.4 work is on `main`.** That's the
from-scratch transformer and training pipeline, the original six-tool
interpretability toolkit, and the newer SAE, steering and attribution tools.
It also covers multi-GPU training, working CLI entry points, and 7 runnable
notebooks. Test coverage is ~99%, with `ruff`, `black`, and `mypy` all clean.

The [roadmap](#roadmap) and [issue tracker](https://github.com/RithvikReddy0-0/KAMUI/issues) reflect active development.

---

## Why KAMUI exists

Most interpretability research is done on pretrained models (GPT-2,
LLaMA) using tools that weren't designed for transparency. This creates
two problems:

1. **The model is a black box**: you can probe it, but you don't know
   what choices were made in training, initialisation, or architecture.

2. **The tools are abstractions**: `model.run_with_cache()` hides the
   hook system. `AutoModelForCausalLM` hides the architecture.

KAMUI removes both layers of opacity. You train the model yourself.
You read every line of every tool.

---

## What makes KAMUI different

|  | nanoGPT | TransformerLens | **KAMUI** |
|--|---------|----------------|-----------|
| Implemented from scratch | ✅ | ❌ | ✅ |
| Trains from scratch | ✅ | ❌ | ✅ |
| Full interpretability toolkit | ❌ | ✅ | ✅ |
| Context-managed hook system | ❌ | partial | ✅ |
| Educational notebooks (7) | ❌ | ❌ | ✅ |
| Research infrastructure | ❌ | ❌ | ✅ |
| Zero magic abstractions | ✅ | ❌ | ✅ |

---

## Quickstart

```bash
git clone https://github.com/RithvikReddy0-0/KAMUI
cd kamui
pip install -e ".[all]"
pytest
```

This clones the repo, installs all dependencies in editable mode, and runs
the full test suite (866 tests, ~99% coverage).

### API

**Train a model** (or simply: `kamui-train --config configs/nano.yaml --corpus data/corpus.txt`)

```python
import kamui
from kamui.training import DataLoader, TextDataset

config    = kamui.ModelConfig.from_yaml("configs/nano.yaml")
model     = kamui.KAMUITransformer(config)
tokenizer = kamui.BPETokenizer.train("data/corpus.txt", vocab_size=config.vocab_size)

tokens  = tokenizer.encode(open("data/corpus.txt", encoding="utf-8").read())
loader  = DataLoader(TextDataset(tokens, config.context_length), batch_size=16)
trainer = kamui.Trainer(model, loader, config=kamui.TrainingConfig(max_steps=2000))
trainer.train(2000)
```

**Run logit lens**

```python
import torch

ids    = torch.tensor(tokenizer.encode("The Eiffel Tower is located in the city of"))
lens   = kamui.LogitLens(model, tokenizer)
result = lens.run(ids)
result.plot()   # layer × token heatmap — watch the prediction emerge with depth
```

**Find induction heads**

```python
detector = kamui.InductionHeadDetector(model)
scores   = detector.score_all_heads()
detector.plot_scores(scores)   # induction heads typically emerge at layers 1-2
```

**Causal intervention**

```python
patcher   = kamui.ActivationPatcher(model)
clean     = torch.tensor(tokenizer.encode("The Eiffel Tower is in Paris"))
corrupted = torch.tensor(tokenizer.encode("The Eiffel Tower is in Berlin"))
effect    = patcher.patch_all_layers(clean, corrupted)
effect.plot()   # which layer stores the fact?
```

**Which input tokens drove the prediction?**

```python
attr = kamui.GradientAttribution(model, tokenizer)
attr.token_attribution(ids, method="integrated_gradients").plot()
```

**Sparse autoencoder: learn features, read them, steer with them**

```python
from kamui.mechinterp import (
    FeatureSteerer, collect_activations, interpret_features, train_sae,
)

seqs = [torch.tensor(tokenizer.encode(line)) for line in lines]
acts = collect_activations(model, "blocks.1.ffn.output", seqs)    # one row per token
sae  = kamui.SparseAutoencoder(d_model=config.d_model, n_features=8 * config.d_model)
train_sae(sae, acts, epochs=50)

profiles = interpret_features(sae, acts, torch.cat(seqs), top_k=10)   # what each feature detects

steerer = FeatureSteerer(model, sae)
print(steerer.generate_steered_with_feature(       # clamp a feature up, read the text
    tokenizer, "The city of", "blocks.1.ffn.output",
    feature=profiles[0].feature, coefficient=8.0, max_new_tokens=20,
))
```

**A LLaMA-style variant** (rotary positions + RMSNorm) is one config away:

```python
import dataclasses

base   = kamui.ModelConfig.from_yaml("configs/nano.yaml")
config = dataclasses.replace(base, positional_encoding="rope", normalization="rmsnorm")
model  = kamui.KAMUITransformer(config)
```

---

## Architecture

KAMUI is organised into five layers with a strict one-direction dependency:

```
tokenizer  →  model  →  hooks  →  mechinterp  →  evaluate
```

```
text input
    ↓  BPETokenizer (from scratch — no tiktoken)
token_ids  (B, S)
    ↓  Embedding: token (+ learned / sinusoidal positions; RoPE acts inside attention)
residual_stream  (B, S, D)
    ↓  × n_layers:
       Pre-norm → MultiHeadAttention → residual add
       Pre-norm → FeedForward        → residual add
residual_stream  (B, S, D)
    ↓  Final norm → Linear unembedding (weight-tied)
logits  (B, S, V)

norm = LayerNorm (default) or RMSNorm, set by ModelConfig.normalization

HookManager captures any activation above ↑
mechinterp tools use captured activations for analysis
```

---

## Interpretability Toolkit

**Components and circuits** (v0.1)

| Tool | What it answers |
|------|----------------|
| `AttentionVisualizer` | What is each attention head attending to? |
| `LogitLens` | At each layer, what token does the model predict? |
| `LinearProbe` | At each layer, what linguistic properties are encoded? |
| `ActivationPatcher` | Which components are *causally* responsible for a behaviour? |
| `InductionHeadDetector` | Which heads implement in-context pattern matching? |
| `CircuitAblator` | What is the minimal circuit for a behaviour? |
| `GradientAttribution` | Which input tokens drove this prediction? (v0.2) |

**Features and superposition** (v0.2 – v0.4, in `kamui.mechinterp`)

| Tool | What it answers |
|------|----------------|
| `SparseAutoencoder` + `train_sae` | What features is this layer representing? (save / load included) |
| `interpret_features` | Which tokens make each feature fire? |
| `max_activating_examples` | In what contexts does a feature fire most strongly? |
| `feature_cooccurrence` | Which features fire together? |
| `feature_similarity` | Which features point in the same direction? |
| `FeatureSteerer.steer` / `generate_steered` | What happens if I push the model along a feature? |
| `FeatureSteerer.ablate_features` | Does the model actually *use* this feature? |
| `build_steering_vector` | What direction separates two sets of prompts? |

---

## Training & evaluation

| Piece | What it does |
|-------|--------------|
| `Trainer` / `TrainingConfig` | Explicit loop: gradient accumulation, clipping, cosine LR with warmup |
| `kamui.training.distributed` | Multi-GPU data parallelism (DDP), verified with two real processes |
| `kamui.training.EMA` | Exponential moving average of the weights for evaluation / sampling |
| `compute_perplexity` / `compute_accuracy` | Perplexity and top-k next-token accuracy |
| `expected_calibration_error` | Does the model's confidence match its accuracy? |
| `generate` | Greedy / top-k / nucleus / temperature sampling, repetition penalty, stop tokens |

---

## Educational notebooks

| Notebook | What you learn |
|----------|----------------|
| `00_bpe_tokenizer` | Build BPE tokenisation from first principles |
| `01_attention_mechanics` | Visualise attention in a 2-layer model |
| `02_training_dynamics` | Loss curves, gradient norms, LR schedules |
| `03_logit_lens` | Watch predictions evolve layer by layer |
| `04_activation_patching` | Causal interventions — find where facts live |
| `05_induction_heads` | Detect and ablate induction circuits |
| `06_circuit_analysis` | Reverse-engineer a complete behaviour |

---

## Research infrastructure

KAMUI includes first-class research tooling:

```
research/
├── experiments/        # one folder per experiment (config + results + notes)
├── reports/            # written findings and paper drafts
├── figures/            # publication-quality plots
├── future/             # design specs (e.g. the SAE spec, now implemented)
└── RESEARCH_LOG.md     # chronological experiment log
```

Every experiment is reproducible from its folder alone. The research log
becomes the experiments section of your paper.

---

## Roadmap

| Version | Scope | Status |
|---------|-------|--------|
| **v0.1** | Core transformer + 6 interpretability tools | ✅ Released |
| **v0.2** | Sparse autoencoders, gradient attribution, RoPE | ✅ Released |
| **v0.3** | Multi-GPU training (DDP) | ✅ Released |
| **v0.4** | SAE feature analysis & steering, RMSNorm, weight EMA, decoding controls, accuracy metric | 🔄 On `main`, unreleased |

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Installation

KAMUI is not yet on PyPI — install from source:

```bash
# Minimal install (training + inference), pinned to a release
pip install "git+https://github.com/RithvikReddy0-0/KAMUI.git@v0.3.0"

# With visualisation extras (matplotlib, plotly)
pip install "kamui[viz] @ git+https://github.com/RithvikReddy0-0/KAMUI.git@v0.3.0"

# Full development install
git clone https://github.com/RithvikReddy0-0/KAMUI
cd KAMUI
pip install -e ".[all]"
pre-commit install
```

**Requirements**: Python 3.11+, PyTorch 2.1+

---

## Contributing

KAMUI is an open research project. See [CONTRIBUTING.md](CONTRIBUTING.md).

The easiest first contribution is adding a new interpretability tool to
`kamui/mechinterp/` — the hook system handles activation capture, you only
write the analysis logic.

Find open issues on [GitHub Issues](https://github.com/RithvikReddy0-0/KAMUI/issues).

---

## Research philosophy

This project is built on a simple conviction:

> Interpretability is not a feature. It is the prerequisite for trust.

We cannot trust systems we cannot understand. KAMUI is a tool for building
that understanding — one component, one circuit, one forward pass at a time.

The framework is inspired by:
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy's minimal GPT implementation
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — Neel Nanda's interpretability library
- [Anthropic Interpretability Research](https://transformer-circuits.pub) — the circuits thread

---

## Citation

If you use KAMUI in research, please cite:

```bibtex
@software{mukkara2026kamui,
  author    = {Mukkara, Rithvik Reddy},
  title     = {{KAMUI}: {K}nowledge {A}ctivation {M}apping \& {U}nderstanding {I}nterface},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/RithvikReddy0-0/KAMUI},
  license   = {MIT},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="https://github.com/RithvikReddy0-0">Rithvik Reddy Mukkara</a>
<br>
Amrita Vishwa Vidyapeetham · CSE · 2027
</div>