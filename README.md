<div align="center">

# KAMUI

### Knowledge Activation Mapping & Understanding Interface

**A Transformer Interpretability Framework Built From Scratch**

*"To understand a model, you must first see what it sees."*

[![Tests](https://github.com/RithvikReddy0-0/kamui/actions/workflows/ci.yml/badge.svg)](https://github.com/RithvikReddy0-0/kamui/actions)
[![Docs](https://github.com/RithvikReddy0-0/kamui/actions/workflows/docs.yml/badge.svg)](https://rithvikreddy0-0.github.io/kamui)
[![PyPI version](https://badge.fury.io/py/kamui.svg)](https://badge.fury.io/py/kamui)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://rithvikreddy0-0.github.io/kamui) •
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
BPE Tokenizer            ████░░░░░░  ⏳ in progress
Embeddings               ░░░░░░░░░░  ⏳ planned
Attention Mechanism      ░░░░░░░░░░  ⏳ planned
Transformer Architecture ░░░░░░░░░░  ⏳ planned
Training Pipeline        ░░░░░░░░░░  ⏳ planned
Hook System              ░░░░░░░░░░  ⏳ planned
Logit Lens               ░░░░░░░░░░  ⏳ planned
Activation Patching      ░░░░░░░░░░  ⏳ planned
```

The [roadmap](#roadmap) and [issue tracker](https://github.com/RithvikReddy0-0/kamui/issues) reflect active development.

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
git clone https://github.com/RithvikReddy0-0/kamui
cd kamui
pip install -e ".[all]"
pytest
```

This clones the repo, installs all dependencies in editable mode, and runs the test suite.
The tests cover the components that are implemented so far — config, vocabulary, and tokenizer infrastructure.

### Planned API (v0.1)

Once the core components are complete, the intended interface will look like this:

**Train a model**

```python
import kamui

model     = kamui.KAMUITransformer.from_config("configs/small.yaml")
tokenizer = kamui.BPETokenizer.train("data/tinystories.txt", vocab_size=8192)
trainer   = kamui.Trainer(model, tokenizer, config="configs/small.yaml")
trainer.train()
```

**Run logit lens**

```python
lens   = kamui.LogitLens(model, tokenizer)
result = lens.run("The Eiffel Tower is located in the city of")
result.plot()   # layer × token heatmap — watch "Paris" emerge
```

**Find induction heads**

```python
detector = kamui.InductionHeadDetector(model)
scores   = detector.score_all_heads()
detector.plot_scores(scores)
# Expect high scores at layer 1, heads 2 and 5
```

**Causal intervention**

```python
patcher = kamui.ActivationPatcher(model)
effect  = patcher.patch_all_layers(
    clean="The Eiffel Tower is in Paris",
    corrupted="The Eiffel Tower is in Berlin",
)
effect.plot()   # which layer stores "Paris"?
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
    ↓  Embedding: token + positional
residual_stream  (B, S, D)
    ↓  × n_layers:
       Pre-LN → MultiHeadAttention → residual add
       Pre-LN → FeedForward       → residual add
residual_stream  (B, S, D)
    ↓  Final LayerNorm → Linear unembedding
logits  (B, S, V)

HookManager captures any activation above ↑
mechinterp tools use captured activations for analysis
```

---

## Planned Interpretability Toolkit (v0.1)

| Tool | What it answers |
|------|----------------|
| `AttentionVisualizer` | What is each attention head attending to? |
| `LogitLens` | At each layer, what token does the model predict? |
| `LinearProbe` | At each layer, what linguistic properties are encoded? |
| `ActivationPatcher` | Which components are *causally* responsible for a behaviour? |
| `InductionHeadDetector` | Which heads implement in-context pattern matching? |
| `CircuitAblator` | What is the minimal circuit for a behaviour? |

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
├── future/             # v0.2 design specs (SAEs)
└── RESEARCH_LOG.md     # chronological experiment log
```

Every experiment is reproducible from its folder alone. The research log
becomes the experiments section of your paper.

---

## Roadmap

| Version | Scope | Status |
|---------|-------|--------|
| **v0.1** | Core transformer + 6 interpretability tools | 🔄 In progress |
| **v0.2** | Sparse autoencoders, gradient attribution, RoPE | 📋 Designed |

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

---

## Installation

```bash
# Minimal install (training + inference)
pip install kamui

# With visualisation tools
pip install "kamui[viz]"

# With Jupyter notebooks
pip install "kamui[notebooks]"

# Full development install
git clone https://github.com/RithvikReddy0-0/kamui
cd kamui
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

Find open issues on [GitHub Issues](https://github.com/RithvikReddy0-0/kamui/issues).

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
  url       = {https://github.com/RithvikReddy0-0/kamui},
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