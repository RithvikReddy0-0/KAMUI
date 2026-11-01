# Mechanistic Interpretability — Overview

## What is mechanistic interpretability?

Mechanistic interpretability asks: **which model components are causally
responsible for which behaviours, and how do they work?**

It is not satisfied by knowing that a model predicts the correct token.
It wants to know *which attention heads* contributed, *which layers* stored
the relevant information, and *what algorithm* the model is running.

The goal is to reverse-engineer a trained neural network the way a
compiler engineer reverse-engineers a binary: understand the internal
computation, not just the input-output behaviour.

---

## KAMUI's tools

The core tools are classes importable from `kamui.mechinterp`, and they're designed
to be used in sequence.

**Tools that observe**

| Tool | Question it answers | Since |
|------|---------------------|-------|
| `AttentionVisualizer` | What is each head attending to? | v0.1 |
| `LogitLens` | At each layer, what does the model predict? | v0.1 |
| `LinearProbe` | At each layer, what properties are encoded? | v0.1 |
| `GradientAttribution` | Which input tokens is this output sensitive to? | v0.2 |
| `SparseAutoencoder` | What separate features are packed into a layer? | v0.2 |

**Tools that intervene**

| Tool | Question it answers | Since |
|------|---------------------|-------|
| `ActivationPatcher` | Which components *cause* a behaviour? | v0.1 |
| `CircuitAblator` | What is the minimal set of components for a behaviour? | v0.1 |
| `InductionHeadDetector` | Which heads implement in-context pattern matching? | v0.1 |
| `FeatureSteerer` | What changes if a feature is pushed up or down? | v0.4 |

`InductionHeadDetector` does both: its scores are observations, and
`ablate_and_measure` is an intervention.

---

## Suggested workflow for a new behaviour

When you find an interesting model behaviour, investigate it in this order:

```text
1. AttentionVisualizer
   → "Which heads are active on this input?"

2. LogitLens
   → "At which layer does the correct answer emerge?"

3. ActivationPatcher (layer-level)
   → "Which layer is causally critical?"

4. ActivationPatcher (head-level)
   → "Which heads within that layer are critical?"

5. CircuitAblator
   → "What is the minimal circuit?"

6. LinearProbe (optional)
   → "What linguistic property does each layer encode?"

7. SparseAutoencoder (optional)
   → "What features is that layer made of?"

8. FeatureSteerer (optional)
   → "Does pushing on that feature change behaviour?"
```

[Your First Interpretability Experiment](../tutorials/your_first_interpretability_experiment.md)
walks through steps 1 to 5 on a real model. Notebook
[`07_sparse_autoencoders.ipynb`](https://github.com/RithvikReddy0-0/KAMUI/blob/main/notebooks/07_sparse_autoencoders.ipynb)
covers features, and the [API reference](../api/index.md) lists everything else.

---

## The causal vs. correlational distinction

This is the most important concept in mechanistic interpretability.

**Correlation**: head X is active when behaviour Y occurs.
**Causation**: removing head X causes behaviour Y to disappear.

`AttentionVisualizer`, `LogitLens`, `LinearProbe` and `GradientAttribution` show
correlation or local sensitivity. `ActivationPatcher`, `CircuitAblator` and
`FeatureSteerer` show causation.

Always verify with a causal tool before claiming you have found a circuit.
