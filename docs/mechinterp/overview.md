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

## KAMUI's six tools (v0.1)

Each tool is a standalone class in `kamui/mechinterp/`. They are designed
to be used in sequence, with each one building on the previous.

| Tool | Question it answers | Start here |
|------|--------------------|----|
| `AttentionVisualizer` | What is each head attending to? | Week 14 |
| `LogitLens` | At each layer, what does the model predict? | Week 15 |
| `LinearProbe` | At each layer, what properties are encoded? | Week 16 |
| `ActivationPatcher` | Which components *cause* a behaviour? | Week 17 |
| `InductionHeadDetector` | Which heads implement in-context pattern matching? | Week 18 |
| `CircuitAblator` | What is the minimal set of components for a behaviour? | Week 19 |

---

## Suggested workflow for a new behaviour

When you find an interesting model behaviour, investigate it in this order:

```
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
```

---

## The causal vs. correlational distinction

This is the most important concept in mechanistic interpretability.

**Correlation**: head X is active when behaviour Y occurs.
**Causation**: removing head X causes behaviour Y to disappear.

`AttentionVisualizer` and `LogitLens` show correlation.
`ActivationPatcher` and `CircuitAblator` show causation.

Always verify with a causal tool before claiming you have found a circuit.

---

## v0.2 scope

Sparse autoencoders (SAEs) for feature decomposition are planned for v0.2.
See `research/future/sae_design.md` for the design specification.
