# v0.2 Design: Sparse Autoencoders for Feature Decomposition

**Status:** Planned for v0.2. Not in v0.1 scope.
**Prerequisite:** v0.1 complete, trained checkpoint available.

---

## Motivation

Transformers are hypothesised to represent more features than they have
dimensions by using *superposition*: multiple features are encoded in
overlapping directions in activation space.  A linear basis cannot
disentangle them.

Sparse autoencoders (SAEs) learn an overcomplete dictionary of features
by training a two-layer network with a sparsity penalty on the hidden
activations.  The learned features tend to be monosemantic (each neuron
activates for a coherent concept) even when the original MLP neurons are
polysemantic.

Reference: Anthropic (2023). Towards Monosemanticity: Decomposing Language
Models With Dictionary Learning.
https://transformer-circuits.pub/2023/monosemantic-features

---

## Planned Architecture

```
SparseAutoencoder(d_model, n_features, l1_coeff):
    encode: Linear(d_model → n_features) + ReLU
    decode: Linear(n_features → d_model)
    loss:   reconstruction_loss + l1_coeff * sparsity_loss
```

Where:
- `n_features = k * d_model` for k ∈ {4, 8, 16} (overcomplete)
- `sparsity_loss = mean(|hidden_activations|)` (L1 penalty)
- `reconstruction_loss = MSE(input, decode(encode(input)))`

---

## Planned Module Location

`kamui/mechinterp/superposition.py` — added in v0.2

---

## Training Procedure

1. Load a trained KAMUI checkpoint (small config, step 5000)
2. Cache MLP activations at every layer for 50K sequences (using HookManager)
3. Train one SAE per layer: 50 epochs, Adam, lr=1e-4
4. Evaluate: reconstruction loss, % dead features (never activate),
   mean L0 sparsity (how many features active per token)

---

## Research Questions

1. Do SAE features at layer 3 correspond to interpretable linguistic concepts?
2. Do features overlap across layers (same concept encoded at multiple depths)?
3. How does feature sparsity vary with layer depth?
4. Can we find features corresponding to induction-head-related concepts?

---

## Decision to defer to v0.2

SAE training is a second full research project:
- Separate training pipeline (different loss, different convergence criteria)
- Separate evaluation methodology
- Requires a trained base model as prerequisite

Including SAEs in v0.1 would create three simultaneous unfinished projects.
The architectural decision is documented here so v0.2 can begin immediately
after v0.1 is shipped without rethinking the design.
