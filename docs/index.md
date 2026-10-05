# KAMUI Documentation

**Knowledge Activation Mapping & Understanding Interface**

> "To understand a model, you must first see what it sees."

---

## What is KAMUI?

KAMUI is a decoder-only transformer and mechanistic interpretability
framework built entirely from scratch in PyTorch.  No HuggingFace Trainer.
No opaque abstractions.  Every weight, activation, and attention pattern
is exposed and documented.

## Where to start

**If you want to understand transformers:**
→ Start with [Architecture Overview](architecture/transformer.md)
→ Then work through the [Notebooks](https://github.com/RithvikReddy0-0/KAMUI/tree/main/notebooks) in order

**If you want to run interpretability experiments:**
→ Start with the [Quickstart](tutorials/quickstart.md)
→ Then read [Logit Lens](mechinterp/logit_lens.md) and [Activation Patching](mechinterp/activation_patching.md)

**If you want to contribute:**
→ Read [CONTRIBUTING.md](https://github.com/RithvikReddy0-0/KAMUI/blob/main/CONTRIBUTING.md)
→ Find a `good-first-issue` on GitHub

## Architecture in one diagram

```
text input
    ↓  [BPETokenizer]
token_ids  (B, S)
    ↓  [Embedding: token + positional]
residual_stream  (B, S, D)
    ↓  ×n_layers [TransformerBlock: Pre-LN → Attention → residual → Pre-LN → FFN → residual]
residual_stream  (B, S, D)
    ↓  [Final norm (LayerNorm or RMSNorm) → Unembed]
logits  (B, S, V)
    ↓  [HookManager captures any activation above]
mechinterp tools: LogitLens | ActivationPatcher | InductionHeadDetector | CircuitAblator
                  GradientAttribution | SparseAutoencoder | FeatureSteerer
```

## Feature scope

| Feature | Since |
|---------|-------|
| BPE tokeniser, from-scratch transformer, explicit training loop | v0.1 |
| Hook system | v0.1 |
| Logit lens, attention visualisation, linear probing | v0.1 |
| Activation patching, induction-head detection, circuit ablation | v0.1 |
| RoPE positional encoding | v0.2 |
| Gradient attribution (input×grad, integrated gradients) | v0.2 |
| Sparse autoencoders | v0.2 |
| Multi-GPU training (DDP) | v0.3 |
| SAE feature analysis (interpretation, max-activating examples, co-occurrence, similarity) | v0.4 (on `main`) |
| Steering, steered generation, contrastive vectors, feature ablation | v0.4 (on `main`) |
| RMSNorm option, weight EMA, decoding controls, top-k accuracy, KV-cache, gradient checkpointing | v0.4 (on `main`) |
