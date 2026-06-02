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
→ Then work through the [Notebooks](../notebooks/) in order

**If you want to run interpretability experiments:**
→ Start with the [Quickstart](tutorials/quickstart.md)
→ Then read [Logit Lens](mechinterp/logit_lens.md) and [Activation Patching](mechinterp/activation_patching.md)

**If you want to contribute:**
→ Read [CONTRIBUTING.md](../CONTRIBUTING.md)
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
    ↓  [LayerNorm → Unembed]
logits  (B, S, V)
    ↓  [HookManager captures any activation above]
mechinterp tools: LogitLens | ActivationPatcher | InductionHeadDetector | CircuitAblator
```

## v0.1 feature scope

| Feature | Status |
|---------|--------|
| BPE tokeniser | Phase 1 |
| Transformer model (from scratch) | Phase 1 |
| Explicit training loop | Phase 2 |
| Hook system | Phase 3 |
| Logit lens | Phase 4 |
| Attention visualisation | Phase 4 |
| Activation patching | Phase 4 |
| Linear probing | Phase 4 |
| Induction head detection | Phase 4 |
| Circuit ablation | Phase 4 |
| Sparse autoencoders | v0.2 |
