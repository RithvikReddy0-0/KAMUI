# Activation Patching

## What it is

Activation patching is the primary causal intervention tool in mechanistic
interpretability. It establishes not just *where* information is present,
but *whether* that information is causally responsible for a behaviour.

---

## The technique

```text
1. Run model on clean input  → cache all activations
2. Run model on corrupt input
   → at hook point H, replace activation with cached clean activation
3. Measure: how much did the output recover toward the clean output?

Effect ≈ 0.0  →  hook point H is NOT on the causal path
Effect ≈ 1.0  →  hook point H IS on the causal path
```

---

## Worked example

The examples on this page use a `model` and two token ID tensors of the same length,
`clean_ids` and `corrupted_ids`. In the [Quickstart](../tutorials/quickstart.md) they
are the encoded `"the cat sat on the "` and `"the cat mat . the "`.

```python
from kamui.mechinterp import ActivationPatcher

patcher = ActivationPatcher(model)

# Which block's attention output carries the answer?
result = patcher.patch_all_layers(clean_ids, corrupted_ids, component="attn")

print(result.effects)      # one effect per block
print(result.best_layer()) # the block with the largest effect
result.plot()              # bar chart of the effects
```

Set `component="ffn"` to test the feed-forward networks instead. On the Quickstart's
model, the attention effects stay near 0 and the feed-forward effects are
`[0.9948, 0.0767]`, so the first block's feed-forward network is on the causal path.

### Patching a specific head

```python
result = patcher.patch_all_heads(clean_ids, corrupted_ids)

print(result.best_head())  # (layer, head) with the largest effect
result.plot()              # (n_layers × n_heads) heatmap of causal effects
```

---

## The logit difference metric

The effect is measured as **logit difference recovery** at the last position:

```text
logit_diff(logits) = logit(answer_token) - logit(corrupt_token)

effect = (logit_diff(patched) - logit_diff(corrupted)) /
         (logit_diff(clean)   - logit_diff(corrupted))
```

By default, `answer_token` is the clean run's top prediction and `corrupt_token` is
the corrupted run's. Pass either one by keyword to compare specific tokens.

An effect of 0 means patching changed nothing, and 1 means full recovery. The value
isn't clamped, so it can fall slightly below 0 or above 1.

---

## Designing good patching experiments

**Clean and corrupt inputs must have the same shape.**

Patching swaps activations position by position, so both inputs must encode to the
same number of tokens. BPE merges can span words, so check the lengths after encoding.

**Clean and corrupt inputs must differ in exactly one thing.**

Good: `"Paris" → "Berlin"` (one factual change)
Bad: `"The Eiffel Tower is in Paris" → "Big Ben is in London"` (multiple changes)

If multiple things differ between clean and corrupt, you cannot attribute
the patching effect to any single variable.

**The corrupt run must produce a different answer.**

If both runs give the same answer, there is nothing to recover. When the two logit
differences are effectively equal, `ActivationPatcher` reports an effect of 0.

---

## References

Meng et al. (2022). Locating and Editing Factual Associations in GPT.
NeurIPS 2022. https://arxiv.org/abs/2202.05262

Wang et al. (2022). Interpretability in the Wild: a Circuit for
Indirect Object Identification in GPT-2 small.
https://arxiv.org/abs/2211.00593
