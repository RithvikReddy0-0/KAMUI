# Activation Patching

## What it is

Activation patching is the primary causal intervention tool in mechanistic
interpretability. It establishes not just *where* information is present,
but *whether* that information is causally responsible for a behaviour.

---

## The technique

```
1. Run model on clean input  → cache all activations
2. Run model on corrupt input
   → at hook point H, replace activation with cached clean activation
3. Measure: how much did the output recover toward the clean output?

Effect = 0.0  →  hook point H is NOT on the causal path
Effect = 1.0  →  hook point H IS on the causal path
```

---

## Worked example

```python
from kamui.mechinterp import ActivationPatcher

patcher = ActivationPatcher(model)

# Does layer 3's attention output store "Paris" vs "Berlin"?
result = patcher.patch_all_layers(
    clean="The Eiffel Tower is in Paris",
    corrupted="The Eiffel Tower is in Berlin",
    component="attn",
)

result.plot()
# Bar chart: layer with highest bar = where "Paris" is stored
```

### Patching a specific head

```python
result = patcher.patch_all_heads(
    clean_ids=clean_ids,
    corrupted_ids=corrupted_ids,
)
result.plot()   # (n_layers × n_heads) heatmap of causal effects
```

---

## The logit difference metric

For binary comparisons (correct token vs. incorrect token), the effect is
measured as **logit difference recovery**:

```
logit_diff(logits) = logit(correct_token) - logit(incorrect_token)

effect = (logit_diff(patched) - logit_diff(corrupted)) /
         (logit_diff(clean)   - logit_diff(corrupted))
```

This normalises to `[0, 1]`: 0 = patching had no effect; 1 = full recovery.

---

## Designing good patching experiments

**Clean and corrupt inputs must differ in exactly one thing.**

Good: `"Paris" → "Berlin"` (one factual change)
Bad: `"The Eiffel Tower is in Paris" → "Big Ben is in London"` (multiple changes)

If multiple things differ between clean and corrupt, you cannot attribute
the patching effect to any single variable.

**The corrupt run must produce wrong output.**

If the corrupted model already gets the right answer (by coincidence), the
logit difference metric is undefined.

---

## References

Meng et al. (2022). Locating and Editing Factual Associations in GPT.
NeurIPS 2022. https://arxiv.org/abs/2202.05262

Wang et al. (2022). Interpretability in the Wild: a Circuit for
Indirect Object Identification in GPT-2 small.
https://arxiv.org/abs/2211.00593
