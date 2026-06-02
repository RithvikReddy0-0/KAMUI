# Induction Heads

## What they are

Induction heads are the canonical interpretable circuit in transformers.
They implement a simple but powerful algorithm:

> "Find the previous occurrence of the current token. Predict that what
> came after it last time will come after it this time."

Given the sequence `[A][B]...[A]`, an induction head at the second `[A]`
attends to `[B]` (the token that followed the first `[A]`) and boosts its
probability.

This is the mechanism underlying **in-context learning**: the ability to
follow patterns from the context without any weight updates.

---

## The two-head circuit

Induction is a two-component circuit:

```
Layer 0: "Previous token" head
    At position t: attends to position t-1
    Copies token[t-1] into the residual stream at position t

Layer 1+: "Induction" head
    At position t: uses the "what came before me?" information
    to attend back to the previous occurrence of token[t]
    Boosts the probability of whatever followed it
```

The two heads must work together — neither alone produces induction
behaviour. This is the first documented **circuit** in transformers.

---

## Detection

```python
from kamui.mechinterp import InductionHeadDetector

detector = InductionHeadDetector(model)

# Score all (layer, head) pairs
scores = detector.score_all_heads()
# Returns: {(0,0): 0.04, (0,1): 0.07, (1,2): 0.83, (1,5): 0.71, ...}

# Visualise
detector.plot_scores(scores)
# Heatmap: rows=layers, cols=heads, cell=induction score

# Find heads above threshold
induction_heads = [(l, h) for (l, h), s in scores.items() if s > 0.5]
```

### The scoring method

Construct sequences of the form `[random × 50][same random × 50]`.
At each position `t` in the second half, measure the attention weight
on position `t - prefix_length` (the "induction position").

A high average weight on the induction position = induction head.

---

## Causal verification

Detection alone is correlation. Verify causally with ablation:

```python
# How much does removing induction heads hurt in-context learning?
baseline_icl = measure_icl(model)

ablated_icl = detector.ablate_and_measure(
    heads=induction_heads,
    task="repeat_pattern",
)

print(f"ICL degradation: {(baseline_icl - ablated_icl) / baseline_icl:.1%}")
# Expected: significant drop (30-70%) — induction heads drive ICL
```

---

## Expected findings in KAMUI small model

Based on Olsson et al. (2022) and replications in similar-scale models:

- Layer 0: contains "previous token" heads (low induction score, high
  attention to position t-1)
- Layer 1: contains induction heads (high induction score, > 0.5)
- Layers 2+: may contain weaker induction heads or other circuits

---

## References

Olsson et al. (2022). In-context Learning and Induction Heads.
Transformer Circuits Thread.
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
