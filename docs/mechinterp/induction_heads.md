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

```text
Earlier layer: "Previous token" head
    At position t: attends to position t-1
    Copies token[t-1] into the residual stream at position t

Later layer: "Induction" head
    At position t: uses the "what came before me?" information
    to attend back to the previous occurrence of token[t]
    Boosts the probability of whatever followed it
```

The two heads must work together — neither alone produces induction
behaviour, so a model needs at least two layers.

---

## Detection

The examples on this page use the `model` and `tokens` from the
[Quickstart](../tutorials/quickstart.md).

```python
from kamui.mechinterp import InductionHeadDetector

# Sample test sequences from tokens the model has seen
detector = InductionHeadDetector(model, token_ids=set(tokens))

# Score all (layer, head) pairs: {(layer, head): score}, each in [0, 1]
scores = detector.score_all_heads(prefix_len=12, n_samples=4)

# Visualise
detector.plot_scores(scores)
# Heatmap: rows=layers, cols=heads, cell=induction score

# Find heads above threshold
induction_heads = [(l, h) for (l, h), s in scores.items() if s > 0.5]
```

Without `token_ids`, the detector samples from the whole vocabulary, including tokens
the model never saw in training.

### The scoring method

Construct sequences of the form `[prefix][same prefix]`, where the prefix has length
`L` (`prefix_len`, which defaults to `context_length // 2`).

At position `t` in the second half, the current token last appeared at `t - L`, so
what followed it is at `t - L + 1`. The score is the mean attention weight on that
position, across every `t` in the second half and every sample.

A high average weight on the induction position suggests an induction head.

---

## Causal verification

Detection alone is correlation. Verify causally with ablation, which zeroes the
chosen heads and measures the loss on the second half of a repeated sequence:

```python
best_head = max(scores, key=scores.get)

# Rise in in-context loss: above 0 means removing the head hurt
print(detector.ablate_and_measure(heads=[best_head], prefix_len=12))
```

If a head scores high and removing it barely changes the loss, the score alone isn't
evidence of a circuit.

---

## What KAMUI's models show so far

KAMUI has not yet produced a trustworthy induction head. The Quickstart's model peaks
at a score of about 0.08, which makes sense because its corpus doesn't reward copying
from context. Notebook
[`05_induction_heads.ipynb`](https://github.com/RithvikReddy0-0/KAMUI/blob/main/notebooks/05_induction_heads.ipynb)
reports the same null result.

Olsson et al. (2022) found induction heads across a range of larger models. Treat their
findings as what to look for, not as what a given KAMUI model will show.

---

## References

Olsson et al. (2022). In-context Learning and Induction Heads.
Transformer Circuits Thread.
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
