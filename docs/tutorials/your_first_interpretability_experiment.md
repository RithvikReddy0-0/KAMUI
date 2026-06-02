# Your First Interpretability Experiment

This tutorial walks through a complete interpretability investigation:
finding and verifying an induction circuit in the KAMUI small model.

You need a trained checkpoint from [Training Your First Model](training_your_first_model.md).

---

## Setup

```python
import kamui

model     = kamui.KAMUITransformer.from_checkpoint("checkpoints/small/best.pt")
tokenizer = kamui.BPETokenizer.load("checkpoints/small/tokenizer.json")

model.eval()   # disable dropout for deterministic results
```

---

## Step 1: Look at attention patterns

Before forming a hypothesis, look at what the model is doing.

```python
from kamui.mechinterp import AttentionVisualizer

viz = AttentionVisualizer(model, tokenizer)
result = viz.run("the cat sat on the mat the cat")
result.plot_all()
```

Look for:
- **Previous-token heads**: strong attention on the diagonal shifted left by 1
- **Heads with sharp, non-diagonal patterns**: these are interesting

Write down which (layer, head) pairs look interesting before running any
quantitative analysis. This is good scientific practice — it prevents
p-hacking.

---

## Step 2: Test the induction hypothesis

Do any heads show induction behaviour (attending to repeated patterns)?

```python
from kamui.mechinterp import InductionHeadDetector

detector = InductionHeadDetector(model)
scores = detector.score_all_heads()
detector.plot_scores(scores)

# Print top-5 heads by induction score
top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
for (layer, head), score in top5:
    print(f"  Layer {layer}, Head {head}: {score:.3f}")
```

Expected output:
```
  Layer 1, Head 2: 0.831
  Layer 1, Head 5: 0.714
  Layer 1, Head 0: 0.612
  Layer 0, Head 3: 0.094   ← this is probably a prev-token head
  Layer 2, Head 1: 0.088
```

---

## Step 3: Verify causally with ablation

Correlation is not causation. Verify that these heads actually drive
in-context learning performance.

```python
induction_heads = [(l, h) for (l, h), s in scores.items() if s > 0.5]
print(f"Induction heads: {induction_heads}")

# Measure ICL degradation when induction heads are ablated
baseline_icl  = detector.measure_icl(model)
ablated_icl   = detector.ablate_and_measure(heads=induction_heads)

degradation = (baseline_icl - ablated_icl) / baseline_icl
print(f"ICL degradation from ablating induction heads: {degradation:.1%}")
# Expected: 40-70% degradation
```

If degradation is large (> 30%), you have causal evidence that these heads
drive in-context learning.

---

## Step 4: Use logit lens to understand what they contribute

```python
from kamui.mechinterp import LogitLens

lens = LogitLens(model, tokenizer)

# Use a sequence with a repeated pattern
result = lens.run("cat dog cat dog cat dog cat")
result.plot()
```

Look for: at which layer does the repeated pattern start to be predicted
correctly? Is it the same layer where induction heads live?

---

## Step 5: Document your findings

Create an experiment folder:

```bash
cp -r research/experiments/README.md research/experiments/exp_001_induction_circuit/
```

Fill in:
- `config.yaml`: exact model config used
- `results.json`: induction scores, ICL degradation percentages
- `notes.md`: your observations

Add an entry to `research/RESEARCH_LOG.md`.

---

## What you just did

You completed a full mechanistic interpretability investigation:

1. ✅ Observed attention patterns (exploratory)
2. ✅ Formed a hypothesis (induction heads at layer 1)
3. ✅ Quantified the hypothesis (induction scores)
4. ✅ Tested causality (ablation experiment)
5. ✅ Interpreted the finding (logit lens)
6. ✅ Documented everything (research log)

This is the same workflow used in published interpretability papers.
The next step is to find a circuit that has *not* been documented before —
something specific to your model or your training data.
