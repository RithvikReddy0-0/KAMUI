# Experiment Template

Copy this folder for each new experiment:
```bash
cp -r research/experiments/TEMPLATE research/experiments/exp_NNN_short_description
```

Each experiment folder must contain exactly three files:

---

## config.yaml

The exact model and experiment configuration used.
Copy from `configs/` and pin every value — no "default" references.

```yaml
# Experiment: exp_001_induction_head_detection
# Date: YYYY-MM-DD
# Based on: configs/small.yaml

experiment:
  id: "exp_001"
  name: "induction_head_detection"
  hypothesis: "Induction heads appear in layers 1-2 of the 6-layer small model"
  checkpoint: "checkpoints/small/step_005000.pt"

model:
  n_layers: 6
  d_model: 256
  n_heads: 8
  d_ff: 1024
  vocab_size: 8192
  context_length: 256

analysis:
  tool: "InductionHeadDetector"
  n_sequences: 100
  prefix_length: 50
  threshold: 0.5
```

---

## results.json

Numerical results in a machine-readable format.

```json
{
  "experiment_id": "exp_001",
  "date": "YYYY-MM-DD",
  "hypothesis": "Induction heads appear in layers 1-2",
  "metrics": {
    "head_scores": {
      "(0,0)": 0.04, "(0,1)": 0.08, "(0,2)": 0.02,
      "(1,0)": 0.71, "(1,2)": 0.83, "(1,5)": 0.67,
      "(2,1)": 0.31
    },
    "max_score_layer_0": 0.08,
    "max_score_layer_1": 0.83,
    "heads_above_threshold": ["(1,0)", "(1,2)", "(1,5)"]
  },
  "conclusion": "Confirmed. Peak induction scores at layer 1.",
  "next_experiment": "exp_002_ablate_induction_heads"
}
```

---

## notes.md

Free-form observations.  Write immediately after running — do not wait.
This becomes the Discussion section of your paper.

```markdown
## Experiment exp_001 — Induction Head Detection

**Date:** YYYY-MM-DD
**Time to run:** ~3 minutes on T4

### What I expected
Induction heads at layers 1-2, consistent with Olsson et al. (2022).

### What I found
[Your observations here]

### What confused me
[Anything that didn't match expectations]

### Figures
- `figures/induction_scores_heatmap.png` — (layer, head) heatmap

### Next steps
- [ ] exp_002: ablate the three induction heads and measure ICL drop
```
