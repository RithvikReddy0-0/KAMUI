# Quickstart

This guide gets you from zero to a trained model with logit lens running
in under 20 minutes on a free Google Colab T4 GPU.

---

## 1. Install

```bash
pip install kamui
```

Or from source (recommended for development):

```bash
git clone https://github.com/RithvikReddy0-0/kamui
cd kamui
pip install -e ".[all]"
```

---

## 2. Download data

```bash
# TinyShakespeare — 1MB, trains in minutes
wget -O data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

## 3. Train a nano model

```bash
python scripts/train.py --config configs/nano.yaml
```

Expected output:
```
step=0    train_loss=8.941  lr=0.00e+00  grad_norm=2.341
step=50   train_loss=5.123  lr=1.50e-04  grad_norm=1.832
step=100  train_loss=3.891  lr=3.00e-04  grad_norm=1.234
...
step=2000 train_loss=1.621  val_loss=1.743  tokens_per_sec=9200
Training complete. Checkpoint saved: checkpoints/nano/step_0002000.pt
```

Training takes ~10 minutes on a T4 GPU, ~30 minutes on CPU.

---

## 4. Generate text

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/nano/best.pt \
  --generate "HAMLET:"
```

---

## 5. Run logit lens

```python
import kamui

# Load model and tokenizer
model = kamui.KAMUITransformer.from_checkpoint("checkpoints/nano/best.pt")
tokenizer = kamui.BPETokenizer.load("checkpoints/nano/tokenizer.json")

# Run logit lens
lens = kamui.LogitLens(model, tokenizer)
result = lens.run("To be, or not to be, that is the")
result.plot()
```

This renders a heatmap showing what the model predicts at each layer
and position. You can watch "question" emerge as the predicted next token
as the model processes the sequence.

---

## 6. Find induction heads

```python
detector = kamui.InductionHeadDetector(model)
scores = detector.score_all_heads()
detector.plot_scores(scores)
```

Expected: high scores (> 0.5) at layer 1, heads 2–5.

---

## Next steps

- [Train the small model](training_your_first_model.md) for better results
- [Run your first interpretability experiment](your_first_interpretability_experiment.md)
- [Read the architecture docs](../architecture/transformer.md)
- [Browse the educational notebooks](../../notebooks/)
