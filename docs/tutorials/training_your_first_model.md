# Training Your First Model

This guide walks through training the small model — the one used for all
KAMUI interpretability experiments — from scratch.

---

## Prerequisites

- GPU with at least 8GB VRAM (T4, RTX 3080, or better)
- 4GB disk space for data and checkpoints
- KAMUI installed: `pip install -e ".[all]"`

---

## Step 1: Download and prepare TinyStories

TinyStories (Eldan & Li, 2023) is a dataset of simple children's stories.
It is ideal for training interpretability-research models because:
- Stories have clear narrative structure (useful for probing experiments)
- Short sequences mean fast training
- Simple language means patterns emerge clearly

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
with open('data/tinystories.txt', 'w') as f:
    for item in ds:
        f.write(item['text'] + '\n<|endoftext|>\n')
print(f'Written {len(ds)} stories.')
"
```

---

## Step 2: Train the BPE tokenizer

```bash
python -c "
from kamui.tokenizer import BPETokenizer
tokenizer = BPETokenizer.train('data/tinystories.txt', vocab_size=8192)
tokenizer.save('checkpoints/small/tokenizer.json')
print('Tokenizer trained and saved.')
"
```

---

## Step 3: Pre-tokenize the corpus

Pre-tokenization is done once and saves time on subsequent training runs.

```bash
python scripts/tokenize_corpus.py \
    --input data/tinystories.txt \
    --tokenizer checkpoints/small/tokenizer.json \
    --output data/tinystories_tokens.bin
```

---

## Step 4: Train

```bash
python scripts/train.py --config configs/small.yaml
```

Training will log to stdout:
```
step=0      train_loss=9.012  lr=0.00e+00   grad_norm=2.891
step=100    train_loss=4.231  lr=1.50e-04   grad_norm=1.432
step=500    train_loss=2.891  lr=2.95e-04   val_loss=2.934
step=1000   train_loss=2.341  lr=2.72e-04   val_loss=2.389
...
step=5000   train_loss=1.782  lr=3.00e-05   val_loss=1.891
Training complete.
```

Expected training time: ~2 hours on T4.
Expected final val loss: 1.8–2.0 nats.
Expected val perplexity: 6–8.

---

## Step 5: Verify the checkpoint

```bash
python scripts/inspect_checkpoint.py \
    --checkpoint checkpoints/small/best.pt
```

Output:
```
Model: KAMUITransformer
  n_layers:  6
  d_model:   256
  n_heads:   8
  vocab_size: 8192
  Parameters: 12,431,360

Training:
  Step:       5000
  Train loss: 1.782
  Val loss:   1.891
  Val PPL:    6.63
```

---

## Step 6: Generate text (sanity check)

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/small/best.pt \
    --generate "Once upon a time there was a little girl"
```

If the output is coherent (even if simple), training succeeded.
If it is random garbage, the model did not converge — re-run with a lower
learning rate or check for the common bugs listed in
[Training Loop docs](../architecture/training.md).

---

## Next: run interpretability experiments

The small model trained here is the one used in all interpretability
notebooks. See [Your First Interpretability Experiment](your_first_interpretability_experiment.md).
