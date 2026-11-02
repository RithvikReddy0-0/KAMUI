# Training Your First Model

This guide trains a model from the command line, saves a checkpoint, and loads it
back. It uses the `tiny` config, which trains in seconds on a CPU, so you can see every
step work before you commit to a longer run. CI runs this page's commands in order, up
to the section on real text.

---

## Prerequisites

Follow the install step in the [Quickstart](quickstart.md), and run these commands from
the repository root.

!!! note "Using a GPU"
    `kamui-train` uses the best device it finds, but KAMUI hasn't been run on a real
    GPU yet. See [Where Training Runs](../architecture/training.md#where-training-runs).

---

## Step 1: Make a corpus

Save this as a script, or paste it into Python. It writes the same grammar as the
Quickstart to `data/animals.txt`.

```python
import random
from pathlib import Path

rng = random.Random(0)
animals, verbs = ["cat", "dog", "bird", "fish"], ["sat", "ran", "slept", "hid"]
preps, objects = ["on", "under", "near"], ["mat", "log", "rug", "box"]
corpus = " ".join(
    f"the {rng.choice(animals)} {rng.choice(verbs)} {rng.choice(preps)} the {rng.choice(objects)} ."
    for _ in range(700)
)

Path("data").mkdir(exist_ok=True)
Path("data/animals.txt").write_text(corpus, encoding="utf-8")
```

---

## Step 2: Train

```bash
python scripts/train.py --config configs/tiny.yaml --corpus data/animals.txt --steps 200
```

If you installed the package, `kamui-train` is the same command.

This trains a tokenizer on the corpus, trains the model for 200 steps, and measures
the validation loss. It logs every step, and the loss falls from about 5.8 to
about 1.8:

```text
step=1 train_loss=5.7982 lr=0.00e+00 grad_norm=2.942
step=2 train_loss=5.7916 lr=3.00e-04 grad_norm=2.975
...
step=200 train_loss=1.7736 lr=2.74e-03 grad_norm=0.666
```

Your exact numbers may differ slightly. Two files appear in `checkpoints/tiny/`:

- `tokenizer.json`: the trained tokenizer
- `step_0000200.pt`: the model, optimizer state and config at the final step

---

## Step 3: Inspect the checkpoint

```bash
python scripts/inspect_checkpoint.py --checkpoint checkpoints/tiny/step_0000200.pt
```

This prints the step, the training and validation losses, the model config, and
every saved tensor with its shape.

---

## Step 4: Generate text

```bash
python scripts/evaluate.py --checkpoint checkpoints/tiny/step_0000200.pt --config configs/tiny.yaml --tokenizer checkpoints/tiny/tokenizer.json --generate "the cat" --max-new-tokens 12
```

`--config` is required, because it describes the model to rebuild, and `--generate`
also needs `--tokenizer`. Sampling is random, but the output should follow the grammar
of the corpus, like this run's `the bird slept near the mat . the fish hid under the
log .` If it does, training worked.

---

## Step 5: Load it in Python

This is how you bring a trained model into the interpretability tools:

```python
import kamui
from kamui.training import load_model_only

config = kamui.ModelConfig.from_yaml("configs/tiny.yaml")
model = kamui.KAMUITransformer(config)
load_model_only("checkpoints/tiny/step_0000200.pt", model)
model.eval()

tokenizer = kamui.BPETokenizer.load("checkpoints/tiny/tokenizer.json")
print(kamui.generate(model, tokenizer, "the cat", max_new_tokens=12))
```

---

## Train on real text

Point `--corpus` at any UTF-8 text file. The `nano` config is the next size up:

```bash
python scripts/train.py --config configs/nano.yaml --corpus data/your_text.txt
```

Before you start a long run, know these limits:

- **The tokenizer sets the pace.** It's pure Python and retrains on every run.
  With a vocabulary of 4,096, 1 MB of text took about 14 minutes to train the
  tokenizer and 4 more to encode on a laptop CPU, all before the first training step.
- **The corpus must be large for its vocabulary.** A larger `vocab_size` lets BPE merge
  more, so a small corpus can collapse into very few tokens. A 21 KB corpus with
  `nano`'s vocabulary of 4,096 became 60 tokens, which stops training with
  `need at least context_length + 1 tokens`.
- **You get one checkpoint.** `kamui-train` saves only when training finishes, so an
  interrupted run saves nothing.
- **There are no reference numbers yet.** This project hasn't published training times
  or final losses for `nano`, `small` or `medium`.

See [What `kamui-train` Reads](../architecture/training.md#what-kamui-train-reads) for
the config keys that apply. `data/README.md` lists public datasets.

---

## Next: run interpretability experiments

[Your First Interpretability Experiment](your_first_interpretability_experiment.md)
walks through a full investigation, and it works the same on a model you trained here.
