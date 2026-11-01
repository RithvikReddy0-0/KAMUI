# Quickstart

Build a tiny language model on your CPU, watch a prediction form inside it, and
find the part that decides. The whole page runs in under a minute with no GPU and
no downloads. It uses the same model as the
[README's story](https://github.com/RithvikReddy0-0/KAMUI#readme), and CI runs every
code block here.

---

## 1. Install

```bash
git clone https://github.com/RithvikReddy0-0/KAMUI
cd KAMUI
pip install -e ".[all]"
```

!!! danger "Do not run `pip install kamui`"
    KAMUI is not published on PyPI. The `kamui` package there is an unrelated
    phase-unwrapping library that happens to share the name.

    To use KAMUI as a dependency without cloning it, install the tagged release
    from GitHub:

    ```bash
    pip install "git+https://github.com/RithvikReddy0-0/KAMUI.git@v0.4.0"
    ```

---

## 2. Build and train a model

The corpus is seven hundred sentences with real grammar, written by the script
itself. The model has two blocks and about 90,000 parameters.

```python
import random

import torch

import kamui
from kamui.training import DataLoader, TextDataset

torch.manual_seed(0)
rng = random.Random(0)

# Sentences with real grammar: "the ANIMAL VERB PREP the OBJECT ."
animals, verbs = ["cat", "dog", "bird", "fish"], ["sat", "ran", "slept", "hid"]
preps, objects = ["on", "under", "near"], ["mat", "log", "rug", "box"]
corpus = " ".join(
    f"the {rng.choice(animals)} {rng.choice(verbs)} {rng.choice(preps)} the {rng.choice(objects)} ."
    for _ in range(700)
)

config = kamui.ModelConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128,
                           vocab_size=320, context_length=32)
tokenizer = kamui.BPETokenizer.train(corpus, vocab_size=config.vocab_size)
tokens = tokenizer.encode(corpus)

model = kamui.KAMUITransformer(config)
trainer = kamui.Trainer(
    model,
    DataLoader(TextDataset(tokens, config.context_length), batch_size=8, seed=0),
    config=kamui.TrainingConfig(max_lr=3e-3, warmup_steps=10, max_steps=1000),
)
trainer.train(200)
model.eval()
```

---

## 3. Watch a prediction form

The logit lens asks what the model would predict if it stopped early. Row 0 is
before any block has run, and each later row adds one block.

```python
ids = torch.tensor(tokenizer.encode("the cat sat on the "))
result = kamui.LogitLens(model, tokenizer).run(ids)
for layer in range(result.probs.shape[0]):
    print(layer, repr(tokenizer.decode([result.probs[layer, -1].argmax().item()])))
# 0 'ran on the '
# 1 'mat . the '
# 2 'mat . the '
```

The tokenizer learned merges that span words, so one token can be several words.
Before any block runs, the model guesses a verb phrase, which is the wrong kind of
word here. After the first block it guesses an object, and it doesn't change its
mind after that. Exact values can vary slightly by machine.

In a notebook, `result.plot()` draws the same thing for every position.

---

## 4. Find the part that decides

Seeing where an answer appears doesn't prove what caused it. Activation patching
tests that directly: run a broken input, put one clean part back, and see whether
the answer returns. An effect of 1 means the answer fully returned, and 0 means
nothing changed.

```python
corrupted = torch.tensor(tokenizer.encode("the cat mat . the "))
patcher = kamui.ActivationPatcher(model)
print(patcher.patch_all_layers(ids, corrupted, component="ffn").effects)
# tensor([0.9948, 0.0767])
```

Restoring the first block's feed-forward brings the answer back on its own.

---

## Next steps

- [Run a full investigation of this result](your_first_interpretability_experiment.md)
- [Train from the command line](training_your_first_model.md)
- [Read the architecture docs](../architecture/transformer.md)
- [Browse the notebooks](https://github.com/RithvikReddy0-0/KAMUI/tree/main/notebooks)
