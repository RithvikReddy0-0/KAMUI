# Your First Interpretability Experiment

The [Quickstart](quickstart.md) showed that the first block's feed-forward network
matters on one input. This tutorial turns that into a proper investigation, with a
question written down first, a causal test, a second independent test, and a rival
explanation that fails.

Stay in the same Python session as the Quickstart, because this page uses its
`model`, `tokenizer`, `tokens`, `ids`, `corrupted` and `patcher`. If you closed the
session, run the Quickstart's code again. It only takes seconds.

---

## Step 1: Write down the question

Do this before you look at any data, so the results can't quietly change what you
claim you were asking.

> After `the cat sat on the `, the model's top guess is `mat . the `. After
> `the cat mat . the `, it is `dog `. Which component turns the last token into that
> choice?

A reasonable first guess is attention, because attention is what lets a token look
at the words before it. Write that guess down too.

---

## Step 2: Look at attention patterns

```python
from kamui.mechinterp import AttentionVisualizer

attention = AttentionVisualizer(model, tokenizer).run(ids)
print(attention.weights.shape)  # torch.Size([2, 4, 3, 3]): layers, heads, positions, positions
```

In a notebook, `attention.plot_all()` draws every head. Note any heads that look
interesting, but don't conclude anything yet: this shows where attention goes, not
whether it matters.

---

## Step 3: Locate the layer

The Quickstart's logit lens already answered this. The prediction changes between
row 0 and row 1, so the choice is made somewhere inside the first block. That block
holds both an attention layer and a feed-forward network, so the question is still
open.

---

## Step 4: Intervene

Patch each kind of component in turn:

```python
for component in ("attn", "ffn"):
    print(component, patcher.patch_all_layers(ids, corrupted, component=component).effects)
# attn tensor([-5.1457e-03,  3.7194e-05])
# ffn tensor([0.9948, 0.0767])
```

Restoring attention does almost nothing, while restoring the first feed-forward
network restores the answer. The first guess was wrong. Checking each attention head
separately agrees:

```python
heads = patcher.patch_all_heads(ids, corrupted)
print(heads.effects.abs().max())  # under 0.02: no single head matters here
```

---

## Step 5: Confirm with a different method

One method can mislead you, so test the finding a second way. Ablation switches a
component off and measures what the model loses, which is a different question from
"does restoring it fix a broken input?"

```python
from kamui.mechinterp import CircuitAblator

target = tokenizer.encode("mat . the ")[0]


def score(logits):
    return logits[0, -1, target].item()


ablator = CircuitAblator(model)
for name in (
    "blocks.0.attn.output",
    "blocks.0.ffn.output",
    "blocks.1.attn.output",
    "blocks.1.ffn.output",
):
    print(f"{name:22} {ablator.ablate([name], ids, score).delta:+.2f}")
# blocks.0.attn.output   -0.25
# blocks.0.ffn.output    -10.00
# blocks.1.attn.output   -0.02
# blocks.1.ffn.output    -0.13
```

The two methods agree. Zeroing the first feed-forward network drops the answer's
logit by 10, from 7.16 to −2.84, and no other component moves it by more than 0.25.

---

## Step 6: Find the smallest circuit

`find_minimal_circuit` removes components greedily for as long as the score stays
above a chosen fraction of the full model's score.

```python
from kamui.mechinterp.circuits import find_minimal_circuit

circuit = find_minimal_circuit(model, ids, score, threshold=0.8)
print(circuit)  # ['blocks.0.ffn.output']
print(find_minimal_circuit(model, ids, score, threshold=0.95))
# ['blocks.0.ffn.output', 'blocks.1.ffn.output']
```

One component keeps 80% of the score, and adding the second block's feed-forward
network keeps 95%.

---

## Step 7: Test a rival explanation

Induction heads are the most famous circuit in transformers. Could one be involved?

```python
detector = kamui.InductionHeadDetector(model, token_ids=set(tokens))
scores = detector.score_all_heads(prefix_len=12, n_samples=4)
print(max(scores.values()))  # about 0.08, on a scale where 1 is a perfect induction head
```

No head comes close. That's a real result: nothing in this corpus rewards copying
earlier text, and a null result ruled out a tempting story.

---

## Step 8: Write it down

Create a folder for the experiment:

```bash
mkdir research/experiments/exp_001_first_block_ffn
```

Add the three files described in
[`research/experiments/README.md`](https://github.com/RithvikReddy0-0/KAMUI/blob/main/research/experiments/README.md):

- `config.yaml`: the exact model and experiment settings
- `results.json`: the patching effects, ablation deltas and induction scores
- `notes.md`: what you expected, what happened, and what you would test next

Then add an entry to `research/RESEARCH_LOG.md`.

---

## What you found

On this input, for this model, the first block's feed-forward network maps the last
token to the kind of word that comes next. Attention isn't needed, because the last
token alone carries the information.

Keep the claim that narrow. It comes from one input to a tiny model, so a stronger
claim needs many inputs. [Activation Patching](../mechinterp/activation_patching.md)
and [Circuit Analysis](../mechinterp/circuits.md) describe the tools in more detail.
