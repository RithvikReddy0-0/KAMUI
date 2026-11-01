# Logit Lens

## What it is

The logit lens projects the residual stream at each layer back to
vocabulary space using the model's final unembedding matrix. It answers:

> "If the model were to output right now (at layer L), what would it predict?"

This reveals how predictions evolve with depth — some facts are resolved
early, others require the full depth of the network.

---

## How it works

```text
for layer L in 0..n_layers:
    residual_L = residual_stream_at_layer_L      # (1, S, D)
    probs_L    = softmax(unembed(final_ln(residual_L)))  # (1, S, V)
```

The key insight: the final unembedding matrix can be applied to the
residual stream at *any* layer, not just the last one. This works because
the residual stream always has shape `(B, S, D)` — the same at every layer.

---

## Usage

The examples on this page use the `model` and `tokenizer` from the
[Quickstart](../tutorials/quickstart.md). `run` takes token IDs, not a string.

```python
import torch

from kamui.mechinterp import LogitLens

lens = LogitLens(model, tokenizer)

# Run on a prompt
ids = torch.tensor(tokenizer.encode("the cat sat on the "))
result = lens.run(ids)

# Plot: rows = layers, columns = token positions, cell = top-1 prediction
result.plot()

# Inspect a specific position across all layers
last = len(result.tokens) - 1
result.plot_position(pos=last)   # last token — where the prediction is made

# Get raw probabilities
probs = result.probs    # (n_layers+1, S, V)
```

---

## What to look for

Row 0 is the residual stream before any block has run, and row L is the stream after
L blocks.

In the Quickstart's model, row 0 predicts a verb phrase and row 1 predicts an object,
so the choice is made in the first block.

In a model large enough to store facts, you can run the same check on a factual
completion:

- Early layers predict high-frequency tokens, such as function words
- At some layer L\*, the correct answer first appears near the top
- By the final layer, the correct answer dominates

The layer L\* is a good place to look for the association, but the lens only shows
where an answer becomes visible. Use `ActivationPatcher` to verify it causally.

---

## Interpreting the output

`LogitLensResult` fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `probs` | `(L+1, S, V)` | Probability distribution at each layer and position |
| `top_tokens` | `(L+1, S, k)` | Top-k predicted tokens at each layer and position |
| `tokens` | `list[str]` | Input token strings (for axis labels) |
| `tokenizer` | | The tokenizer, used to label plots |

`run` takes `top_k` to set `k`, which defaults to 5.

---

## Reference

nostalgebraist (2020). Interpreting GPT: the logit lens.
LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru
