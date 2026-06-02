# Logit Lens

## What it is

The logit lens projects the residual stream at each layer back to
vocabulary space using the model's final unembedding matrix. It answers:

> "If the model were to output right now (at layer L), what would it predict?"

This reveals how predictions evolve with depth — some facts are resolved
early, others require the full depth of the network.

---

## How it works

```
for layer L in 0..n_layers:
    residual_L = residual_stream_at_layer_L      # (B, S, D)
    probs_L    = softmax(unembed(final_ln(residual_L)))  # (B, S, V)
```

The key insight: the final unembedding matrix can be applied to the
residual stream at *any* layer, not just the last one. This works because
the residual stream always has shape `(B, S, D)` — the same at every layer.

---

## Usage

```python
from kamui.mechinterp import LogitLens

lens = LogitLens(model, tokenizer)

# Run on a prompt
result = lens.run("The Eiffel Tower is located in the city of")

# Plot: rows = layers, columns = token positions, cell = top-1 prediction
result.plot()

# Inspect a specific position across all layers
result.plot_position(pos=-1)   # last token — where the prediction is made

# Get raw probabilities
probs = result.probs    # (n_layers+1, S, V)
```

---

## What to look for

**Factual completion** (`"The Eiffel Tower is in [Paris]"`):
- At layer 0: model predicts high-frequency tokens (function words)
- At layer L*: the correct answer (Paris) first appears in top-3
- At final layer: Paris dominates

The layer L* where the correct answer emerges is where the factual
association is likely stored. Use `ActivationPatcher` to verify causally.

**Syntax vs. semantics**:
- Part-of-speech patterns typically emerge early (layers 0–2)
- Semantic content emerges later (layers 3+)

---

## Interpreting the output

`LogitLensResult` fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `probs` | `(L+1, S, V)` | Probability distribution at each layer and position |
| `top_tokens` | `(L+1, S, k)` | Top-k predicted tokens at each layer and position |
| `tokens` | `list[str]` | Input token strings (for axis labels) |

---

## Reference

nostalgebraist (2020). Interpreting GPT: the logit lens.
LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru
