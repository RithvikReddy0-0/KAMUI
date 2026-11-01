# Circuit Analysis

## What is a circuit?

A circuit is the **minimal subset of model components** (attention heads,
MLP layers) that is causally sufficient to implement a specific behaviour.

Finding a circuit means:
1. You have identified *which* components matter
2. You have verified this *causally* (ablating non-circuit components
   doesn't hurt performance; ablating circuit components does)
3. You understand *what algorithm* the circuit implements

---

## CircuitAblator

The examples on this page use a `model`, a `token_ids` tensor and a `metric`
function. A metric turns logits into one number, where higher is better:

```python
target = tokenizer.encode("mat . the ")[0]


def metric(logits):
    return logits[0, -1, target].item()
```

Components are named `blocks.{i}.attn.output` or `blocks.{i}.ffn.output`:

```python
from kamui.mechinterp import CircuitAblator

ablator = CircuitAblator(model)

# Zero-ablate specific components and measure effect
result = ablator.ablate(
    components=["blocks.0.attn.output", "blocks.1.attn.output"],
    token_ids=token_ids,
    metric=metric,
)
print(result.delta)   # how much the metric changed

# Mean ablation
result = ablator.mean_ablate(
    components=["blocks.0.attn.output"],
    token_ids=token_ids,
    baseline_ids=baseline_ids,
    metric=metric,
)
```

`AblationResult` holds `value`, `baseline` and `components`, and `delta` is `value`
minus `baseline`. A negative delta means removing the components hurt.

### Zero ablation vs. mean ablation

**Zero ablation**: replace the component's output with zeros.

- Fast and simple
- Can produce out-of-distribution activations (the model has never
  seen a zero activation stream at that point)

**Mean ablation**: replace with the mean activation over a baseline dataset.

- Removes the task-specific contribution while keeping baseline statistics intact
- Usually the better choice for results you intend to report

---

## Circuit discovery workflow

```text
Step 1: Identify candidate layers
    → ActivationPatcher.patch_all_layers()
    → Find layers with a large effect (0.3 is a common rule of thumb)

Step 2: Identify candidate heads within those layers
    → ActivationPatcher.patch_all_heads()
    → Find (layer, head) pairs with a large effect

Step 3: Verify the circuit
    → CircuitAblator: ablate everything EXCEPT the circuit
    → Verify performance is preserved (circuit is sufficient)
    → CircuitAblator: ablate ONLY the circuit components
    → Verify performance drops (circuit is necessary)

Step 4: Understand the algorithm
    → AttentionVisualizer: what does each head attend to?
    → LogitLens: what does each component add to the prediction?
    → Write a natural-language description of the algorithm
```

`CircuitAblator` works on whole `attn` and `ffn` components. To remove individual
heads, use `InductionHeadDetector.ablate_and_measure`.

---

## Automatic circuit discovery

`find_minimal_circuit` implements a greedy search:

```python
from kamui.mechinterp.circuits import find_minimal_circuit

circuit = find_minimal_circuit(
    model=model,
    task_ids=token_ids,
    task_metric=metric,
    threshold=0.95,   # preserve 95% of full-model performance
)
print(circuit)
```

On the [tutorial's](../tutorials/your_first_interpretability_experiment.md) model and
input, it returns `['blocks.0.ffn.output', 'blocks.1.ffn.output']`.

The threshold is a fraction of the full model's metric, so the metric should be
positive and meaningful on its own. A log-probability or a difference that can change
sign won't work as a threshold.

**Warning**: greedy search is approximate. It may miss circuits where
multiple components together matter but individually don't. Always verify
the result manually.

---

## Example: the induction circuit

This is the circuit as described by Olsson et al. (2022). KAMUI hasn't reproduced it,
as [Induction Heads](induction_heads.md) explains.

```text
Components: a previous-token head, and an induction head in a later layer

Algorithm:
    Earlier layer, previous-token head:
        At position t, attends to t-1
        Writes a "previous token" signal to the residual stream

    Later layer, induction head:
        Reads the "previous token" signal
        Attends to the token after the last occurrence of the current token
        Boosts that token's next-token prediction
```

---

## References

Wang et al. (2022). Interpretability in the Wild.
https://arxiv.org/abs/2211.00593

Conmy et al. (2023). Towards Automated Circuit Discovery.
NeurIPS 2023. https://arxiv.org/abs/2304.14997
