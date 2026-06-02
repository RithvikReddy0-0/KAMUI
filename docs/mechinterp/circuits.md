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

```python
from kamui.mechinterp import CircuitAblator

ablator = CircuitAblator(model)

# Zero-ablate specific components and measure effect
result = ablator.ablate(
    components=["blocks.1.attn", "blocks.2.attn"],
    token_ids=token_ids,
    metric=logit_difference,
)
print(result.delta)   # how much the metric changed

# Mean ablation (more principled than zero ablation)
result = ablator.mean_ablate(
    components=["blocks.1.attn"],
    token_ids=token_ids,
    baseline_ids=random_baseline_ids,
    metric=logit_difference,
)
```

### Zero ablation vs. mean ablation

**Zero ablation**: replace the component's output with zeros.
- Fast and simple
- Can produce out-of-distribution activations (the model has never
  seen a zero activation stream at that point)

**Mean ablation**: replace with the mean activation over a baseline dataset.
- More principled: removes the task-specific contribution while
  keeping baseline statistics intact
- Preferred for published results

---

## Circuit discovery workflow

```
Step 1: Identify candidate layers
    → ActivationPatcher.patch_all_layers()
    → Find layers with effect > 0.3

Step 2: Identify candidate heads within those layers
    → ActivationPatcher.patch_all_heads()
    → Find (layer, head) pairs with effect > 0.3

Step 3: Verify the circuit
    → CircuitAblator: ablate ONLY the circuit components
    → Verify performance is preserved (circuit is sufficient)
    → CircuitAblator: ablate everything EXCEPT the circuit
    → Verify performance drops (circuit is necessary)

Step 4: Understand the algorithm
    → AttentionVisualizer: what does each head attend to?
    → LogitLens: what does each head contribute to the residual stream?
    → Write a natural-language description of the algorithm
```

---

## Automatic circuit discovery

`find_minimal_circuit` implements a greedy search:

```python
from kamui.mechinterp.circuits import find_minimal_circuit

circuit = find_minimal_circuit(
    model=model,
    task_ids=task_token_ids,
    task_metric=logit_difference,
    threshold=0.95,   # preserve 95% of full-model performance
)
print(circuit)
# ["blocks.1.attn.head_2", "blocks.1.attn.head_5", "blocks.0.attn.head_0"]
```

**Warning**: greedy search is approximate. It may miss circuits where
multiple components together matter but individually don't. Always verify
the result manually.

---

## Example: the induction circuit

```
Components: {(layer=0, head=?), (layer=1, head=2), (layer=1, head=5)}

Algorithm:
    Layer 0, prev-token head:
        At position t, attends to t-1
        Writes "previous token" signal to residual stream

    Layer 1, induction heads (2 and 5):
        Read "previous token" signal
        Attend to the last occurrence of (current token preceded by prev token)
        Boost the next-token prediction accordingly

This is sufficient to explain in-context copying behaviour.
```

---

## References

Wang et al. (2022). Interpretability in the Wild.
https://arxiv.org/abs/2211.00593

Conmy et al. (2023). Towards Automated Circuit Discovery.
NeurIPS 2023. https://arxiv.org/abs/2304.14997
