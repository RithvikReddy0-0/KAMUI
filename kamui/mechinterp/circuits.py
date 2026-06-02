"""Circuit detection and ablation utilities.

A circuit is a minimal subset of model components (attention heads, MLP
layers) that are causally sufficient to implement a specific behaviour.
Identifying circuits is the ultimate goal of mechanistic interpretability.

Responsibilities:
    - ``CircuitAblator``:
        Ablate (zero out) any set of components and measure the effect on
        model output for a given task.

        ``CircuitAblator.ablate(
              components: list[str],
              token_ids: Tensor,
              metric: Callable
          ) -> AblationResult``:
            Zero-ablate the specified components (by hook point string).
            Run the model. Compute ``metric(logits)``.
            Return ``AblationResult`` with the metric value and the delta
            from the full-model baseline.

        ``CircuitAblator.mean_ablate(
              components: list[str],
              token_ids: Tensor,
              baseline_ids: Tensor,
              metric: Callable
          ) -> AblationResult``:
            Replace the specified component's activation with its mean
            activation over a baseline dataset, rather than zeroing it.
            Mean ablation is more principled than zero ablation: it removes
            the component's task-specific contribution while preserving
            baseline statistics.

    - ``find_minimal_circuit(
          model, task_ids, task_metric, threshold=0.95
      ) -> list[str]``:
        Greedy search for the minimal set of components that achieves
        ``threshold`` fraction of full-model performance on ``task_metric``.
        Start with all components. Iteratively remove the component whose
        ablation causes the smallest performance drop. Stop when removing
        any component drops performance below ``threshold``.

        Note: this is exponentially hard in general.  The greedy heuristic
        is approximate but effective for small circuits.

Circuit analysis workflow:
    1. Define a task and a scalar metric (e.g. logit difference between
       the correct and incorrect token)
    2. Use ActivationPatcher to identify which layers are causally relevant
    3. Use CircuitAblator to test which heads within those layers matter
    4. Verify the identified circuit by ablating everything EXCEPT the
       circuit and checking that performance is preserved

References:
    Wang, K. et al. (2022). Interpretability in the Wild: a Circuit for
    Indirect Object Identification in GPT-2 small.
    https://arxiv.org/abs/2211.00593

    Conmy, A. et al. (2023). Towards Automated Circuit Discovery for
    Mechanistic Interpretability. NeurIPS 2023.
    https://arxiv.org/abs/2304.14997

Implemented in: Phase 4, Weeks 19–20
"""

# Implementation begins in Phase 4.
