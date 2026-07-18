"""Circuit detection and ablation utilities.

A circuit is a minimal subset of model components (attention sublayers, FFN
sublayers) that is causally sufficient to implement a specific behaviour.
Identifying circuits is the ultimate goal of mechanistic interpretability.

Responsibilities:
    - ``CircuitAblator.ablate(components, token_ids, metric)``:
        Zero-ablate the listed components (by output hook-point string, e.g.
        ``"blocks.2.attn.output"``), run the model, and return an
        ``AblationResult`` with the metric value and its delta from the
        un-ablated baseline.

    - ``CircuitAblator.mean_ablate(components, token_ids, baseline_ids, metric)``:
        Replace each component's activation with its mean activation over a
        baseline batch instead of zeros.  Mean ablation removes the
        component's task-specific contribution while preserving its baseline
        statistics — more principled than zeroing.

    - ``find_minimal_circuit(model, task_ids, task_metric, threshold)``:
        Greedy search for a minimal component set that retains ``threshold``
        fraction of full-model performance (``task_metric`` is
        higher-is-better).  Starts from all components and repeatedly drops
        the one whose removal (i.e. ablation) hurts least.  Greedy and
        approximate — exact minimal circuits are exponentially hard.

Circuit analysis workflow:
    1. Define a task and a scalar metric (e.g. correct-vs-incorrect logit diff).
    2. Use ``ActivationPatcher`` to find causally relevant layers.
    3. Use ``CircuitAblator`` to test which components within them matter.
    4. Verify by ablating everything EXCEPT the circuit and checking the
       metric is preserved — exactly what ``find_minimal_circuit`` automates.

References:
    Wang, K. et al. (2022). Interpretability in the Wild (IOI).
    https://arxiv.org/abs/2211.00593
    Conmy, A. et al. (2023). Towards Automated Circuit Discovery.
    https://arxiv.org/abs/2304.14997

Implemented in: Phase 4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kamui.model.transformer import KAMUITransformer

#: A metric maps logits ``(1, S, V)`` to a scalar (higher = better).
Metric = Callable[[Tensor], float]


@dataclass
class AblationResult:
    """The effect of one ablation experiment.

    Attributes:
        value:      Metric value with the components ablated.
        baseline:   Metric value of the intact model.
        components: The ablated hook points.
    """

    value: float
    baseline: float
    components: list[str]

    @property
    def delta(self) -> float:
        """Metric change caused by the ablation (``value - baseline``)."""
        return self.value - self.baseline


def _validate_components(model: KAMUITransformer, components: list[str]) -> list[str]:
    """Check each component is an ablatable output point; return module paths."""
    if not components:
        raise ValueError("components must be a non-empty list of hook points")
    n_layers = model.config.n_layers
    valid = {f"blocks.{i}.{c}.output" for i in range(n_layers) for c in ("attn", "ffn")}
    paths = []
    for component in components:
        if component not in valid:
            raise ValueError(
                f"'{component}' is not an ablatable component "
                f"(use blocks.{{i}}.attn.output / blocks.{{i}}.ffn.output)"
            )
        paths.append(component.rsplit(".", 1)[0])
    return paths


def _as_batch(ids: Tensor) -> Tensor:
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError(f"token_ids must be (S,) or (B, S), got shape {tuple(ids.shape)}")
    return ids


class CircuitAblator:
    """Zero- and mean-ablation of model components.

    Attributes:
        model: A trained ``KAMUITransformer``.
    """

    def __init__(self, model: KAMUITransformer) -> None:
        """Create an ablator over ``model``.

        Args:
            model: A ``KAMUITransformer``.
        """
        self.model = model

    def _run_ablated(
        self, ids: Tensor, module_paths: list[str], replacements: dict[str, Tensor] | None
    ) -> Tensor:
        """Run the model with each listed module's output replaced.

        ``replacements`` maps module path → replacement tensor (broadcastable
        to the output).  When None, outputs are zeroed.
        """
        modules = dict(self.model.named_modules())
        handles = []
        for path in module_paths:

            def _make(path: str) -> Callable[[nn.Module, tuple, Tensor], Tensor]:
                def hook(_m: nn.Module, _inp: tuple, out: Tensor) -> Tensor:
                    if replacements is None:
                        return torch.zeros_like(out)
                    return replacements[path].expand_as(out)

                return hook

            handles.append(modules[path].register_forward_hook(_make(path)))
        try:
            return self.model(ids)
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def ablate(self, components: list[str], token_ids: Tensor, metric: Metric) -> AblationResult:
        """Zero-ablate ``components`` and measure the metric change.

        Args:
            components: Output hook points to ablate
                (``"blocks.{i}.attn.output"`` / ``"blocks.{i}.ffn.output"``).
            token_ids:  Input token IDs ``(S,)`` or ``(B, S)``.
            metric:     Maps logits to a scalar (higher = better).

        Returns:
            An ``AblationResult``.

        Raises:
            ValueError: If ``components`` is empty or contains an invalid point.
        """
        paths = _validate_components(self.model, components)
        ids = _as_batch(token_ids)
        self.model.eval()

        baseline = metric(self.model(ids))
        value = metric(self._run_ablated(ids, paths, replacements=None))
        return AblationResult(value=value, baseline=baseline, components=list(components))

    @torch.no_grad()
    def mean_ablate(
        self,
        components: list[str],
        token_ids: Tensor,
        baseline_ids: Tensor,
        metric: Metric,
    ) -> AblationResult:
        """Replace components' activations with their baseline-batch means.

        Args:
            components:   Output hook points to ablate.
            token_ids:    Task input ``(S,)`` or ``(B, S)``.
            baseline_ids: Baseline batch ``(S,)`` or ``(B, S)`` used to compute
                each component's mean activation (averaged over batch and
                positions, broadcast back at ablation time).
            metric:       Maps logits to a scalar (higher = better).

        Returns:
            An ``AblationResult``.

        Raises:
            ValueError: If ``components`` is empty or contains an invalid point.
        """
        paths = _validate_components(self.model, components)
        ids = _as_batch(token_ids)
        base = _as_batch(baseline_ids)
        self.model.eval()

        # Cache mean activations over the baseline batch.
        means: dict[str, Tensor] = {}
        modules = dict(self.model.named_modules())
        handles = []
        for path in paths:

            def _make(path: str) -> Callable[[nn.Module, tuple, Tensor], None]:
                def hook(_m: nn.Module, _inp: tuple, out: Tensor) -> None:
                    means[path] = out.detach().mean(dim=(0, 1), keepdim=True)  # (1, 1, D)

                return hook

            handles.append(modules[path].register_forward_hook(_make(path)))
        try:
            self.model(base)
        finally:
            for handle in handles:
                handle.remove()

        baseline_metric = metric(self.model(ids))
        value = metric(self._run_ablated(ids, paths, replacements=means))
        return AblationResult(value=value, baseline=baseline_metric, components=list(components))


def find_minimal_circuit(
    model: KAMUITransformer,
    task_ids: Tensor,
    task_metric: Metric,
    threshold: float = 0.95,
) -> list[str]:
    """Greedy search for a minimal circuit preserving task performance.

    The circuit starts as every ``attn``/``ffn`` output component.  At each
    round, ablating all *non-circuit* components, the search drops the kept
    component whose additional removal leaves the highest metric — as long as
    that metric stays at or above ``threshold * full_performance``.

    Args:
        model:       A ``KAMUITransformer``.
        task_ids:    Task input token IDs ``(S,)`` or ``(B, S)``.
        task_metric: Maps logits to a scalar (higher = better).
        threshold:   Fraction of full-model performance to preserve, in (0, 1].

    Returns:
        The kept components (a subset of all ``blocks.{i}.{attn,ffn}.output``).

    Raises:
        ValueError: If ``threshold`` is out of range.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    ablator = CircuitAblator(model)
    n_layers = model.config.n_layers
    all_components = [f"blocks.{i}.{c}.output" for i in range(n_layers) for c in ("attn", "ffn")]

    ids = _as_batch(task_ids)
    model.eval()
    with torch.no_grad():
        full_performance = task_metric(model(ids))
    floor = threshold * full_performance

    circuit = list(all_components)
    while len(circuit) > 1:
        best_candidate: str | None = None
        best_value = float("-inf")
        for candidate in circuit:
            ablated = [c for c in all_components if c not in circuit or c == candidate]
            value = ablator.ablate(ablated, ids, task_metric).value
            if value > best_value:
                best_value = value
                best_candidate = candidate
        if best_value >= floor and best_candidate is not None:
            circuit.remove(best_candidate)
        else:
            break
    return circuit
