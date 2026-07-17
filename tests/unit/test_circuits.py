"""Unit tests for kamui.mechinterp.circuits.

Coverage target:
    kamui/mechinterp/circuits.py — 100%
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from kamui.mechinterp.circuits import AblationResult, CircuitAblator, find_minimal_circuit
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=40,
        context_length=12,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _model(**overrides: object) -> KAMUITransformer:
    torch.manual_seed(0)
    return KAMUITransformer(_config(**overrides)).eval()


def _metric(logits: Tensor) -> float:
    """A simple higher-is-better scalar: the max final-position logit."""
    return logits[0, -1].max().item()


def _ids(seq: int = 6) -> Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 40, (1, seq))


class TestAblate:
    def test_result_fields(self) -> None:
        ablator = CircuitAblator(_model())
        result = ablator.ablate(["blocks.0.attn.output"], _ids(), _metric)
        assert isinstance(result, AblationResult)
        assert result.components == ["blocks.0.attn.output"]
        assert result.delta == pytest.approx(result.value - result.baseline)

    def test_ablating_everything_equals_embedding_only_model(self) -> None:
        # With every sublayer output zeroed, each block is an identity, so the
        # model reduces to unembed(final_ln(embed(ids))) — exact equality.
        model = _model(n_layers=2)
        ablator = CircuitAblator(model)
        ids = _ids()
        components = [f"blocks.{i}.{c}.output" for i in range(2) for c in ("attn", "ffn")]
        result = ablator.ablate(components, ids, _metric)
        with torch.no_grad():
            expected_logits = model.unembed(model.final_ln(model.embed(ids)))
        assert result.value == pytest.approx(_metric(expected_logits), abs=1e-5)

    def test_hooks_removed_after_ablation(self) -> None:
        model = _model()
        CircuitAblator(model).ablate(["blocks.0.ffn.output"], _ids(), _metric)
        assert len(model.blocks[0].ffn._forward_hooks) == 0

    def test_accepts_1d_ids(self) -> None:
        ablator = CircuitAblator(_model())
        result = ablator.ablate(["blocks.0.attn.output"], _ids()[0], _metric)
        assert isinstance(result.value, float)

    def test_empty_components_raises(self) -> None:
        ablator = CircuitAblator(_model())
        with pytest.raises(ValueError, match="non-empty"):
            ablator.ablate([], _ids(), _metric)

    def test_invalid_component_raises(self) -> None:
        ablator = CircuitAblator(_model())
        with pytest.raises(ValueError, match="not an ablatable"):
            ablator.ablate(["blocks.0.attn.weights"], _ids(), _metric)

    def test_out_of_range_layer_raises(self) -> None:
        ablator = CircuitAblator(_model(n_layers=2))
        with pytest.raises(ValueError, match="not an ablatable"):
            ablator.ablate(["blocks.7.attn.output"], _ids(), _metric)

    def test_bad_ids_rank_raises(self) -> None:
        ablator = CircuitAblator(_model())
        with pytest.raises(ValueError, match="token_ids must be"):
            ablator.ablate(["blocks.0.attn.output"], torch.zeros(1, 2, 3).long(), _metric)


class TestMeanAblate:
    def test_result_fields(self) -> None:
        ablator = CircuitAblator(_model())
        result = ablator.mean_ablate(["blocks.0.attn.output"], _ids(), _ids(), _metric)
        assert isinstance(result, AblationResult)
        assert isinstance(result.delta, float)

    def test_mean_ablation_with_identical_baseline_differs_from_zero(self) -> None:
        # Mean ablation replaces with the mean activation, not zeros — for a
        # non-degenerate model the two interventions give different metrics.
        ablator = CircuitAblator(_model())
        ids = _ids()
        zero = ablator.ablate(["blocks.0.attn.output"], ids, _metric)
        mean = ablator.mean_ablate(["blocks.0.attn.output"], ids, ids, _metric)
        assert mean.value != pytest.approx(zero.value, abs=1e-8)

    def test_hooks_removed(self) -> None:
        model = _model()
        CircuitAblator(model).mean_ablate(["blocks.1.ffn.output"], _ids(), _ids(), _metric)
        assert len(model.blocks[1].ffn._forward_hooks) == 0

    def test_invalid_component_raises(self) -> None:
        ablator = CircuitAblator(_model())
        with pytest.raises(ValueError, match="not an ablatable"):
            ablator.mean_ablate(["embed.output"], _ids(), _ids(), _metric)


class TestFindMinimalCircuit:
    def test_returns_subset_of_components(self) -> None:
        model = _model(n_layers=2)
        circuit = find_minimal_circuit(model, _ids(), _metric, threshold=0.5)
        all_components = {f"blocks.{i}.{c}.output" for i in range(2) for c in ("attn", "ffn")}
        assert set(circuit).issubset(all_components)
        assert len(circuit) >= 1

    def test_threshold_one_keeps_more_than_loose_threshold(self) -> None:
        # A stricter threshold can never yield a smaller circuit.
        model = _model(n_layers=2)
        strict = find_minimal_circuit(model, _ids(), _metric, threshold=1.0)
        loose = find_minimal_circuit(model, _ids(), _metric, threshold=0.01)
        assert len(strict) >= len(loose)

    def test_no_component_removable_keeps_full_circuit(self) -> None:
        # Metric = negative distance from the intact model's logits.  Full
        # performance is exactly 0; any ablation makes it negative, dropping
        # below the floor — so the greedy loop must stop with everything kept.
        model = _model(n_layers=2)
        ids = _ids()
        with torch.no_grad():
            full = model(ids)

        def exact_metric(logits: Tensor) -> float:
            return -float((logits - full).abs().sum().item())

        circuit = find_minimal_circuit(model, ids, exact_metric, threshold=1.0)
        assert len(circuit) == 4  # all attn/ffn components of 2 layers

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be"):
            find_minimal_circuit(_model(), _ids(), _metric, threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be"):
            find_minimal_circuit(_model(), _ids(), _metric, threshold=0.0)
