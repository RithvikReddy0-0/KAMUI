"""Unit tests for kamui.training.ema (exponential moving average of weights).

Coverage target:
    kamui/training/ema.py — 100%
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kamui.training.ema import EMA


def _model() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(3, 2)


def _params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


class TestInit:
    @pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
    def test_invalid_decay_raises(self, decay: float) -> None:
        with pytest.raises(ValueError, match="decay must be in"):
            EMA(_model(), decay=decay)

    def test_shadow_starts_as_a_copy_of_the_weights(self) -> None:
        model = _model()
        ema = EMA(model)
        for name, param in model.named_parameters():
            assert torch.equal(ema.shadow[name], param)
            assert ema.shadow[name] is not param  # a copy, not an alias
        assert ema.num_updates == 0

    def test_frozen_parameters_are_not_tracked(self) -> None:
        model = _model()
        model.bias.requires_grad_(False)
        ema = EMA(model)
        assert set(ema.shadow) == {"weight"}


class TestUpdate:
    def test_update_matches_formula(self) -> None:
        model = _model()
        ema = EMA(model, decay=0.9)
        old = _params(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)
        for name, param in model.named_parameters():
            expected = 0.9 * old[name] + 0.1 * param
            assert torch.allclose(ema.shadow[name], expected, atol=1e-6)
        assert ema.num_updates == 1

    def test_zero_decay_tracks_weights_exactly(self) -> None:
        model = _model()
        ema = EMA(model, decay=0.0)
        with torch.no_grad():
            model.weight.fill_(7.0)
        ema.update(model)
        assert torch.equal(ema.shadow["weight"], model.weight)

    def test_update_does_not_change_the_model(self) -> None:
        model = _model()
        ema = EMA(model, decay=0.5)
        before = _params(model)
        ema.update(model)
        for name, param in model.named_parameters():
            assert torch.equal(param, before[name])


class TestSwap:
    def _diverged(self) -> tuple[nn.Linear, EMA, dict[str, torch.Tensor]]:
        """A model whose raw weights differ from its EMA shadow."""
        model = _model()
        ema = EMA(model, decay=0.9)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        return model, ema, _params(model)

    def test_copy_to_loads_shadow(self) -> None:
        model, ema, _ = self._diverged()
        ema.copy_to(model)
        for name, param in model.named_parameters():
            assert torch.equal(param, ema.shadow[name])

    def test_store_restore_roundtrip(self) -> None:
        model, ema, raw = self._diverged()
        ema.store(model)
        ema.copy_to(model)
        ema.restore(model)
        for name, param in model.named_parameters():
            assert torch.equal(param, raw[name])

    def test_restore_without_store_raises(self) -> None:
        model, ema, _ = self._diverged()
        with pytest.raises(RuntimeError, match="nothing to restore"):
            ema.restore(model)

    def test_average_parameters_swaps_in_and_out(self) -> None:
        model, ema, raw = self._diverged()
        with ema.average_parameters(model):
            for name, param in model.named_parameters():
                assert torch.equal(param, ema.shadow[name])
        for name, param in model.named_parameters():
            assert torch.equal(param, raw[name])

    def test_average_parameters_restores_on_exception(self) -> None:
        model, ema, raw = self._diverged()
        with pytest.raises(KeyError), ema.average_parameters(model):
            raise KeyError("boom")
        for name, param in model.named_parameters():
            assert torch.equal(param, raw[name])


class TestStateDict:
    def test_roundtrip(self) -> None:
        model = _model()
        ema = EMA(model, decay=0.9)
        with torch.no_grad():
            model.weight.add_(2.0)
        ema.update(model)

        other = EMA(_model(), decay=0.5)
        other.load_state_dict(ema.state_dict())
        assert other.decay == 0.9
        assert other.num_updates == 1
        for name in ema.shadow:
            assert torch.equal(other.shadow[name], ema.shadow[name])

    def test_state_dict_is_a_copy(self) -> None:
        ema = EMA(_model())
        state = ema.state_dict()
        state["shadow"]["weight"].fill_(99.0)
        assert not torch.equal(ema.shadow["weight"], state["shadow"]["weight"])

    def test_mismatched_names_raise(self) -> None:
        ema = EMA(_model())
        bad = {"decay": 0.9, "num_updates": 0, "shadow": {"other": torch.zeros(1)}}
        with pytest.raises(ValueError, match="parameter names differ"):
            ema.load_state_dict(bad)
