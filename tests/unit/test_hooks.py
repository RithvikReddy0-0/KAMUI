"""Unit tests for kamui.hooks (registry + manager).

The most critical property of the hook system is transparency: attaching
hooks must never change model output.  These tests enforce that, plus the
capture, teardown, and error contracts.

Coverage target:
    kamui/hooks/registry.py — 100%
    kamui/hooks/manager.py  — 100%
"""

from __future__ import annotations

import pytest
import torch

from kamui.hooks import HookManager, HookRegistry
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=50,
        context_length=12,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _model() -> KAMUITransformer:
    return KAMUITransformer(_config()).eval()


def _ids() -> torch.Tensor:
    return torch.randint(0, 50, (2, 7))


# ===========================================================================
# HookRegistry
# ===========================================================================


class TestHookRegistry:
    def test_all_points_structure(self) -> None:
        points = HookRegistry.all_points(_config(n_layers=2))
        assert points == [
            "embed.output",
            "blocks.0.attn.output",
            "blocks.0.attn.weights",
            "blocks.0.ffn.mid",
            "blocks.0.ffn.output",
            "blocks.1.attn.output",
            "blocks.1.attn.weights",
            "blocks.1.ffn.mid",
            "blocks.1.ffn.output",
            "unembed.input",
        ]

    def test_all_points_scales_with_layers(self) -> None:
        p1 = HookRegistry.all_points(_config(n_layers=1))
        p3 = HookRegistry.all_points(_config(n_layers=3))
        assert len(p1) == 2 + 4 * 1
        assert len(p3) == 2 + 4 * 3

    def test_validate_true(self) -> None:
        cfg = _config(n_layers=2)
        assert HookRegistry.validate("blocks.1.attn.weights", cfg)
        assert HookRegistry.validate("embed.output", cfg)
        assert HookRegistry.validate("unembed.input", cfg)

    def test_validate_false_typo(self) -> None:
        assert not HookRegistry.validate("blocks.1.att.output", _config(n_layers=2))

    def test_validate_false_out_of_range_layer(self) -> None:
        assert not HookRegistry.validate("blocks.5.attn.output", _config(n_layers=2))


# ===========================================================================
# HookManager — capture
# ===========================================================================


class TestHookManagerCapture:
    def test_embed_output(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("embed", "output")
            model(_ids())
            act = hooks.get("embed.output")
        assert act.shape == (2, 7, 16)

    def test_attn_output(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("blocks.0.attn", "output")
            model(_ids())
            assert hooks.get("blocks.0.attn.output").shape == (2, 7, 16)

    def test_attn_weights(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("blocks.1.attn", "weights")
            model(_ids())
            w = hooks.get("blocks.1.attn.weights")
        assert w.shape == (2, 4, 7, 7)  # (B, H, S, S)
        mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
        assert torch.all(w[0, 0][mask] == 0.0)  # causal

    def test_ffn_mid(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("blocks.0.ffn", "mid")
            model(_ids())
            assert hooks.get("blocks.0.ffn.mid").shape == (2, 7, 32)  # (B, S, F)

    def test_ffn_output(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("blocks.0.ffn", "output")
            model(_ids())
            assert hooks.get("blocks.0.ffn.output").shape == (2, 7, 16)

    def test_unembed_input(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("unembed", "input")
            model(_ids())
            assert hooks.get("unembed.input").shape == (2, 7, 16)

    def test_multiple_points_and_get_all(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("embed", "output")
            hooks.attach("blocks.0.attn", "weights")
            hooks.attach("blocks.1.ffn", "output")
            model(_ids())
            cache = hooks.get_all()
        assert set(cache) == {
            "embed.output",
            "blocks.0.attn.weights",
            "blocks.1.ffn.output",
        }

    def test_attach_is_chainable(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("embed", "output").attach("unembed", "input")
            model(_ids())
            assert "embed.output" in hooks.get_all()
            assert "unembed.input" in hooks.get_all()


# ===========================================================================
# HookManager — invariants
# ===========================================================================


class TestHookManagerInvariants:
    def test_hooks_do_not_change_output(self) -> None:
        model = _model()
        ids = _ids()
        clean = model(ids)
        with HookManager(model) as hooks:
            hooks.attach("embed", "output")
            hooks.attach("blocks.0.attn", "weights")
            hooks.attach("blocks.1.ffn", "mid")
            hooked = model(ids)
        assert torch.allclose(clean, hooked, atol=1e-6)

    def test_hooks_removed_on_exit(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("blocks.0.attn", "output")
            hooks.attach("blocks.0.attn", "weights")
        assert len(model.blocks[0].attn._forward_hooks) == 0
        assert "forward" not in model.blocks[0].attn.__dict__

    def test_hooks_removed_on_exception(self) -> None:
        model = _model()
        attn = model.blocks[0].attn
        with pytest.raises(RuntimeError), HookManager(model) as hooks:
            hooks.attach("blocks.0.attn", "output")
            raise RuntimeError("boom")
        assert len(attn._forward_hooks) == 0

    def test_cache_survives_exit(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("embed", "output")
            model(_ids())
        assert hooks.get("embed.output").shape == (2, 7, 16)

    def test_clear_empties_cache(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("embed", "output")
            model(_ids())
            assert hooks.get_all()
            hooks.clear()
            assert hooks.get_all() == {}

    def test_double_weights_attach_is_idempotent(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("blocks.0.attn", "weights")
            hooks.attach("blocks.0.attn", "weights")  # no double-wrap
            model(_ids())
            assert hooks.get("blocks.0.attn.weights").shape == (2, 4, 7, 7)
        assert "forward" not in model.blocks[0].attn.__dict__


# ===========================================================================
# HookManager — errors
# ===========================================================================


class TestHookManagerErrors:
    def test_invalid_hook_point_raises(self) -> None:
        model = _model()
        with HookManager(model) as hooks, pytest.raises(ValueError, match="not a valid hook point"):
            hooks.attach("blocks.0.attn", "banana")

    def test_get_missing_raises(self) -> None:
        model = _model()
        with HookManager(model) as hooks:
            hooks.attach("embed", "output")
            with pytest.raises(KeyError, match="not in the cache"):
                hooks.get("embed.output")

    def test_out_of_range_layer_raises(self) -> None:
        model = _model()
        with HookManager(model) as hooks, pytest.raises(ValueError, match="not a valid hook point"):
            hooks.attach("blocks.9.attn", "output")
