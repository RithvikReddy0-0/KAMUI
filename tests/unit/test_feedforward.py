"""Unit tests for kamui.model.feedforward.

Coverage target:
    kamui/model/feedforward.py — 100%

Test categories:
    - construction: submodule types, shapes, dropout probability
    - forward: shape preservation, op ordering, GELU nonlinearity, dropout
      train/eval behaviour, error paths
    - architecture constraints: no normalization layers, parameter count
      matches ModelConfig.feedforward_parameters
    - gradient flow, repr
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kamui.model.config import ModelConfig
from kamui.model.feedforward import FeedForward
from kamui.model.normalization import LayerNorm, RMSNorm


def _small_config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ff=64,
        vocab_size=50,
        context_length=12,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


# ===========================================================================
# Construction
# ===========================================================================

class TestFeedForwardConstruction:
    def test_fc_in_is_expansion_linear(self) -> None:
        ff = FeedForward(_small_config(d_model=16, d_ff=64))
        assert isinstance(ff.fc_in, nn.Linear)
        assert ff.fc_in.in_features == 16
        assert ff.fc_in.out_features == 64

    def test_fc_out_is_projection_linear(self) -> None:
        ff = FeedForward(_small_config(d_model=16, d_ff=64))
        assert isinstance(ff.fc_out, nn.Linear)
        assert ff.fc_out.in_features == 64
        assert ff.fc_out.out_features == 16

    def test_activation_is_gelu(self) -> None:
        ff = FeedForward(_small_config())
        assert isinstance(ff.activation, nn.GELU)

    def test_dropout_probability_matches_config(self) -> None:
        ff = FeedForward(_small_config(dropout=0.3))
        assert isinstance(ff.dropout, nn.Dropout)
        assert ff.dropout.p == 0.3

    def test_both_linears_have_bias(self) -> None:
        ff = FeedForward(_small_config())
        assert ff.fc_in.bias is not None
        assert ff.fc_out.bias is not None


# ===========================================================================
# Forward
# ===========================================================================

class TestFeedForwardForward:
    def test_output_shape_preserved_3d(self) -> None:
        ff = FeedForward(_small_config(d_model=16))
        out = ff(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_output_shape_preserved_2d(self) -> None:
        # FFN is position-wise: works on any (..., d_model).
        ff = FeedForward(_small_config(d_model=16))
        out = ff(torch.randn(7, 16))
        assert out.shape == (7, 16)

    def test_hidden_dimension_is_d_ff(self) -> None:
        ff = FeedForward(_small_config(d_model=16, d_ff=64))
        h = ff.fc_in(torch.randn(2, 5, 16))
        assert h.shape == (2, 5, 64)

    def test_op_ordering_matches_manual(self) -> None:
        # eval() disables dropout, so output == fc_out(gelu(fc_in(x))).
        ff = FeedForward(_small_config(d_model=16, d_ff=64))
        ff.eval()
        x = torch.randn(3, 4, 16)
        expected = ff.fc_out(ff.activation(ff.fc_in(x)))
        assert torch.allclose(ff(x), expected, atol=1e-6)

    def test_gelu_does_not_zero_negatives(self) -> None:
        # Unlike ReLU, GELU keeps a small negative response near zero.
        # Verify the activation itself matches reference GELU on a negative.
        ff = FeedForward(_small_config())
        neg = torch.tensor([-1.0])
        out = ff.activation(neg)
        assert out.item() < 0.0          # not clamped to 0 like ReLU
        assert out.item() == pytest.approx(F.gelu(neg).item(), abs=1e-6)

    def test_dropout_active_in_train_mode(self) -> None:
        ff = FeedForward(_small_config(d_model=16, d_ff=512, dropout=0.5))
        ff.train()
        torch.manual_seed(0)
        x = torch.randn(4, 8, 16)
        # Two forward passes differ when dropout is active.
        out1 = ff(x)
        out2 = ff(x)
        assert not torch.allclose(out1, out2)

    def test_dropout_inactive_in_eval_mode(self) -> None:
        ff = FeedForward(_small_config(dropout=0.5))
        ff.eval()
        x = torch.randn(2, 5, 16)
        assert torch.allclose(ff(x), ff(x))

    def test_type_error_on_non_tensor(self) -> None:
        ff = FeedForward(_small_config())
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            ff([1.0] * 16)  # type: ignore[arg-type]

    def test_value_error_on_dim_mismatch(self) -> None:
        ff = FeedForward(_small_config(d_model=16))
        with pytest.raises(ValueError, match="does not match d_model"):
            ff(torch.randn(2, 5, 8))


# ===========================================================================
# Architecture constraints
# ===========================================================================

class TestFeedForwardArchitecture:
    def test_no_normalization_layers(self) -> None:
        # The FFN must NOT contain any normalization — that belongs to block.py.
        ff = FeedForward(_small_config())
        for module in ff.modules():
            assert not isinstance(
                module,
                (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, LayerNorm, RMSNorm),
            )

    def test_parameter_count_matches_config(self) -> None:
        # With n_layers=1, config.feedforward_parameters is the per-layer count.
        cfg = _small_config(n_layers=1, d_model=16, d_ff=64)
        ff = FeedForward(cfg)
        n_params = sum(p.numel() for p in ff.parameters())
        assert n_params == cfg.feedforward_parameters

    def test_parameter_count_formula(self) -> None:
        cfg = _small_config(d_model=16, d_ff=64)
        ff = FeedForward(cfg)
        n_params = sum(p.numel() for p in ff.parameters())
        # 2 * D * F + F + D  (two weight matrices + two biases)
        assert n_params == 2 * 16 * 64 + 64 + 16

    def test_gradient_flow(self) -> None:
        ff = FeedForward(_small_config())
        x = torch.randn(2, 5, 16, requires_grad=True)
        ff(x).sum().backward()
        assert ff.fc_in.weight.grad is not None
        assert ff.fc_in.bias.grad is not None
        assert ff.fc_out.weight.grad is not None
        assert ff.fc_out.bias.grad is not None
        assert x.grad is not None

    def test_repr(self) -> None:
        ff = FeedForward(_small_config(d_model=16, d_ff=64, dropout=0.1))
        r = repr(ff)
        assert "FeedForward" in r
        assert "d_model=16" in r
        assert "d_ff=64" in r
