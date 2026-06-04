"""Unit tests for kamui.model.normalization.

Coverage target:
    kamui/model/normalization.py — 100%

Test categories:
    - LayerNorm: construction, parameter init, normalization correctness,
      affine transform, numerical stability, reference parity, error paths,
      gradient flow, repr
    - RMSNorm: construction, parameter init, no mean-centering, no bias,
      numerical stability, error paths, gradient flow, repr
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kamui.model.normalization import LayerNorm, RMSNorm


# ===========================================================================
# LayerNorm
# ===========================================================================

class TestLayerNorm:
    def test_weight_and_bias_shape(self) -> None:
        ln = LayerNorm(normalized_shape=16)
        assert ln.weight.shape == (16,)
        assert ln.bias.shape == (16,)

    def test_weight_init_ones_bias_init_zeros(self) -> None:
        ln = LayerNorm(normalized_shape=8)
        assert torch.equal(ln.weight, torch.ones(8))
        assert torch.equal(ln.bias, torch.zeros(8))

    def test_params_are_learnable(self) -> None:
        ln = LayerNorm(normalized_shape=8)
        assert isinstance(ln.weight, nn.Parameter)
        assert isinstance(ln.bias, nn.Parameter)
        assert ln.weight.requires_grad
        assert ln.bias.requires_grad
        assert len(list(ln.parameters())) == 2

    def test_parameter_count(self) -> None:
        ln = LayerNorm(normalized_shape=32)
        # gamma + beta = 2 * D
        assert sum(p.numel() for p in ln.parameters()) == 2 * 32

    def test_eps_stored(self) -> None:
        ln = LayerNorm(normalized_shape=8, eps=1e-6)
        assert ln.eps == 1e-6

    def test_default_eps(self) -> None:
        ln = LayerNorm(normalized_shape=8)
        assert ln.eps == 1e-5

    def test_output_shape_preserved_2d(self) -> None:
        ln = LayerNorm(normalized_shape=16)
        out = ln(torch.randn(4, 16))
        assert out.shape == (4, 16)

    def test_output_shape_preserved_3d(self) -> None:
        ln = LayerNorm(normalized_shape=16)
        out = ln(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_normalizes_to_zero_mean_unit_variance(self) -> None:
        # With default gamma=1, beta=0, each token's features are standardised.
        ln = LayerNorm(normalized_shape=64)
        x = torch.randn(8, 64) * 5 + 3  # arbitrary mean/scale
        out = ln(x)
        per_token_mean = out.mean(dim=-1)
        per_token_var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(per_token_mean, torch.zeros(8), atol=1e-5)
        assert torch.allclose(per_token_var, torch.ones(8), atol=1e-3)

    def test_known_values(self) -> None:
        # x = [1, 2, 3, 4]: mean=2.5, var=1.25, std=sqrt(1.25)
        ln = LayerNorm(normalized_shape=4, eps=0.0 + 1e-12)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = ln(x)
        std = (1.25) ** 0.5
        expected = torch.tensor([[(v - 2.5) / std for v in (1.0, 2.0, 3.0, 4.0)]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_affine_transform_applied(self) -> None:
        ln = LayerNorm(normalized_shape=4)
        with torch.no_grad():
            ln.weight.fill_(2.0)
            ln.bias.fill_(1.0)
        x = torch.randn(3, 4)
        # out = 2 * normalized + 1
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + ln.eps)
        assert torch.allclose(ln(x), 2.0 * norm + 1.0, atol=1e-6)

    def test_matches_torch_reference(self) -> None:
        torch.manual_seed(0)
        ln = LayerNorm(normalized_shape=32, eps=1e-5)
        with torch.no_grad():
            ln.weight.normal_()
            ln.bias.normal_()
        x = torch.randn(4, 7, 32)
        ref = F.layer_norm(x, (32,), ln.weight, ln.bias, eps=1e-5)
        assert torch.allclose(ln(x), ref, atol=1e-5)

    def test_numerical_stability_constant_input(self) -> None:
        # Constant features → variance 0; epsilon prevents division by zero.
        ln = LayerNorm(normalized_shape=8)
        x = torch.full((3, 8), 7.0)
        out = ln(x)
        assert torch.isfinite(out).all()
        # (x - mean) = 0, so output equals beta (zeros).
        assert torch.allclose(out, torch.zeros(3, 8))

    def test_numerical_stability_large_values(self) -> None:
        ln = LayerNorm(normalized_shape=16)
        x = torch.randn(4, 16) * 1e6
        out = ln(x)
        assert torch.isfinite(out).all()

    def test_invalid_normalized_shape(self) -> None:
        with pytest.raises(ValueError, match="normalized_shape must be > 0"):
            LayerNorm(normalized_shape=0)

    def test_invalid_eps(self) -> None:
        with pytest.raises(ValueError, match="eps must be > 0"):
            LayerNorm(normalized_shape=8, eps=0.0)

    def test_invalid_eps_negative(self) -> None:
        with pytest.raises(ValueError, match="eps must be > 0"):
            LayerNorm(normalized_shape=8, eps=-1e-5)

    def test_type_error_on_non_tensor(self) -> None:
        ln = LayerNorm(normalized_shape=4)
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            ln([1.0, 2.0, 3.0, 4.0])  # type: ignore[arg-type]

    def test_value_error_on_dim_mismatch(self) -> None:
        ln = LayerNorm(normalized_shape=16)
        with pytest.raises(ValueError, match="does not match normalized_shape"):
            ln(torch.randn(4, 8))

    def test_gradient_flow(self) -> None:
        ln = LayerNorm(normalized_shape=16)
        x = torch.randn(2, 16, requires_grad=True)
        out = ln(x)
        out.sum().backward()
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None
        assert x.grad is not None

    def test_repr(self) -> None:
        ln = LayerNorm(normalized_shape=16, eps=1e-5)
        r = repr(ln)
        assert "LayerNorm" in r
        assert "normalized_shape=16" in r
        assert "eps=1e-05" in r


# ===========================================================================
# RMSNorm
# ===========================================================================

class TestRMSNorm:
    def test_weight_shape_and_init(self) -> None:
        rms = RMSNorm(normalized_shape=16)
        assert rms.weight.shape == (16,)
        assert torch.equal(rms.weight, torch.ones(16))

    def test_no_bias_attribute(self) -> None:
        rms = RMSNorm(normalized_shape=8)
        # RMSNorm has gamma only — no learnable bias.
        assert not hasattr(rms, "bias")
        assert len(list(rms.parameters())) == 1

    def test_param_is_learnable(self) -> None:
        rms = RMSNorm(normalized_shape=8)
        assert isinstance(rms.weight, nn.Parameter)
        assert rms.weight.requires_grad

    def test_eps_stored(self) -> None:
        rms = RMSNorm(normalized_shape=8, eps=1e-6)
        assert rms.eps == 1e-6

    def test_output_shape_preserved(self) -> None:
        rms = RMSNorm(normalized_shape=16)
        out = rms(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_known_values(self) -> None:
        # x = [3, 4]: mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5)
        rms = RMSNorm(normalized_shape=2, eps=1e-12)
        x = torch.tensor([[3.0, 4.0]])
        out = rms(x)
        denom = (12.5 + 1e-12) ** 0.5
        expected = torch.tensor([[3.0 / denom, 4.0 / denom]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_no_mean_centering(self) -> None:
        # Unlike LayerNorm, a constant nonzero input is NOT zeroed — it is
        # scaled by 1/rms (here rms == |c|), so output ≈ sign(c) per element.
        rms = RMSNorm(normalized_shape=8, eps=1e-12)
        x = torch.full((1, 8), 5.0)
        out = rms(x)
        assert torch.allclose(out, torch.ones(1, 8), atol=1e-5)

    def test_unit_rms_output(self) -> None:
        # With gamma=1, the RMS of the output is ~1.
        rms = RMSNorm(normalized_shape=64)
        x = torch.randn(8, 64) * 10
        out = rms(x)
        out_rms = torch.sqrt((out ** 2).mean(dim=-1))
        assert torch.allclose(out_rms, torch.ones(8), atol=1e-3)

    def test_numerical_stability_zero_input(self) -> None:
        # All-zero input → mean-square 0; epsilon prevents division by zero.
        rms = RMSNorm(normalized_shape=8)
        out = rms(torch.zeros(3, 8))
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros(3, 8))

    def test_numerical_stability_large_values(self) -> None:
        rms = RMSNorm(normalized_shape=16)
        out = rms(torch.randn(4, 16) * 1e6)
        assert torch.isfinite(out).all()

    def test_invalid_normalized_shape(self) -> None:
        with pytest.raises(ValueError, match="normalized_shape must be > 0"):
            RMSNorm(normalized_shape=-4)

    def test_invalid_eps(self) -> None:
        with pytest.raises(ValueError, match="eps must be > 0"):
            RMSNorm(normalized_shape=8, eps=0.0)

    def test_type_error_on_non_tensor(self) -> None:
        rms = RMSNorm(normalized_shape=4)
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            rms("not a tensor")  # type: ignore[arg-type]

    def test_value_error_on_dim_mismatch(self) -> None:
        rms = RMSNorm(normalized_shape=16)
        with pytest.raises(ValueError, match="does not match normalized_shape"):
            rms(torch.randn(4, 32))

    def test_gradient_flow(self) -> None:
        rms = RMSNorm(normalized_shape=16)
        x = torch.randn(2, 16, requires_grad=True)
        rms(x).sum().backward()
        assert rms.weight.grad is not None
        assert x.grad is not None

    def test_repr(self) -> None:
        rms = RMSNorm(normalized_shape=16, eps=1e-5)
        r = repr(rms)
        assert "RMSNorm" in r
        assert "normalized_shape=16" in r
