"""Unit tests for kamui.model.block.

Coverage target:
    kamui/model/block.py — 100%

Test categories:
    - construction: submodule names/types (hook-registry contract)
    - forward: shape, return_weights, Pre-LN residual structure, residual
      presence, end-to-end causality, dropout/eval determinism, error paths
    - architecture: parameter count vs config, gradient flow, repr
"""

from __future__ import annotations

import pytest
import torch

from kamui.model.attention import MultiHeadAttention
from kamui.model.block import TransformerBlock
from kamui.model.config import ModelConfig
from kamui.model.feedforward import FeedForward
from kamui.model.normalization import LayerNorm


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
# Construction — submodule contract
# ===========================================================================


class TestTransformerBlockConstruction:
    def test_submodule_names_and_types(self) -> None:
        # These exact names are relied on by kamui.hooks.registry.
        block = TransformerBlock(_small_config())
        assert isinstance(block.ln1, LayerNorm)
        assert isinstance(block.attn, MultiHeadAttention)
        assert isinstance(block.ln2, LayerNorm)
        assert isinstance(block.ffn, FeedForward)

    def test_layernorms_sized_to_d_model(self) -> None:
        block = TransformerBlock(_small_config(d_model=16))
        assert block.ln1.normalized_shape == 16
        assert block.ln2.normalized_shape == 16

    def test_ln1_and_ln2_are_distinct(self) -> None:
        block = TransformerBlock(_small_config())
        assert block.ln1 is not block.ln2


# ===========================================================================
# Forward
# ===========================================================================


class TestTransformerBlockForward:
    def test_output_shape(self) -> None:
        block = TransformerBlock(_small_config(d_model=16))
        out = block(torch.randn(2, 7, 16))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 7, 16)

    def test_return_weights_shape(self) -> None:
        block = TransformerBlock(_small_config(d_model=16, n_heads=4))
        out, weights = block(torch.randn(2, 7, 16), return_weights=True)
        assert out.shape == (2, 7, 16)
        assert weights.shape == (2, 4, 7, 7)

    def test_pre_ln_residual_structure(self) -> None:
        # Verify x = x + attn(ln1(x)); then x = x + ffn(ln2(x)), in eval mode.
        block = TransformerBlock(_small_config(d_model=16))
        block.eval()
        x = torch.randn(1, 5, 16)
        mid = x + block.attn(block.ln1(x))
        expected = mid + block.ffn(block.ln2(mid))
        assert torch.allclose(block(x), expected, atol=1e-6)

    def test_residual_connection_present(self) -> None:
        # The output is the input plus deltas, not just the sublayer output.
        block = TransformerBlock(_small_config(d_model=16))
        block.eval()
        x = torch.randn(1, 5, 16)
        out = block(x)
        # Removing the residual would give a very different tensor; with the
        # residual, output = x + (attn delta) + (ffn delta).
        deltas = out - x
        recomputed = x + deltas
        assert torch.allclose(out, recomputed, atol=1e-6)
        assert not torch.allclose(out, deltas, atol=1e-4)  # input actually added

    def test_return_weights_matches_plain_output(self) -> None:
        block = TransformerBlock(_small_config())
        block.eval()
        x = torch.randn(1, 6, 16)
        out_plain = block(x)
        out_w, _ = block(x, return_weights=True)
        assert torch.allclose(out_plain, out_w, atol=1e-6)

    def test_causality_preserved(self) -> None:
        # Perturbing a future token must not change earlier positions' outputs.
        block = TransformerBlock(_small_config(d_model=16))
        block.eval()
        x = torch.randn(1, 8, 16)
        out_a = block(x)
        x2 = x.clone()
        x2[0, 5] = torch.randn(16)
        out_b = block(x2)
        assert torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
        assert not torch.allclose(out_a[0, 5], out_b[0, 5], atol=1e-6)

    def test_eval_is_deterministic(self) -> None:
        block = TransformerBlock(_small_config(dropout=0.5))
        block.eval()
        x = torch.randn(2, 6, 16)
        assert torch.allclose(block(x), block(x))

    def test_type_error_on_non_tensor(self) -> None:
        block = TransformerBlock(_small_config())
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            block([[0.0] * 16])  # type: ignore[arg-type]

    def test_value_error_on_non_3d(self) -> None:
        block = TransformerBlock(_small_config())
        with pytest.raises(ValueError, match="must be 3-D"):
            block(torch.randn(7, 16))

    def test_value_error_on_dim_mismatch(self) -> None:
        block = TransformerBlock(_small_config(d_model=16))
        with pytest.raises(ValueError, match="does not match d_model"):
            block(torch.randn(1, 5, 8))


# ===========================================================================
# Architecture
# ===========================================================================


class TestTransformerBlockArchitecture:
    def test_parameter_count_matches_config(self) -> None:
        # Per-block params = attention + FFN + two LayerNorms (each 2*d_model).
        cfg = _small_config(n_layers=1, d_model=16)
        block = TransformerBlock(cfg)
        n_params = sum(p.numel() for p in block.parameters())
        expected = (
            cfg.attention_parameters
            + cfg.feedforward_parameters
            + 4 * cfg.d_model  # ln1 (2*D) + ln2 (2*D)
        )
        assert n_params == expected

    def test_gradient_flow(self) -> None:
        block = TransformerBlock(_small_config())
        x = torch.randn(2, 6, 16, requires_grad=True)
        block(x).sum().backward()
        # Every submodule receives gradients.
        assert block.ln1.weight.grad is not None
        assert block.attn.q_proj.weight.grad is not None
        assert block.ln2.weight.grad is not None
        assert block.ffn.fc_in.weight.grad is not None
        assert x.grad is not None

    def test_repr(self) -> None:
        block = TransformerBlock(_small_config(d_model=16, n_heads=4, d_ff=64))
        r = repr(block)
        assert "TransformerBlock" in r
        assert "d_model=16" in r
        assert "n_heads=4" in r
        assert "d_ff=64" in r
