"""Unit tests for kamui.model.attention.

Coverage target:
    kamui/model/attention.py — 100%

Test categories:
    - scaled_dot_product_attention: shapes, softmax normalisation, scaling,
      causal masking (future positions get exactly zero weight), single-token
      exactness, autograd gradient check, permutation equivariance, known values
    - MultiHeadAttention: construction, causal-mask buffer, forward shape,
      return_weights, end-to-end causality, parameter count vs config,
      no normalization, gradient flow, dropout, error paths, repr
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kamui.model.attention import MultiHeadAttention, scaled_dot_product_attention
from kamui.model.config import ModelConfig
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
# scaled_dot_product_attention
# ===========================================================================


class TestScaledDotProductAttention:
    def test_output_and_weights_shapes(self) -> None:
        q = torch.randn(2, 4, 5, 8)  # (B, H, S, Dh)
        k = torch.randn(2, 4, 5, 8)
        v = torch.randn(2, 4, 5, 8)
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 4, 5, 8)
        assert weights.shape == (2, 4, 5, 5)

    def test_weights_are_probabilities(self) -> None:
        q = torch.randn(1, 1, 6, 8)
        k = torch.randn(1, 1, 6, 8)
        v = torch.randn(1, 1, 6, 8)
        _, weights = scaled_dot_product_attention(q, k, v)
        # Each query's weights form a probability distribution over keys.
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 6), atol=1e-5)
        assert (weights >= 0).all()

    def test_single_key_returns_value_exactly(self) -> None:
        # With one key, the only attention weight is 1.0 and output == value.
        q = torch.randn(1, 1, 1, 4)
        k = torch.randn(1, 1, 1, 4)
        v = torch.randn(1, 1, 1, 4)
        out, weights = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(weights, torch.ones(1, 1, 1, 1), atol=1e-6)
        assert torch.allclose(out, v, atol=1e-6)

    def test_scaling_applied(self) -> None:
        # score for key0 = (1*2)/sqrt(4) = 1.0; key1 = 0. softmax([1, 0]).
        q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])  # (1,1,1,4)
        k = torch.tensor([[[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]])  # (1,1,2,4)
        v = torch.tensor([[[[1.0], [0.0]]]])  # (1,1,2,1)
        _, weights = scaled_dot_product_attention(q, k, v)
        expected_w0 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
        assert weights[0, 0, 0, 0].item() == pytest.approx(expected_w0, abs=1e-6)

    def test_causal_mask_zeroes_future(self) -> None:
        # -inf masking (not 0) means future positions get EXACTLY zero weight.
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        assert torch.all(weights[0, 0][mask] == 0.0)
        # Position 0 attends only to itself.
        assert weights[0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_masked_rows_still_sum_to_one(self) -> None:
        q = torch.randn(1, 1, 5, 8)
        k = torch.randn(1, 1, 5, 8)
        v = torch.randn(1, 1, 5, 8)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 5), atol=1e-5)

    def test_known_values_uniform_when_keys_equal(self) -> None:
        # Equal keys → equal scores → uniform weights → output is mean of values.
        q = torch.randn(1, 1, 1, 4)
        k = torch.ones(1, 1, 3, 4)
        v = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        out, weights = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(weights, torch.full((1, 1, 1, 3), 1 / 3), atol=1e-6)
        assert out[0, 0, 0, 0].item() == pytest.approx(2.0, abs=1e-6)

    def test_permutation_equivariance(self) -> None:
        # Permuting keys and values together permutes the weights but leaves
        # the output unchanged (same Q-KV associations).
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 3, 4)
        v = torch.randn(1, 1, 3, 4)
        out_a, _ = scaled_dot_product_attention(q, k, v)
        perm = torch.tensor([2, 0, 1])
        out_b, _ = scaled_dot_product_attention(q, k[:, :, perm], v[:, :, perm])
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_gradient_check(self) -> None:
        # Analytical (autograd) vs numerical gradient in float64.
        torch.manual_seed(0)
        q = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda q, k, v: scaled_dot_product_attention(q, k, v)[0],
            (q, k, v),
            eps=1e-6,
            atol=1e-4,
        )


# ===========================================================================
# MultiHeadAttention — construction
# ===========================================================================


class TestMultiHeadAttentionConstruction:
    def test_head_geometry(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16, n_heads=4))
        assert mha.n_heads == 4
        assert mha.d_head == 4

    def test_projections_are_linear(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16))
        for proj in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
            assert isinstance(proj, nn.Linear)
            assert proj.in_features == 16
            assert proj.out_features == 16

    def test_causal_mask_buffer(self) -> None:
        mha = MultiHeadAttention(_small_config(context_length=12))
        assert mha.causal_mask.shape == (12, 12)
        assert mha.causal_mask.dtype == torch.bool
        assert torch.equal(
            mha.causal_mask, torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
        )
        # It is a buffer, not a parameter.
        assert "causal_mask" in dict(mha.named_buffers())
        assert "causal_mask" not in dict(mha.named_parameters())

    def test_dropout_probability(self) -> None:
        mha = MultiHeadAttention(_small_config(dropout=0.25))
        assert mha.dropout.p == 0.25


# ===========================================================================
# MultiHeadAttention — forward
# ===========================================================================


class TestMultiHeadAttentionForward:
    def test_output_shape(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16))
        out = mha(torch.randn(2, 7, 16))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 7, 16)

    def test_return_weights_shape(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16, n_heads=4))
        out, weights = mha(torch.randn(2, 7, 16), return_weights=True)
        assert out.shape == (2, 7, 16)
        assert weights.shape == (2, 4, 7, 7)

    def test_weights_are_causal(self) -> None:
        mha = MultiHeadAttention(_small_config())
        _, weights = mha(torch.randn(1, 6, 16), return_weights=True)
        mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        for h in range(weights.shape[1]):
            assert torch.all(weights[0, h][mask] == 0.0)

    def test_causality_future_tokens_dont_affect_past(self) -> None:
        # Changing a token at position t must not change outputs at positions < t.
        mha = MultiHeadAttention(_small_config(d_model=16))
        mha.eval()
        x = torch.randn(1, 8, 16)
        out_a = mha(x)
        x2 = x.clone()
        x2[0, 5] = torch.randn(16)  # perturb position 5
        out_b = mha(x2)
        assert torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
        assert not torch.allclose(out_a[0, 5], out_b[0, 5], atol=1e-6)

    def test_shorter_sequence_than_context(self) -> None:
        mha = MultiHeadAttention(_small_config(context_length=12))
        out = mha(torch.randn(1, 3, 16))
        assert out.shape == (1, 3, 16)

    def test_dropout_active_in_train(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16, dropout=0.5))
        mha.train()
        torch.manual_seed(0)
        x = torch.randn(2, 6, 16)
        assert not torch.allclose(mha(x), mha(x))

    def test_dropout_inactive_in_eval(self) -> None:
        mha = MultiHeadAttention(_small_config(dropout=0.5))
        mha.eval()
        x = torch.randn(2, 6, 16)
        assert torch.allclose(mha(x), mha(x))

    def test_type_error_on_non_tensor(self) -> None:
        mha = MultiHeadAttention(_small_config())
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            mha([[0.0] * 16])  # type: ignore[arg-type]

    def test_value_error_on_non_3d(self) -> None:
        mha = MultiHeadAttention(_small_config())
        with pytest.raises(ValueError, match="must be 3-D"):
            mha(torch.randn(7, 16))

    def test_value_error_on_dim_mismatch(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16))
        with pytest.raises(ValueError, match="does not match d_model"):
            mha(torch.randn(1, 5, 8))

    def test_value_error_on_too_long_sequence(self) -> None:
        mha = MultiHeadAttention(_small_config(context_length=12))
        with pytest.raises(ValueError, match="exceeds context_length"):
            mha(torch.randn(1, 13, 16))


# ===========================================================================
# MultiHeadAttention — architecture constraints
# ===========================================================================


class TestMultiHeadAttentionArchitecture:
    def test_parameter_count_matches_config(self) -> None:
        cfg = _small_config(n_layers=1, d_model=16)
        mha = MultiHeadAttention(cfg)
        n_params = sum(p.numel() for p in mha.parameters())
        assert n_params == cfg.attention_parameters

    def test_parameter_count_formula(self) -> None:
        cfg = _small_config(d_model=16)
        mha = MultiHeadAttention(cfg)
        n_params = sum(p.numel() for p in mha.parameters())
        # 4 linear layers, each d_model x d_model weight + d_model bias.
        assert n_params == 4 * (16 * 16 + 16)

    def test_no_normalization_layers(self) -> None:
        mha = MultiHeadAttention(_small_config())
        for module in mha.modules():
            assert not isinstance(
                module,
                (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, LayerNorm, RMSNorm),
            )

    def test_gradient_flow(self) -> None:
        mha = MultiHeadAttention(_small_config())
        x = torch.randn(2, 6, 16, requires_grad=True)
        mha(x).sum().backward()
        for proj in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
            assert proj.weight.grad is not None
            assert proj.bias.grad is not None
        assert x.grad is not None

    def test_repr(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16, n_heads=4))
        r = repr(mha)
        assert "MultiHeadAttention" in r
        assert "n_heads=4" in r
        assert "d_head=4" in r


# ===========================================================================
# MultiHeadAttention — rotary positional encoding (RoPE)
# ===========================================================================


class TestMultiHeadAttentionRoPE:
    def test_rope_module_built_only_for_rope_config(self) -> None:
        assert MultiHeadAttention(_small_config(positional_encoding="rope")).rope is not None
        assert MultiHeadAttention(_small_config(positional_encoding="learned")).rope is None

    def test_rope_output_shape(self) -> None:
        mha = MultiHeadAttention(_small_config(d_model=16, positional_encoding="rope"))
        assert mha(torch.randn(2, 7, 16)).shape == (2, 7, 16)

    def test_rope_is_still_causal(self) -> None:
        mha = MultiHeadAttention(_small_config(positional_encoding="rope"))
        _, weights = mha(torch.randn(1, 6, 16), return_weights=True)
        mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        assert torch.all(weights[0, :, mask] == 0.0)

    def test_rope_changes_attention_vs_no_rope(self) -> None:
        # With the same weights and input, rotating Q/K must change attention.
        torch.manual_seed(0)
        rope_mha = MultiHeadAttention(_small_config(positional_encoding="rope")).eval()
        plain_mha = MultiHeadAttention(_small_config(positional_encoding="learned")).eval()
        # Copy the projections so the only difference is RoPE.
        plain_mha.load_state_dict(
            {k: v for k, v in rope_mha.state_dict().items() if "rope" not in k},
            strict=False,
        )
        x = torch.randn(1, 6, 16)
        assert not torch.allclose(rope_mha(x), plain_mha(x), atol=1e-5)

    def test_rope_gradient_flow(self) -> None:
        mha = MultiHeadAttention(_small_config(positional_encoding="rope"))
        x = torch.randn(2, 6, 16, requires_grad=True)
        mha(x).sum().backward()
        assert x.grad is not None
        assert mha.q_proj.weight.grad is not None
