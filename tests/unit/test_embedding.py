"""Unit tests for kamui.model.embedding.

Coverage target:
    kamui/model/embedding.py — 100%

Test categories:
    - TokenEmbedding: construction, init shape, lookup, error paths
    - SinusoidalPositionalEncoding: known values, shape, non-learnable buffer,
      odd d_model, range errors
    - LearnedPositionalEncoding: learnable weight, shape, range errors
    - Embedding (combined): learned + sinusoidal paths, additivity, dropout,
      output shape, validation errors, repr, parameter count
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kamui.model.config import ModelConfig
from kamui.model.embedding import (
    Embedding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)


# ===========================================================================
# TokenEmbedding
# ===========================================================================

class TestTokenEmbedding:
    def test_weight_shape(self) -> None:
        emb = TokenEmbedding(vocab_size=50, d_model=16)
        assert emb.weight.shape == (50, 16)

    def test_weight_is_parameter(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        assert isinstance(emb.weight, nn.Parameter)
        assert emb.weight.requires_grad

    def test_forward_output_shape(self) -> None:
        emb = TokenEmbedding(vocab_size=50, d_model=16)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        out = emb(ids)
        assert out.shape == (2, 3, 16)

    def test_forward_is_row_lookup(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        ids = torch.tensor([[0, 3]], dtype=torch.long)
        out = emb(ids)
        assert torch.equal(out[0, 0], emb.weight[0])
        assert torch.equal(out[0, 1], emb.weight[3])

    def test_forward_1d_input(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        out = emb(torch.tensor([1, 2, 3], dtype=torch.long))
        assert out.shape == (3, 4)

    def test_init_std_respected(self) -> None:
        torch.manual_seed(0)
        emb = TokenEmbedding(vocab_size=5000, d_model=64, init_std=0.02)
        # Mean ~ 0, std ~ 0.02 over a large matrix.
        assert abs(emb.weight.mean().item()) < 0.01
        assert abs(emb.weight.std().item() - 0.02) < 0.005

    def test_accepts_int32(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        out = emb(torch.tensor([[1, 2]], dtype=torch.int32))
        assert out.shape == (1, 2, 4)

    def test_invalid_vocab_size(self) -> None:
        with pytest.raises(ValueError, match="vocab_size must be > 0"):
            TokenEmbedding(vocab_size=0, d_model=4)

    def test_invalid_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be > 0"):
            TokenEmbedding(vocab_size=4, d_model=-1)

    def test_type_error_on_non_tensor(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            emb([1, 2, 3])  # type: ignore[arg-type]

    def test_type_error_on_float_dtype(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        with pytest.raises(TypeError, match="integer dtype"):
            emb(torch.tensor([[1.0, 2.0]]))

    def test_value_error_on_negative_id(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        with pytest.raises(ValueError, match=r"outside \[0, 10\)"):
            emb(torch.tensor([[-1, 2]], dtype=torch.long))

    def test_value_error_on_too_large_id(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        with pytest.raises(ValueError, match=r"outside \[0, 10\)"):
            emb(torch.tensor([[10]], dtype=torch.long))

    def test_repr(self) -> None:
        emb = TokenEmbedding(vocab_size=10, d_model=4)
        r = repr(emb)
        assert "TokenEmbedding" in r
        assert "vocab_size=10" in r
        assert "d_model=4" in r


# ===========================================================================
# SinusoidalPositionalEncoding
# ===========================================================================

class TestSinusoidalPositionalEncoding:
    def test_buffer_shape(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=32, d_model=16)
        assert pe.pe.shape == (32, 16)

    def test_buffer_is_not_a_parameter(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        # Non-learnable: registered as a buffer, not a parameter.
        assert list(pe.parameters()) == []
        assert "pe" in dict(pe.named_buffers())

    def test_forward_shape(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=32, d_model=16)
        out = pe(10)
        assert out.shape == (10, 16)

    def test_known_values_position_zero(self) -> None:
        # At pos=0: sin(0)=0 for even dims, cos(0)=1 for odd dims.
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        row0 = pe(1)[0]
        assert row0[0].item() == pytest.approx(0.0)
        assert row0[1].item() == pytest.approx(1.0)
        assert row0[2].item() == pytest.approx(0.0)
        assert row0[3].item() == pytest.approx(1.0)

    def test_known_values_first_dim_is_sin_pos(self) -> None:
        # div_term for i=0 is 10000^0 = 1, so PE(pos, 0) = sin(pos).
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        col0 = pe(5)[:, 0]
        for pos in range(5):
            assert col0[pos].item() == pytest.approx(math.sin(pos), abs=1e-5)

    def test_values_bounded(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=64, d_model=32)
        table = pe(64)
        assert table.max().item() <= 1.0 + 1e-6
        assert table.min().item() >= -1.0 - 1e-6

    def test_odd_d_model(self) -> None:
        # Odd d_model: one more sine column than cosine column.
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=5)
        assert pe.pe.shape == (8, 5)
        out = pe(3)
        assert out.shape == (3, 5)

    def test_full_context_length(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        assert pe(8).shape == (8, 4)

    def test_invalid_context_length(self) -> None:
        with pytest.raises(ValueError, match="context_length must be > 0"):
            SinusoidalPositionalEncoding(context_length=0, d_model=4)

    def test_invalid_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be > 0"):
            SinusoidalPositionalEncoding(context_length=8, d_model=0)

    def test_seq_len_too_small(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            pe(0)

    def test_seq_len_too_large(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        with pytest.raises(ValueError, match="exceeds context_length"):
            pe(9)

    def test_repr(self) -> None:
        pe = SinusoidalPositionalEncoding(context_length=8, d_model=4)
        r = repr(pe)
        assert "SinusoidalPositionalEncoding" in r
        assert "context_length=8" in r


# ===========================================================================
# LearnedPositionalEncoding
# ===========================================================================

class TestLearnedPositionalEncoding:
    def test_weight_shape(self) -> None:
        pe = LearnedPositionalEncoding(context_length=32, d_model=16)
        assert pe.weight.shape == (32, 16)

    def test_weight_is_learnable(self) -> None:
        pe = LearnedPositionalEncoding(context_length=8, d_model=4)
        assert isinstance(pe.weight, nn.Parameter)
        assert pe.weight.requires_grad
        assert len(list(pe.parameters())) == 1

    def test_forward_shape(self) -> None:
        pe = LearnedPositionalEncoding(context_length=32, d_model=16)
        assert pe(10).shape == (10, 16)

    def test_forward_is_row_slice(self) -> None:
        pe = LearnedPositionalEncoding(context_length=8, d_model=4)
        out = pe(3)
        assert torch.equal(out, pe.weight[:3])

    def test_init_std_respected(self) -> None:
        torch.manual_seed(0)
        pe = LearnedPositionalEncoding(context_length=4096, d_model=64, init_std=0.02)
        assert abs(pe.weight.std().item() - 0.02) < 0.005

    def test_invalid_context_length(self) -> None:
        with pytest.raises(ValueError, match="context_length must be > 0"):
            LearnedPositionalEncoding(context_length=-1, d_model=4)

    def test_invalid_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be > 0"):
            LearnedPositionalEncoding(context_length=8, d_model=0)

    def test_seq_len_too_small(self) -> None:
        pe = LearnedPositionalEncoding(context_length=8, d_model=4)
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            pe(0)

    def test_seq_len_too_large(self) -> None:
        pe = LearnedPositionalEncoding(context_length=8, d_model=4)
        with pytest.raises(ValueError, match="exceeds context_length"):
            pe(100)

    def test_repr(self) -> None:
        pe = LearnedPositionalEncoding(context_length=8, d_model=4)
        r = repr(pe)
        assert "LearnedPositionalEncoding" in r
        assert "context_length=8" in r


# ===========================================================================
# Embedding (combined)
# ===========================================================================

def _small_config(**overrides: object) -> ModelConfig:
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


class TestEmbeddingCombined:
    def test_learned_path_selected(self) -> None:
        emb = Embedding(_small_config(positional_encoding="learned"))
        assert isinstance(emb.positional_encoding, LearnedPositionalEncoding)

    def test_sinusoidal_path_selected(self) -> None:
        emb = Embedding(_small_config(positional_encoding="sinusoidal"))
        assert isinstance(emb.positional_encoding, SinusoidalPositionalEncoding)

    def test_output_shape_learned(self) -> None:
        emb = Embedding(_small_config(positional_encoding="learned"))
        ids = torch.randint(0, 50, (4, 10))
        assert emb(ids).shape == (4, 10, 16)

    def test_output_shape_sinusoidal(self) -> None:
        emb = Embedding(_small_config(positional_encoding="sinusoidal"))
        ids = torch.randint(0, 50, (4, 10))
        assert emb(ids).shape == (4, 10, 16)

    def test_output_is_token_plus_position(self) -> None:
        # With dropout=0 and eval mode, output == token_emb + pos_emb exactly.
        emb = Embedding(_small_config(positional_encoding="sinusoidal"))
        emb.eval()
        ids = torch.randint(0, 50, (2, 7))
        expected = emb.token_embedding(ids) + emb.positional_encoding(7)
        assert torch.allclose(emb(ids), expected)

    def test_position_broadcast_over_batch(self) -> None:
        # Same token at the same position in different batch rows → same vector.
        emb = Embedding(_small_config(positional_encoding="learned"))
        emb.eval()
        ids = torch.tensor([[5, 9, 1], [5, 9, 1]], dtype=torch.long)
        out = emb(ids)
        assert torch.allclose(out[0], out[1])

    def test_dropout_active_in_train_mode(self) -> None:
        emb = Embedding(_small_config(dropout=0.5))
        emb.train()
        torch.manual_seed(0)
        ids = torch.randint(0, 50, (4, 10))
        out = emb(ids)
        # With p=0.5 dropout some activations are zeroed during training.
        assert (out == 0).any()

    def test_dropout_inactive_in_eval_mode(self) -> None:
        emb = Embedding(_small_config(dropout=0.5))
        emb.eval()
        ids = torch.randint(0, 50, (4, 10))
        expected = emb.token_embedding(ids) + emb.positional_encoding(10)
        assert torch.allclose(emb(ids), expected)

    def test_max_context_length_ok(self) -> None:
        cfg = _small_config(context_length=12)
        emb = Embedding(cfg)
        ids = torch.randint(0, 50, (1, 12))
        assert emb(ids).shape == (1, 12, 16)

    def test_type_error_on_non_tensor(self) -> None:
        emb = Embedding(_small_config())
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            emb([[1, 2, 3]])  # type: ignore[arg-type]

    def test_value_error_on_non_2d(self) -> None:
        emb = Embedding(_small_config())
        with pytest.raises(ValueError, match="must be 2-D"):
            emb(torch.randint(0, 50, (4,)))

    def test_value_error_on_too_long_sequence(self) -> None:
        emb = Embedding(_small_config(context_length=12))
        ids = torch.randint(0, 50, (1, 13))
        with pytest.raises(ValueError, match="exceeds context_length"):
            emb(ids)

    def test_parameter_count_matches_config_learned(self) -> None:
        cfg = _small_config(positional_encoding="learned")
        emb = Embedding(cfg)
        n_params = sum(p.numel() for p in emb.parameters())
        assert n_params == cfg.embedding_parameters

    def test_parameter_count_matches_config_sinusoidal(self) -> None:
        cfg = _small_config(positional_encoding="sinusoidal")
        emb = Embedding(cfg)
        # Sinusoidal adds 0 trainable params, so only the token table counts.
        n_params = sum(p.numel() for p in emb.parameters())
        assert n_params == cfg.embedding_parameters
        assert n_params == cfg.vocab_size * cfg.d_model

    def test_gradients_flow(self) -> None:
        emb = Embedding(_small_config(positional_encoding="learned"))
        ids = torch.randint(0, 50, (2, 6))
        out = emb(ids)
        out.sum().backward()
        assert emb.token_embedding.weight.grad is not None
        assert emb.positional_encoding.weight.grad is not None  # type: ignore[union-attr]

    def test_repr(self) -> None:
        emb = Embedding(_small_config(positional_encoding="sinusoidal"))
        r = repr(emb)
        assert "Embedding" in r
        assert "positional_encoding='sinusoidal'" in r
