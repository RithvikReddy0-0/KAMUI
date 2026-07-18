"""Unit tests for tensor shape contracts across all model components.

Shape tests are the fastest and most valuable tests in an ML codebase.
A wrong shape silently produces wrong results; these tests catch that
immediately.

Shape contract reference (from kamui.model.config):
    B  = batch size
    S  = sequence length
    D  = d_model
    H  = n_heads
    Dh = d_head = D / H
    F  = d_ff
    V  = vocab_size
"""

from __future__ import annotations

import math

import pytest
import torch

from kamui.model.attention import MultiHeadAttention
from kamui.model.config import ModelConfig
from kamui.model.embedding import Embedding
from kamui.model.feedforward import FeedForward
from kamui.model.transformer import KAMUITransformer

_B, _S, _D, _H, _F, _V = 2, 10, 32, 4, 64, 100


def _config() -> ModelConfig:
    return ModelConfig(
        n_layers=2,
        d_model=_D,
        n_heads=_H,
        d_ff=_F,
        vocab_size=_V,
        context_length=16,
        dropout=0.0,
    )


def test_attention_output_shape() -> None:
    attn = MultiHeadAttention(_config())
    out = attn(torch.randn(_B, _S, _D))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (_B, _S, _D)


def test_attention_weights_shape() -> None:
    attn = MultiHeadAttention(_config())
    _, weights = attn(torch.randn(_B, _S, _D), return_weights=True)
    assert weights.shape == (_B, _H, _S, _S)


def test_attention_weights_sum_to_one() -> None:
    attn = MultiHeadAttention(_config())
    _, weights = attn(torch.randn(_B, _S, _D), return_weights=True)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_causal_mask_no_future_leakage() -> None:
    attn = MultiHeadAttention(_config())
    _, weights = attn(torch.randn(1, _S, _D), return_weights=True)
    mask = torch.triu(torch.ones(_S, _S, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[0, :, mask] == 0.0)


def test_ffn_output_shape() -> None:
    ffn = FeedForward(_config())
    assert ffn(torch.randn(_B, _S, _D)).shape == (_B, _S, _D)


def test_ffn_hidden_shape() -> None:
    ffn = FeedForward(_config())
    hidden = ffn.activation(ffn.fc_in(torch.randn(_B, _S, _D)))
    assert hidden.shape == (_B, _S, _F)


def test_embedding_output_shape() -> None:
    embed = Embedding(_config())
    ids = torch.randint(0, _V, (_B, _S))
    assert embed(ids).shape == (_B, _S, _D)


def test_model_logits_shape() -> None:
    model = KAMUITransformer(_config())
    ids = torch.randint(0, _V, (_B, _S))
    assert model(ids).shape == (_B, _S, _V)


def test_model_loss_is_scalar() -> None:
    model = KAMUITransformer(_config())
    ids = torch.randint(0, _V, (_B, _S))
    loss = model(ids, targets=torch.randint(0, _V, (_B, _S)))
    assert loss.ndim == 0


def test_model_loss_near_log_vocab_size_at_init() -> None:
    torch.manual_seed(0)
    model = KAMUITransformer(_config()).eval()
    ids = torch.randint(0, _V, (4, _S))
    loss = model(ids, targets=torch.randint(0, _V, (4, _S)))
    assert loss.item() == pytest.approx(math.log(_V), abs=0.6)
