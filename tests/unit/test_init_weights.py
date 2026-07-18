"""Unit tests for kamui.model.init_weights.

Coverage target:
    kamui/model/init_weights.py — 100%
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kamui.model.block import TransformerBlock
from kamui.model.config import ModelConfig
from kamui.model.embedding import Embedding
from kamui.model.init_weights import init_weights


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=4,
        d_model=32,
        n_heads=4,
        d_ff=128,
        vocab_size=200,
        context_length=16,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


class TestInitWeights:
    def test_returns_same_model(self) -> None:
        block = TransformerBlock(_config())
        assert init_weights(block, n_layers=4) is block

    def test_invalid_n_layers(self) -> None:
        block = TransformerBlock(_config())
        with pytest.raises(ValueError, match="n_layers must be > 0"):
            init_weights(block, n_layers=0)

    def test_linear_bias_zeroed(self) -> None:
        block = TransformerBlock(_config())
        init_weights(block, n_layers=4)
        assert torch.all(block.attn.q_proj.bias == 0.0)
        assert torch.all(block.ffn.fc_in.bias == 0.0)

    def test_layernorm_reset_to_ones_zeros(self) -> None:
        block = TransformerBlock(_config())
        with torch.no_grad():
            block.ln1.weight.fill_(5.0)
            block.ln1.bias.fill_(3.0)
        init_weights(block, n_layers=4)
        assert torch.all(block.ln1.weight == 1.0)
        assert torch.all(block.ln1.bias == 0.0)

    def test_embedding_init_std(self) -> None:
        cfg = _config(vocab_size=5000, d_model=64)
        emb = Embedding(cfg)
        init_weights(emb, n_layers=4, std=0.02)
        assert abs(emb.token_embedding.weight.std().item() - 0.02) < 0.005

    def test_qkv_weights_use_base_std(self) -> None:
        # Non-residual linears keep std ~ 0.02 (large layers to reduce noise).
        cfg = _config(d_model=256, d_ff=256, n_layers=4)
        block = TransformerBlock(cfg)
        init_weights(block, n_layers=4, std=0.02)
        assert abs(block.attn.q_proj.weight.std().item() - 0.02) < 0.004

    def test_residual_projections_scaled(self) -> None:
        # out_proj (W_O) and ffn.fc_out (W_2) use std = 0.02 / sqrt(2*n_layers).
        n_layers = 8
        cfg = _config(d_model=256, d_ff=256, n_layers=n_layers)
        block = TransformerBlock(cfg)
        init_weights(block, n_layers=n_layers, std=0.02)
        expected = 0.02 / math.sqrt(2 * n_layers)
        assert abs(block.attn.out_proj.weight.std().item() - expected) < 0.002
        assert abs(block.ffn.fc_out.weight.std().item() - expected) < 0.002

    def test_residual_scaling_smaller_than_base(self) -> None:
        n_layers = 8
        cfg = _config(d_model=256, d_ff=256, n_layers=n_layers)
        block = TransformerBlock(cfg)
        init_weights(block, n_layers=n_layers, std=0.02)
        # out_proj std must be clearly smaller than q_proj std.
        assert block.attn.out_proj.weight.std().item() < block.attn.q_proj.weight.std().item()

    def test_rmsnorm_scale_reset_to_ones(self) -> None:
        from kamui.model.normalization import RMSNorm

        net = nn.Sequential(RMSNorm(8), nn.Linear(8, 8))
        with torch.no_grad():
            net[0].weight.fill_(5.0)
        init_weights(net, n_layers=2)
        assert torch.all(net[0].weight == 1.0)  # RMSNorm has no bias to reset

    def test_works_on_plain_linear_tree(self) -> None:
        # A module with no custom types still initialises (Linear branch only).
        net = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        init_weights(net, n_layers=2)
        for layer in net:
            assert torch.all(layer.bias == 0.0)
