"""Unit tests for kamui.model.transformer.

Coverage target:
    kamui/model/transformer.py — 100%
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from kamui.model.block import TransformerBlock
from kamui.model.config import ModelConfig
from kamui.model.embedding import Embedding
from kamui.model.normalization import LayerNorm
from kamui.model.transformer import KAMUITransformer


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=100,
        context_length=16,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


# ===========================================================================
# Construction / structure
# ===========================================================================

class TestKAMUITransformerStructure:
    def test_named_submodules(self) -> None:
        model = KAMUITransformer(_config(n_layers=3))
        assert isinstance(model.embed, Embedding)
        assert isinstance(model.blocks, nn.ModuleList)
        assert len(model.blocks) == 3
        assert all(isinstance(b, TransformerBlock) for b in model.blocks)
        assert isinstance(model.final_ln, LayerNorm)
        assert isinstance(model.unembed, nn.Linear)

    def test_unembed_shape_and_no_bias(self) -> None:
        model = KAMUITransformer(_config(d_model=32, vocab_size=100))
        assert model.unembed.weight.shape == (100, 32)
        assert model.unembed.bias is None

    def test_weight_tying(self) -> None:
        model = KAMUITransformer(_config())
        # The unembedding matrix IS the token embedding matrix.
        assert model.unembed.weight is model.embed.token_embedding.weight

    def test_from_config(self) -> None:
        cfg = _config()
        model = KAMUITransformer.from_config(cfg)
        assert isinstance(model, KAMUITransformer)
        assert model.config is cfg

    def test_from_yaml(self, tmp_path) -> None:
        cfg = _config()
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)
        model = KAMUITransformer.from_yaml(path)
        assert model.config.n_layers == cfg.n_layers
        assert model.config.d_model == cfg.d_model

    def test_repr(self) -> None:
        r = repr(KAMUITransformer(_config()))
        assert "KAMUITransformer" in r
        assert "n_layers=3" in r


# ===========================================================================
# Forward — logits
# ===========================================================================

class TestKAMUITransformerForward:
    def test_logits_shape(self) -> None:
        model = KAMUITransformer(_config(vocab_size=100))
        ids = torch.randint(0, 100, (2, 10))
        logits = model(ids)
        assert logits.shape == (2, 10, 100)

    def test_logits_are_finite(self) -> None:
        model = KAMUITransformer(_config())
        logits = model(torch.randint(0, 100, (2, 8)))
        assert torch.isfinite(logits).all()

    def test_causality_end_to_end(self) -> None:
        # Perturbing a later token must not change earlier positions' logits.
        model = KAMUITransformer(_config())
        model.eval()
        ids = torch.randint(0, 100, (1, 12))
        logits_a = model(ids)
        ids2 = ids.clone()
        ids2[0, 7] = (ids2[0, 7] + 1) % 100
        logits_b = model(ids2)
        assert torch.allclose(logits_a[0, :7], logits_b[0, :7], atol=1e-5)
        assert not torch.allclose(logits_a[0, 7], logits_b[0, 7], atol=1e-5)

    def test_shorter_than_context(self) -> None:
        model = KAMUITransformer(_config(context_length=16))
        logits = model(torch.randint(0, 100, (1, 4)))
        assert logits.shape == (1, 4, 100)


# ===========================================================================
# Forward — loss
# ===========================================================================

class TestKAMUITransformerLoss:
    def test_loss_is_scalar(self) -> None:
        model = KAMUITransformer(_config(vocab_size=100))
        ids = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))
        loss = model(ids, targets=targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_near_ln_vocab_at_init(self) -> None:
        # At init, logits are ~uniform so loss ≈ ln(vocab_size).
        cfg = _config(vocab_size=100)
        model = KAMUITransformer(cfg)
        model.eval()
        ids = torch.randint(0, 100, (4, 12))
        targets = torch.randint(0, 100, (4, 12))
        loss = model(ids, targets=targets)
        assert abs(loss.item() - math.log(100)) < 0.6

    def test_loss_backward(self) -> None:
        model = KAMUITransformer(_config())
        ids = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        loss = model(ids, targets=targets)
        loss.backward()
        assert model.embed.token_embedding.weight.grad is not None

    def test_targets_type_error(self) -> None:
        model = KAMUITransformer(_config())
        ids = torch.randint(0, 100, (2, 8))
        with pytest.raises(TypeError, match="targets must be a torch.Tensor"):
            model(ids, targets=[[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_targets_shape_mismatch(self) -> None:
        model = KAMUITransformer(_config())
        ids = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 7))
        with pytest.raises(ValueError, match="must match token_ids"):
            model(ids, targets=targets)


# ===========================================================================
# Parameter counting
# ===========================================================================

class TestKAMUITransformerParameters:
    def test_num_parameters_positive(self) -> None:
        model = KAMUITransformer(_config())
        assert model.num_parameters() > 0

    def test_weight_tying_not_double_counted(self) -> None:
        # The tied matrix is counted once, so total params < naive sum.
        model = KAMUITransformer(_config())
        counted = model.num_parameters()
        naive = sum(p.numel() for p in model.parameters())
        # parameters() already dedupes shared params, so these agree — the
        # tied weight appears once in both.
        assert counted == naive

    def test_num_parameters_matches_config_estimate(self) -> None:
        # config.estimated_total_parameters assumes weight tying (unembed adds
        # 0 params) — the actual model must match it.
        cfg = _config()
        model = KAMUITransformer(cfg)
        assert model.num_parameters() == cfg.estimated_total_parameters

    def test_trainable_only_flag(self) -> None:
        model = KAMUITransformer(_config())
        for p in model.parameters():
            p.requires_grad_(False)
        assert model.num_parameters(trainable_only=True) == 0
        assert model.num_parameters(trainable_only=False) > 0
