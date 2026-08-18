"""Unit tests for kamui.mechinterp.superposition (sparse autoencoders).

Coverage target:
    kamui/mechinterp/superposition.py — 100%
"""

from __future__ import annotations

import pytest
import torch

from kamui.mechinterp.superposition import (
    FeatureProfile,
    SAELoss,
    SAEMetrics,
    SparseAutoencoder,
    collect_activations,
    interpret_features,
    sae_feature_metrics,
    train_sae,
)
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer


def _structured_activations(
    d_model: int = 8, n_atoms: int = 16, n: int = 512, seed: int = 0
) -> torch.Tensor:
    """A sparse linear mixture of a random dictionary — learnable by an SAE."""
    torch.manual_seed(seed)
    dictionary = torch.randn(n_atoms, d_model)
    codes = (torch.rand(n, n_atoms) < 0.15).float() * torch.rand(n, n_atoms)
    return codes @ dictionary


# ===========================================================================
# SparseAutoencoder
# ===========================================================================


class TestSparseAutoencoder:
    def test_invalid_d_model(self) -> None:
        with pytest.raises(ValueError, match="d_model must be > 0"):
            SparseAutoencoder(0, 32)

    def test_invalid_n_features(self) -> None:
        with pytest.raises(ValueError, match="n_features must be > 0"):
            SparseAutoencoder(8, 0)

    def test_invalid_l1_coeff(self) -> None:
        with pytest.raises(ValueError, match="l1_coeff must be >= 0"):
            SparseAutoencoder(8, 32, l1_coeff=-1.0)

    def test_encode_decode_forward_shapes(self) -> None:
        sae = SparseAutoencoder(8, 32)
        x = torch.randn(5, 8)
        features = sae.encode(x)
        assert features.shape == (5, 32)
        assert sae.decode(features).shape == (5, 8)
        recon, feats = sae(x)
        assert recon.shape == (5, 8)
        assert feats.shape == (5, 32)

    def test_features_are_non_negative(self) -> None:
        sae = SparseAutoencoder(8, 32)
        features = sae.encode(torch.randn(10, 8))
        assert (features >= 0).all()  # ReLU output

    def test_decode_of_one_hot_is_dictionary_atom(self) -> None:
        # Decoding a one-hot feature returns that feature's dictionary direction
        # plus the decoder pre-bias — the defining SAE dictionary property.
        sae = SparseAutoencoder(8, 32)
        one_hot = torch.zeros(1, 32)
        one_hot[0, 5] = 1.0
        assert torch.allclose(sae.decode(one_hot)[0], sae.W_dec[5] + sae.b_dec, atol=1e-6)

    def test_decoder_rows_are_unit_norm_at_init(self) -> None:
        sae = SparseAutoencoder(8, 32)
        assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(32), atol=1e-5)

    def test_normalize_decoder_rescales(self) -> None:
        sae = SparseAutoencoder(8, 16)
        with torch.no_grad():
            sae.W_dec.mul_(7.0)  # break the unit norm
        sae.normalize_decoder()
        assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(16), atol=1e-5)

    def test_loss_decomposition(self) -> None:
        sae = SparseAutoencoder(8, 32, l1_coeff=0.05)
        loss = sae.loss(torch.randn(20, 8))
        assert isinstance(loss, SAELoss)
        assert torch.allclose(
            loss.total, loss.reconstruction + sae.l1_coeff * loss.sparsity, atol=1e-6
        )

    def test_repr(self) -> None:
        r = repr(SparseAutoencoder(16, 64, l1_coeff=1e-3))
        assert "SparseAutoencoder" in r
        assert "n_features=64" in r


# ===========================================================================
# train_sae
# ===========================================================================


class TestTrainSAE:
    def test_reconstruction_improves(self) -> None:
        # On structured (sparse-mixture) data, an overcomplete SAE must learn
        # to reconstruct — final MSE well below the initial epoch's.
        acts = _structured_activations(d_model=8, n_atoms=16, n=512)
        sae = SparseAutoencoder(8, 32, l1_coeff=1e-4)
        history = train_sae(sae, acts, epochs=40, lr=3e-3, batch_size=128, seed=0)
        assert history[-1]["reconstruction"] < 0.5 * history[0]["reconstruction"]

    def test_history_shape_and_keys(self) -> None:
        acts = _structured_activations(n=128)
        sae = SparseAutoencoder(8, 16)
        history = train_sae(sae, acts, epochs=3, batch_size=64)
        assert len(history) == 3
        assert set(history[0]) == {"epoch", "total", "reconstruction", "sparsity"}

    def test_renormalize_false_path(self) -> None:
        acts = _structured_activations(n=128)
        sae = SparseAutoencoder(8, 16)
        history = train_sae(sae, acts, epochs=2, batch_size=64, renormalize=False)
        assert len(history) == 2  # runs without renormalising the decoder

    def test_bad_activation_rank(self) -> None:
        sae = SparseAutoencoder(8, 32)
        with pytest.raises(ValueError, match="must be 2-D"):
            train_sae(sae, torch.randn(5, 8, 2))

    def test_dim_mismatch(self) -> None:
        sae = SparseAutoencoder(8, 32)
        with pytest.raises(ValueError, match="must match d_model"):
            train_sae(sae, torch.randn(10, 16))

    def test_bad_epochs(self) -> None:
        sae = SparseAutoencoder(8, 32)
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            train_sae(sae, torch.randn(10, 8), epochs=0)

    def test_bad_batch_size(self) -> None:
        sae = SparseAutoencoder(8, 32)
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            train_sae(sae, torch.randn(10, 8), batch_size=0)


# ===========================================================================
# collect_activations
# ===========================================================================


class TestCollectActivations:
    def _model(self) -> KAMUITransformer:
        torch.manual_seed(0)
        cfg = ModelConfig(
            n_layers=2, d_model=16, n_heads=4, d_ff=32, vocab_size=40, context_length=8
        )
        return KAMUITransformer(cfg).eval()

    def test_shape_is_flattened_tokens_by_d_model(self) -> None:
        model = self._model()
        seqs = [torch.randint(0, 40, (2, 8)) for _ in range(3)]
        acts = collect_activations(model, "blocks.0.ffn.output", seqs)
        assert acts.shape == (3 * 2 * 8, 16)

    def test_accepts_1d_sequences(self) -> None:
        model = self._model()
        acts = collect_activations(model, "embed.output", [torch.randint(0, 40, (8,))])
        assert acts.shape == (8, 16)

    def test_empty_sequences_raises(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="no activations"):
            collect_activations(model, "embed.output", [])

    def test_feeds_sae_training(self) -> None:
        # End-to-end: real model activations train an SAE without error.
        model = self._model()
        seqs = [torch.randint(0, 40, (4, 8)) for _ in range(4)]
        acts = collect_activations(model, "blocks.1.ffn.output", seqs)
        sae = SparseAutoencoder(16, 64, l1_coeff=1e-4)
        history = train_sae(sae, acts, epochs=5, lr=1e-3, batch_size=32)
        assert history[-1]["reconstruction"] <= history[0]["reconstruction"]


# ===========================================================================
# sae_feature_metrics
# ===========================================================================


class TestSAEMetrics:
    def test_fields_and_ranges(self) -> None:
        acts = _structured_activations(n=256)
        sae = SparseAutoencoder(8, 32, l1_coeff=1e-4)
        train_sae(sae, acts, epochs=10, lr=3e-3, batch_size=128)
        metrics = sae_feature_metrics(sae, acts)
        assert isinstance(metrics, SAEMetrics)
        assert 0.0 <= metrics.dead_feature_fraction <= 1.0
        assert 0.0 <= metrics.mean_l0 <= 32
        assert metrics.mean_l1 >= 0.0
        assert metrics.reconstruction_mse >= 0.0

    def test_all_features_dead_when_bias_very_negative(self) -> None:
        # Force every ReLU off → all features dead, zero L0. Deterministic.
        sae = SparseAutoencoder(8, 32)
        with torch.no_grad():
            sae.b_enc.fill_(-1e9)
        metrics = sae_feature_metrics(sae, torch.randn(50, 8))
        assert metrics.dead_feature_fraction == 1.0
        assert metrics.mean_l0 == 0.0

    def test_bad_shape_raises(self) -> None:
        sae = SparseAutoencoder(8, 32)
        with pytest.raises(ValueError, match="must be"):
            sae_feature_metrics(sae, torch.randn(10, 16))


def _identity_sae(d: int) -> SparseAutoencoder:
    """An SAE whose encode is exactly ReLU(x): feature f activation == relu(x[f])."""
    sae = SparseAutoencoder(d_model=d, n_features=d, l1_coeff=0.0)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(d))
        sae.W_dec.copy_(torch.eye(d))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


# ===========================================================================
# interpret_features
# ===========================================================================


class TestInterpretFeatures:
    def test_top_activating_tokens_are_recovered_in_order(self) -> None:
        # With the identity SAE, feature f activation == relu(activation[:, f]).
        sae = _identity_sae(3)
        activations = torch.tensor(
            [
                [5.0, 0.0, 0.0],  # token 7 — strongest on feature 0
                [3.0, 0.0, 0.0],  # token 7 — weaker on feature 0
                [0.0, 9.0, 0.0],  # token 2 — feature 1
                [0.0, 0.0, 1.0],  # token 5 — feature 2
            ]
        )
        tokens = torch.tensor([7, 7, 2, 5])

        (profile,) = interpret_features(sae, activations, tokens, top_k=2, features=[0])
        assert isinstance(profile, FeatureProfile)
        assert profile.feature == 0
        assert profile.top_token_ids == [7, 7]  # row 0 then row 1, descending
        assert profile.top_activations == [5.0, 3.0]
        assert profile.density == 2 / 4  # feature 0 fires on 2 of 4 tokens
        assert profile.mean_activation == pytest.approx((5.0 + 3.0) / 2)

    def test_dead_feature_reports_empty(self) -> None:
        sae = _identity_sae(1)
        activations = torch.tensor([[-1.0], [-2.0]])  # ReLU → always 0
        tokens = torch.tensor([4, 9])
        (profile,) = interpret_features(sae, activations, tokens, top_k=3, features=[0])
        assert profile.density == 0.0
        assert profile.mean_activation == 0.0
        assert profile.top_token_ids == []
        assert profile.top_activations == []

    def test_top_k_clamps_to_active_count(self) -> None:
        sae = _identity_sae(1)
        activations = torch.tensor([[2.0], [0.0], [0.0]])  # only one active token
        tokens = torch.tensor([1, 2, 3])
        (profile,) = interpret_features(sae, activations, tokens, top_k=10, features=[0])
        assert profile.top_token_ids == [1]  # never pads with inactive tokens
        assert profile.density == pytest.approx(1 / 3)

    def test_defaults_to_all_features(self) -> None:
        sae = _identity_sae(4)
        profiles = interpret_features(sae, torch.rand(6, 4), torch.arange(6), top_k=2)
        assert [p.feature for p in profiles] == [0, 1, 2, 3]

    def test_bad_activations_shape_raises(self) -> None:
        sae = _identity_sae(3)
        with pytest.raises(ValueError, match="activations must be"):
            interpret_features(sae, torch.rand(5, 8), torch.arange(5))

    def test_token_ids_length_mismatch_raises(self) -> None:
        sae = _identity_sae(3)
        with pytest.raises(ValueError, match="token_ids must be"):
            interpret_features(sae, torch.rand(5, 3), torch.arange(4))

    def test_bad_top_k_raises(self) -> None:
        sae = _identity_sae(3)
        with pytest.raises(ValueError, match="top_k must be"):
            interpret_features(sae, torch.rand(5, 3), torch.arange(5), top_k=0)

    def test_feature_out_of_range_raises(self) -> None:
        sae = _identity_sae(3)
        with pytest.raises(ValueError, match="feature must be"):
            interpret_features(sae, torch.rand(5, 3), torch.arange(5), features=[9])
