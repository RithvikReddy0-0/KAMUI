"""Unit tests for kamui.mechinterp.steering (activation / feature steering).

Coverage target:
    kamui/mechinterp/steering.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.evaluate.generation import generate  # noqa: E402
from kamui.mechinterp.steering import (  # noqa: E402
    FeatureSteerer,
    SteeringResult,
    _steering_hook,
    build_steering_vector,
)
from kamui.mechinterp.superposition import (  # noqa: E402
    SparseAutoencoder,
    collect_activations,
)
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


class _CharTokenizer:
    """A trivial tokenizer (encode/decode) for generation tests — vocab < 16."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 16 for c in text] or [1]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def _tiny_model(seed: int = 0) -> KAMUITransformer:
    torch.manual_seed(seed)
    config = ModelConfig(
        n_layers=2,
        d_model=8,
        n_heads=2,
        d_ff=16,
        vocab_size=16,
        context_length=8,
        dropout=0.0,
    )
    model = KAMUITransformer(config)
    model.eval()
    return model


def _ids(length: int = 5) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 16, (length,))


# ===========================================================================
# The intervention math (pure, model-free)
# ===========================================================================


class TestSteeringHook:
    def test_adds_at_every_position(self) -> None:
        add = torch.tensor([1.0, 2.0, 3.0, 4.0])
        hook = _steering_hook(add, position=None)
        out = torch.zeros(2, 3, 4)
        result = hook(None, (), out)  # type: ignore[arg-type]
        assert torch.equal(result, add.expand(2, 3, 4))

    def test_adds_only_at_one_position(self) -> None:
        add = torch.tensor([1.0, 2.0, 3.0, 4.0])
        hook = _steering_hook(add, position=1)
        out = torch.zeros(1, 3, 4)
        result = hook(None, (), out)  # type: ignore[arg-type]
        assert torch.equal(result[0, 1], add)
        assert result[0, 0].abs().sum() == 0.0  # other positions untouched
        assert result[0, 2].abs().sum() == 0.0
        assert out.abs().sum() == 0.0  # input tensor was not mutated


# ===========================================================================
# SteeringResult
# ===========================================================================


class TestSteeringResult:
    def _result(self) -> SteeringResult:
        baseline = torch.zeros(1, 1, 5)
        steered = torch.tensor([[[3.0, -2.0, 1.0, -4.0, 5.0]]])
        return SteeringResult(baseline, steered, "embed.output", 2.0)

    def test_logit_delta(self) -> None:
        delta = self._result().logit_delta
        assert torch.equal(delta, torch.tensor([3.0, -2.0, 1.0, -4.0, 5.0]))

    def test_top_promoted(self) -> None:
        assert self._result().top_promoted(2) == [(4, 5.0), (0, 3.0)]

    def test_top_suppressed(self) -> None:
        assert self._result().top_suppressed(2) == [(3, -4.0), (1, -2.0)]

    def test_top_k_clamps_to_vocab(self) -> None:
        assert len(self._result().top_promoted(100)) == 5

    def test_plot_returns_figure(self) -> None:
        fig = self._result().plot(k=3)
        assert fig is not None
        plt.close(fig)


# ===========================================================================
# FeatureSteerer.steer
# ===========================================================================


class TestSteer:
    def test_zero_coefficient_is_identity(self) -> None:
        model = _tiny_model()
        steerer = FeatureSteerer(model)
        direction = torch.randn(8)
        result = steerer.steer(_ids(), "blocks.0.attn.output", direction, coefficient=0.0)
        assert torch.equal(result.baseline_logits, result.steered_logits)

    def test_nonzero_coefficient_changes_output(self) -> None:
        model = _tiny_model()
        steerer = FeatureSteerer(model)
        direction = torch.randn(8)
        result = steerer.steer(_ids(), "embed.output", direction, coefficient=5.0)
        assert not torch.equal(result.baseline_logits, result.steered_logits)
        assert result.hook_point == "embed.output"
        assert result.coefficient == 5.0
        assert result.steered_logits.shape == (1, 5, 16)

    def test_accepts_1xs_input(self) -> None:
        model = _tiny_model()
        steerer = FeatureSteerer(model)
        result = steerer.steer(_ids().unsqueeze(0), "blocks.1.ffn.output", torch.randn(8))
        assert result.steered_logits.shape == (1, 5, 16)

    def test_position_scoped_steer_changes_output(self) -> None:
        model = _tiny_model()
        steerer = FeatureSteerer(model)
        result = steerer.steer(_ids(), "embed.output", torch.randn(8), coefficient=5.0, position=0)
        assert not torch.equal(result.baseline_logits, result.steered_logits)

    def test_invalid_hook_point_raises(self) -> None:
        steerer = FeatureSteerer(_tiny_model())
        with pytest.raises(ValueError, match="not a steerable point"):
            steerer.steer(_ids(), "blocks.0.attn.weights", torch.randn(8))

    def test_bad_direction_shape_raises(self) -> None:
        steerer = FeatureSteerer(_tiny_model())
        with pytest.raises(ValueError, match="direction must have shape"):
            steerer.steer(_ids(), "embed.output", torch.randn(4))

    def test_bad_input_shape_raises(self) -> None:
        steerer = FeatureSteerer(_tiny_model())
        with pytest.raises(ValueError, match="single sequence"):
            steerer.steer(torch.randint(0, 16, (2, 5)), "embed.output", torch.randn(8))


# ===========================================================================
# FeatureSteerer.steer_with_feature
# ===========================================================================


class TestSteerWithFeature:
    def test_matches_manual_decoder_direction(self) -> None:
        model = _tiny_model()
        sae = SparseAutoencoder(d_model=8, n_features=32)
        steerer = FeatureSteerer(model, sae)

        by_feature = steerer.steer_with_feature(_ids(), "embed.output", feature=3, coefficient=4.0)
        by_direction = steerer.steer(_ids(), "embed.output", sae.W_dec[3].detach(), coefficient=4.0)
        assert torch.equal(by_feature.steered_logits, by_direction.steered_logits)

    def test_requires_sae(self) -> None:
        steerer = FeatureSteerer(_tiny_model())  # no SAE
        with pytest.raises(ValueError, match="requires an SAE"):
            steerer.steer_with_feature(_ids(), "embed.output", feature=0)

    def test_feature_out_of_range_raises(self) -> None:
        sae = SparseAutoencoder(d_model=8, n_features=32)
        steerer = FeatureSteerer(_tiny_model(), sae)
        with pytest.raises(ValueError, match="feature must be"):
            steerer.steer_with_feature(_ids(), "embed.output", feature=99)


# ===========================================================================
# FeatureSteerer.generate_steered
# ===========================================================================


class TestGenerateSteered:
    def test_zero_coefficient_matches_plain_generate(self) -> None:
        model = _tiny_model()
        tok = _CharTokenizer()
        steerer = FeatureSteerer(model)
        baseline = generate(model, tok, "hello", max_new_tokens=6)
        steered = steerer.generate_steered(
            tok, "hello", "embed.output", torch.randn(8), coefficient=0.0, max_new_tokens=6
        )
        assert steered == baseline

    def test_strong_steering_changes_text(self) -> None:
        model = _tiny_model()
        tok = _CharTokenizer()
        steerer = FeatureSteerer(model)
        baseline = generate(model, tok, "hello", max_new_tokens=6)
        torch.manual_seed(7)
        steered = steerer.generate_steered(
            tok, "hello", "embed.output", torch.randn(8), coefficient=1000.0, max_new_tokens=6
        )
        assert steered != baseline

    def test_hook_is_removed_after_generation(self) -> None:
        model = _tiny_model()
        tok = _CharTokenizer()
        steerer = FeatureSteerer(model)
        baseline = generate(model, tok, "hello", max_new_tokens=6)
        steerer.generate_steered(
            tok, "hello", "embed.output", torch.randn(8), coefficient=500.0, max_new_tokens=6
        )
        # If the steering hook leaked, this plain call would differ.
        assert generate(model, tok, "hello", max_new_tokens=6) == baseline

    def test_with_feature_matches_manual_direction(self) -> None:
        model = _tiny_model()
        tok = _CharTokenizer()
        sae = SparseAutoencoder(d_model=8, n_features=32)
        steerer = FeatureSteerer(model, sae)
        by_feature = steerer.generate_steered_with_feature(
            tok, "hi", "embed.output", feature=5, coefficient=50.0, max_new_tokens=6
        )
        by_direction = steerer.generate_steered(
            tok, "hi", "embed.output", sae.W_dec[5].detach(), coefficient=50.0, max_new_tokens=6
        )
        assert by_feature == by_direction

    def test_invalid_hook_point_raises(self) -> None:
        steerer = FeatureSteerer(_tiny_model())
        with pytest.raises(ValueError, match="not a steerable point"):
            steerer.generate_steered(_CharTokenizer(), "hi", "unembed.input", torch.randn(8))

    def test_bad_direction_shape_raises(self) -> None:
        steerer = FeatureSteerer(_tiny_model())
        with pytest.raises(ValueError, match="direction must have shape"):
            steerer.generate_steered(_CharTokenizer(), "hi", "embed.output", torch.randn(3))

    def test_with_feature_requires_sae(self) -> None:
        steerer = FeatureSteerer(_tiny_model())  # no SAE
        with pytest.raises(ValueError, match="requires an SAE"):
            steerer.generate_steered_with_feature(_CharTokenizer(), "hi", "embed.output", feature=0)


# ===========================================================================
# build_steering_vector (ActAdd contrastive direction)
# ===========================================================================


class TestBuildSteeringVector:
    def _seqs(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        torch.manual_seed(3)
        positive = [torch.randint(0, 16, (5,)), torch.randint(0, 16, (5,))]
        negative = [torch.randint(0, 16, (5,))]
        return positive, negative

    def test_shape_is_d_model(self) -> None:
        model = _tiny_model()
        pos, neg = self._seqs()
        assert build_steering_vector(model, "blocks.0.ffn.output", pos, neg).shape == (8,)

    def test_matches_mean_activation_difference(self) -> None:
        model = _tiny_model()
        pos, neg = self._seqs()
        expected = collect_activations(model, "blocks.0.ffn.output", pos).mean(0) - (
            collect_activations(model, "blocks.0.ffn.output", neg).mean(0)
        )
        actual = build_steering_vector(model, "blocks.0.ffn.output", pos, neg)
        assert torch.equal(actual, expected)

    def test_self_contrast_is_zero(self) -> None:
        model = _tiny_model()
        pos, _ = self._seqs()
        direction = build_steering_vector(model, "blocks.1.ffn.output", pos, pos)
        assert torch.equal(direction, torch.zeros(8))

    def test_antisymmetry(self) -> None:
        model = _tiny_model()
        pos, neg = self._seqs()
        d_pn = build_steering_vector(model, "embed.output", pos, neg)
        d_np = build_steering_vector(model, "embed.output", neg, pos)
        assert torch.equal(d_pn, -d_np)

    def test_normalize_gives_unit_norm(self) -> None:
        model = _tiny_model()
        pos, neg = self._seqs()
        direction = build_steering_vector(model, "embed.output", pos, neg, normalize=True)
        assert direction.norm().item() == pytest.approx(1.0, abs=1e-5)

    def test_composes_with_steer(self) -> None:
        model = _tiny_model()
        pos, neg = self._seqs()
        steerer = FeatureSteerer(model)
        direction = build_steering_vector(model, "embed.output", pos, neg)
        result = steerer.steer(_ids(), "embed.output", direction, coefficient=10.0)
        assert not torch.equal(result.baseline_logits, result.steered_logits)

    def test_empty_positive_raises(self) -> None:
        model = _tiny_model()
        _, neg = self._seqs()
        with pytest.raises(ValueError, match="no activations"):
            build_steering_vector(model, "embed.output", [], neg)
