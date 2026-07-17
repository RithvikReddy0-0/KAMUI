"""Unit tests for kamui.mechinterp.induction.

Coverage target:
    kamui/mechinterp/induction.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.mechinterp.induction import InductionHeadDetector  # noqa: E402
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=40,
        context_length=16,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _detector(**overrides: object) -> InductionHeadDetector:
    torch.manual_seed(0)
    return InductionHeadDetector(KAMUITransformer(_config(**overrides)).eval())


class TestScoreAllHeads:
    def test_scores_all_pairs(self) -> None:
        detector = _detector(n_layers=2, n_heads=4)
        scores = detector.score_all_heads(prefix_len=6, n_samples=2)
        assert set(scores) == {(layer, head) for layer in range(2) for head in range(4)}

    def test_scores_in_unit_interval(self) -> None:
        detector = _detector()
        scores = detector.score_all_heads(prefix_len=6, n_samples=2)
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_default_prefix_len(self) -> None:
        detector = _detector()
        scores = detector.score_all_heads(n_samples=1)  # defaults to ctx // 2
        assert len(scores) == 2 * 4

    def test_deterministic_given_seed(self) -> None:
        detector = _detector()
        a = detector.score_all_heads(prefix_len=6, n_samples=2, seed=3)
        b = detector.score_all_heads(prefix_len=6, n_samples=2, seed=3)
        assert a == b

    def test_prefix_len_out_of_range(self) -> None:
        detector = _detector()
        with pytest.raises(ValueError, match="prefix_len must be"):
            detector.score_all_heads(prefix_len=100)
        with pytest.raises(ValueError, match="prefix_len must be"):
            detector.score_all_heads(prefix_len=1)

    def test_bad_n_samples(self) -> None:
        detector = _detector()
        with pytest.raises(ValueError, match="n_samples must be"):
            detector.score_all_heads(n_samples=0)


class TestPlotScores:
    def test_plot_returns_figure(self) -> None:
        detector = _detector()
        scores = detector.score_all_heads(prefix_len=6, n_samples=1)
        fig = detector.plot_scores(scores)
        assert fig is not None
        plt.close(fig)


class TestAblateAndMeasure:
    def test_returns_float(self) -> None:
        detector = _detector()
        delta = detector.ablate_and_measure([(0, 0)], prefix_len=6)
        assert isinstance(delta, float)

    def test_multiple_heads_multiple_layers(self) -> None:
        detector = _detector()
        delta = detector.ablate_and_measure([(0, 0), (0, 1), (1, 2)], prefix_len=6)
        assert isinstance(delta, float)

    def test_ablation_changes_loss(self) -> None:
        # Zero-ablating every head of every layer must change the output
        # (the model degenerates to embeddings + FFN only).
        detector = _detector(n_layers=2, n_heads=4)
        all_heads = [(layer, head) for layer in range(2) for head in range(4)]
        delta = detector.ablate_and_measure(all_heads, prefix_len=6)
        assert delta != 0.0

    def test_hooks_removed_after_call(self) -> None:
        detector = _detector()
        detector.ablate_and_measure([(0, 0)], prefix_len=6)
        out_proj = detector.model.blocks[0].attn.out_proj
        assert len(out_proj._forward_pre_hooks) == 0

    def test_empty_heads_raises(self) -> None:
        detector = _detector()
        with pytest.raises(ValueError, match="non-empty"):
            detector.ablate_and_measure([])

    def test_out_of_range_head_raises(self) -> None:
        detector = _detector(n_layers=2, n_heads=4)
        with pytest.raises(ValueError, match="out of range"):
            detector.ablate_and_measure([(5, 0)])
        with pytest.raises(ValueError, match="out of range"):
            detector.ablate_and_measure([(0, 9)])
