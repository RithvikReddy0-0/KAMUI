"""Unit tests for kamui.mechinterp.attention_viz.

Coverage target:
    kamui/mechinterp/attention_viz.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.mechinterp.attention_viz import (  # noqa: E402
    AttentionResult,
    AttentionVisualizer,
    head_summary_stats,
)
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=40,
        context_length=12,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


class _Tok:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (i % 26)) for i in ids)


def _viz(**overrides: object) -> AttentionVisualizer:
    torch.manual_seed(0)
    model = KAMUITransformer(_config(**overrides)).eval()
    return AttentionVisualizer(model, _Tok())


class TestAttentionVisualizerRun:
    def test_weights_shape(self) -> None:
        result = _viz(n_layers=2, n_heads=4).run(torch.randint(0, 40, (6,)))
        assert result.weights.shape == (2, 4, 6, 6)
        assert result.n_layers == 2
        assert result.n_heads == 4

    def test_weights_are_causal_probabilities(self) -> None:
        result = _viz().run(torch.randint(0, 40, (6,)))
        # Rows sum to 1 and the upper triangle is exactly zero.
        sums = result.weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
        assert torch.all(result.weights[..., mask] == 0.0)

    def test_tokens_decoded(self) -> None:
        result = _viz().run(torch.randint(0, 40, (5,)))
        assert len(result.tokens) == 5

    def test_accepts_2d_single_row(self) -> None:
        result = _viz().run(torch.randint(0, 40, (1, 4)))
        assert result.weights.shape[-1] == 4

    def test_batch_gt_one_raises(self) -> None:
        with pytest.raises(ValueError, match="single sequence"):
            _viz().run(torch.randint(0, 40, (2, 4)))

    def test_undecodable_token_fallback(self) -> None:
        class _Bad:
            def decode(self, ids: list[int]) -> str:
                raise ValueError("no")

        torch.manual_seed(0)
        model = KAMUITransformer(_config()).eval()
        result = AttentionVisualizer(model, _Bad()).run(torch.randint(0, 40, (3,)))
        assert all(t.startswith("<") for t in result.tokens)


class TestAttentionResultPlots:
    def test_plot_single_head(self) -> None:
        result = _viz().run(torch.randint(0, 40, (5,)))
        fig = result.plot(1, 2)
        assert fig is not None
        plt.close(fig)

    def test_plot_out_of_range(self) -> None:
        result = _viz(n_layers=2, n_heads=4).run(torch.randint(0, 40, (4,)))
        with pytest.raises(IndexError, match="layer"):
            result.plot(5, 0)
        with pytest.raises(IndexError, match="head"):
            result.plot(0, 9)

    def test_plot_all_grid(self) -> None:
        result = _viz().run(torch.randint(0, 40, (4,)))
        fig = result.plot_all()
        assert fig is not None
        plt.close(fig)

    def test_plot_interactive_returns_plotly(self) -> None:
        import plotly.graph_objects as go

        result = _viz(n_layers=2, n_heads=2).run(torch.randint(0, 40, (4,)))
        fig = result.plot_interactive()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2 * 2  # one trace per (layer, head)


class TestHeadSummaryStats:
    def test_columns_and_lengths(self) -> None:
        result = _viz(n_layers=2, n_heads=4).run(torch.randint(0, 40, (6,)))
        stats = head_summary_stats(result)
        assert set(stats) == {"layer", "head", "entropy", "self_frac", "prev_frac"}
        assert all(len(col) == 2 * 4 for col in stats.values())

    def test_pure_previous_token_head_detected(self) -> None:
        # Hand-built pattern: every query attends fully to position t-1
        # (position 0 attends to itself) → prev_frac ≈ 1, entropy ≈ 0.
        seq = 5
        w = torch.zeros(1, 1, seq, seq)
        w[0, 0, 0, 0] = 1.0
        for t in range(1, seq):
            w[0, 0, t, t - 1] = 1.0
        result = AttentionResult(weights=w, tokens=list("abcde"))
        stats = head_summary_stats(result)
        assert stats["prev_frac"][0] == pytest.approx(1.0, abs=1e-5)
        assert stats["entropy"][0] == pytest.approx(0.0, abs=1e-4)
        assert stats["self_frac"][0] == pytest.approx(1 / seq, abs=1e-5)

    def test_pure_diagonal_head_detected(self) -> None:
        # Identity attention → self_frac = 1.
        seq = 4
        w = torch.eye(seq).view(1, 1, seq, seq)
        result = AttentionResult(weights=w, tokens=list("abcd"))
        stats = head_summary_stats(result)
        assert stats["self_frac"][0] == pytest.approx(1.0, abs=1e-5)

    def test_single_token_sequence_prev_frac_zero(self) -> None:
        w = torch.ones(1, 1, 1, 1)
        result = AttentionResult(weights=w, tokens=["a"])
        stats = head_summary_stats(result)
        assert stats["prev_frac"][0] == 0.0
