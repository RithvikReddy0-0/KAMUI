"""Unit tests for kamui.mechinterp.attribution.

Coverage target:
    kamui/mechinterp/attribution.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.mechinterp.attribution import AttributionResult, GradientAttribution  # noqa: E402
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=50,
        context_length=16,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


class _Tok:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (i % 26)) for i in ids)


def _attr(with_tokenizer: bool = True, **overrides: object) -> GradientAttribution:
    torch.manual_seed(0)
    model = KAMUITransformer(_config(**overrides)).eval()
    return GradientAttribution(model, _Tok() if with_tokenizer else None)


def _ids(seq: int = 10, vocab: int = 50) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, vocab, (seq,))


class TestInputXGrad:
    def test_scores_shape_and_finite(self) -> None:
        result = _attr().token_attribution(_ids(), method="input_x_grad")
        assert result.scores.shape == (10,)
        assert torch.isfinite(result.scores).all()

    def test_scores_are_nonzero(self) -> None:
        # A non-trivial model gives nonzero input-gradient attribution.
        result = _attr().token_attribution(_ids(), method="input_x_grad")
        assert result.scores.abs().sum() > 0

    def test_metrics_are_none(self) -> None:
        result = _attr().token_attribution(_ids(), method="input_x_grad")
        assert result.input_metric is None
        assert result.baseline_metric is None

    def test_accepts_2d_single_row(self) -> None:
        result = _attr().token_attribution(_ids().unsqueeze(0))
        assert result.scores.shape == (10,)

    def test_explicit_target_token(self) -> None:
        result = _attr().token_attribution(_ids(), target_token=7)
        assert result.target_token == 7

    def test_default_target_is_argmax(self) -> None:
        attr = _attr()
        ids = _ids()
        result = attr.token_attribution(ids)
        with torch.no_grad():
            expected = int(attr.model(ids.unsqueeze(0))[0, -1].argmax())
        assert result.target_token == expected


class TestIntegratedGradients:
    def test_scores_shape_and_finite(self) -> None:
        result = _attr().token_attribution(_ids(), method="integrated_gradients", steps=32)
        assert result.scores.shape == (10,)
        assert torch.isfinite(result.scores).all()

    def test_completeness_axiom(self) -> None:
        # The defining IG property: attributions sum to metric(input) - metric(baseline).
        result = _attr().token_attribution(_ids(), method="integrated_gradients", steps=128)
        assert result.input_metric is not None and result.baseline_metric is not None
        gap = result.input_metric - result.baseline_metric
        assert result.scores.sum().item() == pytest.approx(gap, abs=0.1)

    def test_more_steps_reduces_completeness_error(self) -> None:
        attr = _attr()
        ids = _ids()

        def err(steps: int) -> float:
            r = attr.token_attribution(ids, method="integrated_gradients", steps=steps)
            return abs(r.scores.sum().item() - (r.input_metric - r.baseline_metric))

        assert err(128) <= err(8) + 1e-6

    def test_bad_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="steps must be >= 1"):
            _attr().token_attribution(_ids(), method="integrated_gradients", steps=0)


class TestErrorsAndLabels:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            _attr().token_attribution(_ids(), method="shap")

    def test_batch_gt_one_raises(self) -> None:
        with pytest.raises(ValueError, match="single sequence"):
            _attr().token_attribution(torch.randint(0, 50, (2, 6)))

    def test_labels_use_tokenizer(self) -> None:
        result = _attr(with_tokenizer=True).token_attribution(_ids(5))
        assert len(result.tokens) == 5
        assert all(t.isalpha() for t in result.tokens)  # decoded to letters

    def test_labels_without_tokenizer_are_ids(self) -> None:
        result = _attr(with_tokenizer=False).token_attribution(_ids(4))
        assert all(t.startswith("<") for t in result.tokens)

    def test_decode_failure_falls_back(self) -> None:
        class _BadTok:
            def decode(self, ids: list[int]) -> str:
                raise ValueError("nope")

        torch.manual_seed(0)
        model = KAMUITransformer(_config()).eval()
        result = GradientAttribution(model, _BadTok()).token_attribution(_ids(3))
        assert all(t.startswith("<") for t in result.tokens)


class TestPlot:
    def test_plot_returns_figure(self) -> None:
        result = _attr().token_attribution(_ids(6))
        fig = result.plot()
        assert fig is not None
        plt.close(fig)

    def test_result_is_dataclass(self) -> None:
        result = _attr().token_attribution(_ids(5))
        assert isinstance(result, AttributionResult)
        assert result.method == "input_x_grad"
