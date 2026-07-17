"""Unit tests for kamui.evaluate.calibration.

Coverage target:
    kamui/evaluate/calibration.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.evaluate.calibration import (  # noqa: E402
    expected_calibration_error,
    reliability_diagram,
    temperature_scaling,
)
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


def _model() -> KAMUITransformer:
    torch.manual_seed(0)
    return KAMUITransformer(
        ModelConfig(
            n_layers=1,
            d_model=16,
            n_heads=4,
            d_ff=32,
            vocab_size=20,
            context_length=8,
            dropout=0.0,
        )
    ).eval()


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_is_zero(self) -> None:
        # One-hot predictions that are always correct: confidence 1, accuracy 1.
        probs = torch.eye(4)
        labels = torch.arange(4)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-6)

    def test_fully_overconfident_known_value(self) -> None:
        # One-hot predictions, half correct: confidence 1.0, accuracy 0.5 → ECE 0.5.
        probs = torch.eye(2).repeat(2, 1)  # predicts class 0,1,0,1
        labels = torch.tensor([0, 1, 1, 0])  # half wrong
        assert expected_calibration_error(probs, labels) == pytest.approx(0.5, abs=1e-6)

    def test_in_unit_interval(self) -> None:
        torch.manual_seed(0)
        probs = F.softmax(torch.randn(50, 10), dim=-1)
        labels = torch.randint(0, 10, (50,))
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_bad_n_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins must be"):
            expected_calibration_error(torch.eye(2), torch.arange(2), n_bins=0)

    def test_bad_probs_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="probs must be 2-D"):
            expected_calibration_error(torch.rand(4), torch.arange(4))

    def test_label_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="labels must be 1-D"):
            expected_calibration_error(torch.eye(4), torch.arange(3))


class TestReliabilityDiagram:
    def test_returns_figure_with_ece_title(self) -> None:
        torch.manual_seed(0)
        probs = F.softmax(torch.randn(30, 5), dim=-1)
        labels = torch.randint(0, 5, (30,))
        fig = reliability_diagram(probs, labels)
        assert fig is not None
        assert "ECE" in fig.axes[0].get_title()
        plt.close(fig)

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(ValueError, match="probs must be 2-D"):
            reliability_diagram(torch.rand(4), torch.arange(4))


class TestTemperatureScaling:
    def test_returns_positive_float(self) -> None:
        model = _model()
        loader = [torch.randint(0, 20, (2, 8)) for _ in range(2)]
        t = temperature_scaling(model, loader)
        assert isinstance(t, float)
        assert t > 0

    def test_optimum_no_worse_than_t1(self) -> None:
        # NLL at the chosen temperature must be <= NLL at T=1 (1.0 is on the
        # grid via the default linspace being dense around it — verify directly).
        model = _model()
        loader = [torch.randint(0, 20, (2, 8)) for _ in range(2)]
        t = temperature_scaling(model, loader, temperatures=torch.tensor([0.5, 1.0, 2.0]))
        logits, targets = [], []
        for batch in loader:
            out = model(batch[:, :-1])
            logits.append(out.reshape(-1, out.shape[-1]))
            targets.append(batch[:, 1:].reshape(-1))
        logit_cat, target_cat = torch.cat(logits), torch.cat(targets)
        nll_best = F.cross_entropy(logit_cat / t, target_cat).item()
        nll_t1 = F.cross_entropy(logit_cat, target_cat).item()
        assert nll_best <= nll_t1 + 1e-6

    def test_pair_batches_supported(self) -> None:
        model = _model()
        loader = [(torch.randint(0, 20, (2, 7)), torch.randint(0, 20, (2, 7)))]
        t = temperature_scaling(model, loader)
        assert t > 0

    def test_empty_loader_raises(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="no tokens"):
            temperature_scaling(model, [])

    def test_restores_training_mode(self) -> None:
        model = _model()
        model.train()
        temperature_scaling(model, [torch.randint(0, 20, (1, 8))])
        assert model.training
