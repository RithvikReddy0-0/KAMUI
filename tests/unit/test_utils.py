"""Unit tests for kamui.utils (reproducibility, logging, plotting).

Coverage target:
    kamui/utils/*.py — 100%
"""

from __future__ import annotations

import logging

import matplotlib
import pytest
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.utils.logging import (  # noqa: E402
    TrainingLogger,
    get_logger,
    log_model_stats,
    parse_log_line,
)
from kamui.utils.plotting import heatmap, layer_plot, save_figure, token_heatmap  # noqa: E402
from kamui.utils.reproducibility import get_device, set_deterministic, set_seed  # noqa: E402

# ===========================================================================
# Reproducibility
# ===========================================================================


class TestReproducibility:
    def test_set_seed_makes_torch_deterministic(self) -> None:
        set_seed(42)
        a = torch.randn(8)
        set_seed(42)
        b = torch.randn(8)
        assert torch.equal(a, b)

    def test_set_seed_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="seed must be >= 0"):
            set_seed(-1)

    def test_set_deterministic_toggle(self) -> None:
        set_deterministic(True)
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        set_deterministic(False)  # restore default for other tests
        assert not torch.backends.cudnn.deterministic

    def test_get_device_returns_device(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")


# ===========================================================================
# Logging
# ===========================================================================


class TestLogging:
    def test_get_logger_formats_and_caches_handler(self) -> None:
        logger = get_logger("kamui.test_x")
        logger2 = get_logger("kamui.test_x")
        assert logger is logger2
        assert len(logger.handlers) == 1  # not duplicated

    def test_training_logger_line_format(self) -> None:
        tlog = TrainingLogger("kamui.test_train")
        line = tlog.log_step(500, train_loss=2.341, lr=3e-4, grad_norm=1.834)
        assert line.startswith("step=500 ")
        assert "train_loss=2.3410" in line
        assert "lr=3.00e-04" in line
        assert tlog.step == 500

    def test_parse_log_line_roundtrip(self) -> None:
        tlog = TrainingLogger("kamui.test_parse")
        line = tlog.log_step(7, train_loss=1.5, lr=1e-3)
        parsed = parse_log_line(line)
        assert parsed["step"] == 7.0
        assert parsed["train_loss"] == pytest.approx(1.5)
        assert parsed["lr"] == pytest.approx(1e-3)

    def test_parse_log_line_malformed(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            parse_log_line("step=1 garbage")

    def test_log_model_stats(self, caplog: pytest.LogCaptureFixture) -> None:
        model = nn.Linear(4, 4)
        logger = logging.getLogger("kamui.test_stats")
        with caplog.at_level(logging.INFO, logger="kamui.test_stats"):
            log_model_stats(model, logger)
        assert "model parameters" in caplog.text
        assert "20" in caplog.text  # 4*4 + 4


# ===========================================================================
# Plotting
# ===========================================================================


class TestPlotting:
    def test_heatmap_returns_figure(self) -> None:
        fig = heatmap(torch.rand(3, 4), ["a", "b", "c"], ["w", "x", "y", "z"], title="t")
        assert fig is not None
        plt.close(fig)

    def test_heatmap_no_labels(self) -> None:
        fig = heatmap(torch.rand(2, 2))
        plt.close(fig)

    def test_heatmap_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            heatmap(torch.rand(3))

    def test_layer_plot_list_and_tensor(self) -> None:
        fig1 = layer_plot([0.1, 0.5, 0.9], ylabel="acc", title="t")
        fig2 = layer_plot(torch.tensor([0.1, 0.5]))
        plt.close(fig1)
        plt.close(fig2)

    def test_token_heatmap(self) -> None:
        fig = token_heatmap(["the", "cat"], [0.2, 0.8], title="importance")
        plt.close(fig)

    def test_token_heatmap_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            token_heatmap(["a"], [0.1, 0.2])

    def test_save_figure_absolute(self, tmp_path) -> None:
        fig = layer_plot([1.0, 2.0])
        out = save_figure(fig, tmp_path / "fig.png")
        assert out.exists()
        plt.close(fig)

    def test_save_figure_relative_goes_to_figures_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        fig = layer_plot([1.0])
        out = save_figure(fig, "sub/fig.png")
        assert out.exists()
        assert "figures" in str(out)
        plt.close(fig)
