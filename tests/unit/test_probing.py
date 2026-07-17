"""Unit tests for kamui.mechinterp.probing.

Coverage target:
    kamui/mechinterp/probing.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kamui.mechinterp.probing import LayerProbeResult, LinearProbe, ProbeResult  # noqa: E402
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=20,
        context_length=8,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _probe(**overrides: object) -> LinearProbe:
    torch.manual_seed(0)
    return LinearProbe(KAMUITransformer(_config(**overrides)).eval())


def _token_class_dataset(n: int = 40) -> tuple[list[torch.Tensor], list[int]]:
    """A linearly separable task: label = whether the LAST token id < 10.

    The last-position residual stream contains that token's embedding, so a
    linear probe must be able to decode this well above chance.
    """
    torch.manual_seed(2)
    dataset = [torch.randint(0, 20, (5,)) for _ in range(n)]
    labels = [int(ids[-1].item() < 10) for ids in dataset]
    # Ensure both classes appear.
    dataset[0][-1] = 3
    labels[0] = 1
    dataset[1][-1] = 15
    labels[1] = 0
    return dataset, labels


class TestLinearProbeTrain:
    def test_result_fields(self) -> None:
        probe = _probe()
        dataset, labels = _token_class_dataset()
        result = probe.train("blocks.0.ffn.output", dataset, labels, epochs=50)
        assert isinstance(result, ProbeResult)
        assert result.hook_point == "blocks.0.ffn.output"
        assert 0.0 <= result.train_acc <= 1.0
        assert 0.0 <= result.val_acc <= 1.0
        assert result.weights.shape == (2, 16)  # (n_classes, d_model)

    def test_separable_task_beats_chance(self) -> None:
        # Last-token identity is linearly present in the embedding output.
        probe = _probe()
        dataset, labels = _token_class_dataset(n=60)
        result = probe.train("embed.output", dataset, labels, epochs=300, lr=0.1)
        assert result.train_acc > 0.9

    def test_empty_dataset_raises(self) -> None:
        probe = _probe()
        with pytest.raises(ValueError, match="non-empty"):
            probe.train("embed.output", [], [])

    def test_length_mismatch_raises(self) -> None:
        probe = _probe()
        with pytest.raises(ValueError, match="differ in length"):
            probe.train("embed.output", [torch.randint(0, 20, (4,))], [0, 1])

    def test_single_class_raises(self) -> None:
        probe = _probe()
        dataset = [torch.randint(0, 20, (4,)) for _ in range(4)]
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            probe.train("embed.output", dataset, [1, 1, 1, 1])


class TestProbeAllLayers:
    def test_accuracies_per_depth(self) -> None:
        probe = _probe(n_layers=2)
        dataset, labels = _token_class_dataset(n=30)
        result = probe.probe_all_layers(dataset, labels, epochs=50)
        assert isinstance(result, LayerProbeResult)
        assert len(result.val_accs) == 3  # embedding + 2 layers
        assert len(result.train_accs) == 3
        assert all(0.0 <= a <= 1.0 for a in result.val_accs)

    def test_best_layer_in_range(self) -> None:
        probe = _probe(n_layers=2)
        dataset, labels = _token_class_dataset(n=30)
        result = probe.probe_all_layers(dataset, labels, epochs=50)
        assert 0 <= result.best_layer() <= 2

    def test_plot_returns_figure(self) -> None:
        result = LayerProbeResult(val_accs=[0.5, 0.7, 0.9], train_accs=[0.6, 0.8, 1.0])
        fig = result.plot()
        assert fig is not None
        plt.close(fig)

    def test_invalid_labels_raise(self) -> None:
        probe = _probe()
        dataset = [torch.randint(0, 20, (4,)) for _ in range(4)]
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            probe.probe_all_layers(dataset, [0, 0, 0, 0])
