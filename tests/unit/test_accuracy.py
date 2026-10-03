"""Unit tests for kamui.evaluate.accuracy (top-k next-token accuracy).

Coverage target:
    kamui/evaluate/accuracy.py — 100%
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kamui.evaluate.accuracy import compute_accuracy
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer

# Distinct scores (no ties): the model's ranking is always 0 > 1 > 2 > 3 > ...
_LOGITS = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]


class _FixedLogitModel(nn.Module):
    """Predicts the same ranked logits at every position."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("row", torch.tensor(_LOGITS))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch, seq = ids.shape
        return self.row.expand(batch, seq, -1)


def _pair_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Four positions whose targets are ranked 1st, 2nd, 3rd and 4th by the model.
    inputs = torch.zeros(1, 4, dtype=torch.long)
    targets = torch.tensor([[0, 1, 2, 3]])
    return [(inputs, targets)]


class TestExactValues:
    @pytest.mark.parametrize(("k", "expected"), [(1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)])
    def test_top_k_accuracy(self, k: int, expected: float) -> None:
        acc = compute_accuracy(_FixedLogitModel(), _pair_loader(), k=k)  # type: ignore[arg-type]
        assert acc == expected

    def test_k_larger_than_vocab_is_clamped(self) -> None:
        acc = compute_accuracy(_FixedLogitModel(), _pair_loader(), k=100)  # type: ignore[arg-type]
        assert acc == 1.0

    def test_single_tensor_batches_are_shifted(self) -> None:
        # [5, 0, 1, 2, 3] -> inputs [5, 0, 1, 2], targets [0, 1, 2, 3]
        loader = [torch.tensor([[5, 0, 1, 2, 3]])]
        acc = compute_accuracy(_FixedLogitModel(), loader, k=2)  # type: ignore[arg-type]
        assert acc == 0.5

    def test_accumulates_across_batches(self) -> None:
        loader = _pair_loader() + [(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4).long())]
        # first batch: 1/4 correct at top-1; second batch: all 4 targets are token 0.
        acc = compute_accuracy(_FixedLogitModel(), loader, k=1)  # type: ignore[arg-type]
        assert acc == 5 / 8


class TestRealModel:
    def _model(self) -> KAMUITransformer:
        torch.manual_seed(0)
        cfg = ModelConfig(
            n_layers=1, d_model=16, n_heads=4, d_ff=32, vocab_size=20, context_length=8
        )
        return KAMUITransformer(cfg)

    def test_accuracy_is_monotone_in_k(self) -> None:
        model = self._model()
        torch.manual_seed(1)
        loader = [torch.randint(0, 20, (4, 9)) for _ in range(3)]
        accs = [compute_accuracy(model, loader, k=k) for k in (1, 3, 5, 10, 20)]
        assert all(0.0 <= a <= 1.0 for a in accs)
        assert accs == sorted(accs)
        assert accs[-1] == 1.0  # k == vocab_size always contains the target

    def test_restores_training_mode(self) -> None:
        model = self._model()
        model.train()
        compute_accuracy(model, [torch.randint(0, 20, (2, 5))])
        assert model.training


class TestErrors:
    def test_bad_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            compute_accuracy(_FixedLogitModel(), _pair_loader(), k=0)  # type: ignore[arg-type]

    def test_empty_loader_raises(self) -> None:
        with pytest.raises(ValueError, match="no tokens"):
            compute_accuracy(_FixedLogitModel(), [])  # type: ignore[arg-type]
