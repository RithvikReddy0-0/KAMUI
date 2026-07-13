"""Unit tests for kamui.mechinterp.logit_lens.

Coverage target:
    kamui/mechinterp/logit_lens.py — 100%
"""

from __future__ import annotations

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # headless backend for plot tests
import matplotlib.pyplot as plt  # noqa: E402

from kamui.mechinterp.logit_lens import LogitLens  # noqa: E402
from kamui.model.config import ModelConfig  # noqa: E402
from kamui.model.transformer import KAMUITransformer  # noqa: E402


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=3,
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


def _lens(**overrides: object) -> LogitLens:
    model = KAMUITransformer(_config(**overrides)).eval()
    return LogitLens(model, _Tok())


class TestLogitLensRun:
    def test_probs_shape(self) -> None:
        lens = _lens(n_layers=3, vocab_size=40)
        result = lens.run(torch.randint(0, 40, (7,)))
        assert result.probs.shape == (4, 7, 40)  # (n_layers+1, S, V)

    def test_probs_are_distributions(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (5,)))
        sums = result.probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_final_layer_matches_model_logits(self) -> None:
        # The last reconstructed stream, projected, must equal the model's own
        # output distribution — validating the residual reconstruction.
        model = KAMUITransformer(_config()).eval()
        lens = LogitLens(model, _Tok())
        ids = torch.randint(0, 40, (1, 6))
        result = lens.run(ids)
        model_probs = torch.softmax(model(ids), dim=-1)[0]  # (S, V)
        assert torch.allclose(result.probs[-1], model_probs, atol=1e-5)

    def test_top_tokens_shape(self) -> None:
        lens = _lens(n_layers=3)
        result = lens.run(torch.randint(0, 40, (5,)), top_k=4)
        assert result.top_tokens.shape == (4, 5, 4)

    def test_top_k_clamped_to_vocab(self) -> None:
        lens = _lens(vocab_size=40)
        result = lens.run(torch.randint(0, 40, (3,)), top_k=1000)
        assert result.top_tokens.shape[-1] == 40

    def test_tokens_labels_length(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (5,)))
        assert len(result.tokens) == 5
        assert all(isinstance(t, str) for t in result.tokens)

    def test_accepts_2d_single_row(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (1, 6)))
        assert result.probs.shape[1] == 6

    def test_top_k_too_small(self) -> None:
        lens = _lens()
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            lens.run(torch.randint(0, 40, (5,)), top_k=0)

    def test_batch_gt_one_raises(self) -> None:
        lens = _lens()
        with pytest.raises(ValueError, match="single sequence"):
            lens.run(torch.randint(0, 40, (2, 5)))

    def test_n_layers_property(self) -> None:
        lens = _lens(n_layers=3)
        result = lens.run(torch.randint(0, 40, (4,)))
        assert result.n_layers == 3


class TestLogitLensResultHelpers:
    def test_top1_token_ids_shape(self) -> None:
        lens = _lens(n_layers=2)
        result = lens.run(torch.randint(0, 40, (5,)))
        assert result.top1_token_ids().shape == (3, 5)

    def test_top1_labels(self) -> None:
        lens = _lens(n_layers=2)
        result = lens.run(torch.randint(0, 40, (5,)))
        labels = result.top1_labels()
        assert len(labels) == 3 and len(labels[0]) == 5

    def test_decode_fallback_on_bad_token(self) -> None:
        class _BadTok:
            def decode(self, ids: list[int]) -> str:
                raise ValueError("cannot decode")

        model = KAMUITransformer(_config()).eval()
        result = LogitLens(model, _BadTok()).run(torch.randint(0, 40, (4,)))
        # Undecodable tokens fall back to "<id>" form.
        assert all(t.startswith("<") for t in result.tokens)

    def test_decode_empty_fallback(self) -> None:
        class _EmptyTok:
            def decode(self, ids: list[int]) -> str:
                return ""

        model = KAMUITransformer(_config()).eval()
        result = LogitLens(model, _EmptyTok()).run(torch.randint(0, 40, (3,)))
        assert all(t.startswith("<") for t in result.tokens)


class TestLogitLensPlots:
    def test_plot_returns_figure(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (5,)))
        fig = result.plot()
        assert fig is not None
        plt.close(fig)

    def test_plot_custom_figsize(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (4,)))
        fig = result.plot(figsize=(8, 6))
        plt.close(fig)

    def test_plot_position_returns_figure(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (5,)))
        fig = result.plot_position(2)
        assert fig is not None
        plt.close(fig)

    def test_plot_position_out_of_range(self) -> None:
        lens = _lens()
        result = lens.run(torch.randint(0, 40, (5,)))
        with pytest.raises(IndexError):
            result.plot_position(10)
