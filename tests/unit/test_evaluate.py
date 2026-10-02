"""Unit tests for kamui.evaluate (perplexity + generation).

Coverage target:
    kamui/evaluate/perplexity.py — 100%
    kamui/evaluate/generation.py — 100%
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kamui.evaluate.generation import (
    GenerationResult,
    _apply_repetition_penalty,
    generate,
    generate_with_probs,
)
from kamui.evaluate.perplexity import (
    compute_perplexity,
    compute_sequence_perplexity,
    compute_token_loss,
)
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=40,
        context_length=8,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _model(**overrides: object) -> KAMUITransformer:
    return KAMUITransformer(_config(**overrides)).eval()


class _ByteTokenizer:
    """A trivial reversible tokenizer over a small vocab, for generation tests."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 40 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (i % 26)) for i in ids)


# ===========================================================================
# Perplexity
# ===========================================================================


class TestPerplexity:
    def test_token_loss_shape(self) -> None:
        model = _model()
        ids = torch.randint(0, 40, (6,))
        loss = compute_token_loss(model, ids)
        assert loss.shape == (5,)

    def test_token_loss_accepts_2d_single_row(self) -> None:
        model = _model()
        ids = torch.randint(0, 40, (1, 6))
        assert compute_token_loss(model, ids).shape == (5,)

    def test_token_loss_type_error(self) -> None:
        model = _model()
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            compute_token_loss(model, [1, 2, 3])  # type: ignore[arg-type]

    def test_token_loss_rank_error(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="must be 1-D"):
            compute_token_loss(model, torch.randint(0, 40, (2, 3)))

    def test_token_loss_too_short(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="at least 2 tokens"):
            compute_token_loss(model, torch.tensor([5]))

    def test_sequence_perplexity_short(self) -> None:
        model = _model(context_length=8)
        ids = torch.randint(0, 40, (6,))
        ppl = compute_sequence_perplexity(model, ids)
        expected = math.exp(compute_token_loss(model, ids).mean().item())
        assert ppl == pytest.approx(expected, rel=1e-5)

    def test_sequence_perplexity_near_vocab_at_init(self) -> None:
        model = _model(vocab_size=40)
        ids = torch.randint(0, 40, (7,))
        ppl = compute_sequence_perplexity(model, ids)
        assert abs(ppl - 40) < 20  # ~ uniform at init

    def test_sequence_perplexity_sliding_window_runs(self) -> None:
        # Sequence longer than context_length triggers the sliding window.
        model = _model(context_length=8)
        ids = torch.randint(0, 40, (25,))
        ppl = compute_sequence_perplexity(model, ids, stride=4)
        assert math.isfinite(ppl) and ppl > 0

    def test_sequence_perplexity_default_stride(self) -> None:
        model = _model(context_length=8)
        ids = torch.randint(0, 40, (20,))
        ppl = compute_sequence_perplexity(model, ids)  # stride defaults
        assert math.isfinite(ppl) and ppl > 0

    def test_sequence_perplexity_tiny_final_window(self) -> None:
        # stride == ctx makes the final window land on a single token; the
        # guard skips it rather than feeding a length-1 window to the model.
        model = _model(context_length=8)
        ids = torch.randint(0, 40, (17,))
        ppl = compute_sequence_perplexity(model, ids, stride=8)
        assert math.isfinite(ppl) and ppl > 0

    def test_sequence_perplexity_restores_training(self) -> None:
        model = _model(context_length=8)
        model.train()
        compute_sequence_perplexity(model, torch.randint(0, 40, (25,)), stride=4)
        assert model.training

    def test_compute_perplexity_restores_training(self) -> None:
        model = _model()
        model.train()
        compute_perplexity(model, [torch.randint(0, 40, (2, 8))])
        assert model.training

    def test_compute_perplexity_tensor_batches(self) -> None:
        model = _model()
        loader = [torch.randint(0, 40, (2, 8)) for _ in range(3)]
        ppl = compute_perplexity(model, loader)
        assert math.isfinite(ppl) and ppl > 0

    def test_compute_perplexity_pair_batches(self) -> None:
        model = _model()
        loader = [(torch.randint(0, 40, (2, 7)), torch.randint(0, 40, (2, 7))) for _ in range(3)]
        ppl = compute_perplexity(model, loader)
        assert math.isfinite(ppl) and ppl > 0

    def test_compute_perplexity_empty_raises(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="no tokens"):
            compute_perplexity(model, [])

    def test_split_batch_type_error(self) -> None:
        model = _model()
        with pytest.raises(TypeError, match="must be a"):
            compute_perplexity(model, [123])

    def test_restores_training_mode(self) -> None:
        model = _model()
        model.train()
        compute_token_loss(model, torch.randint(0, 40, (5,)))
        assert model.training  # restored


# ===========================================================================
# Generation
# ===========================================================================


class TestGeneration:
    def test_greedy_is_deterministic(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        a = generate(model, tok, "hello", max_new_tokens=6, strategy="greedy")
        b = generate(model, tok, "hello", max_new_tokens=6, strategy="greedy")
        assert a == b
        assert isinstance(a, str)

    def test_output_extends_prompt(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        out = generate(model, tok, "hi", max_new_tokens=5, strategy="greedy")
        # decoded length = prompt tokens + new tokens
        assert len(out) == len("hi") + 5

    def test_top_k_sampling_seeded_reproducible(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        a = generate(model, tok, "abc", max_new_tokens=5, strategy="top_k", top_k=5, seed=0)
        b = generate(model, tok, "abc", max_new_tokens=5, strategy="top_k", top_k=5, seed=0)
        assert a == b

    def test_nucleus_sampling_runs(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        out = generate(model, tok, "abc", max_new_tokens=5, strategy="nucleus", top_p=0.9, seed=1)
        assert isinstance(out, str)

    def test_temperature_sampling_runs(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        out = generate(
            model,
            tok,
            "abc",
            max_new_tokens=5,
            strategy="temperature",
            temperature=0.8,
            seed=2,
        )
        assert isinstance(out, str)

    def test_crops_to_context_length(self) -> None:
        # Generating well past context_length must not error (window cropping).
        model = _model(context_length=8)
        tok = _ByteTokenizer()
        out = generate(model, tok, "hello", max_new_tokens=20, strategy="greedy")
        assert isinstance(out, str)

    def test_unknown_strategy_raises(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="unknown strategy"):
            generate(model, tok, "hi", strategy="beam")

    def test_bad_temperature_raises(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="temperature must be"):
            generate(model, tok, "hi", temperature=0.0)

    def test_bad_top_k_raises(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="top_k must be"):
            generate(model, tok, "hi", strategy="top_k", top_k=0)

    def test_bad_top_p_raises(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="top_p must be"):
            generate(model, tok, "hi", strategy="nucleus", top_p=1.5)

    def test_empty_prompt_raises(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="non-empty"):
            generate(model, tok, "", max_new_tokens=3)

    def test_generate_with_probs(self) -> None:
        model = _model(vocab_size=40)
        tok = _ByteTokenizer()
        result = generate_with_probs(model, tok, "hello", n_tokens=4)
        assert isinstance(result, GenerationResult)
        assert len(result.token_ids) == 4
        assert result.probs.shape == (4, 40)
        # Each step is a probability distribution.
        assert torch.allclose(result.probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_generate_with_probs_empty_prompt_raises(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        with pytest.raises(ValueError, match="non-empty"):
            generate_with_probs(model, tok, "", n_tokens=3)

    def test_generate_restores_training_mode(self) -> None:
        model = _model()
        model.train()
        generate(model, _ByteTokenizer(), "hi", max_new_tokens=2)
        assert model.training

    def test_generate_with_probs_restores_training_mode(self) -> None:
        model = _model()
        model.train()
        generate_with_probs(model, _ByteTokenizer(), "hi", n_tokens=2)
        assert model.training


# ===========================================================================
# Decoding controls: repetition penalty and stop tokens
# ===========================================================================


class _FixedLogitModel(nn.Module):
    """Always predicts the same next-token logits, so decoding is exact to reason about."""

    def __init__(self, logits: list[float], context_length: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=context_length)
        self.register_buffer("row", torch.tensor(logits))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch, seq = ids.shape
        return self.row.expand(batch, seq, -1)


class _DigitTokenizer:
    """Single-digit token IDs: "5" <-> [5]."""

    def encode(self, text: str) -> list[int]:
        return [int(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(str(i) for i in ids)


# Token 0 narrowly beats token 1; everything else is far behind (vocab of 8).
_LOGITS = [1.0, 0.9, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]


class TestRepetitionPenaltyMath:
    def test_divides_positive_and_multiplies_negative(self) -> None:
        logits = torch.tensor([2.0, -2.0, 1.0])
        out = _apply_repetition_penalty(logits, torch.tensor([0, 1]), penalty=2.0)
        assert torch.equal(out, torch.tensor([1.0, -4.0, 1.0]))

    def test_repeated_token_penalised_once(self) -> None:
        logits = torch.tensor([2.0, -2.0, 1.0])
        out = _apply_repetition_penalty(logits, torch.tensor([0, 0, 0]), penalty=2.0)
        assert torch.equal(out, torch.tensor([1.0, -2.0, 1.0]))

    def test_input_not_mutated(self) -> None:
        logits = torch.tensor([2.0, -2.0, 1.0])
        _apply_repetition_penalty(logits, torch.tensor([0]), penalty=2.0)
        assert torch.equal(logits, torch.tensor([2.0, -2.0, 1.0]))


class TestDecodingControls:
    def test_no_penalty_repeats_the_top_token(self) -> None:
        model = _FixedLogitModel(_LOGITS)
        out = generate(model, _DigitTokenizer(), "5", max_new_tokens=4)  # type: ignore[arg-type]
        assert out == "50000"

    def test_penalty_breaks_the_repetition(self) -> None:
        # Step 1: 0 wins (1.0 vs 0.9). Step 2: 0 was seen -> 0.5 < 0.9, so 1.
        # Steps 3-4: both seen -> 0.5 vs 0.45, so 0.
        model = _FixedLogitModel(_LOGITS)
        out = generate(
            model, _DigitTokenizer(), "5", max_new_tokens=4, repetition_penalty=2.0  # type: ignore[arg-type]
        )
        assert out == "50100"

    def test_stop_token_ends_generation(self) -> None:
        model = _FixedLogitModel(_LOGITS)
        out = generate(
            model, _DigitTokenizer(), "5", max_new_tokens=4, stop_token_ids=[0]  # type: ignore[arg-type]
        )
        assert out == "50"  # stops right after producing token 0 (kept in output)

    def test_unproduced_stop_token_has_no_effect(self) -> None:
        model = _FixedLogitModel(_LOGITS)
        out = generate(
            model, _DigitTokenizer(), "5", max_new_tokens=4, stop_token_ids=[3]  # type: ignore[arg-type]
        )
        assert out == "50000"

    def test_penalty_of_one_matches_default(self) -> None:
        model = _model()
        tok = _ByteTokenizer()
        default = generate(model, tok, "hello", max_new_tokens=6)
        explicit = generate(model, tok, "hello", max_new_tokens=6, repetition_penalty=1.0)
        assert explicit == default

    @pytest.mark.parametrize("penalty", [0.0, -1.0])
    def test_bad_penalty_raises(self, penalty: float) -> None:
        with pytest.raises(ValueError, match="repetition_penalty must be"):
            generate(_model(), _ByteTokenizer(), "hi", repetition_penalty=penalty)
