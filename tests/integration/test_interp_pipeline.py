"""Integration test: end-to-end interpretability pipeline.

Verifies that the full pipeline — train → hook → interpret — works without
errors and produces outputs of the correct type and shape, running all six
tools on one briefly-trained model.

These tests do not verify research findings (those belong in
research/experiments/); they verify that the code runs correctly end-to-end.
"""

from __future__ import annotations

# Seeded word-salad: varied enough that BPE cannot collapse it (see
# test_train_nano for why a perfectly periodic corpus breaks the dataloader).
import random as _random

import pytest
import torch
from torch import Tensor

from kamui.mechinterp import (
    ActivationPatcher,
    AttentionVisualizer,
    CircuitAblator,
    InductionHeadDetector,
    LinearProbe,
    LogitLens,
)
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.tokenizer.bpe import BPETokenizer
from kamui.training.data import DataLoader, TextDataset
from kamui.training.trainer import Trainer, TrainingConfig

_WORDS = ["the", "cat", "dog", "sat", "ran", "on", "to", "mat", "log", "sun"]
_RNG = _random.Random(0)
_CORPUS = " ".join(_RNG.choice(_WORDS) for _ in range(3000))


@pytest.fixture(scope="module")
def trained() -> tuple[KAMUITransformer, BPETokenizer]:
    """A briefly-trained small model + its tokenizer, shared across tests."""
    torch.manual_seed(0)
    config = ModelConfig(
        n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=280, context_length=16
    )
    tokenizer = BPETokenizer.train(_CORPUS, vocab_size=config.vocab_size)
    tokens = tokenizer.encode(_CORPUS)
    loader = DataLoader(TextDataset(tokens, 16), batch_size=8, seed=0)
    trainer = Trainer(
        KAMUITransformer(config),
        loader,
        config=TrainingConfig(max_lr=3e-3, warmup_steps=5, max_steps=500),
    )
    trainer.train(40)
    trainer.model.eval()
    return trainer.model, tokenizer


def _ids(tokenizer: BPETokenizer, text: str = "the cat sat on") -> Tensor:
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


@pytest.mark.slow
def test_logit_lens_runs(trained: tuple[KAMUITransformer, BPETokenizer]) -> None:
    """LogitLens must return per-layer distributions of the right shape."""
    model, tokenizer = trained
    ids = _ids(tokenizer)
    result = LogitLens(model, tokenizer).run(ids)
    assert result.probs.shape == (model.config.n_layers + 1, len(ids), model.config.vocab_size)
    sums = result.probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.slow
def test_attention_visualizer_runs(trained: tuple[KAMUITransformer, BPETokenizer]) -> None:
    model, tokenizer = trained
    ids = _ids(tokenizer)
    result = AttentionVisualizer(model, tokenizer).run(ids)
    assert result.weights.shape[:2] == (model.config.n_layers, model.config.n_heads)


@pytest.mark.slow
def test_activation_patcher_runs(trained: tuple[KAMUITransformer, BPETokenizer]) -> None:
    """Per-layer patching effects must be finite; embed patch fully recovers."""
    model, tokenizer = trained
    clean = _ids(tokenizer, "the cat sat on")
    corrupted = _ids(tokenizer, "the dog sat on")
    length = min(len(clean), len(corrupted))
    patcher = ActivationPatcher(model)
    result = patcher.patch_all_layers(clean[:length], corrupted[:length])
    assert torch.isfinite(result.effects).all()
    full = patcher.patch_single(clean[:length], corrupted[:length], "embed.output")
    # Patching the embedding restores the clean run exactly (recovery 1.0) —
    # unless the trained model predicts identically for both prompts, in which
    # case the metric is degenerate and the documented result is 0.0.  The
    # deterministic 1.0 anchor lives in tests/unit/test_activation_patch.py.
    assert full == pytest.approx(1.0, abs=1e-4) or full == 0.0


@pytest.mark.slow
def test_induction_detector_runs(trained: tuple[KAMUITransformer, BPETokenizer]) -> None:
    model, _ = trained
    scores = InductionHeadDetector(model).score_all_heads(prefix_len=8, n_samples=2)
    assert len(scores) == model.config.n_layers * model.config.n_heads
    assert all(0.0 <= s <= 1.0 for s in scores.values())


@pytest.mark.slow
def test_circuit_ablator_runs(trained: tuple[KAMUITransformer, BPETokenizer]) -> None:
    model, tokenizer = trained
    ids = _ids(tokenizer)
    result = CircuitAblator(model).ablate(
        ["blocks.0.attn.output"], ids, lambda logits: logits[0, -1].max().item()
    )
    assert isinstance(result.delta, float)


@pytest.mark.slow
def test_linear_probe_runs(trained: tuple[KAMUITransformer, BPETokenizer]) -> None:
    model, tokenizer = trained
    dataset = [
        _ids(tokenizer, "the cat sat"),
        _ids(tokenizer, "the dog sat"),
    ] * 10
    labels = [0, 1] * 10
    result = LinearProbe(model).train("embed.output", dataset, labels, epochs=100)
    assert 0.0 <= result.val_acc <= 1.0
