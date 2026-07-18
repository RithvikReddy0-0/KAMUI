"""Integration test: train a small model end-to-end on a synthetic corpus.

This is the primary end-to-end test.  It verifies that the full pipeline
(tokeniser → data → model → training loop → checkpoint → generation) works
correctly, on CPU, in under a minute.
"""

from __future__ import annotations

#: A seeded word-salad corpus: repetitive enough to learn quickly, varied
#: enough that BPE cannot collapse it into a handful of giant merged tokens
#: (a perfectly periodic corpus compresses to ~10 tokens and starves the
#: dataloader).
import random as _random
from pathlib import Path

import pytest
import torch

from kamui.evaluate.generation import generate
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.scripts.train import main as train_main
from kamui.tokenizer.bpe import BPETokenizer
from kamui.training.checkpointing import load_model_only, save_checkpoint
from kamui.training.data import DataLoader, TextDataset, train_val_split
from kamui.training.trainer import Trainer, TrainingConfig

_WORDS = ["the", "cat", "dog", "sat", "ran", "on", "to", "mat", "log", "sun"]
_RNG = _random.Random(0)
_CORPUS = " ".join(_RNG.choice(_WORDS) for _ in range(4000))

Pipeline = tuple[KAMUITransformer, BPETokenizer, Trainer, list[dict[str, float]]]


def _config() -> ModelConfig:
    return ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=128,
        vocab_size=300,
        context_length=32,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    """Train the full pipeline once and share it across tests."""
    torch.manual_seed(0)
    config = _config()
    tokenizer = BPETokenizer.train(_CORPUS, vocab_size=config.vocab_size)
    tokens = tokenizer.encode(_CORPUS)
    train_tokens, val_tokens = train_val_split(tokens, val_fraction=0.1)

    train_loader = DataLoader(TextDataset(train_tokens, 32), batch_size=8, seed=0)
    val_loader = DataLoader(TextDataset(val_tokens, 32), batch_size=8, shuffle=False)
    trainer = Trainer(
        model=KAMUITransformer(config),
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainingConfig(max_lr=3e-3, warmup_steps=10, max_steps=1000),
    )
    records = trainer.train(150)
    return trainer.model, tokenizer, trainer, records


@pytest.mark.slow
def test_loss_decreases_at_least_30_percent(pipeline: Pipeline) -> None:
    """A correct pipeline must cut loss by >= 30% in 150 steps on easy data.

    Failure here usually means a broken causal mask, a gradient-flow bug, or
    a tokeniser producing wrong IDs.
    """
    _, _, _, records = pipeline
    first, last = records[0]["train_loss"], records[-1]["train_loss"]
    assert last < first * 0.7, f"loss {first:.3f} -> {last:.3f} (< 30% drop)"


@pytest.mark.slow
def test_validation_loss_is_finite_and_low(pipeline: Pipeline) -> None:
    _, _, trainer, records = pipeline
    val = trainer.evaluate()
    # Far below the ~log(vocab) of a random model, and below the initial loss.
    assert val < records[0]["train_loss"]


@pytest.mark.slow
def test_checkpoint_roundtrip_preserves_behaviour(pipeline: Pipeline, tmp_path: Path) -> None:
    """Reloading a checkpoint must reproduce the exact same logits."""
    model, _, trainer, _ = pipeline
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, trainer.optimizer, trainer.scheduler, step=trainer.step)

    fresh = KAMUITransformer(_config())
    load_model_only(path, fresh)
    ids = torch.randint(0, 300, (1, 16))
    model.eval()
    fresh.eval()
    assert torch.allclose(model(ids), fresh(ids), atol=1e-6)


@pytest.mark.slow
def test_trained_model_generates_text(pipeline: Pipeline) -> None:
    """Generation must extend the prompt and round-trip through the tokenizer."""
    model, tokenizer, _, _ = pipeline
    out = generate(model, tokenizer, "the cat ", max_new_tokens=20, strategy="greedy")
    assert out.startswith("the cat ")
    assert len(out) > len("the cat ")


@pytest.mark.slow
def test_train_script_end_to_end(tmp_path: Path) -> None:
    """The ``kamui-train`` CLI must run a config-driven training end to end."""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(_CORPUS, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    _config().to_yaml(config_path)

    out_dir = tmp_path / "run"
    train_main(
        [
            "--config",
            str(config_path),
            "--corpus",
            str(corpus_path),
            "--steps",
            "3",
            "--out",
            str(out_dir),
            "--seed",
            "0",
        ]
    )
    assert (out_dir / "tokenizer.json").exists()
    checkpoints = list(out_dir.glob("step_*.pt"))
    assert len(checkpoints) == 1
