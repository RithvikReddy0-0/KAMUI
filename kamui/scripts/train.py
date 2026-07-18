"""Entry point for training a KAMUI transformer (``kamui-train``).

Usage:
    kamui-train --config configs/nano.yaml --corpus data/corpus.txt
    python scripts/train.py --config configs/nano.yaml --corpus data/corpus.txt

This script is intentionally minimal — it is a thin wrapper around
``kamui.training.Trainer``.  All training logic lives in the library; none
lives here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.tokenizer.bpe import BPETokenizer
from kamui.training.checkpointing import save_checkpoint
from kamui.training.data import DataLoader, TextDataset, train_val_split
from kamui.training.trainer import Trainer, TrainingConfig
from kamui.utils.logging import get_logger
from kamui.utils.reproducibility import set_seed

logger = get_logger("kamui.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a KAMUI transformer")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--corpus", type=str, required=True, help="UTF-8 text corpus path")
    parser.add_argument("--steps", type=int, default=None, help="override training steps")
    parser.add_argument("--out", type=str, default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args(argv)


def _training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Translate the YAML ``training`` section into a ``TrainingConfig``."""
    training = raw.get("training", {})
    return TrainingConfig(
        max_lr=float(training.get("learning_rate", 3e-4)),
        min_lr=float(training.get("min_lr", 3e-5)),
        warmup_steps=int(training.get("warmup_steps", 100)),
        max_steps=int(training.get("max_steps", 5000)),
        grad_accum_steps=int(training.get("grad_accum_steps", 1)),
        max_grad_norm=float(training.get("grad_clip", 1.0)),
        weight_decay=float(training.get("weight_decay", 0.1)),
        eval_interval=int(training.get("eval_interval", 0)),
        log=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Train a model from a YAML config and a text corpus."""
    args = parse_args(argv)
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_config = ModelConfig.from_yaml(args.config)
    training_config = _training_config(raw)
    if args.steps is not None:
        training_config.max_steps = max(training_config.max_steps, args.steps)

    out_dir = Path(args.out or raw.get("logging", {}).get("output_dir", "checkpoints/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("training tokenizer (vocab_size=%d)", model_config.vocab_size)
    tokenizer = BPETokenizer.train(Path(args.corpus), vocab_size=model_config.vocab_size)
    tokenizer.save(out_dir / "tokenizer.json")

    tokens = tokenizer.encode(Path(args.corpus).read_text(encoding="utf-8"))
    train_fraction = float(raw.get("data", {}).get("train_split", 0.95))
    train_tokens, val_tokens = train_val_split(tokens, val_fraction=1.0 - train_fraction)

    batch_size = int(raw.get("training", {}).get("batch_size", 16))
    train_loader = DataLoader(
        TextDataset(train_tokens, model_config.context_length),
        batch_size=batch_size,
        seed=args.seed,
    )
    val_loader = DataLoader(
        TextDataset(val_tokens, model_config.context_length),
        batch_size=batch_size,
        shuffle=False,
    )

    model = KAMUITransformer(model_config)
    logger.info("model parameters: %s", f"{model.num_parameters():,}")

    trainer = Trainer(model, train_loader, val_loader=val_loader, config=training_config)
    n_steps = args.steps if args.steps is not None else training_config.max_steps
    trainer.train(n_steps)

    val_loss = trainer.evaluate()
    ckpt_path = out_dir / f"step_{trainer.step:07d}.pt"
    save_checkpoint(
        ckpt_path,
        model,
        trainer.optimizer,
        trainer.scheduler,
        step=trainer.step,
        config=model_config,
        train_loss=trainer.history[-1]["train_loss"],
        val_loss=val_loss,
    )
    logger.info("saved checkpoint to %s (val_loss=%.4f)", ckpt_path, val_loss)


if __name__ == "__main__":
    main()
