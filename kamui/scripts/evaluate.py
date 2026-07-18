"""Evaluate a trained KAMUI checkpoint (``kamui-eval``).

Usage:
    kamui-eval --checkpoint checkpoints/run/step_0000200.pt --config configs/nano.yaml
    kamui-eval --checkpoint ... --config ... --tokenizer checkpoints/run/tokenizer.json \\
        --generate "Once upon a time"
"""

from __future__ import annotations

import argparse

from kamui.evaluate.generation import generate
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.tokenizer.bpe import BPETokenizer
from kamui.training.checkpointing import load_model_only
from kamui.utils.logging import get_logger

logger = get_logger("kamui.eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a KAMUI checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint .pt path")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer.json path")
    parser.add_argument("--generate", type=str, default=None, help="prompt to continue")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="nucleus")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load a checkpoint, print model stats, and optionally generate text."""
    args = parse_args(argv)

    config = ModelConfig.from_yaml(args.config)
    model = KAMUITransformer(config)
    load_model_only(args.checkpoint, model)
    model.eval()
    logger.info("loaded %s", args.checkpoint)
    logger.info("model parameters: %s", f"{model.num_parameters():,}")

    if args.generate is not None:
        if args.tokenizer is None:
            raise SystemExit("--generate requires --tokenizer")
        tokenizer = BPETokenizer.load(args.tokenizer)
        text = generate(
            model,
            tokenizer,
            args.generate,
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
        )
        logger.info("generation: %s", text)


if __name__ == "__main__":
    main()
