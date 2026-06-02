"""Entry point for training a KAMUI transformer.

Usage:
    python scripts/train.py --config configs/small.yaml
    python scripts/train.py --config configs/small.yaml --resume checkpoints/small/step_0002000.pt
    kamui-train --config configs/small.yaml

This script is intentionally minimal — it is a thin wrapper around
``kamui.training.Trainer``.  All training logic lives in the library;
none lives here.  This makes the training loop testable without invoking
a subprocess.

Implemented in: Phase 2
"""

# Implementation begins in Phase 2.
# This file will contain:
#
#   def parse_args() -> argparse.Namespace:
#       parser = argparse.ArgumentParser(description="Train a KAMUI transformer")
#       parser.add_argument("--config", type=str, required=True)
#       parser.add_argument("--resume", type=str, default=None)
#       parser.add_argument("--seed", type=int, default=None)
#       return parser.parse_args()
#
#
#   def main() -> None:
#       args = parse_args()
#       config = ModelConfig.from_yaml(args.config)
#       set_seed(config.training.seed if args.seed is None else args.seed)
#       model = KAMUITransformer(config)
#       trainer = Trainer(model, config)
#       if args.resume:
#           trainer.load_checkpoint(args.resume)
#       trainer.train()
#
#
#   if __name__ == "__main__":
#       main()
