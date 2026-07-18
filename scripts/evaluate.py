"""Evaluate a trained KAMUI checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/run/step_0000200.pt \
        --config configs/nano.yaml
    kamui-eval --checkpoint ... --config ... --tokenizer ... --generate "Once upon a time"

All logic lives in ``kamui.scripts.evaluate`` so it is importable and testable.
"""

from kamui.scripts.evaluate import main

if __name__ == "__main__":
    main()
