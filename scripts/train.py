"""Entry point for training a KAMUI transformer.

Usage:
    python scripts/train.py --config configs/nano.yaml --corpus data/corpus.txt
    kamui-train --config configs/nano.yaml --corpus data/corpus.txt

All logic lives in ``kamui.scripts.train`` so it is importable and testable.
"""

from kamui.scripts.train import main

if __name__ == "__main__":
    main()
