"""Inspect a KAMUI checkpoint: print architecture, parameter count, training history.

Usage:
    python scripts/inspect_checkpoint.py --checkpoint checkpoints/small/best.pt

Output includes:
    - Model config (all hyperparameters)
    - Parameter count (total, embedding, attention, FFN)
    - Training step, train loss, val loss at checkpoint
    - List of all named modules and their shapes

Implemented in: Phase 2
"""

# Implementation begins in Phase 2.
