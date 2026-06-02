"""Evaluate a trained KAMUI checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/small/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/small/best.pt --generate "Once upon a time"
    kamui-eval --checkpoint checkpoints/small/best.pt

Implemented in: Phase 2
"""

# Implementation begins in Phase 2.
# This script will:
#   1. Load a checkpoint
#   2. Compute validation perplexity
#   3. Optionally generate text from a prompt
#   4. Print a summary of model stats
