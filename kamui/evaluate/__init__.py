"""Evaluation subpackage for KAMUI.

Responsibilities:
    - Compute perplexity on a held-out dataset
    - Generate text with configurable sampling strategies
    - Measure calibration (confidence vs. accuracy alignment)

Public API (available after Phase 4):
    compute_perplexity  — token-level perplexity on a dataset
    generate            — text generation with top-k, nucleus, greedy sampling

Implemented in: Phase 4, Weeks 11–12 (training evaluation) and
                Phase 4, Week 15 (generation, calibration)
"""

# from kamui.evaluate.perplexity import compute_perplexity
# from kamui.evaluate.generation import generate

__all__: list[str] = []
