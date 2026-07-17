"""Evaluation subpackage for KAMUI.

Responsibilities:
    - Compute perplexity on a held-out dataset
    - Generate text with configurable sampling strategies
    - Measure calibration (confidence vs. accuracy alignment)

Public API:
    compute_perplexity  — token-level perplexity on a dataset
    generate            — text generation with top-k, nucleus, greedy sampling

Implemented in: Phase 4, Weeks 11–12 (training evaluation) and
                Phase 4, Week 15 (generation, calibration)
"""

from kamui.evaluate.calibration import (
    expected_calibration_error,
    reliability_diagram,
    temperature_scaling,
)
from kamui.evaluate.generation import (
    GenerationResult,
    generate,
    generate_with_probs,
)
from kamui.evaluate.perplexity import (
    compute_perplexity,
    compute_sequence_perplexity,
    compute_token_loss,
)

__all__: list[str] = [
    "compute_perplexity",
    "compute_sequence_perplexity",
    "compute_token_loss",
    "generate",
    "generate_with_probs",
    "GenerationResult",
    "expected_calibration_error",
    "reliability_diagram",
    "temperature_scaling",
]
