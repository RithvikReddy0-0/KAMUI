"""Shared utility subpackage for KAMUI.

Responsibilities:
    - Structured logging with step/epoch context
    - Reproducibility: seed setting, determinism configuration
    - Matplotlib and plotly helpers for interpretability visualisations

Design constraint:
    This package must not import from any other kamui subpackage.
    It is the base of the dependency graph.  All other subpackages
    may import from utils; utils imports from nothing within kamui.

Implemented in: Phase 0 (stubs), progressively during Phases 1–4
"""

from kamui.utils.logging import TrainingLogger, get_logger, log_model_stats, parse_log_line
from kamui.utils.plotting import heatmap, layer_plot, save_figure, token_heatmap
from kamui.utils.reproducibility import get_device, set_deterministic, set_seed

__all__: list[str] = [
    "get_logger",
    "TrainingLogger",
    "log_model_stats",
    "parse_log_line",
    "heatmap",
    "layer_plot",
    "token_heatmap",
    "save_figure",
    "set_seed",
    "set_deterministic",
    "get_device",
]
