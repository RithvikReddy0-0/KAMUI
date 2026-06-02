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

__all__: list[str] = []
