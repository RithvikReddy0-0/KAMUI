"""Mechanistic interpretability subpackage for KAMUI.

This is the scientific core of the project.  Every tool here asks a
causal question about a trained transformer: which components are
responsible for which behaviours?

v0.1 scope (six tools):
    1. AttentionVisualizer  — extract and plot attention weight matrices
    2. LogitLens            — project residual stream to vocabulary at each layer
    3. LinearProbe          — train linear classifiers on cached activations
    4. ActivationPatcher    — causal interventions via activation replacement
    5. InductionHeadDetector— score all (layer, head) pairs for induction behaviour
    6. CircuitAblator       — zero-ablate components and measure output change

v0.2 scope (not in this module):
    SparseAutoencoder       — feature decomposition via sparse dictionary learning
                              See research/future/sae_design.md

Design principle:
    Every tool is a standalone class that depends only on:
    - a trained ``KAMUITransformer``
    - a ``BPETokenizer``
    - ``kamui.hooks.HookManager`` for activation capture

    Tools do NOT depend on each other.  They are composable but not coupled.

    Every tool exposes a ``.run()`` method that returns a typed result
    object, and a ``.plot()`` method on the result object for visualisation.
    This two-step design separates computation from display, making it easy
    to run tools in headless environments and visualise separately.

Dependency:
    kamui.mechinterp depends on kamui.hooks and kamui.model.
    It does NOT depend on kamui.training (training is complete before
    interpretability analysis begins).

Implemented in: Phase 4, Weeks 15–22
"""

# from kamui.mechinterp.attention_viz import AttentionVisualizer
from kamui.mechinterp.activation_patch import (
    ActivationPatcher,
    HeadPatchingResult,
    PatchingResult,
)
from kamui.mechinterp.logit_lens import LogitLens, LogitLensResult

# from kamui.mechinterp.probing import LinearProbe
# from kamui.mechinterp.induction import InductionHeadDetector
# from kamui.mechinterp.circuits import CircuitAblator

__all__: list[str] = [
    "LogitLens",
    "LogitLensResult",
    "ActivationPatcher",
    "PatchingResult",
    "HeadPatchingResult",
]
