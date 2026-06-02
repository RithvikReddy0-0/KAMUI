"""Shared matplotlib and plotly helpers for interpretability visualisations.

Responsibilities:
    - ``heatmap(matrix, row_labels, col_labels, title, **kwargs) -> Figure``:
        Render a 2-D tensor as a labelled heatmap.
        Used by AttentionVisualizer and CircuitAblator.

    - ``layer_plot(values_by_layer, ylabel, title) -> Figure``:
        Line or bar chart of a scalar value at each layer.
        Used by LogitLens, LinearProbe, and ActivationPatcher.

    - ``token_heatmap(token_strings, values, title) -> Figure``:
        Colour-code a sequence of tokens by a scalar value per token.
        Used to highlight which tokens drive a behaviour.

    - ``save_figure(fig, path, dpi=150)``:
        Save a matplotlib figure to path with consistent settings.
        Always saves to ``research/figures/`` if path is relative.

Style constants:
    KAMUI uses a consistent visual style across all plots:
    - Font: IBM Plex Mono (matches the documentation site)
    - Colormap: "RdBu_r" for diverging, "Blues" for sequential
    - Figure width: 10 inches (single column), 20 inches (double column)
    - DPI: 150 for screen, 300 for publication

Implemented in: Phase 4 (when interpretability tools are built)
"""

# Implementation begins in Phase 4.
