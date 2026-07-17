"""Shared matplotlib helpers for interpretability visualisations.

Responsibilities:
    - ``heatmap(matrix, ...)``:      labelled 2-D heatmap (attention, ablation).
    - ``layer_plot(values, ...)``:   bar chart of a scalar per layer.
    - ``token_heatmap(tokens, values, ...)``: colour a token sequence by value.
    - ``save_figure(fig, path)``:    save with consistent DPI; relative paths
      resolve into ``research/figures/``.

Style constants:
    - Colormap: ``"RdBu_r"`` for diverging data, ``"Blues"`` for sequential.
    - DPI: 150 for screen use.

Implemented in: Phase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Default save resolution.
_DPI: int = 150

#: Default directory for relative figure paths.
_FIGURES_DIR: str = "research/figures"


def heatmap(
    matrix: Tensor,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str = "",
    cmap: str = "Blues",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Render a 2-D tensor as a labelled heatmap.

    Args:
        matrix:     A 2-D tensor.
        row_labels: Optional y-axis tick labels.
        col_labels: Optional x-axis tick labels.
        title:      Plot title.
        cmap:       Matplotlib colormap name.
        figsize:    Optional figure size.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If ``matrix`` is not 2-D.
    """
    import matplotlib.pyplot as plt

    if matrix.dim() != 2:
        raise ValueError(f"matrix must be 2-D, got shape {tuple(matrix.shape)}")

    data = matrix.detach().cpu().numpy()
    rows, cols = data.shape
    fig, ax = plt.subplots(figsize=figsize or (max(5, cols * 0.5), max(4, rows * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    if col_labels is not None:
        ax.set_xticks(range(cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if row_labels is not None:
        ax.set_yticks(range(rows))
        ax.set_yticklabels(row_labels)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def layer_plot(
    values_by_layer: list[float] | Tensor,
    ylabel: str = "value",
    title: str = "",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Bar chart of a scalar value at each layer.

    Args:
        values_by_layer: One scalar per layer.
        ylabel:          Y-axis label.
        title:           Plot title.
        figsize:         Optional figure size.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    values = values_by_layer.tolist() if isinstance(values_by_layer, Tensor) else values_by_layer
    fig, ax = plt.subplots(figsize=figsize or (max(6, len(values) * 0.6), 4))
    ax.bar(range(len(values)), values, color="steelblue")
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticks(range(len(values)))
    fig.tight_layout()
    return fig


def token_heatmap(
    token_strings: list[str],
    values: list[float] | Tensor,
    title: str = "",
    cmap: str = "RdBu_r",
) -> Figure:
    """Colour-code a token sequence by a scalar per token.

    Args:
        token_strings: The decoded tokens.
        values:        One scalar per token.
        title:         Plot title.
        cmap:          Matplotlib colormap name.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If lengths differ.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    vals = values.tolist() if isinstance(values, Tensor) else list(values)
    if len(token_strings) != len(vals):
        raise ValueError(
            f"token_strings ({len(token_strings)}) and values ({len(vals)}) differ in length"
        )

    data = np.asarray(vals, dtype=float)[None, :]  # (1, S)
    fig, ax = plt.subplots(figsize=(max(6, len(vals) * 0.7), 1.8))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(token_strings)))
    ax.set_xticklabels(token_strings, rotation=45, ha="right")
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.35)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = _DPI) -> Path:
    """Save a figure with consistent settings.

    Relative paths are resolved into ``research/figures/``; parent directories
    are created as needed.

    Args:
        fig:  The figure to save.
        path: Destination path (absolute, or relative to ``research/figures/``).
        dpi:  Save resolution.

    Returns:
        The absolute path the figure was written to.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path(_FIGURES_DIR) / path_obj
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_obj, dpi=dpi, bbox_inches="tight")
    return path_obj.resolve()
