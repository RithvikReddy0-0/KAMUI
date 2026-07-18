"""Attention weight extraction and visualisation.

The simplest interpretability tool and the natural first step: extract what
each attention head is attending to and display it.

Responsibilities:
    - ``AttentionVisualizer.run(token_ids) -> AttentionResult``:
        One forward pass with hooks on every ``blocks.{i}.attn.weights``
        point; returns an ``AttentionResult`` with the full
        ``(n_layers, n_heads, S, S)`` weight tensor and decoded token labels.

    - ``AttentionResult.plot(layer, head)``:      single-head heatmap.
    - ``AttentionResult.plot_all()``:             grid of heatmaps (layers × heads).
    - ``AttentionResult.plot_interactive()``:     plotly heatmap with a
      layer/head dropdown selector.

    - ``head_summary_stats(result) -> dict[str, list]``:
        Per-head statistics for identifying head types (returned as a plain
        column dict — pandas is not a KAMUI dependency):
        - ``entropy``:   mean entropy of the attention distribution
          (low = sharp / interpretable)
        - ``self_frac``: fraction of attention on the diagonal
        - ``prev_frac``: fraction of attention on position t-1
          (high = previous-token head)

What to look for:
    - Diagonal heads: attend to the current token.
    - Previous-token heads: attend strongly to position t-1.
    - Induction heads: attend to the previous occurrence of the current
      token (see ``induction.py`` for formal detection).
    - Diffuse heads: near-uniform attention, minimal information content.

Implemented in: Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from kamui.hooks.manager import HookManager
from kamui.model.transformer import KAMUITransformer

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from plotly.graph_objects import Figure as PlotlyFigure

#: Small constant to keep log() finite when computing entropies.
_ENTROPY_EPS: float = 1e-12


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token ID to a label, tolerating undecodable bytes."""
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        return f"<{token_id}>"
    return text if text else f"<{token_id}>"


@dataclass
class AttentionResult:
    """The output of ``AttentionVisualizer.run``.

    Attributes:
        weights: ``(n_layers, n_heads, S, S)`` attention probabilities.
        tokens:  Decoded input token labels, length ``S``.
    """

    weights: Tensor
    tokens: list[str]

    @property
    def n_layers(self) -> int:
        """Number of layers captured."""
        return self.weights.shape[0]

    @property
    def n_heads(self) -> int:
        """Number of heads per layer."""
        return self.weights.shape[1]

    def _check(self, layer: int, head: int) -> None:
        if not (0 <= layer < self.n_layers):
            raise IndexError(f"layer {layer} out of range [0, {self.n_layers})")
        if not (0 <= head < self.n_heads):
            raise IndexError(f"head {head} out of range [0, {self.n_heads})")

    def plot(self, layer: int, head: int, figsize: tuple[float, float] | None = None) -> Figure:
        """Heatmap of one head's attention pattern, axes labelled by token.

        Args:
            layer:   Layer index.
            head:    Head index.
            figsize: Optional figure size.

        Returns:
            The matplotlib ``Figure``.

        Raises:
            IndexError: If ``layer`` or ``head`` is out of range.
        """
        import matplotlib.pyplot as plt

        self._check(layer, head)
        data = self.weights[layer, head].cpu().numpy()
        seq = len(self.tokens)
        fig, ax = plt.subplots(figsize=figsize or (max(5, seq * 0.6), max(4, seq * 0.6)))
        im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(seq))
        ax.set_xticklabels(self.tokens, rotation=45, ha="right")
        ax.set_yticks(range(seq))
        ax.set_yticklabels(self.tokens)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        ax.set_title(f"layer {layer}, head {head}")
        fig.colorbar(im, ax=ax, label="attention weight")
        fig.tight_layout()
        return fig

    def plot_all(self, figsize: tuple[float, float] | None = None) -> Figure:
        """Grid of heatmaps — rows are layers, columns are heads.

        Returns:
            The matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        rows, cols = self.n_layers, self.n_heads
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize or (cols * 2.2, rows * 2.2), squeeze=False
        )
        for layer in range(rows):
            for head in range(cols):
                ax = axes[layer][head]
                ax.imshow(self.weights[layer, head].cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"L{layer}H{head}", fontsize=8)
        fig.suptitle("Attention patterns (rows = layers, cols = heads)")
        fig.tight_layout()
        return fig

    def plot_interactive(self) -> PlotlyFigure:
        """Interactive plotly heatmap with a (layer, head) dropdown selector.

        Returns:
            A ``plotly.graph_objects.Figure``.
        """
        import plotly.graph_objects as go

        fig = go.Figure()
        buttons = []
        n_traces = self.n_layers * self.n_heads
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                idx = layer * self.n_heads + head
                fig.add_trace(
                    go.Heatmap(
                        z=self.weights[layer, head].cpu().numpy(),
                        x=self.tokens,
                        y=self.tokens,
                        colorscale="Blues",
                        zmin=0.0,
                        zmax=1.0,
                        visible=(idx == 0),
                    )
                )
                visibility = [i == idx for i in range(n_traces)]
                buttons.append(
                    {
                        "label": f"L{layer}H{head}",
                        "method": "update",
                        "args": [{"visible": visibility}],
                    }
                )
        fig.update_layout(
            updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.15, "y": 1.0}],
            title="Attention pattern",
            xaxis_title="key position",
            yaxis_title="query position",
        )
        fig.update_yaxes(autorange="reversed")
        return fig


class AttentionVisualizer:
    """Extract every head's attention pattern for a given input.

    Attributes:
        model:     A trained ``KAMUITransformer``.
        tokenizer: A tokenizer with ``decode`` (for axis labels).
    """

    def __init__(self, model: KAMUITransformer, tokenizer: Any) -> None:
        """Create a visualizer over ``model``.

        Args:
            model:     A ``KAMUITransformer``.
            tokenizer: A tokenizer with ``decode(list[int]) -> str``.
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def run(self, token_ids: Tensor) -> AttentionResult:
        """Capture all attention weights for a single sequence.

        Args:
            token_ids: Token IDs of shape ``(S,)`` or ``(1, S)``.

        Returns:
            An ``AttentionResult``.

        Raises:
            ValueError: If ``token_ids`` is not a single sequence.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2 or token_ids.shape[0] != 1:
            raise ValueError(
                f"token_ids must be a single sequence (S,) or (1, S), "
                f"got shape {tuple(token_ids.shape)}"
            )

        self.model.eval()
        n_layers = self.model.config.n_layers
        with HookManager(self.model) as hooks:
            for i in range(n_layers):
                hooks.attach(f"blocks.{i}.attn", "weights")
            self.model(token_ids)
            weights = torch.stack(
                [hooks.get(f"blocks.{i}.attn.weights")[0] for i in range(n_layers)]
            )  # (n_layers, n_heads, S, S)

        tokens = [_decode_token(self.tokenizer, int(t)) for t in token_ids[0].tolist()]
        return AttentionResult(weights=weights, tokens=tokens)


def head_summary_stats(result: AttentionResult) -> dict[str, list[float | int]]:
    """Compute per-head statistics that help identify head types.

    Args:
        result: An ``AttentionResult``.

    Returns:
        A column dict with one row per (layer, head):
            - ``layer`` / ``head``: indices
            - ``entropy``:   mean attention entropy (low = sharp)
            - ``self_frac``: mean attention on the diagonal (self-attention)
            - ``prev_frac``: mean attention on position t-1 (previous-token)
    """
    weights = result.weights  # (L, H, S, S)
    n_layers, n_heads, seq, _ = weights.shape

    stats: dict[str, list[float | int]] = {
        "layer": [],
        "head": [],
        "entropy": [],
        "self_frac": [],
        "prev_frac": [],
    }
    diag = torch.arange(seq)
    for layer in range(n_layers):
        for head in range(n_heads):
            w = weights[layer, head]  # (S, S)
            entropy = -(w * (w + _ENTROPY_EPS).log()).sum(dim=-1).mean().item()
            self_frac = w[diag, diag].mean().item()
            # Attention on t-1, defined for query positions 1..S-1.
            prev_frac = w[diag[1:], diag[:-1]].mean().item() if seq > 1 else 0.0
            stats["layer"].append(layer)
            stats["head"].append(head)
            stats["entropy"].append(entropy)
            stats["self_frac"].append(self_frac)
            stats["prev_frac"].append(prev_frac)
    return stats
