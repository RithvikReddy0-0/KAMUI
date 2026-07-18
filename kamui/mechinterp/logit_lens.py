"""Logit lens: project the residual stream to vocabulary at every layer.

The logit lens (nostalgebraist, 2020) answers: at each layer, if the model
were to output now, what would it predict?

It applies the model's final LayerNorm and unembedding matrix to the residual
stream at each layer, converting the D-dimensional vector at each
(layer, position) to a probability distribution over the vocabulary.

Responsibilities:
    - ``LogitLens``:
        ``LogitLens.run(token_ids) -> LogitLensResult``:
            Reconstruct the residual stream at every layer (via hooks), apply
            ``final_ln + unembed`` to each, and return a ``LogitLensResult``
            with per-layer probability distributions and top-k tokens.

        ``LogitLensResult.plot()``:
            Heatmap of the top-1 predicted token at each (layer, position),
            coloured by the probability assigned to that token.

        ``LogitLensResult.plot_position(pos)``:
            For one token position, plot how the probability of the final
            prediction evolves with depth.

How the per-layer residual stream is reconstructed:
    KAMUI is Pre-LN, so each block adds its attention and FFN outputs to the
    stream: ``x = x + attn_out; x = x + ffn_out``.  Therefore the residual
    stream after block i equals ``embed.output`` plus the running sum of every
    block's ``attn.output`` and ``ffn.output`` — all of which are standard hook
    points.  This keeps the lens fully decoupled from model internals.

Key insight:
    Some facts resolve early (low layers), others late.  For a prompt like
    "The Eiffel Tower is in [Paris]", the correct answer emerges at a specific
    layer, identifying where the association is stored.

References:
    nostalgebraist (2020). Interpreting GPT: the logit lens.
    https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru

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


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode a single token ID to a label, tolerating undecodable bytes."""
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        return f"<{token_id}>"
    return text if text else f"<{token_id}>"


@dataclass
class LogitLensResult:
    """The output of ``LogitLens.run``.

    Attributes:
        probs:      ``(n_layers + 1, S, V)`` per-layer probability distributions
            (row 0 is the embedding, row L is the final layer).
        top_tokens: ``(n_layers + 1, S, k)`` top-k token IDs per (layer, pos).
        tokens:     Decoded input token labels, length ``S``.
        tokenizer:  The tokenizer, used to decode predicted tokens for plots.
    """

    probs: Tensor
    top_tokens: Tensor
    tokens: list[str]
    tokenizer: Any

    @property
    def n_layers(self) -> int:
        """Number of transformer layers (rows = n_layers + 1)."""
        return self.probs.shape[0] - 1

    def top1_token_ids(self) -> Tensor:
        """Return the top-1 predicted token ID at each (layer, position)."""
        return self.top_tokens[:, :, 0]

    def top1_labels(self) -> list[list[str]]:
        """Decoded top-1 predicted token at each (layer, position)."""
        ids = self.top1_token_ids()
        return [[_decode_token(self.tokenizer, int(t)) for t in row] for row in ids]

    def plot(self, figsize: tuple[float, float] | None = None) -> Figure:
        """Heatmap of top-1 predictions coloured by their probability.

        Returns:
            A matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        top1_prob = self.probs.max(dim=-1).values.cpu().numpy()  # (L+1, S)
        labels = self.top1_labels()
        rows, cols = top1_prob.shape

        fig, ax = plt.subplots(figsize=figsize or (max(6, cols), max(4, rows * 0.5)))
        im = ax.imshow(top1_prob, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("token position")
        ax.set_ylabel("layer (0 = embedding)")
        ax.set_xticks(range(cols))
        ax.set_xticklabels(self.tokens, rotation=45, ha="right")
        ax.set_yticks(range(rows))
        for r in range(rows):
            for c in range(cols):
                ax.text(
                    c,
                    r,
                    labels[r][c],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
        fig.colorbar(im, ax=ax, label="top-1 probability")
        fig.tight_layout()
        return fig

    def plot_position(self, pos: int, figsize: tuple[float, float] | None = None) -> Figure:
        """Plot how the final prediction's probability evolves across layers.

        Args:
            pos: Token position to inspect.

        Returns:
            A matplotlib ``Figure``.

        Raises:
            IndexError: If ``pos`` is out of range.
        """
        import matplotlib.pyplot as plt

        seq_len = self.probs.shape[1]
        if not (0 <= pos < seq_len):
            raise IndexError(f"pos {pos} out of range for sequence length {seq_len}")

        final_token = int(self.top1_token_ids()[-1, pos])
        curve = self.probs[:, pos, final_token].cpu().numpy()
        label = _decode_token(self.tokenizer, final_token)

        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        ax.plot(range(len(curve)), curve, marker="o")
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_ylabel(f"P('{label}')")
        ax.set_title(f"Prediction emergence at position {pos}")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        return fig


class LogitLens:
    """Project the residual stream to vocabulary at every layer.

    Attributes:
        model:     A trained ``KAMUITransformer``.
        tokenizer: A tokenizer with ``decode`` (for token labels).
    """

    def __init__(self, model: KAMUITransformer, tokenizer: Any) -> None:
        """Create a logit lens over ``model``.

        Args:
            model:     A ``KAMUITransformer``.
            tokenizer: A tokenizer with a ``decode(list[int]) -> str`` method.
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def run(self, token_ids: Tensor, top_k: int = 5) -> LogitLensResult:
        """Run the logit lens over a single sequence.

        Args:
            token_ids: Token IDs of shape ``(S,)`` or ``(1, S)``.
            top_k:     Number of top tokens to record per (layer, position).

        Returns:
            A ``LogitLensResult``.

        Raises:
            ValueError: If ``token_ids`` is not a single sequence, or ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2 or token_ids.shape[0] != 1:
            raise ValueError(
                f"token_ids must be a single sequence (S,) or (1, S), "
                f"got shape {tuple(token_ids.shape)}"
            )

        n_layers = self.model.config.n_layers
        self.model.eval()

        with HookManager(self.model) as hooks:
            hooks.attach("embed", "output")
            for i in range(n_layers):
                hooks.attach(f"blocks.{i}.attn", "output")
                hooks.attach(f"blocks.{i}.ffn", "output")
            self.model(token_ids)

            stream = hooks.get("embed.output")  # (1, S, D)
            streams = [stream]
            for i in range(n_layers):
                stream = (
                    stream
                    + hooks.get(f"blocks.{i}.attn.output")
                    + hooks.get(f"blocks.{i}.ffn.output")
                )
                streams.append(stream)

        # Project each residual stream through final_ln + unembed.
        probs_per_layer = []
        for stream in streams:
            logits = self.model.unembed(self.model.final_ln(stream))  # (1, S, V)
            probs_per_layer.append(torch.softmax(logits, dim=-1)[0])  # (S, V)
        probs = torch.stack(probs_per_layer)  # (L+1, S, V)

        vocab = probs.shape[-1]
        top_tokens = probs.topk(min(top_k, vocab), dim=-1).indices  # (L+1, S, k)
        labels = [_decode_token(self.tokenizer, int(t)) for t in token_ids[0].tolist()]

        return LogitLensResult(
            probs=probs, top_tokens=top_tokens, tokens=labels, tokenizer=self.tokenizer
        )
