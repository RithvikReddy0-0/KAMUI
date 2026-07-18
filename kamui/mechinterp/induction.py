"""Induction head detection and scoring.

Induction heads are the canonical example of interpretable circuits in
transformers.  They implement in-context pattern matching: given a sequence
like ``[A][B]...[A]``, an induction head at the second ``[A]`` attends to
``[B]`` and boosts the probability of ``[B]`` as the next token.

Detection method (Olsson et al., 2022):
    Feed the model a random sequence repeated twice: ``[x_0..x_{L-1}] [x_0..
    x_{L-1}]``.  For a query position ``t`` in the second half, the current
    token's previous occurrence is at ``t - L``, so an induction head attends
    to position ``t - L + 1`` — the token that followed it last time.  A
    head's induction score is its mean attention weight on that offset over
    all second-half positions.  Scores are in ``[0, 1]``; > 0.5 strongly
    indicates induction behaviour.

Responsibilities:
    - ``InductionHeadDetector.score_all_heads()``:
        Score every (layer, head) pair; returns ``{(layer, head): score}``.
    - ``InductionHeadDetector.plot_scores(scores)``:
        Heatmap of induction scores by (layer, head).
    - ``InductionHeadDetector.ablate_and_measure(heads)``:
        Zero-ablate the given heads and return the increase in second-half
        (in-context) loss — causal confirmation that those heads drive ICL.

References:
    Olsson, C. et al. (2022). In-context Learning and Induction Heads.
    https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/

Implemented in: Phase 4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kamui.hooks.manager import HookManager
from kamui.model.transformer import KAMUITransformer

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class InductionHeadDetector:
    """Score and causally test every attention head for induction behaviour.

    Attributes:
        model: A trained ``KAMUITransformer``.
    """

    def __init__(self, model: KAMUITransformer) -> None:
        """Create a detector over ``model``.

        Args:
            model: A ``KAMUITransformer``.
        """
        self.model = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _repeated_sequence(self, prefix_len: int, seed: int) -> Tensor:
        """Return a ``(1, 2 * prefix_len)`` random sequence repeated twice."""
        generator = torch.Generator().manual_seed(seed)
        vocab = self.model.config.vocab_size
        prefix = torch.randint(0, vocab, (prefix_len,), generator=generator)
        return torch.cat([prefix, prefix]).unsqueeze(0)

    def _resolve_prefix_len(self, prefix_len: int | None) -> int:
        ctx = self.model.config.context_length
        if prefix_len is None:
            prefix_len = ctx // 2
        if not (2 <= prefix_len <= ctx // 2):
            raise ValueError(
                f"prefix_len must be in [2, context_length // 2 = {ctx // 2}], got {prefix_len}"
            )
        return prefix_len

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_all_heads(
        self, prefix_len: int | None = None, n_samples: int = 4, seed: int = 0
    ) -> dict[tuple[int, int], float]:
        """Compute the induction score for every (layer, head) pair.

        Args:
            prefix_len: Length of the repeated prefix (defaults to
                ``context_length // 2``).
            n_samples:  Number of random repeated sequences to average over.
            seed:       Base RNG seed (sample ``i`` uses ``seed + i``).

        Returns:
            ``{(layer, head): score}`` with scores in ``[0, 1]``.

        Raises:
            ValueError: If ``prefix_len`` is out of range or ``n_samples < 1``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        prefix_len = self._resolve_prefix_len(prefix_len)

        self.model.eval()
        n_layers = self.model.config.n_layers
        n_heads = self.model.config.n_heads
        totals = torch.zeros(n_layers, n_heads)

        for sample in range(n_samples):
            ids = self._repeated_sequence(prefix_len, seed + sample)
            with HookManager(self.model) as hooks:
                for i in range(n_layers):
                    hooks.attach(f"blocks.{i}.attn", "weights")
                self.model(ids)
                # Mean attention on the induction offset t -> t - L + 1 for
                # every query position t in the second half.
                queries = torch.arange(prefix_len, 2 * prefix_len)
                keys = queries - prefix_len + 1
                for i in range(n_layers):
                    w = hooks.get(f"blocks.{i}.attn.weights")[0]  # (H, S, S)
                    totals[i] += w[:, queries, keys].mean(dim=-1)

        totals /= n_samples
        return {
            (layer, head): totals[layer, head].item()
            for layer in range(n_layers)
            for head in range(n_heads)
        }

    def plot_scores(
        self,
        scores: dict[tuple[int, int], float],
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Heatmap of induction scores by (layer, head).

        Args:
            scores:  The dict returned by ``score_all_heads``.
            figsize: Optional figure size.

        Returns:
            The matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt

        n_layers = self.model.config.n_layers
        n_heads = self.model.config.n_heads
        grid = torch.zeros(n_layers, n_heads)
        for (layer, head), score in scores.items():
            grid[layer, head] = score

        fig, ax = plt.subplots(figsize=figsize or (max(5, n_heads), max(4, n_layers)))
        im = ax.imshow(grid.numpy(), aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title("Induction scores")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        fig.colorbar(im, ax=ax, label="induction score")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Causal confirmation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ablate_and_measure(
        self,
        heads: list[tuple[int, int]],
        prefix_len: int | None = None,
        seed: int = 0,
    ) -> float:
        """Zero-ablate ``heads`` and measure the rise in in-context loss.

        In-context performance is the mean next-token loss over the second
        half of a repeated random sequence (predictable purely from context).
        A positive return value means ablation hurt in-context learning.

        Args:
            heads:      ``(layer, head)`` pairs to zero-ablate.
            prefix_len: Repeated-prefix length (defaults to ``context_length // 2``).
            seed:       RNG seed for the sequence.

        Returns:
            ``ablated_loss - baseline_loss`` on second-half tokens.

        Raises:
            ValueError: If ``heads`` is empty or any index is out of range.
        """
        if not heads:
            raise ValueError("heads must be a non-empty list of (layer, head) pairs")
        n_layers = self.model.config.n_layers
        n_heads = self.model.config.n_heads
        for layer, head in heads:
            if not (0 <= layer < n_layers and 0 <= head < n_heads):
                raise ValueError(f"(layer={layer}, head={head}) out of range")

        prefix_len = self._resolve_prefix_len(prefix_len)
        self.model.eval()
        ids = self._repeated_sequence(prefix_len, seed)

        baseline = self._second_half_loss(ids, prefix_len)

        # Zero the ablated heads' slices of each affected layer's out_proj input.
        d_head = self.model.config.d_head
        by_layer: dict[int, list[int]] = {}
        for layer, head in heads:
            by_layer.setdefault(layer, []).append(head)

        handles = []
        modules = dict(self.model.named_modules())
        for layer, layer_heads in by_layer.items():
            out_proj = modules[f"blocks.{layer}.attn.out_proj"]

            def _make(
                layer_heads: list[int],
            ) -> Callable[[nn.Module, tuple[Tensor, ...]], tuple[Tensor]]:
                def pre_hook(_m: nn.Module, inp: tuple[Tensor, ...]) -> tuple[Tensor]:
                    x = inp[0]
                    b, s, _ = x.shape
                    x = x.clone().view(b, s, n_heads, d_head)
                    for h in layer_heads:
                        x[:, :, h, :] = 0.0
                    return (x.view(b, s, n_heads * d_head),)

                return pre_hook

            handles.append(out_proj.register_forward_pre_hook(_make(layer_heads)))

        try:
            ablated = self._second_half_loss(ids, prefix_len)
        finally:
            for handle in handles:
                handle.remove()

        return ablated - baseline

    def _second_half_loss(self, ids: Tensor, prefix_len: int) -> float:
        """Mean next-token loss over the second half of ``ids``."""
        logits = self.model(ids)  # (1, 2L, V)
        # Predict tokens at positions prefix_len..2L-1 from their predecessors.
        preds = logits[0, prefix_len - 1 : -1]  # (L, V)
        targets = ids[0, prefix_len:]  # (L,)
        return F.cross_entropy(preds, targets).item()
