"""Activation steering: causally push the model along a feature direction.

Interpretation tells you *what* a direction in activation space represents;
steering tests that reading **causally**.  You add a direction ``d`` (scaled by
a coefficient ``c``) into the residual stream during the forward pass::

    residual  ->  residual + c * d

and measure how the output logits move.  If ``d`` is a sparse-autoencoder
feature's decoder direction (a row of ``W_dec``), a positive ``c`` amplifies
that feature and a negative ``c`` suppresses it — the causal complement of
``interpret_features``: first read what a feature detects, then clamp it up or
down and watch the prediction change.

Technique (activation addition / feature clamping):
    1. Run the model once to get baseline logits.
    2. Register a forward hook on a residual-stream point that returns
       ``output + c * d`` (optionally only at one position).
    3. Run again; the difference in logits is the causal effect of the direction.

Responsibilities:
    - ``SteeringResult``:
        Baseline vs. steered logits, with the last-position logit delta and the
        tokens a steer most promotes / suppresses.
    - ``FeatureSteerer.steer``:
        Steer along an arbitrary direction at a residual-stream point.
    - ``FeatureSteerer.steer_with_feature``:
        Steer along an SAE feature's decoder direction (``W_dec[feature]``).
    - ``FeatureSteerer.generate_steered`` / ``generate_steered_with_feature``:
        Autoregressively generate text with the intervention active at every
        step — the tangible payoff (steer, then read the continuation).
    - ``build_steering_vector``:
        Derive a steering direction from contrasting example sequences
        (positive-minus-negative mean activations) — ActAdd, no SAE required.

References:
    Turner, A. et al. (2023). Activation Addition: Steering Language Models
    Without Optimization. https://arxiv.org/abs/2308.10248
    Templeton, A. et al. (2024). Scaling Monosemanticity: Extracting
    Interpretable Features from Claude 3 Sonnet.
    https://transformer-circuits.pub/2024/scaling-monosemanticity

Implemented in: v0.4 (builds on kamui.mechinterp.superposition).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from kamui.evaluate.generation import TokenizerLike, generate
from kamui.mechinterp.superposition import SparseAutoencoder, collect_activations
from kamui.model.transformer import KAMUITransformer

#: Floor for the steering-vector norm, to avoid divide-by-zero on normalisation.
_NORM_EPS: float = 1e-8

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class SteeringResult:
    """The effect of one steering intervention.

    Attributes:
        baseline_logits: ``(1, S, V)`` logits with no intervention.
        steered_logits:  ``(1, S, V)`` logits with the direction added.
        hook_point:      The residual-stream point that was steered.
        coefficient:     The scalar the direction was multiplied by.
    """

    baseline_logits: Tensor
    steered_logits: Tensor
    hook_point: str
    coefficient: float

    @property
    def logit_delta(self) -> Tensor:
        """The change in final-position logits, ``steered - baseline`` — shape ``(V,)``."""
        return self.steered_logits[0, -1] - self.baseline_logits[0, -1]

    def top_promoted(self, k: int = 5) -> list[tuple[int, float]]:
        """The ``k`` token IDs whose logit rose the most, as ``(token_id, delta)``."""
        delta = self.logit_delta
        k = min(k, delta.shape[0])
        values, indices = torch.topk(delta, k)
        return list(zip(indices.tolist(), values.tolist(), strict=True))

    def top_suppressed(self, k: int = 5) -> list[tuple[int, float]]:
        """The ``k`` token IDs whose logit fell the most, as ``(token_id, delta)``."""
        delta = self.logit_delta
        k = min(k, delta.shape[0])
        values, indices = torch.topk(-delta, k)
        return [(int(i), -float(v)) for i, v in zip(indices.tolist(), values.tolist(), strict=True)]

    def plot(self, k: int = 8, figsize: tuple[float, float] | None = None) -> Figure:
        """Bar chart of the tokens most promoted (up) and suppressed (down)."""
        import matplotlib.pyplot as plt

        promoted = self.top_promoted(k)
        suppressed = self.top_suppressed(k)
        entries = suppressed[::-1] + promoted  # most-negative to most-positive
        labels = [str(tid) for tid, _ in entries]
        deltas = [d for _, d in entries]
        colors = ["indianred" if d < 0 else "seagreen" for d in deltas]

        fig, ax = plt.subplots(figsize=figsize or (max(6, len(entries) * 0.5), 4))
        ax.bar(range(len(deltas)), deltas, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlabel("token id")
        ax.set_ylabel("logit delta (steered - baseline)")
        ax.set_title(f"Steering {self.hook_point} by {self.coefficient:+.2f}")
        ax.axhline(0.0, color="black", linewidth=0.6)
        fig.tight_layout()
        return fig


def _steering_hook(
    add: Tensor, position: int | None
) -> Callable[[nn.Module, tuple[Any, ...], Tensor], Tensor]:
    """A forward hook that returns ``output + add`` (broadcast over ``(B, S, D)``).

    If ``position`` is given, only that sequence position is shifted; otherwise
    the direction is added at every position.
    """

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> Tensor:
        if position is None:
            return output + add
        shifted = output.clone()
        shifted[:, position, :] = shifted[:, position, :] + add
        return shifted

    return hook


class FeatureSteerer:
    """Causally steer a model along activation-space directions.

    Attributes:
        model: A trained ``KAMUITransformer``.
        sae:   An optional ``SparseAutoencoder`` whose decoder rows supply
            feature directions for ``steer_with_feature``.
    """

    def __init__(self, model: KAMUITransformer, sae: SparseAutoencoder | None = None) -> None:
        """Create a steerer.

        Args:
            model: A ``KAMUITransformer``.
            sae:   Optional SAE providing feature directions (``W_dec``).
        """
        self.model = model
        self.sae = sae

    @staticmethod
    def _as_batch(ids: Tensor) -> Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be a single sequence (S,) or (1, S), got shape {tuple(ids.shape)}"
            )
        return ids

    @staticmethod
    def _is_steerable(hook_point: str) -> bool:
        return (
            hook_point == "embed.output"
            or hook_point.endswith(".attn.output")
            or hook_point.endswith(".ffn.output")
        )

    def _resolve(self, module_path: str) -> nn.Module:
        return dict(self.model.named_modules())[module_path]

    def _validate(self, hook_point: str, direction: Tensor) -> None:
        if not self._is_steerable(hook_point):
            raise ValueError(
                f"'{hook_point}' is not a steerable point "
                f"(use embed.output / blocks.i.attn.output / blocks.i.ffn.output)"
            )
        d_model = self.model.config.d_model
        if direction.shape != (d_model,):
            raise ValueError(
                f"direction must have shape (d_model={d_model},), got {tuple(direction.shape)}"
            )

    def _feature_direction(self, feature: int) -> Tensor:
        if self.sae is None:
            raise ValueError(
                "steering with a feature requires an SAE; pass one to FeatureSteerer(...)"
            )
        if not (0 <= feature < self.sae.n_features):
            raise ValueError(f"feature must be in [0, {self.sae.n_features}), got {feature}")
        return self.sae.W_dec[feature].detach()

    @torch.no_grad()
    def steer(
        self,
        input_ids: Tensor,
        hook_point: str,
        direction: Tensor,
        coefficient: float = 1.0,
        position: int | None = None,
    ) -> SteeringResult:
        """Add ``coefficient * direction`` to a residual-stream point and rerun.

        Args:
            input_ids:   Token IDs ``(S,)`` or ``(1, S)``.
            hook_point:  A steerable output point: ``"embed.output"``,
                ``"blocks.{i}.attn.output"``, or ``"blocks.{i}.ffn.output"``.
            direction:   A ``(d_model,)`` direction to add.
            coefficient: The scalar the direction is multiplied by (may be
                negative to suppress).
            position:    If given, steer only this sequence position; otherwise
                steer every position.

        Returns:
            A ``SteeringResult`` with baseline and steered logits.

        Raises:
            ValueError: If ``hook_point`` is not steerable, ``input_ids`` is not
                a single sequence, or ``direction`` is not shape ``(d_model,)``.
        """
        self._validate(hook_point, direction)

        ids = self._as_batch(input_ids)
        self.model.eval()
        baseline_logits = self.model(ids)

        add = coefficient * direction.to(baseline_logits.dtype)
        module = self._resolve(hook_point.rsplit(".", 1)[0])
        handle = module.register_forward_hook(_steering_hook(add, position))
        try:
            steered_logits = self.model(ids)
        finally:
            handle.remove()

        return SteeringResult(baseline_logits, steered_logits, hook_point, coefficient)

    def steer_with_feature(
        self,
        input_ids: Tensor,
        hook_point: str,
        feature: int,
        coefficient: float = 1.0,
        position: int | None = None,
    ) -> SteeringResult:
        """Steer along an SAE feature's decoder direction (``W_dec[feature]``).

        Args:
            input_ids:   Token IDs ``(S,)`` or ``(1, S)``.
            hook_point:  A steerable residual-stream point.
            feature:     The SAE feature index whose direction to add.
            coefficient: The scalar the direction is multiplied by.
            position:    If given, steer only this sequence position.

        Returns:
            A ``SteeringResult``.

        Raises:
            ValueError: If no SAE was provided or ``feature`` is out of range.
        """
        direction = self._feature_direction(feature)
        return self.steer(input_ids, hook_point, direction, coefficient, position)

    @torch.no_grad()
    def generate_steered(
        self,
        tokenizer: TokenizerLike,
        prompt: str,
        hook_point: str,
        direction: Tensor,
        coefficient: float = 1.0,
        **generate_kwargs: Any,
    ) -> str:
        """Generate text while steering along ``direction`` at every step.

        The direction is added at every position of every forward pass during
        autoregressive generation, so the whole continuation is produced under
        the intervention — the tangible payoff of steering (compare against a
        plain ``generate`` call to see the effect on the text).

        Args:
            tokenizer:       An object with ``encode`` / ``decode``.
            prompt:          The (non-empty) prompt string.
            hook_point:      A steerable residual-stream point.
            direction:       A ``(d_model,)`` direction to add.
            coefficient:     The scalar the direction is multiplied by.
            **generate_kwargs: Forwarded to ``kamui.evaluate.generate``
                (``max_new_tokens``, ``strategy``, ``temperature``, ``top_k``,
                ``top_p``, ``seed``).

        Returns:
            The full decoded string (prompt + steered continuation).

        Raises:
            ValueError: If ``hook_point`` is not steerable or ``direction`` is
                not shape ``(d_model,)``.
        """
        self._validate(hook_point, direction)
        add = coefficient * direction.detach().to(next(self.model.parameters()).dtype)
        module = self._resolve(hook_point.rsplit(".", 1)[0])
        handle = module.register_forward_hook(_steering_hook(add, position=None))
        try:
            return generate(self.model, tokenizer, prompt, **generate_kwargs)
        finally:
            handle.remove()

    def generate_steered_with_feature(
        self,
        tokenizer: TokenizerLike,
        prompt: str,
        hook_point: str,
        feature: int,
        coefficient: float = 1.0,
        **generate_kwargs: Any,
    ) -> str:
        """Generate text steered along an SAE feature's decoder direction.

        Args:
            tokenizer:       An object with ``encode`` / ``decode``.
            prompt:          The (non-empty) prompt string.
            hook_point:      A steerable residual-stream point.
            feature:         The SAE feature index whose direction to add.
            coefficient:     The scalar the direction is multiplied by.
            **generate_kwargs: Forwarded to ``kamui.evaluate.generate``.

        Returns:
            The full decoded string (prompt + steered continuation).

        Raises:
            ValueError: If no SAE was provided or ``feature`` is out of range.
        """
        direction = self._feature_direction(feature)
        return self.generate_steered(
            tokenizer, prompt, hook_point, direction, coefficient, **generate_kwargs
        )


@torch.no_grad()
def build_steering_vector(
    model: KAMUITransformer,
    hook_point: str,
    positive: Iterable[Tensor],
    negative: Iterable[Tensor],
    normalize: bool = False,
) -> Tensor:
    """Build a contrastive steering direction (ActAdd) from example sequences.

    The direction is the difference of mean activations at ``hook_point``::

        d = mean_token(activation | positive) - mean_token(activation | negative)

    where the mean is taken over every token of every sequence.  Added to the
    residual stream (via ``FeatureSteerer.steer``), ``d`` pushes the model toward
    whatever distinguishes the positive prompts from the negative ones — the
    data-driven alternative to ``steer_with_feature`` when you have contrasting
    examples rather than a trained SAE (Turner et al., 2023).

    Args:
        model:      A ``KAMUITransformer``.
        hook_point: A residual-stream hook point, e.g. ``"blocks.3.ffn.output"``.
        positive:   Sequences exhibiting the target behaviour (each ``(S,)`` or
            ``(B, S)``).
        negative:   Contrasting sequences.
        normalize:  If True, return a unit-norm direction.

    Returns:
        A ``(d_model,)`` steering direction.

    Raises:
        ValueError: If ``positive`` or ``negative`` yields no activations.
    """
    positive_mean = collect_activations(model, hook_point, positive).mean(dim=0)
    negative_mean = collect_activations(model, hook_point, negative).mean(dim=0)
    direction = positive_mean - negative_mean
    if normalize:
        direction = direction / direction.norm().clamp_min(_NORM_EPS)
    return direction
