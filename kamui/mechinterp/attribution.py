"""Gradient-based input attribution: which tokens drove a prediction?

Where ``ActivationPatcher`` localises behaviour to *components* by causal
intervention, gradient attribution localises it to *input tokens* by
differentiating a scalar target through the model.

Two methods are provided, both attributing to the **embedding** of each input
position (a single backward pass each, unlike patching's many forward passes):

    - ``input_x_grad`` (Simonyan et al., 2014):
        ``attr_t = Σ_d  e_{t,d} · ∂metric/∂e_{t,d}``
        where ``e`` is the embedding output.  Fast but only locally faithful.

    - ``integrated_gradients`` (Sundararajan et al., 2017):
        Integrate the gradient along the straight path from a zero-embedding
        baseline to the real embedding.  With baseline ``0``::

            attr_t = Σ_d  e_{t,d} · (1/N) Σ_{k=1..N} ∂metric/∂(α_k e)_{t,d}

        Integrated Gradients satisfies the **completeness axiom**: the
        attributions sum to ``metric(input) − metric(baseline)``.  ``AttributionResult``
        exposes both metrics so this can be checked directly.

The target metric is the final-position logit of ``target_token`` (defaulting to
the model's own top prediction).

References:
    Sundararajan, M. et al. (2017). Axiomatic Attribution for Deep Networks.
    ICML 2017. https://arxiv.org/abs/1703.01365

Implemented in: v0.2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from kamui.model.transformer import KAMUITransformer
from kamui.utils.plotting import token_heatmap

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Supported attribution methods.
_METHODS: tuple[str, ...] = ("input_x_grad", "integrated_gradients")


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token ID to a label, tolerating a missing/failing tokenizer."""
    if tokenizer is None:
        return f"<{token_id}>"
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        return f"<{token_id}>"
    return text if text else f"<{token_id}>"


@dataclass
class AttributionResult:
    """Per-token attribution scores from ``GradientAttribution``.

    Attributes:
        scores:          ``(S,)`` attribution per input position (signed).
        tokens:          Decoded input token labels, length ``S``.
        target_token:    The vocab ID whose final-position logit was attributed.
        method:          ``"input_x_grad"`` or ``"integrated_gradients"``.
        input_metric:    ``metric(input)`` — set for integrated gradients only.
        baseline_metric: ``metric(baseline)`` — set for integrated gradients only.
    """

    scores: Tensor
    tokens: list[str]
    target_token: int
    method: str
    input_metric: float | None = None
    baseline_metric: float | None = None

    def plot(self, **kwargs: Any) -> Figure:
        """Colour-code the token sequence by attribution (red/blue diverging)."""
        return token_heatmap(
            self.tokens,
            self.scores,
            title=f"Token attribution ({self.method}) → token {self.target_token}",
            **kwargs,
        )


class GradientAttribution:
    """Attribute a target prediction to input tokens via gradients.

    Attributes:
        model:     A trained ``KAMUITransformer``.
        tokenizer: Optional tokenizer with ``decode`` (for token labels).
    """

    def __init__(self, model: KAMUITransformer, tokenizer: Any = None) -> None:
        """Create an attributor over ``model``.

        Args:
            model:     A ``KAMUITransformer``.
            tokenizer: Optional tokenizer with ``decode(list[int]) -> str``.
        """
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _as_batch(ids: Tensor) -> Tensor:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(
                f"token_ids must be a single sequence (S,) or (1, S), got shape {tuple(ids.shape)}"
            )
        return ids

    def _scaled_embed_hook(
        self, scale: float, store: dict[str, Tensor]
    ) -> Callable[[nn.Module, tuple[Any, ...], Tensor], Tensor]:
        """Forward hook that scales the embedding output and retains its grad."""

        def hook(_m: nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
            scaled = out * scale
            scaled.retain_grad()
            store["scaled"] = scaled
            store["unscaled"] = out
            return scaled

        return hook

    def _embedding_gradient(
        self, ids: Tensor, target_token: int, scale: float
    ) -> tuple[Tensor, Tensor]:
        """Return ``(embedding, d metric / d scaled_embedding)`` after one backward.

        The embedding output is scaled by ``scale`` (for the IG path); the
        returned embedding is the *unscaled* value used for the ``e · grad``
        product.
        """
        store: dict[str, Tensor] = {}
        handle = self.model.embed.register_forward_hook(self._scaled_embed_hook(scale, store))
        self.model.zero_grad(set_to_none=True)
        try:
            logits = self.model(ids)
            metric = logits[0, -1, target_token]
            metric.backward()
        finally:
            handle.remove()
        grad = store["scaled"].grad
        assert grad is not None  # populated by backward() on the retained tensor
        return store["unscaled"].detach(), grad.detach()

    @torch.no_grad()
    def _metric_at_scale(self, ids: Tensor, target_token: int, scale: float) -> float:
        """Return the target logit with the embedding scaled by ``scale`` (no grad)."""

        def hook(_m: nn.Module, _inp: tuple[Any, ...], out: Tensor) -> Tensor:
            return out * scale

        handle = self.model.embed.register_forward_hook(hook)
        try:
            return float(self.model(ids)[0, -1, target_token])
        finally:
            handle.remove()

    def token_attribution(
        self,
        token_ids: Tensor,
        target_token: int | None = None,
        method: str = "input_x_grad",
        steps: int = 32,
    ) -> AttributionResult:
        """Attribute the target-token logit to each input token.

        Args:
            token_ids:    Token IDs of shape ``(S,)`` or ``(1, S)``.
            target_token: Vocab ID to attribute (defaults to the model's top
                prediction at the final position).
            method:       ``"input_x_grad"`` or ``"integrated_gradients"``.
            steps:        Riemann steps for integrated gradients (>= 1).

        Returns:
            An ``AttributionResult`` with one signed score per input position.

        Raises:
            ValueError: If ``method`` is unknown, ``steps < 1``, or ``token_ids``
                is not a single sequence.
        """
        if method not in _METHODS:
            raise ValueError(f"unknown method '{method}'; expected one of {_METHODS}")
        ids = self._as_batch(token_ids)
        self.model.eval()

        if target_token is None:
            with torch.no_grad():
                target_token = int(self.model(ids)[0, -1].argmax())

        input_metric: float | None = None
        baseline_metric: float | None = None

        if method == "input_x_grad":
            embedding, grad = self._embedding_gradient(ids, target_token, scale=1.0)
            scores = (embedding * grad).sum(dim=-1)[0]
        else:  # integrated_gradients
            if steps < 1:
                raise ValueError(f"steps must be >= 1, got {steps}")
            embedding = torch.zeros(1)
            total_grad: Tensor | None = None
            for k in range(1, steps + 1):
                alpha = k / steps
                embedding, grad = self._embedding_gradient(ids, target_token, scale=alpha)
                total_grad = grad if total_grad is None else total_grad + grad
            avg_grad = total_grad / steps  # type: ignore[operator]
            scores = (embedding * avg_grad).sum(dim=-1)[0]
            input_metric = self._metric_at_scale(ids, target_token, 1.0)
            baseline_metric = self._metric_at_scale(ids, target_token, 0.0)

        tokens = [_decode_token(self.tokenizer, int(t)) for t in ids[0].tolist()]
        return AttributionResult(
            scores=scores.detach(),
            tokens=tokens,
            target_token=target_token,
            method=method,
            input_metric=input_metric,
            baseline_metric=baseline_metric,
        )
