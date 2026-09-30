"""Exponential moving average (EMA) of model weights.

During training the raw weights jitter from step to step.  An EMA keeps a
second, smoothed copy (the *shadow* weights)::

    shadow <- decay * shadow + (1 - decay) * weights

updated after every optimiser step.  Evaluating or sampling with the shadow
weights is usually a little better and noticeably more stable than using the
raw weights, at the cost of one extra copy of the parameters.

Usage (explicit, like the rest of the training loop)::

    ema = EMA(model, decay=0.999)
    for _ in range(n_steps):
        trainer.train(1)
        ema.update(model)

    with ema.average_parameters(model):   # model temporarily holds EMA weights
        val_loss = trainer.evaluate()
    # raw training weights are back here

Responsibilities:
    - ``EMA.update``              — fold the current weights into the shadow.
    - ``EMA.copy_to``             — overwrite a model's weights with the shadow.
    - ``EMA.store`` / ``restore`` — save and bring back the raw weights.
    - ``EMA.average_parameters``  — context manager: swap in, then always swap out.
    - ``EMA.state_dict`` / ``load_state_dict`` — checkpoint the shadow.

Notes:
    Only trainable parameters are tracked, keyed by ``named_parameters()`` name
    (weight-tied parameters appear once).  For a DDP-wrapped model, pass
    ``unwrap_model(model)`` so the names carry no ``module.`` prefix.

Implemented in: v0.4.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


class EMA:
    """An exponential moving average of a model's trainable parameters.

    Attributes:
        decay:       The smoothing factor in ``[0, 1)``; higher is smoother.
        num_updates: How many times ``update`` has been called.
        shadow:      Name → smoothed parameter tensor.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """Start an EMA from the model's current weights.

        Args:
            model: The model whose trainable parameters are tracked.
            decay: Smoothing factor in ``[0, 1)``.

        Raises:
            ValueError: If ``decay`` is outside ``[0, 1)``.
        """
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict[str, Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold the model's current weights into the shadow.

        Args:
            model: The model being trained (same parameter names as at init).
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's tracked parameters with the shadow weights."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])

    def store(self, model: nn.Module) -> None:
        """Save the model's current tracked parameters so ``restore`` can bring them back."""
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Bring back the parameters saved by ``store``.

        Raises:
            RuntimeError: If ``store`` has not been called.
        """
        if not self._backup:
            raise RuntimeError("nothing to restore; call store() first")
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily run the model with EMA weights.

        The raw weights are restored on exit, even if the block raises.
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict[str, Any]:
        """Return the EMA state (decay, update count, and a copy of the shadow)."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: tensor.clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load a state produced by ``state_dict``.

        Raises:
            ValueError: If the saved parameter names do not match this EMA's.
        """
        shadow = state["shadow"]
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA state does not match: parameter names differ")
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow = {name: tensor.clone() for name, tensor in shadow.items()}
