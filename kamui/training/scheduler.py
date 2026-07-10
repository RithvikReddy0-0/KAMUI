"""Learning rate scheduler: linear warmup + cosine decay, from scratch.

Responsibilities:
    - ``CosineWithWarmup``:
        Implements the standard transformer LR schedule:

        Phase 1 — Linear warmup (steps 0 → warmup_steps):
            lr = max_lr * (step / warmup_steps)

        Phase 2 — Cosine decay (steps warmup_steps → max_steps):
            lr = min_lr + 0.5 * (max_lr - min_lr) *
                 (1 + cos(π * (step - warmup_steps) / (max_steps - warmup_steps)))

        Phase 3 — Floor (steps >= max_steps):
            lr = min_lr

Why warmup?
    At the start of training, gradients are large and noisy.  A cold-start
    at max_lr causes the loss to spike or diverge.  Warmup ramps the LR
    slowly, allowing the model to reach a reasonable parameter regime before
    the full learning rate is applied.

Why cosine decay?
    Cosine decay is smooth (no sharp LR drops that cause loss spikes) and
    has a well-defined endpoint (min_lr).  It outperforms step decay and
    linear decay empirically on language modelling.

Implementation note:
    This is NOT a PyTorch LRScheduler subclass.  It is a standalone class
    that returns a float ``lr`` given a ``step`` integer, which makes it
    trivially testable (no optimiser needed).  The Trainer applies the
    returned LR to the optimiser.

Implemented in: Phase 5.
"""

from __future__ import annotations

import math


class CosineWithWarmup:
    """Linear-warmup + cosine-decay learning-rate schedule.

    Attributes:
        max_lr:       Peak learning rate (reached at the end of warmup).
        min_lr:       Floor learning rate (reached at and after ``max_steps``).
        warmup_steps: Number of linear-warmup steps.
        max_steps:    Step at which cosine decay reaches ``min_lr``.
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        """Create the schedule.

        Args:
            max_lr:       Peak learning rate (> 0).
            min_lr:       Floor learning rate (0 <= min_lr <= max_lr).
            warmup_steps: Linear-warmup step count (>= 0).
            max_steps:    Total steps; must be > ``warmup_steps``.

        Raises:
            ValueError: If any argument is out of range.
        """
        if max_lr <= 0:
            raise ValueError(f"max_lr must be > 0, got {max_lr}")
        if not (0.0 <= min_lr <= max_lr):
            raise ValueError(f"min_lr must be in [0, max_lr], got {min_lr}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if max_steps <= warmup_steps:
            raise ValueError(
                f"max_steps ({max_steps}) must be > warmup_steps ({warmup_steps})"
            )

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """Return the learning rate for ``step``.

        Args:
            step: The (non-negative) global optimisation step.

        Returns:
            The learning rate for this step.
        """
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def state_dict(self) -> dict[str, float | int]:
        """Return the schedule's hyperparameters (for checkpointing)."""
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        """Restore the schedule's hyperparameters from ``state``."""
        self.max_lr = float(state["max_lr"])
        self.min_lr = float(state["min_lr"])
        self.warmup_steps = int(state["warmup_steps"])
        self.max_steps = int(state["max_steps"])

    def __repr__(self) -> str:
        return (
            f"CosineWithWarmup(max_lr={self.max_lr}, min_lr={self.min_lr}, "
            f"warmup_steps={self.warmup_steps}, max_steps={self.max_steps})"
        )
