"""Explicit training loop for KAMUITransformer.

This module contains the core training logic, written as readable Python
rather than a framework abstraction.  Every step of training is visible in
``Trainer.train()``: forward, scaled backward, gradient accumulation, clip,
LR schedule, optimiser step.

Responsibilities:
    - ``TrainingConfig``:
        Typed dataclass of all training hyperparameters.

    - ``Trainer``:
        Owns the model, optimiser (AdamW with decay separation), and LR
        schedule (cosine + warmup).  ``train(n_steps)`` runs ``n_steps``
        optimiser updates, each accumulating ``grad_accum_steps`` micro-batches;
        ``evaluate()`` computes the mean validation loss with no gradients.

Key decisions:
    - Gradient accumulation is explicit: gradients accumulate over
      ``grad_accum_steps`` micro-batches before each weight update, simulating
      a larger batch without the memory cost.
    - The pre-clip gradient norm is recorded each update; a persistently large
      norm signals an unstable LR.
    - The LR for each update comes from the scheduler, applied to every
      optimiser parameter group before ``optimizer.step()``.

Implemented in: Phase 5.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kamui.training.optimizer import build_optimizer
from kamui.training.scheduler import CosineWithWarmup

#: A batch is an ``(inputs, targets)`` pair of ``(B, S)`` tensors.
Batch = tuple[Tensor, Tensor]


@dataclass
class TrainingConfig:
    """All training hyperparameters.

    Attributes:
        max_lr / min_lr:  Peak / floor learning rates for the schedule.
        warmup_steps:     Linear-warmup steps.
        max_steps:        Cosine-decay horizon (LR floors after this).
        grad_accum_steps: Micro-batches accumulated per optimiser update.
        max_grad_norm:    Gradient-clipping norm (<= 0 disables clipping).
        weight_decay:     AdamW weight decay for weight matrices.
        betas / eps:      AdamW hyperparameters.
        eval_interval:    Steps between validation evaluations (0 disables).
        log:              Whether to print structured log lines.
    """

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    eval_interval: int = 0
    log: bool = False

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval must be >= 0, got {self.eval_interval}")


def _infinite(loader: Iterable[Batch]) -> Iterator[Batch]:
    """Yield batches from ``loader`` forever, restarting at each epoch end."""
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise ValueError("training dataloader is empty")


class Trainer:
    """An explicit training loop over a ``KAMUITransformer``.

    Attributes:
        model:      The model being trained.
        optimizer:  AdamW with decayed / non-decayed parameter groups.
        scheduler:  The cosine-with-warmup LR schedule.
        step:       The global optimiser-step counter.
        history:    A list of per-step log dicts (step, train_loss, lr, grad_norm).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        """Create a trainer.

        Args:
            model:        The model to train.
            train_loader: An iterable of ``(inputs, targets)`` batches.
            val_loader:   Optional iterable for validation.
            config:       Training hyperparameters (defaults used if None).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        self.optimizer = build_optimizer(
            model,
            lr=self.config.max_lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        self.scheduler = CosineWithWarmup(
            max_lr=self.config.max_lr,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
        )
        self.step = 0
        self.history: list[dict[str, float]] = []
        self._data_iter = _infinite(train_loader)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self, n_steps: int) -> list[dict[str, float]]:
        """Run ``n_steps`` optimiser updates.

        Each update accumulates ``grad_accum_steps`` micro-batches, clips the
        gradient, applies the scheduled LR, and steps the optimiser.

        Args:
            n_steps: Number of optimiser updates to run.

        Returns:
            The list of per-step log dicts recorded during this call.
        """
        self.model.train()
        records: list[dict[str, float]] = []

        for _ in range(n_steps):
            self.optimizer.zero_grad()
            accum_loss = 0.0
            for _ in range(self.config.grad_accum_steps):
                inputs, targets = next(self._data_iter)
                loss = self.model(inputs, targets=targets)
                (loss / self.config.grad_accum_steps).backward()
                accum_loss += loss.item() / self.config.grad_accum_steps

            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                ).item()
            else:
                grad_norm = _grad_norm(self.model)

            lr = self.scheduler.get_lr(self.step)
            self._set_lr(lr)
            self.optimizer.step()
            self.step += 1

            record: dict[str, float] = {
                "step": float(self.step),
                "train_loss": accum_loss,
                "lr": lr,
                "grad_norm": grad_norm,
            }
            if (
                self.config.eval_interval
                and self.val_loader is not None
                and self.step % self.config.eval_interval == 0
            ):
                record["val_loss"] = self.evaluate()

            if self.config.log:
                print(
                    f"step={self.step} train_loss={accum_loss:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm:.3f}"
                )
            records.append(record)
            self.history.append(record)

        return records

    @torch.no_grad()
    def evaluate(self) -> float:
        """Return the mean validation loss over ``val_loader``.

        Returns:
            The token-weighted mean cross-entropy loss.

        Raises:
            ValueError: If no validation loader was provided, or it is empty.
        """
        if self.val_loader is None:
            raise ValueError("no val_loader was provided")

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in self.val_loader:
            loss = self.model(inputs, targets=targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
        if was_training:
            self.model.train()

        if total_tokens == 0:
            raise ValueError("val_loader produced no tokens")
        return total_loss / total_tokens


def _grad_norm(model: nn.Module) -> float:
    """Return the total L2 norm of all parameter gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total**0.5
