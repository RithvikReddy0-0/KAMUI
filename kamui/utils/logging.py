"""Structured logging with training step context.

Responsibilities:
    - ``get_logger(name)``:
        Return a configured Python logger that formats messages as
        ``[2025-01-15 14:32:01] [kamui.training] message``.

    - ``TrainingLogger``:
        Stateful logger that tracks the current step and emits
        machine-parseable log lines::

            step=500 train_loss=2.3410 lr=3.00e-04 grad_norm=1.834

        These lines can be parsed back into dicts with ``parse_log_line`` to
        regenerate training curves from a log file.

    - ``log_model_stats(model, logger)``:
        Log a model summary: parameter count and estimated memory footprint.

Implemented in: Phase 1 (basic logger), Phase 5 (TrainingLogger).
"""

from __future__ import annotations

import logging
import sys

from torch import nn

_FORMAT = "[%(asctime)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with KAMUI's standard format attached once.

    Args:
        name:  Logger name (conventionally ``"kamui.<subpackage>"``).
        level: Logging level (default ``INFO``).

    Returns:
        A configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    return logger


class TrainingLogger:
    """Emit machine-parseable ``key=value`` training log lines.

    Attributes:
        logger: The underlying Python logger.
        step:   The most recently logged step.
    """

    def __init__(self, name: str = "kamui.training") -> None:
        """Create a training logger.

        Args:
            name: Underlying logger name.
        """
        self.logger = get_logger(name)
        self.step = 0

    def log_step(self, step: int, **metrics: float) -> str:
        """Log one training step as a ``key=value`` line.

        Args:
            step:    The global step number.
            metrics: Scalar metrics (e.g. ``train_loss=2.3, lr=3e-4``).

        Returns:
            The formatted line (also emitted at INFO level).
        """
        self.step = step
        parts = [f"step={step}"]
        for key, value in metrics.items():
            if key == "lr":
                parts.append(f"lr={value:.2e}")
            else:
                parts.append(f"{key}={value:.4f}")
        line = " ".join(parts)
        self.logger.info(line)
        return line


def parse_log_line(line: str) -> dict[str, float]:
    """Parse a ``key=value`` line produced by ``TrainingLogger`` back to a dict.

    Args:
        line: A line such as ``"step=500 train_loss=2.3410 lr=3.00e-04"``.

    Returns:
        A dict mapping each key to its float value.

    Raises:
        ValueError: If any token is not a ``key=value`` pair of floats.
    """
    result: dict[str, float] = {}
    for token in line.split():
        if "=" not in token:
            raise ValueError(f"malformed log token: '{token}'")
        key, value = token.split("=", 1)
        result[key] = float(value)
    return result


def log_model_stats(model: nn.Module, logger: logging.Logger) -> None:
    """Log a model's parameter count and rough fp32 memory footprint.

    Args:
        model:  Any ``nn.Module``.
        logger: The logger to emit to.
    """
    n_params = sum(p.numel() for p in model.parameters())
    mem_mb = n_params * 4 / (1024**2)  # fp32 bytes → MiB
    logger.info("model parameters: %s (~%.1f MiB fp32)", f"{n_params:,}", mem_mb)
