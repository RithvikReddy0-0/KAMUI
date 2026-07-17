"""Reproducibility utilities: seeding and determinism.

Responsibilities:
    - ``set_seed(seed)``:   seed ``random``, NumPy, and torch (CPU + CUDA).
    - ``set_deterministic()``: configure PyTorch for fully deterministic ops.
    - ``get_device()``:     return the best available device (CUDA > MPS > CPU).

Why this module exists:
    Unreproducible results are a common failure mode in ML research.  A fixed
    seed and deterministic ops guarantee that two runs with the same config
    produce identical outputs — critical for debugging, regression tests, and
    paper reproducibility.

Implemented in: Phase 1.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger("kamui.utils")


def set_seed(seed: int) -> None:
    """Seed every random number generator KAMUI uses.

    Args:
        seed: The seed value.  Must be a non-negative integer.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled: bool = True) -> None:
    """Configure PyTorch for fully deterministic operation.

    Deterministic mode is slower than the default; use it for debugging and
    regression testing rather than production training.

    Args:
        enabled: Turn determinism on (True) or off (False).
    """
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    torch.use_deterministic_algorithms(enabled)


def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU).

    Logs which device was selected at INFO level.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")  # pragma: no cover — needs CUDA hardware
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")  # pragma: no cover — needs Apple Silicon
    else:
        device = torch.device("cpu")
    logger.info("selected device: %s", device)
    return device
