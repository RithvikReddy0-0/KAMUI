"""Checkpoint saving and loading for full training state.

Responsibilities:
    - ``save_checkpoint(path, model, optimizer, scheduler, step, ...)``:
        Save the complete training state to a ``.pt`` file (step, model,
        optimiser, scheduler, config, and train/val loss), then read it back
        and verify the model keys match before returning.

    - ``load_checkpoint(path, model, optimizer=None, scheduler=None) -> int``:
        Restore state and return the saved step so training resumes exactly.

    - ``load_model_only(path, model)``:
        Load only the model weights, for inference / interpretability.

Design notes:
    Resuming exactly requires more than model weights: the optimiser momentum
    buffers and the scheduler/step must also be restored, or the loss curve
    and LR will not continue smoothly.

    Every save immediately reads the file back and verifies the model
    state_dict keys, catching truncated or corrupt writes before a run is lost.

Implemented in: Phase 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: Any = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
) -> None:
    """Save full training state to ``path`` and verify the write.

    Args:
        path:       Destination ``.pt`` file (parent dirs are created).
        model:      The model.
        optimizer:  The optimiser.
        scheduler:  An object with ``state_dict()`` (e.g. ``CosineWithWarmup``).
        step:       The current global step.
        config:     Optional config object (stored via its ``__dict__``).
        train_loss: Optional last training loss.
        val_loss:   Optional last validation loss.

    Raises:
        IOError: If the read-back verification fails.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": dict(config.__dict__) if config is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path_obj)

    # Verify: read back and confirm the model keys survived the write.
    reloaded = torch.load(path_obj, map_location="cpu", weights_only=False)
    if set(reloaded["model_state"].keys()) != set(checkpoint["model_state"].keys()):
        raise IOError(f"checkpoint verification failed for {path}: model keys differ")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
) -> int:
    """Restore training state from a checkpoint.

    Args:
        path:      The checkpoint file.
        model:     The model to load weights into.
        optimizer: Optional optimiser to restore momentum buffers into.
        scheduler: Optional scheduler to restore.

    Returns:
        The saved global step.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path_obj, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint["step"])


def load_model_only(path: Union[str, Path], model: nn.Module) -> nn.Module:
    """Load only model weights from a checkpoint (for inference / analysis).

    Args:
        path:  The checkpoint file.
        model: The model to load weights into.

    Returns:
        The same ``model`` with weights loaded.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path_obj, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model
