"""Training subpackage for KAMUI.

Responsibilities:
    - Provide an explicit, readable training loop with zero magic
    - Implement learning rate scheduling from scratch
    - Handle checkpointing: save and restore full training state
    - Manage data loading, tokenisation, and batch construction
    - Expose every training decision as visible, modifiable code

Design philosophy:
    KAMUI's training loop is NOT a black box.  There is no ``model.fit()``,
    no callback system that hides behaviour, no framework abstraction.
    The loop is Python: a for-loop over steps, a forward pass, a backward
    pass, a gradient clip, an optimiser step.  Every line is readable.

    If you want to add gradient logging, you add it to the loop.
    If you want to change the LR schedule, you edit the scheduler.
    There is no indirection.

Module layout:
    trainer.py        — Trainer class: the explicit training loop
    scheduler.py      — LR warmup + cosine decay, implemented from scratch
    optimizer.py      — AdamW with correct weight decay separation
    data.py           — DataLoader, tokenisation, sequence packing
    checkpointing.py  — save/load model + optimiser + step state

Public API:
    Trainer           — main training loop
    TrainingConfig    — all training hyperparameters as a typed dataclass
    CosineWithWarmup  — LR schedule
    build_optimizer   — AdamW with weight-decay separation
    TextDataset, DataLoader, tokenise_corpus, train_val_split — data pipeline
    save_checkpoint, load_checkpoint, load_model_only — checkpointing

Implemented in: Phase 2, Weeks 9–12
"""

from kamui.training.checkpointing import (
    load_checkpoint,
    load_model_only,
    save_checkpoint,
)
from kamui.training.data import (
    DataLoader,
    TextDataset,
    tokenise_corpus,
    train_val_split,
)
from kamui.training.optimizer import build_optimizer
from kamui.training.scheduler import CosineWithWarmup
from kamui.training.trainer import Trainer, TrainingConfig

__all__: list[str] = [
    "Trainer",
    "TrainingConfig",
    "CosineWithWarmup",
    "build_optimizer",
    "TextDataset",
    "DataLoader",
    "tokenise_corpus",
    "train_val_split",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_only",
]
