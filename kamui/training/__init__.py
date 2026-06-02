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

Public API (available after Phase 2):
    Trainer           — main training loop
    TrainingConfig    — all training hyperparameters as a typed dataclass

Implemented in: Phase 2, Weeks 9–12
"""

# from kamui.training.trainer import Trainer, TrainingConfig

__all__: list[str] = []
