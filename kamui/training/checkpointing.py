"""Checkpoint saving and loading for full training state.

Responsibilities:
    - ``save_checkpoint(model, optimiser, scheduler, step, config, path)``:
        Save the complete training state to a ``.pt`` file:

            {
                "step":           int,
                "model_state":    model.state_dict(),
                "optimiser_state":optimiser.state_dict(),
                "scheduler_state":scheduler.state_dict(),
                "config":         config.__dict__,
                "train_loss":     float,
                "val_loss":       float,
            }

    - ``load_checkpoint(path, model, optimiser, scheduler)``:
        Restore all state from a checkpoint file.
        Returns the step number so training can resume exactly.

    - ``load_model_only(path, model)``:
        Load only the model weights (not optimiser state).
        Used for inference and interpretability analysis after training.

Design notes:
    What "full state" means:
        A checkpoint is only useful if training can resume exactly where
        it left off — same loss curve, same gradient statistics, same LR.
        This requires saving not just model weights but also optimiser
        momentum buffers and the current scheduler step.

    Checkpoint naming convention:
        checkpoints/step_{step:07d}.pt
        checkpoints/best.pt          (symlink to the checkpoint with lowest val loss)
        checkpoints/last.pt          (symlink to the most recent checkpoint)

    Checkpoint validation:
        Every save immediately reads the file back and verifies that
        model state_dict keys match.  This catches disk corruption and
        incomplete writes before they cause a training run to be lost.

Implemented in: Phase 2, Week 10
"""

# Implementation begins in Phase 2.
