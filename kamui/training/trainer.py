"""Explicit training loop for KAMUITransformer.

This module contains the core training logic.  It is intentionally written
as readable Python, not as a framework abstraction.  Every step of training
is visible in the ``Trainer.train()`` method.

Responsibilities:
    - ``TrainingConfig``:
        Typed dataclass holding all training hyperparameters:
        learning rate, batch size, gradient accumulation steps,
        gradient clip norm, warmup steps, total steps, eval interval,
        checkpoint interval, output directory.

    - ``Trainer``:
        The main training class.  Key methods:

        ``Trainer.train(n_steps)``:
            The training loop.  Pseudocode:

                for step in range(n_steps):
                    batch = next(dataloader)
                    loss = model(batch.ids, targets=batch.targets)

                    loss = loss / grad_accum_steps
                    loss.backward()

                    if (step + 1) % grad_accum_steps == 0:
                        clip_grad_norm_(model.parameters(), max_norm)
                        log_gradient_norm(model)
                        optimiser.step()
                        scheduler.step()
                        optimiser.zero_grad()

                    if step % eval_interval == 0:
                        val_loss = trainer.evaluate()
                        log(step, train_loss, val_loss, lr, grad_norm)

                    if step % ckpt_interval == 0:
                        save_checkpoint(model, optimiser, scheduler, step)

        ``Trainer.evaluate() -> float``:
            Compute validation loss (perplexity) on the held-out split.
            Runs without gradient computation.

Key implementation decisions:
    - Gradient accumulation is implemented explicitly: gradients are
      accumulated over ``grad_accum_steps`` micro-batches before a weight
      update.  This simulates a larger batch size without additional memory.

    - Gradient norm is logged at every update step (before clipping and
      after clipping).  If the pre-clip norm is consistently large (> 10),
      the learning rate is likely too high or the model is unstable.

    - Mixed precision (torch.amp) is controlled by a flag in TrainingConfig.
      When enabled, forward and backward passes use fp16; the optimiser
      step uses fp32 via GradScaler.

    - The training loop emits structured log lines to stdout that can be
      piped to a file: ``step=500 train_loss=2.341 val_loss=2.412 lr=3e-4
      grad_norm=1.23 tokens_per_sec=14500``

Implemented in: Phase 2, Weeks 9–10
"""

# Implementation begins in Phase 2.
