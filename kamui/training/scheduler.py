"""Learning rate scheduler: linear warmup + cosine decay, from scratch.

Responsibilities:
    - ``CosineWithWarmup``:
        Implements the standard transformer LR schedule:

        Phase 1 — Linear warmup (steps 0 → warmup_steps):
            lr = max_lr * (step / warmup_steps)

        Phase 2 — Cosine decay (steps warmup_steps → max_steps):
            lr = min_lr + 0.5 * (max_lr - min_lr) *
                 (1 + cos(π * (step - warmup_steps) / (max_steps - warmup_steps)))

        Phase 3 — Floor (steps > max_steps):
            lr = min_lr

        Typical values:
            max_lr       = 3e-4  (for small models, d_model=256)
            min_lr       = 3e-5  (10% of max_lr)
            warmup_steps = 100   (for 5000-step runs)

Why warmup?
    At the start of training, gradients are large and noisy.  A cold-start
    at max_lr causes the loss to spike or diverge.  Warmup ramps the LR
    slowly, allowing the model to reach a reasonable parameter regime before
    the full learning rate is applied.  This is especially important with
    Pre-LN architectures at large scale.

Why cosine decay?
    Cosine decay is smooth (no sharp LR drops that cause loss spikes) and
    has a well-defined endpoint (min_lr).  It outperforms step decay and
    linear decay empirically on language modelling.

Implementation note:
    This is NOT a PyTorch LRScheduler subclass.  It is a standalone class
    that returns a float ``lr`` given a ``step`` integer.  This makes it
    trivially testable (no optimiser needed) and fully transparent.
    The Trainer is responsible for applying the returned LR to the optimiser.

Implemented in: Phase 2, Week 9
"""

# Implementation begins in Phase 2.
