"""AdamW optimiser configuration with correct weight decay separation.

Responsibilities:
    - ``build_optimizer(model, config) -> torch.optim.AdamW``:
        Construct an AdamW optimiser with two parameter groups:

        Group 1 — weight decay applied:
            All weight matrices (linear layers, embedding tables).
            These have shape with ndim >= 2.

        Group 2 — weight decay NOT applied:
            All biases, LayerNorm γ and β parameters.
            These have shape with ndim < 2.

        Returns a configured ``torch.optim.AdamW`` instance.

Why separate parameter groups?
    AdamW (Loshchilov & Hutter, 2019) applies L2 regularisation directly
    to weights.  Applying weight decay to biases and normalisation
    parameters is incorrect: biases should be unconstrained, and LayerNorm
    parameters are scale/shift parameters that should not be shrunk toward
    zero.  The separation is subtle but measurably improves training
    stability and final perplexity.

    This is the approach used by nanoGPT and all serious transformer
    training implementations.

Typical hyperparameters:
    lr           = set by scheduler (see scheduler.py)
    betas        = (0.9, 0.95)   — standard for transformers
    eps          = 1e-8
    weight_decay = 0.1           — for weight matrices
                   0.0           — for biases and LayerNorm params

References:
    Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay
    Regularization. ICLR 2019. https://arxiv.org/abs/1711.05101

Implemented in: Phase 2, Week 9
"""

# Implementation begins in Phase 2.
