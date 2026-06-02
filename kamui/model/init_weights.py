"""Weight initialisation strategies for KAMUITransformer.

Responsibilities:
    - ``init_weights(module, n_layers)``:
        Apply GPT-2-style scaled initialisation to all parameters in the
        model.  Called once after model construction.

    - Standard init:
        - Embedding matrices:         N(0, 0.02)
        - Linear weights (non-residual): N(0, 0.02)
        - Linear biases:              zeros
        - LayerNorm scale (γ):        ones
        - LayerNorm bias (β):         zeros

    - Scaled residual init:
        The output projection of each attention block (W_O) and the second
        linear layer of each FFN (W_2) are initialised with reduced scale:

            std = 0.02 / sqrt(2 * n_layers)

        Rationale: each residual block adds its output to the stream.  With
        L layers, if each contributes variance σ², the stream variance at
        the final layer is L × σ².  Dividing the output projection std by
        sqrt(2L) keeps the residual stream variance approximately constant
        with depth, preventing activation growth that destabilises training.

        This is the exact strategy used in GPT-2 (Radford et al., 2019).

Design note:
    Weight initialisation is separated into its own module so it can be
    unit-tested independently (verify parameter statistics after init) and
    so alternative strategies can be swapped in without touching the model
    architecture code.

References:
    Radford, A. et al. (2019). Language Models are Unsupervised Multitask
    Learners (GPT-2). OpenAI Blog.

Implemented in: Phase 1, Week 8
"""

# Implementation begins in Phase 1.
