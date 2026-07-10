"""Weight initialisation strategies for KAMUITransformer.

Responsibilities:
    - ``init_weights(model, n_layers)``:
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

Implemented in: Phase 2G.
"""

from __future__ import annotations

import math

from torch import nn

from kamui.model.embedding import LearnedPositionalEncoding, TokenEmbedding
from kamui.model.normalization import LayerNorm, RMSNorm

#: Base standard deviation for GPT-2-style initialisation.
_BASE_STD: float = 0.02

#: Module-name suffixes whose weight is a residual output projection and so
#: receives the depth-scaled initialisation (W_O of attention, W_2 of the FFN).
_RESIDUAL_PROJECTION_SUFFIXES: tuple[str, ...] = ("attn.out_proj", "ffn.fc_out")


def init_weights(model: nn.Module, n_layers: int, std: float = _BASE_STD) -> nn.Module:
    """Apply GPT-2-style scaled initialisation to a model in place.

    Every ``nn.Linear`` weight and every learnable embedding matrix is drawn
    from ``N(0, std)``; biases are zeroed; LayerNorm scale/bias are set to
    one/zero.  Then the residual output projections (attention ``out_proj``
    and FFN ``fc_out``) are re-initialised with the depth-scaled standard
    deviation ``std / sqrt(2 * n_layers)`` to keep the residual-stream
    variance roughly constant with depth.

    Args:
        model:    The model (or any submodule tree) to initialise.
        n_layers: Number of transformer layers, used for the residual scaling.
        std:      Base standard deviation (default 0.02).

    Returns:
        The same ``model`` instance, initialised in place (for chaining).

    Raises:
        ValueError: If ``n_layers`` is not positive.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be > 0, got {n_layers}")

    # Pass 1: standard initialisation by module type.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (TokenEmbedding, LearnedPositionalEncoding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, (LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, "bias"):
                nn.init.zeros_(module.bias)

    # Pass 2: depth-scaled init for residual output projections.
    residual_std = std / math.sqrt(2 * n_layers)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(_RESIDUAL_PROJECTION_SUFFIXES):
            nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    return model
