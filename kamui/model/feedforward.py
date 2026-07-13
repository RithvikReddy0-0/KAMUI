"""Position-wise feed-forward network (FFN), implemented from scratch.

The FFN is applied independently to each token position.  It is the
"computation" counterpart to attention's "communication": attention moves
information between positions; the FFN processes each position's
representation in isolation.

Responsibilities:
    - ``FeedForward``:
        Two linear transformations with a GELU activation in between
        (and dropout on the hidden activation):

            FFN(x) = W_2 · Dropout(GELU(W_1 · x + b_1)) + b_2

        Where:
            W_1 ∈ R^(F × D),  b_1 ∈ R^F     (expansion layer)
            W_2 ∈ R^(D × F),  b_2 ∈ R^D     (projection layer)
            F = d_ff  (typically 4 * d_model)

Design notes:
    Why GELU over ReLU?
        GELU (Gaussian Error Linear Unit) is smooth and has a small
        negative region, providing a soft learned gating effect.
        All modern transformers (GPT-2 onwards, LLaMA, Mistral) use GELU
        or its fast approximation.  ReLU is simpler but performs
        measurably worse on language modelling at equivalent scale.

    Why 4× expansion?
        The 4× factor is empirical and has been stable since the original
        transformer paper.  It gives the FFN sufficient capacity to act as
        a key-value memory (Geva et al., 2021) while keeping parameter
        count manageable.  The exact width is set by ``ModelConfig.d_ff``.

    This module does NOT apply layer normalisation or the residual
    connection — those are the responsibility of ``block.py`` (Pre-LN:
    ``x = x + feed_forward(layer_norm(x))``).

Tensor shapes:
    Input:   x        (B, S, D)     residual stream
    Hidden:  h        (B, S, F)     after W_1 + GELU
    Output:  out      (B, S, D)     after W_2

Hook points (attached externally by kamui.hooks):
    "ffn.mid"     — hidden activation (B, S, F), post-GELU
    "ffn.output"  — FFN output (B, S, D), pre-residual add

References:
    Geva, M. et al. (2021). Transformer Feed-Forward Layers Are Key-Value
    Memories. EMNLP 2021. https://arxiv.org/abs/2012.14913

Implemented in: Phase 2D.
"""

from __future__ import annotations

from torch import Tensor, nn

from kamui.model.config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise two-layer MLP sublayer (expand → GELU → dropout → project).

    Operates independently on each token's ``d_model``-dimensional vector:
    an expansion linear lifts it to ``d_ff`` dimensions, a GELU nonlinearity
    and dropout are applied, and a projection linear maps it back to
    ``d_model``.  Input and output are both the residual-stream shape
    ``(B, S, d_model)``.

    No normalisation and no residual connection are applied here — both are
    owned by the enclosing transformer block (Pre-LN convention).

    Attributes:
        fc_in:      Expansion linear layer ``d_model → d_ff`` (W_1, b_1).
        activation: GELU nonlinearity.
        dropout:    Dropout applied to the hidden (post-GELU) activation.
        fc_out:     Projection linear layer ``d_ff → d_model`` (W_2, b_2).
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the feed-forward network from a model configuration.

        Args:
            config: The model configuration.  ``d_model`` sets the input/output
                width, ``d_ff`` the hidden width, and ``dropout`` the dropout
                probability applied to the hidden activation.
        """
        super().__init__()
        self.config = config
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward transform position-wise.

        Args:
            x: Input tensor of shape ``(..., d_model)`` — typically the
                ``(B, S, d_model)`` residual stream.

        Returns:
            Tensor of the same shape as ``x``.

        Raises:
            TypeError:  If ``x`` is not a tensor.
            ValueError: If the last dimension of ``x`` is not ``d_model``.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"last dimension of x ({x.shape[-1]}) does not match "
                f"d_model ({self.config.d_model})"
            )

        h = self.fc_in(x)  # (..., d_ff)  expansion
        h = self.activation(h)  # (..., d_ff)  GELU
        h = self.dropout(h)  # (..., d_ff)  hidden dropout
        return self.fc_out(h)  # (..., d_model)  projection

    def __repr__(self) -> str:
        return (
            f"FeedForward(d_model={self.config.d_model}, "
            f"d_ff={self.config.d_ff}, dropout={self.config.dropout})"
        )
