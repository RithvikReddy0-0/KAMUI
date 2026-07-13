"""Single transformer block: Pre-LN attention + Pre-LN FFN with residuals.

The transformer block is the repeating unit of the architecture.  It
combines the attention and FFN sublayers with layer normalisation and
residual connections.  KAMUI uses the Pre-LN variant.

Responsibilities:
    - ``TransformerBlock``:
        One complete transformer block:

            # Attention sublayer
            x = x + Attention(LayerNorm(x))

            # FFN sublayer
            x = x + FFN(LayerNorm(x))

        Inputs and outputs are both (B, S, D) — the residual stream shape.
        The block reads from the stream and writes back to it via addition.

Design notes:
    Why is residual connection here and not in attention.py / feedforward.py?
        The residual connection conceptually belongs to the block, not the
        sublayer.  The sublayer computes a *delta* to add to the stream;
        the block applies the addition.  This separation makes it trivial
        to ablate a sublayer (replace its output with zeros) without
        modifying the sublayer's own code.

    Why Pre-LN?
        Post-LN (original Vaswani et al. formulation) places LayerNorm
        *after* the residual addition.  This means the residual path passes
        through LayerNorm, which constrains the gradient flow and requires
        careful warmup.  Pre-LN removes LayerNorm from the residual path,
        resulting in more stable training and allowing larger learning rates.
        See Xiong et al. (2020).

    Named submodules:
        self.ln1    — LayerNorm before attention
        self.attn   — MultiHeadAttention
        self.ln2    — LayerNorm before FFN
        self.ffn    — FeedForward

        These names are used by ``kamui.hooks.registry`` to construct hook
        point strings: "blocks.{L}.attn.output", "blocks.{L}.ffn.output".
        Do not rename them without updating the hook registry.

Tensor shapes:
    Input:   x        (B, S, D)   residual stream
    Output:  x        (B, S, D)   residual stream (updated)

Implemented in: Phase 2F.
"""

from __future__ import annotations

from torch import Tensor, nn

from kamui.model.attention import MultiHeadAttention
from kamui.model.config import ModelConfig
from kamui.model.feedforward import FeedForward
from kamui.model.normalization import LayerNorm


class TransformerBlock(nn.Module):
    """One Pre-LN transformer block (attention sublayer + FFN sublayer).

    Applies two residual sublayers to the ``(B, S, d_model)`` residual stream::

        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))

    LayerNorm is applied to each sublayer's *input* (Pre-LN); the residual
    path itself is a clean identity, and each sublayer contributes a delta
    that the block adds back into the stream.

    Attributes:
        ln1:  LayerNorm applied before attention.
        attn: MultiHeadAttention sublayer.
        ln2:  LayerNorm applied before the feed-forward network.
        ffn:  FeedForward sublayer.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build a transformer block from a model configuration.

        Args:
            config: The model configuration.  Sizes the LayerNorms (``d_model``)
                and is forwarded to the attention and feed-forward sublayers.
        """
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor, return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Run the residual stream through the block.

        Args:
            x:              Residual-stream tensor of shape ``(B, S, d_model)``.
            return_weights: If True, also return the attention probability
                matrix of shape ``(B, n_heads, S, S)`` from this block.

        Returns:
            The updated residual stream of shape ``(B, S, d_model)`` if
            ``return_weights`` is False, otherwise ``(x, weights)``.

        Raises:
            TypeError:  If ``x`` is not a tensor.
            ValueError: If ``x`` is not 3-D or its last dimension is not
                ``d_model``.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"x must be 3-D (B, S, d_model), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(
                f"last dimension of x ({x.shape[-1]}) does not match "
                f"d_model ({self.config.d_model})"
            )

        # Attention sublayer (Pre-LN): normalise input, attend, add to stream.
        if return_weights:
            attn_out, weights = self.attn(self.ln1(x), return_weights=True)
        else:
            attn_out = self.attn(self.ln1(x))
        x = x + attn_out

        # Feed-forward sublayer (Pre-LN): normalise input, transform, add.
        x = x + self.ffn(self.ln2(x))

        if return_weights:
            return x, weights
        return x

    def __repr__(self) -> str:
        return (
            f"TransformerBlock(d_model={self.config.d_model}, "
            f"n_heads={self.config.n_heads}, d_ff={self.config.d_ff})"
        )
