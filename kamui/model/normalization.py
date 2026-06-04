"""Layer normalisation and RMS normalisation, implemented from scratch.

Responsibilities:
    - ``LayerNorm``:
        Normalises activations across the feature dimension (D) for each
        token position independently.

            LayerNorm(x) = γ · (x - μ) / sqrt(σ² + ε) + β

        Where μ and σ² are the mean and (biased / population) variance
        computed over the D-dimensional feature vector at each (batch,
        position) independently.  γ (scale) and β (bias) are learnable
        parameters of shape (D,).

    - ``RMSNorm``:
        A simplified variant that omits the mean-centering step.
        Used by LLaMA and Mistral because it is faster and performs
        equivalently in practice.

            RMSNorm(x) = γ · x / RMS(x),   RMS(x) = sqrt(mean(x²) + ε)

        Provided as an alternative for research comparisons; not used by
        default in KAMUI v0.1.

Design notes:
    KAMUI uses Pre-LN (layer norm applied before the attention and FFN
    sublayers, not after).  This is the modern standard since Xiong et al.
    (2020) showed it enables more stable training at higher learning rates.

    Pre-LN:   x = x + Sublayer(LayerNorm(x))
    Post-LN:  x = LayerNorm(x + Sublayer(x))    ← original Vaswani et al.

    Why implement LayerNorm from scratch rather than using nn.LayerNorm?
        nn.LayerNorm is correct and fast.  KAMUI implements it from scratch
        so students see the formula explicitly and understand that it is
        simply a normalisation + affine transform with no architectural
        mystery.

Tensor shapes:
    Input:   x      (..., D)
    Output:  out    (..., D)   — same shape, values normalised over D

Hook points (attached externally by kamui.hooks):
    "ln.output" — the normalised, affine-transformed activations

References:
    Ba, J. L. et al. (2016). Layer Normalization.
    https://arxiv.org/abs/1607.06450

    Xiong, R. et al. (2020). On Layer Normalization in the Transformer
    Architecture. ICML 2020. https://arxiv.org/abs/2002.04745

    Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization.
    NeurIPS 2019. https://arxiv.org/abs/1910.07467

Implemented in: Phase 2C.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Default epsilon for numerical stability, matching ``torch.nn.LayerNorm``.
_DEFAULT_EPS: float = 1e-5


class LayerNorm(nn.Module):
    """Layer normalisation over the last (feature) dimension, from scratch.

    For each token position, the ``normalized_shape``-dimensional feature
    vector is standardised to zero mean and unit variance, then rescaled and
    shifted by the learnable affine parameters ``weight`` (γ) and ``bias``
    (β)::

        out = γ · (x - μ) / sqrt(σ² + ε) + β

    μ and σ² are computed over the last dimension only, so each (batch,
    position) is normalised independently of every other — no dependence on
    batch size or other examples (unlike BatchNorm).

    Attributes:
        normalized_shape: Size of the feature dimension that is normalised.
        eps:              Epsilon added to the variance for numerical stability.
        weight:           Learnable scale γ, shape ``(normalized_shape,)``,
                          initialised to ones.
        bias:             Learnable shift β, shape ``(normalized_shape,)``,
                          initialised to zeros.
    """

    def __init__(self, normalized_shape: int, eps: float = _DEFAULT_EPS) -> None:
        """Create a LayerNorm over a feature dimension of ``normalized_shape``.

        Args:
            normalized_shape: Size of the last dimension to normalise.  Must be > 0.
            eps:              Numerical-stability epsilon.  Must be > 0.

        Raises:
            ValueError: If ``normalized_shape`` or ``eps`` is not positive.
        """
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError(f"normalized_shape must be > 0, got {normalized_shape}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise ``x`` over its last dimension and apply the affine transform.

        Args:
            x: Input tensor of shape ``(..., normalized_shape)``.

        Returns:
            Tensor of the same shape as ``x``, normalised over the last dimension.

        Raises:
            TypeError:  If ``x`` is not a tensor.
            ValueError: If the last dimension of ``x`` is not ``normalized_shape``.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"last dimension of x ({x.shape[-1]}) does not match "
                f"normalized_shape ({self.normalized_shape})"
            )

        mean = x.mean(dim=-1, keepdim=True)
        # Biased (population) variance over the feature dimension, matching
        # torch.nn.LayerNorm.  Computed explicitly rather than via .var().
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation, from scratch (Zhang & Sennrich 2019).

    A simplified normalisation that rescales by the root-mean-square of the
    feature vector but performs **no mean-centering** and has **no bias**::

        out = γ · x / sqrt(mean(x²) + ε)

    Cheaper than ``LayerNorm`` and used by LLaMA / Mistral.  Provided here for
    research comparison; KAMUI v0.1 uses ``LayerNorm`` by default.

    Attributes:
        normalized_shape: Size of the feature dimension that is normalised.
        eps:              Epsilon added to the mean-square for numerical stability.
        weight:           Learnable scale γ, shape ``(normalized_shape,)``,
                          initialised to ones.  There is no learnable bias.
    """

    def __init__(self, normalized_shape: int, eps: float = _DEFAULT_EPS) -> None:
        """Create an RMSNorm over a feature dimension of ``normalized_shape``.

        Args:
            normalized_shape: Size of the last dimension to normalise.  Must be > 0.
            eps:              Numerical-stability epsilon.  Must be > 0.

        Raises:
            ValueError: If ``normalized_shape`` or ``eps`` is not positive.
        """
        super().__init__()
        if normalized_shape <= 0:
            raise ValueError(f"normalized_shape must be > 0, got {normalized_shape}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise ``x`` by its RMS over the last dimension and scale by γ.

        Args:
            x: Input tensor of shape ``(..., normalized_shape)``.

        Returns:
            Tensor of the same shape as ``x``.

        Raises:
            TypeError:  If ``x`` is not a tensor.
            ValueError: If the last dimension of ``x`` is not ``normalized_shape``.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"last dimension of x ({x.shape[-1]}) does not match "
                f"normalized_shape ({self.normalized_shape})"
            )

        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        x_norm = x / torch.sqrt(mean_square + self.eps)
        return self.weight * x_norm

    def __repr__(self) -> str:
        return f"RMSNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"
