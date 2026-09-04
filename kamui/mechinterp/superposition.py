"""Sparse autoencoders for feature decomposition (superposition analysis).

Transformers are hypothesised to represent more features than they have
dimensions via *superposition*: many features share overlapping directions in
activation space, so a linear basis cannot disentangle them.  A sparse
autoencoder (SAE) learns an **overcomplete dictionary** of feature directions
by reconstructing activations through a wide hidden layer with an L1 sparsity
penalty.  The learned features tend to be monosemantic even when the original
neurons are polysemantic.

Architecture (Anthropic, "Towards Monosemanticity", 2023):
    b_dec:  decoder pre-bias, subtracted before encoding      (d_model,)
    encode: f = ReLU((x - b_dec) @ W_enc + b_enc)             (n_features,)
    decode: x_hat = f @ W_dec + b_dec                         (d_model,)
    loss:   MSE(x, x_hat) + l1_coeff * mean(|f|)

Each feature's decoder direction (a row of ``W_dec``) is kept unit-norm, so the
hidden activation magnitude is the feature's true coefficient rather than an
artefact of decoder scale.

Public API:
    - ``SparseAutoencoder``           — the model (encode / decode / forward / loss / save / load)
    - ``collect_activations``         — cache activations at a hook point via HookManager
    - ``train_sae``                   — train an SAE on cached activations
    - ``sae_feature_metrics``         — reconstruction, dead-feature %, mean L0/L1
    - ``interpret_features``          — top-activating tokens per feature (what it detects)
    - ``feature_cooccurrence``        — co-activation density matrix (which features fire together)

Reference:
    Anthropic (2023). Towards Monosemanticity: Decomposing Language Models
    With Dictionary Learning. https://transformer-circuits.pub/2023/monosemantic-features

Implemented in: v0.2 (see research/future/sae_design.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kamui.hooks.manager import HookManager
from kamui.model.transformer import KAMUITransformer

#: Default L1 sparsity coefficient.
_DEFAULT_L1: float = 1e-3

#: Floor for decoder-column norms, to avoid divide-by-zero on a dead feature.
_NORM_EPS: float = 1e-8


@dataclass
class SAELoss:
    """The three components of the SAE objective (all scalar tensors).

    Attributes:
        total:          ``reconstruction + l1_coeff * sparsity`` (backprop through this).
        reconstruction: MSE between input and reconstruction.
        sparsity:       Mean absolute hidden activation (L1 penalty term).
    """

    total: Tensor
    reconstruction: Tensor
    sparsity: Tensor


@dataclass
class SAEMetrics:
    """Evaluation metrics for a trained SAE.

    Attributes:
        reconstruction_mse:    MSE of the reconstruction on the given data.
        dead_feature_fraction: Fraction of features that never activate.
        mean_l0:               Mean number of active features per token.
        mean_l1:               Mean summed absolute activation per token.
    """

    reconstruction_mse: float
    dead_feature_fraction: float
    mean_l0: float
    mean_l1: float


class SparseAutoencoder(nn.Module):
    """An overcomplete sparse autoencoder over model activations.

    Attributes:
        d_model:    Dimension of the activations being decomposed.
        n_features: Dictionary size (``k * d_model`` for an overcomplete SAE).
        l1_coeff:   Weight of the L1 sparsity penalty.
        W_enc:      ``(d_model, n_features)`` encoder weights.
        b_enc:      ``(n_features,)`` encoder bias.
        W_dec:      ``(n_features, d_model)`` decoder dictionary (unit-norm rows).
        b_dec:      ``(d_model,)`` decoder pre-bias.
    """

    def __init__(self, d_model: int, n_features: int, l1_coeff: float = _DEFAULT_L1) -> None:
        """Create a sparse autoencoder.

        Args:
            d_model:    Activation dimension.  Must be > 0.
            n_features: Dictionary size.  Must be > 0 (typically ``k * d_model``).
            l1_coeff:   L1 sparsity coefficient.  Must be >= 0.

        Raises:
            ValueError: If any argument is out of range.
        """
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if l1_coeff < 0:
            raise ValueError(f"l1_coeff must be >= 0, got {l1_coeff}")

        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.W_dec, std=0.1)
        with torch.no_grad():
            self.normalize_decoder()
            self.W_enc.copy_(self.W_dec.t())  # tied init from the normalised dictionary

    def normalize_decoder(self) -> None:
        """Rescale each feature's decoder direction (row of ``W_dec``) to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)
            self.W_dec.div_(norms)

    def encode(self, x: Tensor) -> Tensor:
        """Encode activations to sparse, non-negative feature coefficients.

        Args:
            x: Activations of shape ``(..., d_model)``.

        Returns:
            Feature activations of shape ``(..., n_features)`` (>= 0 via ReLU).
        """
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, features: Tensor) -> Tensor:
        """Reconstruct activations from feature coefficients.

        Args:
            features: Feature activations of shape ``(..., n_features)``.

        Returns:
            Reconstructed activations of shape ``(..., d_model)``.
        """
        return features @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(reconstruction, features)`` for input activations ``x``."""
        features = self.encode(x)
        return self.decode(features), features

    def loss(self, x: Tensor) -> SAELoss:
        """Compute the SAE objective for a batch of activations.

        Args:
            x: Activations of shape ``(N, d_model)``.

        Returns:
            An ``SAELoss`` with ``total``, ``reconstruction``, and ``sparsity``.
        """
        reconstruction, features = self.forward(x)
        recon = F.mse_loss(reconstruction, x)
        sparsity = features.abs().mean()
        return SAELoss(
            total=recon + self.l1_coeff * sparsity, reconstruction=recon, sparsity=sparsity
        )

    def __repr__(self) -> str:
        return (
            f"SparseAutoencoder(d_model={self.d_model}, "
            f"n_features={self.n_features}, l1_coeff={self.l1_coeff})"
        )

    def save(self, path: str | Path) -> None:
        """Save the SAE's config and weights to a ``.pt`` file.

        Both the architecture (``d_model`` / ``n_features`` / ``l1_coeff``) and
        the learned weights are stored, so ``SparseAutoencoder.load`` can rebuild
        the model without the caller re-specifying its shape.  Parent directories
        are created as needed.

        Args:
            path: Destination ``.pt`` file.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d_model": self.d_model,
                "n_features": self.n_features,
                "l1_coeff": self.l1_coeff,
                "state_dict": self.state_dict(),
            },
            path_obj,
        )

    @classmethod
    def load(cls, path: str | Path) -> SparseAutoencoder:
        """Rebuild an SAE saved by :meth:`save`.

        Loads with ``weights_only=True`` (no arbitrary-code execution — the file
        holds only the config scalars and weight tensors).

        Args:
            path: A ``.pt`` file written by :meth:`save`.

        Returns:
            A ``SparseAutoencoder`` with the saved architecture and weights.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        sae = cls(
            d_model=int(checkpoint["d_model"]),
            n_features=int(checkpoint["n_features"]),
            l1_coeff=float(checkpoint["l1_coeff"]),
        )
        sae.load_state_dict(checkpoint["state_dict"])
        return sae


@torch.no_grad()
def collect_activations(
    model: KAMUITransformer, hook_point: str, sequences: Iterable[Tensor]
) -> Tensor:
    """Cache activations at ``hook_point`` over ``sequences`` into an ``(N, d_model)`` matrix.

    Every ``(B, S, d_model)`` capture is flattened to ``(B*S, d_model)`` and the
    results are concatenated — one row per token — ready for SAE training.

    Args:
        model:     A ``KAMUITransformer``.
        hook_point: A registry hook point, e.g. ``"blocks.0.ffn.output"``.
        sequences: Iterable of token-ID tensors ``(S,)`` or ``(B, S)``.

    Returns:
        A ``(N, d_model)`` tensor of activations.

    Raises:
        ValueError: If ``sequences`` is empty.
    """
    module_path, point = hook_point.rsplit(".", 1)
    model.eval()
    rows: list[Tensor] = []
    for ids in sequences:
        batch = ids.unsqueeze(0) if ids.dim() == 1 else ids
        with HookManager(model) as hooks:
            hooks.attach(module_path, point)
            model(batch)
            activation = hooks.get(hook_point)  # (B, S, d_model)
        rows.append(activation.reshape(-1, activation.shape[-1]))
    if not rows:
        raise ValueError("sequences produced no activations")
    return torch.cat(rows, dim=0)


def train_sae(
    sae: SparseAutoencoder,
    activations: Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    renormalize: bool = True,
) -> list[dict[str, float]]:
    """Train an SAE on cached activations with Adam.

    Args:
        sae:         The ``SparseAutoencoder`` to train (updated in place).
        activations: ``(N, d_model)`` activation matrix.
        epochs:      Number of passes over the data.
        lr:          Adam learning rate.
        batch_size:  Mini-batch size.
        seed:        RNG seed for shuffling and init.
        renormalize: Re-unit-norm the decoder after each step (recommended).

    Returns:
        Per-epoch history: dicts with ``epoch``, ``total``, ``reconstruction``,
        ``sparsity`` (mean over the epoch's batches).

    Raises:
        ValueError: If ``activations`` is not ``(N, d_model)`` matching the SAE,
            or ``epochs`` / ``batch_size`` are not positive.
    """
    if activations.dim() != 2:
        raise ValueError(f"activations must be 2-D (N, d_model), got {tuple(activations.shape)}")
    if activations.shape[1] != sae.d_model:
        raise ValueError(
            f"activations last dim ({activations.shape[1]}) must match d_model ({sae.d_model})"
        )
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n = activations.shape[0]
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        totals = [0.0, 0.0, 0.0]
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = activations[perm[start : start + batch_size]]
            optimizer.zero_grad()
            losses = sae.loss(batch)
            losses.total.backward()
            optimizer.step()
            if renormalize:
                sae.normalize_decoder()
            totals[0] += losses.total.item()
            totals[1] += losses.reconstruction.item()
            totals[2] += losses.sparsity.item()
            n_batches += 1
        history.append(
            {
                "epoch": float(epoch),
                "total": totals[0] / n_batches,
                "reconstruction": totals[1] / n_batches,
                "sparsity": totals[2] / n_batches,
            }
        )
    return history


@torch.no_grad()
def sae_feature_metrics(sae: SparseAutoencoder, activations: Tensor) -> SAEMetrics:
    """Evaluate a trained SAE on activations.

    Args:
        sae:         A ``SparseAutoencoder``.
        activations: ``(N, d_model)`` activation matrix.

    Returns:
        An ``SAEMetrics`` (reconstruction MSE, dead-feature fraction, mean L0/L1).

    Raises:
        ValueError: If ``activations`` is not ``(N, d_model)`` matching the SAE.
    """
    if activations.dim() != 2 or activations.shape[1] != sae.d_model:
        raise ValueError(
            f"activations must be (N, d_model={sae.d_model}), got {tuple(activations.shape)}"
        )
    reconstruction, features = sae.forward(activations)
    active = features > 0  # ReLU output: active iff strictly positive
    return SAEMetrics(
        reconstruction_mse=F.mse_loss(reconstruction, activations).item(),
        dead_feature_fraction=(~active.any(dim=0)).float().mean().item(),
        mean_l0=active.float().sum(dim=-1).mean().item(),
        mean_l1=features.abs().sum(dim=-1).mean().item(),
    )


@dataclass
class FeatureProfile:
    """An interpretation summary for a single SAE feature.

    Attributes:
        feature:         The feature (dictionary-atom) index.
        density:         Fraction of tokens on which the feature is active (> 0).
        mean_activation: Mean activation over the tokens where it fires
            (``0.0`` if the feature never fires).
        top_token_ids:   Token IDs of the strongest-activating tokens, most
            active first (up to ``top_k`` of them).
        top_activations: The matching activation values, in descending order.
    """

    feature: int
    density: float
    mean_activation: float
    top_token_ids: list[int]
    top_activations: list[float]


@torch.no_grad()
def interpret_features(
    sae: SparseAutoencoder,
    activations: Tensor,
    token_ids: Tensor,
    top_k: int = 10,
    features: Iterable[int] | None = None,
) -> list[FeatureProfile]:
    """Profile SAE features by the tokens that most strongly activate them.

    A sparse autoencoder learns a dictionary of feature directions, but a
    feature is only *interpretable* once you see what makes it fire.  For each
    requested feature this returns the tokens whose encoded activation at that
    feature is largest, along with how often the feature fires (density) and how
    strongly (mean activation) — the token-level form of Anthropic's "what does
    this feature detect?" analysis.  Pair ``top_token_ids`` with a tokenizer's
    ``decode`` to read the feature's meaning.

    Args:
        sae:         A (typically trained) ``SparseAutoencoder``.
        activations: ``(N, d_model)`` activations, one row per token.
        token_ids:   ``(N,)`` token ID for each activation row (parallel to
            ``activations`` — usually the tokens passed to ``collect_activations``).
        top_k:       Number of top-activating tokens to keep per feature.
        features:    Feature indices to profile (defaults to every feature).

    Returns:
        One ``FeatureProfile`` per requested feature, in the requested order.

    Raises:
        ValueError: If shapes are inconsistent, ``top_k < 1``, or a requested
            feature index is out of range.
    """
    if activations.dim() != 2 or activations.shape[1] != sae.d_model:
        raise ValueError(
            f"activations must be (N, d_model={sae.d_model}), got {tuple(activations.shape)}"
        )
    if token_ids.dim() != 1 or token_ids.shape[0] != activations.shape[0]:
        raise ValueError(
            f"token_ids must be 1-D with length {activations.shape[0]}, "
            f"got shape {tuple(token_ids.shape)}"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    feature_list = list(range(sae.n_features)) if features is None else list(features)
    coded = sae.encode(activations)  # (N, n_features)
    n_rows = coded.shape[0]
    ids = token_ids.tolist()

    profiles: list[FeatureProfile] = []
    for feature in feature_list:
        if not (0 <= feature < sae.n_features):
            raise ValueError(f"feature must be in [0, {sae.n_features}), got {feature}")
        column = coded[:, feature]
        active = column > 0
        n_active = int(active.sum().item())
        density = n_active / n_rows
        mean_activation = float(column[active].mean().item()) if n_active > 0 else 0.0

        k = min(top_k, n_active)  # never rank in inactive (zero) tokens
        if k > 0:
            values, indices = torch.topk(column, k)
            top_token_ids = [ids[i] for i in indices.tolist()]
            top_activations = [float(v) for v in values.tolist()]
        else:
            top_token_ids = []
            top_activations = []
        profiles.append(
            FeatureProfile(
                feature=feature,
                density=density,
                mean_activation=mean_activation,
                top_token_ids=top_token_ids,
                top_activations=top_activations,
            )
        )
    return profiles


@torch.no_grad()
def feature_cooccurrence(
    sae: SparseAutoencoder,
    activations: Tensor,
    features: Iterable[int] | None = None,
) -> Tensor:
    """Compute how often pairs of SAE features co-activate.

    Monosemantic features rarely fire in isolation — related features (e.g. the
    products of feature splitting) tend to co-occur.  This returns the
    **co-activation density** matrix ``M`` where::

        M[i, j] = fraction of tokens on which features i and j are both active

    ``M`` is symmetric and its diagonal ``M[i, i]`` is feature ``i``'s own firing
    density (as in ``sae_feature_metrics``).  A dead feature yields a zero row and
    column.  Read ``M`` alongside ``interpret_features``: the profiles say what
    each feature detects, ``M`` says which detections travel together.

    Args:
        sae:         A (typically trained) ``SparseAutoencoder``.
        activations: ``(N, d_model)`` activations, one row per token.
        features:    Feature indices to include (defaults to every feature). The
            returned matrix is ordered to match this list.

    Returns:
        A ``(k, k)`` co-activation density matrix for the ``k`` requested
        features (``k == n_features`` by default).

    Raises:
        ValueError: If ``activations`` is not ``(N, d_model)`` matching the SAE,
            or a requested feature index is out of range.
    """
    if activations.dim() != 2 or activations.shape[1] != sae.d_model:
        raise ValueError(
            f"activations must be (N, d_model={sae.d_model}), got {tuple(activations.shape)}"
        )
    feature_list = list(range(sae.n_features)) if features is None else list(features)
    for feature in feature_list:
        if not (0 <= feature < sae.n_features):
            raise ValueError(f"feature must be in [0, {sae.n_features}), got {feature}")

    active = (sae.encode(activations) > 0).float()  # (N, n_features)
    selected = active[:, feature_list]  # (N, k)
    n_rows = selected.shape[0]
    return (selected.t() @ selected) / n_rows
