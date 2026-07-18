"""Linear probing: train classifiers on cached activations to test what each layer knows.

Probing asks: does the residual stream at layer L contain enough information
to predict property P using a *linear* classifier?  If yes, the model has
(linearly) encoded P by layer L.

Responsibilities:
    - ``LinearProbe.train(hook_point, dataset, labels) -> ProbeResult``:
        1. Run the model on each sequence with a hook at ``hook_point`` and
           cache the **last-position** activation.
        2. Train a from-scratch logistic regression (a single linear layer +
           cross-entropy; sklearn is not a KAMUI dependency).
        3. Report train/validation accuracy and the probe weights.

    - ``LinearProbe.probe_all_layers(dataset, labels) -> LayerProbeResult``:
        Train one probe on the residual stream at every depth (embedding +
        after each block, reconstructed from standard hook points exactly as
        in ``logit_lens.py``) and report accuracy by layer.
        ``LayerProbeResult.plot()`` renders accuracy vs. depth — revealing at
        which layer a property becomes linearly decodable.

Implementation note:
    Probes are intentionally linear.  A linear probe that succeeds shows the
    property is *linearly decodable* — a strong claim.  A powerful non-linear
    probe succeeding tells us very little.

References:
    Tenney, I. et al. (2019). BERT Rediscovers the Classical NLP Pipeline.
    https://arxiv.org/abs/1905.05950

Implemented in: Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kamui.hooks.manager import HookManager
from kamui.model.transformer import KAMUITransformer

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Fraction of examples held out for validation.
_VAL_FRACTION: float = 0.2


@dataclass
class ProbeResult:
    """The outcome of training one linear probe.

    Attributes:
        hook_point: Where the activations were captured.
        train_acc:  Accuracy on the training split.
        val_acc:    Accuracy on the validation split.
        weights:    The probe's ``(n_classes, d_model)`` weight matrix.
    """

    hook_point: str
    train_acc: float
    val_acc: float
    weights: Tensor


@dataclass
class LayerProbeResult:
    """Per-depth probe accuracies from ``probe_all_layers``.

    Attributes:
        val_accs:   Validation accuracy at each depth (0 = embedding).
        train_accs: Training accuracy at each depth.
    """

    val_accs: list[float]
    train_accs: list[float]

    def best_layer(self) -> int:
        """Return the depth with the highest validation accuracy."""
        return int(torch.tensor(self.val_accs).argmax().item())

    def plot(self, figsize: tuple[float, float] | None = None) -> Figure:
        """Line chart of probe accuracy vs. layer depth."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (6, 4))
        depths = range(len(self.val_accs))
        ax.plot(depths, self.val_accs, marker="o", label="val")
        ax.plot(depths, self.train_accs, marker="s", linestyle="--", label="train")
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_ylabel("probe accuracy")
        ax.set_title("Linear probe accuracy by depth")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        fig.tight_layout()
        return fig


def _fit_logistic(
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    n_classes: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[float, float, Tensor]:
    """Train a from-scratch logistic regression; return (train_acc, val_acc, W)."""
    torch.manual_seed(seed)
    probe = nn.Linear(x_train.shape[1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(x_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_acc = (probe(x_train).argmax(dim=-1) == y_train).float().mean().item()
        val_acc = (probe(x_val).argmax(dim=-1) == y_val).float().mean().item()
    return train_acc, val_acc, probe.weight.detach().clone()


class LinearProbe:
    """Train linear classifiers on cached activations.

    Attributes:
        model: A trained ``KAMUITransformer``.
    """

    def __init__(self, model: KAMUITransformer) -> None:
        """Create a prober over ``model``.

        Args:
            model: A ``KAMUITransformer``.
        """
        self.model = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(dataset: list[Tensor], labels: list[int]) -> Tensor:
        if not dataset:
            raise ValueError("dataset must be non-empty")
        if len(dataset) != len(labels):
            raise ValueError(
                f"dataset ({len(dataset)}) and labels ({len(labels)}) differ in length"
            )
        label_tensor = torch.tensor(labels, dtype=torch.long)
        if label_tensor.unique().numel() < 2:
            raise ValueError("labels must contain at least 2 distinct classes")
        return label_tensor

    @staticmethod
    def _split(n: int) -> tuple[slice, slice]:
        n_val = max(1, int(n * _VAL_FRACTION))
        return slice(0, n - n_val), slice(n - n_val, n)

    @torch.no_grad()
    def _capture(self, hook_point: str, dataset: list[Tensor]) -> Tensor:
        """Last-position activation at ``hook_point`` for each sequence."""
        module_path, point = hook_point.rsplit(".", 1)
        feats = []
        for ids in dataset:
            batch = ids.unsqueeze(0) if ids.dim() == 1 else ids
            with HookManager(self.model) as hooks:
                hooks.attach(module_path, point)
                self.model(batch)
                feats.append(hooks.get(hook_point)[0, -1])
        return torch.stack(feats)  # (N, D)

    @torch.no_grad()
    def _capture_streams(self, dataset: list[Tensor]) -> list[Tensor]:
        """Last-position residual stream at every depth, per sequence.

        Returns one ``(N, D)`` matrix per depth (0 = embedding, i = after
        block i-1), reconstructed as ``embed + Σ attn_out + ffn_out``.
        """
        n_layers = self.model.config.n_layers
        per_depth: list[list[Tensor]] = [[] for _ in range(n_layers + 1)]
        for ids in dataset:
            batch = ids.unsqueeze(0) if ids.dim() == 1 else ids
            with HookManager(self.model) as hooks:
                hooks.attach("embed", "output")
                for i in range(n_layers):
                    hooks.attach(f"blocks.{i}.attn", "output")
                    hooks.attach(f"blocks.{i}.ffn", "output")
                self.model(batch)
                stream = hooks.get("embed.output")
                per_depth[0].append(stream[0, -1])
                for i in range(n_layers):
                    stream = (
                        stream
                        + hooks.get(f"blocks.{i}.attn.output")
                        + hooks.get(f"blocks.{i}.ffn.output")
                    )
                    per_depth[i + 1].append(stream[0, -1])
        return [torch.stack(feats) for feats in per_depth]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        hook_point: str,
        dataset: list[Tensor],
        labels: list[int],
        epochs: int = 200,
        lr: float = 0.05,
        seed: int = 0,
    ) -> ProbeResult:
        """Train one probe on activations captured at ``hook_point``.

        Args:
            hook_point: A registry hook point (e.g. ``"blocks.1.ffn.output"``).
            dataset:    One 1-D token-ID tensor per example.
            labels:     One integer class label per example.
            epochs:     Full-batch gradient steps for the probe.
            lr:         Probe learning rate.
            seed:       RNG seed for probe init.

        Returns:
            A ``ProbeResult``.

        Raises:
            ValueError: If the dataset/labels are empty, mismatched, or contain
                fewer than 2 classes.
        """
        label_tensor = self._validate(dataset, labels)
        self.model.eval()
        features = self._capture(hook_point, dataset)

        train_slice, val_slice = self._split(len(dataset))
        n_classes = int(label_tensor.max().item()) + 1
        train_acc, val_acc, weights = _fit_logistic(
            features[train_slice],
            label_tensor[train_slice],
            features[val_slice],
            label_tensor[val_slice],
            n_classes,
            epochs,
            lr,
            seed,
        )
        return ProbeResult(
            hook_point=hook_point, train_acc=train_acc, val_acc=val_acc, weights=weights
        )

    def probe_all_layers(
        self,
        dataset: list[Tensor],
        labels: list[int],
        epochs: int = 200,
        lr: float = 0.05,
        seed: int = 0,
    ) -> LayerProbeResult:
        """Train one probe on the residual stream at every depth.

        Args:
            dataset: One 1-D token-ID tensor per example.
            labels:  One integer class label per example.
            epochs:  Full-batch gradient steps per probe.
            lr:      Probe learning rate.
            seed:    RNG seed for probe init.

        Returns:
            A ``LayerProbeResult`` with accuracies at each depth.

        Raises:
            ValueError: If the dataset/labels are invalid.
        """
        label_tensor = self._validate(dataset, labels)
        self.model.eval()
        streams = self._capture_streams(dataset)

        train_slice, val_slice = self._split(len(dataset))
        n_classes = int(label_tensor.max().item()) + 1
        train_accs: list[float] = []
        val_accs: list[float] = []
        for features in streams:
            train_acc, val_acc, _ = _fit_logistic(
                features[train_slice],
                label_tensor[train_slice],
                features[val_slice],
                label_tensor[val_slice],
                n_classes,
                epochs,
                lr,
                seed,
            )
            train_accs.append(train_acc)
            val_accs.append(val_acc)
        return LayerProbeResult(val_accs=val_accs, train_accs=train_accs)
