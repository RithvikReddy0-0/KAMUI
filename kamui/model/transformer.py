"""KAMUITransformer — the full decoder-only language model.

This module assembles all components into the complete model.  It is the
entry point for users: instantiate ``KAMUITransformer``, pass token IDs,
receive logits.

Responsibilities:
    - ``KAMUITransformer``:
        The full model:

            1. Embedding (token + positional)          → (B, S, D)
            2. × n_layers TransformerBlock             → (B, S, D)
            3. Final LayerNorm                         → (B, S, D)
            4. Linear unembedding (D → V)              → (B, S, V)

        Forward pass:
            logits = model(token_ids)                  (B, S, V)
            loss   = model(token_ids, targets=targets) scalar (cross-entropy)

        When ``targets`` is provided, the model computes cross-entropy loss
        over all positions.  Targets are the next-token labels (same shape as
        ``token_ids``); the data pipeline is responsible for the shift.

    - ``KAMUITransformer.from_config(config)``:
        Construct model from a ``ModelConfig`` instance.

    - ``KAMUITransformer.from_yaml(path)``:
        Convenience constructor that loads ``ModelConfig`` from YAML and
        constructs the model.

    - ``KAMUITransformer.num_parameters() -> int``:
        Returns the total number of trainable parameters.  Useful for
        scaling sanity checks.

Design constraints:
    - The forward pass has NO side effects.  It does not log, does not
      store activations internally, and does not call hooks.  All of
      that is handled externally by ``kamui.hooks.HookManager``.
    - The unembedding weight is tied to the token embedding weight
      (weight tying), following GPT-2.  This halves the parameter count
      of the embedding+unembedding pair and empirically improves perplexity.

Named module structure (critical for hook system):
    self.embed              — Embedding module
    self.blocks             — nn.ModuleList of TransformerBlock
    self.blocks[i]          — TransformerBlock at layer i
    self.blocks[i].attn     — MultiHeadAttention
    self.blocks[i].ffn      — FeedForward
    self.final_ln           — final LayerNorm before unembedding
    self.unembed            — linear projection D → V

Tensor shapes:
    Input:   token_ids    (B, S)       int64
    Output:  logits       (B, S, V)    float32
             loss         scalar       float32 (only when targets provided)

Implemented in: Phase 2G.
"""

from __future__ import annotations

from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

from kamui.model.block import TransformerBlock
from kamui.model.config import ModelConfig
from kamui.model.embedding import Embedding
from kamui.model.init_weights import init_weights
from kamui.model.normalization import LayerNorm


class KAMUITransformer(nn.Module):
    """The full decoder-only transformer language model.

    Assembles the embedding, a stack of ``n_layers`` transformer blocks, a
    final LayerNorm, and a weight-tied linear unembedding.  Given token IDs
    it returns next-token logits; given targets as well it returns the
    cross-entropy loss.

    Attributes:
        embed:    Combined token + positional embedding.
        blocks:   ``nn.ModuleList`` of ``TransformerBlock``.
        final_ln: LayerNorm applied before the unembedding.
        unembed:  Linear ``d_model → vocab_size`` (weight tied to the token
            embedding matrix; no bias).
    """

    def __init__(self, config: ModelConfig) -> None:
        """Construct the model from a configuration and initialise weights.

        Args:
            config: The model configuration.
        """
        super().__init__()
        self.config = config

        self.embed = Embedding(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_ln = LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # GPT-2-style scaled initialisation (before tying, so the shared
        # matrix ends up with the standard 0.02 embedding init).
        init_weights(self, config.n_layers)

        # Weight tying: the unembedding shares the token-embedding matrix.
        self.unembed.weight = self.embed.token_embedding.weight

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: ModelConfig) -> KAMUITransformer:
        """Construct a model from a ``ModelConfig`` instance."""
        return cls(config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> KAMUITransformer:
        """Load a ``ModelConfig`` from YAML and construct the model.

        Args:
            path: Path to a YAML file with a ``model`` section.
        """
        return cls(ModelConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, token_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        """Run the model.

        Args:
            token_ids: Integer tensor of shape ``(B, S)``.
            targets:   Optional next-token labels of shape ``(B, S)``.  When
                provided, the return value is the scalar cross-entropy loss.

        Returns:
            Logits of shape ``(B, S, vocab_size)`` if ``targets`` is None,
            otherwise the scalar cross-entropy loss.

        Raises:
            TypeError:  If ``targets`` is provided but is not a tensor.
            ValueError: If ``targets`` is provided but its shape differs from
                ``token_ids``.
        """
        x = self.embed(token_ids)  # (B, S, D)
        for block in self.blocks:
            x = block(x)  # (B, S, D)
        x = self.final_ln(x)  # (B, S, D)
        logits = self.unembed(x)  # (B, S, V)

        if targets is None:
            return logits

        if not isinstance(targets, Tensor):
            raise TypeError(f"targets must be a torch.Tensor, got {type(targets)}")
        if targets.shape != token_ids.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match token_ids "
                f"shape {tuple(token_ids.shape)}"
            )

        # Cross-entropy over all positions; flatten batch and sequence.
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of parameters.

        ``nn.Module.parameters()`` already yields each shared (weight-tied)
        parameter once, so the tied embedding/unembedding matrix is counted
        a single time.

        Args:
            trainable_only: If True (default), count only parameters with
                ``requires_grad=True``.

        Returns:
            The parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def __repr__(self) -> str:
        return (
            f"KAMUITransformer(n_layers={self.config.n_layers}, "
            f"d_model={self.config.d_model}, n_heads={self.config.n_heads}, "
            f"vocab_size={self.config.vocab_size}, "
            f"num_parameters={self.num_parameters():,})"
        )
