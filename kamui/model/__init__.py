"""Transformer model subpackage for KAMUI.

Responsibilities:
    - Define the full decoder-only transformer architecture
    - Own the forward pass: token IDs → logits (or loss)
    - Expose every internal component as a named, inspectable nn.Module
    - Implement weight initialisation following GPT-2 scaled init strategy

Architecture summary:
    token_ids (B, S)
        ↓  [token embedding + positional embedding]
    residual_stream (B, S, D)
        ↓  [× n_layers: LayerNorm → MultiHeadAttention → residual]
        ↓              [LayerNorm → FeedForward → residual         ]
    residual_stream (B, S, D)
        ↓  [LayerNorm → linear unembedding]
    logits (B, S, V)

Design constraints:
    - The model module must be importable with zero knowledge of the hook
      system.  Hooks are attached externally via kamui.hooks.HookManager.
    - No training logic lives here.  The model is a pure function:
      given token IDs, produce logits (and optionally loss).
    - All internal tensors follow the dimension conventions defined in
      kamui.model.config (B, S, D, H, Dh, F, V).

Module layout:
    config.py        — ModelConfig dataclass, dimension conventions
    embedding.py     — token embedding + sinusoidal/learned positional encoding
    attention.py     — scaled dot-product attention + multi-head wrapper
    feedforward.py   — position-wise FFN with GELU activation
    normalization.py — LayerNorm and RMSNorm from scratch
    block.py         — single transformer block (Pre-LN + residuals)
    transformer.py   — KAMUITransformer: full model assembly
    init_weights.py  — scaled weight initialisation strategy

Public API (available after Phase 1):
    ModelConfig         — all hyperparameters as a typed dataclass
    KAMUITransformer    — the full model

Implemented in: Phase 1, Weeks 5–8
"""

# from kamui.model.config import ModelConfig
# from kamui.model.transformer import KAMUITransformer

__all__: list[str] = []
