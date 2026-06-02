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
        over all positions (shift-by-one: predicting token t+1 from t).

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

The naming of self.blocks[i] and its children directly controls the hook
point strings ("blocks.0.attn.output", "blocks.3.ffn.mid", etc.).
Do not change these names without updating ``kamui.hooks.registry``.

Tensor shapes:
    Input:   token_ids    (B, S)       int64
    Output:  logits       (B, S, V)    float32
             loss         scalar       float32 (only when targets provided)

Implemented in: Phase 1, Week 8
"""

# Implementation begins in Phase 1.
