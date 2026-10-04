"""Key/value cache for fast autoregressive generation.

Without a cache, generating each new token re-runs the model over the whole
sequence so far, recomputing every earlier position's attention keys and
values even though they never change.  A KV-cache stores those keys and values
as they are produced, so each step only has to process the newest token::

    step 1 (prefill): run the prompt, store K/V for every prompt position
    step t:           run one token, append its K/V, attend over all stored K/V

The cache is purely an optimisation: logits are the same as a full forward
pass up to floating-point rounding (the test suite checks cached and uncached
generation produce identical tokens).

Responsibilities:
    - ``LayerKVCache`` — the stored keys/values for one attention layer.
    - ``KVCache``      — one ``LayerKVCache`` per transformer block.

Usage::

    cache = model.new_cache()
    logits = model(prompt_ids, cache=cache)          # prefill
    logits = model(next_token_ids, cache=cache)      # one new position

``kamui.evaluate.generate(..., use_cache=True)`` drives this automatically.

Implemented in: v0.4.
"""

from __future__ import annotations

import torch
from torch import Tensor


class LayerKVCache:
    """Accumulated attention keys and values for a single layer.

    Attributes:
        keys:   ``(B, n_heads, T, d_head)`` stored keys, or ``None`` when empty.
        values: ``(B, n_heads, T, d_head)`` stored values, or ``None`` when empty.
    """

    def __init__(self) -> None:
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def length(self) -> int:
        """Number of positions stored (``T``)."""
        return 0 if self.keys is None else self.keys.shape[-2]

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Store new keys/values and return the full (old + new) tensors.

        Args:
            keys:   ``(B, n_heads, S, d_head)`` keys for the new positions.
            values: ``(B, n_heads, S, d_head)`` values for the new positions.

        Returns:
            ``(all_keys, all_values)``, each ``(B, n_heads, T + S, d_head)``.
        """
        if self.keys is None or self.values is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat([self.keys, keys], dim=-2)
            self.values = torch.cat([self.values, values], dim=-2)
        return self.keys, self.values

    def reset(self) -> None:
        """Drop everything stored."""
        self.keys = None
        self.values = None


class KVCache:
    """A key/value cache spanning every transformer block.

    Attributes:
        layers: One ``LayerKVCache`` per block, in order.
    """

    def __init__(self, n_layers: int) -> None:
        """Create an empty cache.

        Args:
            n_layers: Number of transformer blocks (must be > 0).

        Raises:
            ValueError: If ``n_layers`` is not positive.
        """
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        self.layers = [LayerKVCache() for _ in range(n_layers)]

    @property
    def length(self) -> int:
        """Number of positions processed so far (the next token's position)."""
        return self.layers[0].length

    def reset(self) -> None:
        """Empty every layer's cache."""
        for layer in self.layers:
            layer.reset()
