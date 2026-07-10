"""Named hook point registry for all KAMUI model components.

Responsibilities:
    - ``HookRegistry``:
        Maintains the canonical list of all valid hook point strings for a
        given ``KAMUITransformer`` configuration.

        ``HookRegistry.all_points(config) -> list[str]``:
            Return every valid hook point string for a model with
            ``config.n_layers`` layers.  Example output (n_layers=2):

                [
                    "embed.output",
                    "blocks.0.attn.output",
                    "blocks.0.attn.weights",
                    "blocks.0.ffn.mid",
                    "blocks.0.ffn.output",
                    "blocks.1.attn.output",
                    "blocks.1.attn.weights",
                    "blocks.1.ffn.mid",
                    "blocks.1.ffn.output",
                    "unembed.input",
                ]

        ``HookRegistry.validate(hook_point, config) -> bool``:
            Return True if ``hook_point`` is a valid hook point string for the
            given config.  Used by HookManager to raise helpful errors when
            users specify invalid hook points.

Why a registry?
    Without a registry, users must guess hook point strings.  A typo
    ("blocks.3.att.output" instead of "blocks.3.attn.output") silently
    attaches no hook and caches nothing.  The registry makes invalid hook
    points a loud, early error rather than a silent wrong result.

Note — pre_softmax:
    The attention pre-softmax scores are an internal functional computation
    that is not exposed as a module output, and no current interpretability
    tool consumes it.  It is intentionally omitted from the registry so that
    the model stays completely hook-agnostic (no HookPoint modules embedded
    in the architecture).  If it is needed later, attention can expose it via
    an opt-in return value without changing this registry's contract.

Implemented in: Phase 3.
"""

from __future__ import annotations

from kamui.model.config import ModelConfig


class HookRegistry:
    """Canonical registry of valid hook points for a KAMUITransformer."""

    #: Per-layer hook point suffixes, in canonical order.
    _PER_LAYER_POINTS: tuple[str, ...] = (
        "attn.output",
        "attn.weights",
        "ffn.mid",
        "ffn.output",
    )

    @classmethod
    def all_points(cls, config: ModelConfig) -> list[str]:
        """Return every valid hook point string for ``config``.

        Args:
            config: The model configuration (only ``n_layers`` is used).

        Returns:
            An ordered list of hook-point strings: the embedding output, the
            per-layer attention/FFN points for each of ``n_layers`` blocks, and
            the unembedding input.
        """
        points: list[str] = ["embed.output"]
        for layer in range(config.n_layers):
            for suffix in cls._PER_LAYER_POINTS:
                points.append(f"blocks.{layer}.{suffix}")
        points.append("unembed.input")
        return points

    @classmethod
    def validate(cls, hook_point: str, config: ModelConfig) -> bool:
        """Return True if ``hook_point`` is valid for ``config``.

        Args:
            hook_point: A candidate hook-point string.
            config:     The model configuration.

        Returns:
            Whether ``hook_point`` is in ``all_points(config)``.
        """
        return hook_point in set(cls.all_points(config))
