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
                    "blocks.0.attn.pre_softmax",
                    "blocks.0.attn.weights",
                    "blocks.0.attn.output",
                    "blocks.0.ffn.mid",
                    "blocks.0.ffn.output",
                    "blocks.1.attn.pre_softmax",
                    "blocks.1.attn.weights",
                    "blocks.1.attn.output",
                    "blocks.1.ffn.mid",
                    "blocks.1.ffn.output",
                    "unembed.input",
                ]

        ``HookRegistry.validate(hook_point, config) -> bool``:
            Return True if ``hook_point`` is a valid hook point string
            for the given config.  Used by HookManager to raise helpful
            errors when users specify invalid hook points.

Why a registry?
    Without a registry, users must guess hook point strings.  A typo
    ("blocks.3.att.output" instead of "blocks.3.attn.output") silently
    attaches no hook and caches nothing.  The registry makes invalid hook
    points a loud, early error rather than a silent wrong result.

Implemented in: Phase 3, Week 13
"""

# Implementation begins in Phase 3.
