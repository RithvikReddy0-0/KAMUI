"""KAMUI — Knowledge Activation Mapping & Understanding Interface.

A transformer interpretability framework built entirely from scratch.

"To understand a model, you must first see what it sees."

This module exposes the stable public API for KAMUI v0.1.
Only symbols explicitly listed in __all__ are considered public.
Everything else is an implementation detail and may change.

Public API surface (v0.1):
    Model:
        KAMUITransformer    — the decoder-only transformer
        ModelConfig         — all model hyperparameters

    Tokenizer:
        BPETokenizer        — byte-pair encoding tokeniser

    Training:
        Trainer             — explicit training loop
        TrainingConfig      — training hyperparameters

    Hooks:
        HookManager         — context-managed activation capture

    Mechanistic Interpretability:
        LogitLens           — per-layer vocabulary projection
        ActivationPatcher   — causal intervention tool
        InductionHeadDetector — induction head scoring
        CircuitAblator      — component ablation utility
        AttentionVisualizer — attention weight extraction and plotting
        LinearProbe         — probing classifier on cached activations

    Evaluation:
        compute_perplexity  — token-level perplexity
        generate            — text generation with configurable sampling

Example:
    >>> import kamui
    >>> config = kamui.ModelConfig.from_yaml("configs/small.yaml")
    >>> model = kamui.KAMUITransformer(config)
    >>> tokenizer = kamui.BPETokenizer.load("path/to/tokenizer.json")
    >>> # Train (see kamui.Trainer)
    >>> # Interpret (see kamui.LogitLens, kamui.ActivationPatcher, ...)
"""

# Version follows semantic versioning: MAJOR.MINOR.PATCH
# 0.x.y = pre-release; public API not yet stable
__version__ = "0.1.0"
__author__ = "Rithvik Reddy Mukkara"
__email__ = "rithvikreddymukkara@gmail.com"
__license__ = "MIT"

# Public API — each symbol is imported once its implementing phase is complete.
# Tools from not-yet-implemented phases are listed (commented) at the bottom.
from kamui.evaluate.generation import generate
from kamui.evaluate.perplexity import compute_perplexity
from kamui.hooks.manager import HookManager
from kamui.mechinterp.activation_patch import ActivationPatcher
from kamui.mechinterp.logit_lens import LogitLens
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.tokenizer.bpe import BPETokenizer
from kamui.tokenizer.vocab import Vocabulary
from kamui.training.trainer import Trainer, TrainingConfig

# Not yet implemented (future mechinterp tools):
# from kamui.mechinterp.attention_viz import AttentionVisualizer
# from kamui.mechinterp.probing import LinearProbe
# from kamui.mechinterp.induction import InductionHeadDetector
# from kamui.mechinterp.circuits import CircuitAblator

__all__: list[str] = [
    "__version__",
    "ModelConfig",
    "KAMUITransformer",
    "BPETokenizer",
    "Vocabulary",
    "HookManager",
    "compute_perplexity",
    "generate",
    "Trainer",
    "TrainingConfig",
    "LogitLens",
    "ActivationPatcher",
]
