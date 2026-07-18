"""Text generation with configurable sampling strategies.

Responsibilities:
    - ``generate(model, tokenizer, prompt, strategy, **kwargs) -> str``:
        Generate a continuation of ``prompt`` using the specified sampling
        strategy.  Returns the full decoded string (prompt + continuation).

    - Sampling strategies (all implemented from scratch):

        ``greedy``:   pick the argmax token each step (deterministic).
        ``top_k``:    sample from the top-k tokens by probability.
        ``nucleus``:  (top-p) sample from the smallest set whose cumulative
                      probability exceeds p (Holtzman et al., 2019).
        ``temperature``: sample from the full temperature-scaled distribution.

        Temperature is applied to the logits before any top-k / nucleus
        filtering.

    - ``generate_with_probs(model, tokenizer, prompt, n_tokens) -> GenerationResult``:
        Greedily generate ``n_tokens`` and return the decoded text together
        with the per-step probability distributions, for inspecting
        generation dynamics.

References:
    Holtzman, A. et al. (2019). The Curious Case of Neural Text Degeneration.
    ICLR 2020. https://arxiv.org/abs/1904.09751

Implemented in: Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from kamui.model.transformer import KAMUITransformer

#: Supported sampling strategies.
_STRATEGIES: tuple[str, ...] = ("greedy", "top_k", "nucleus", "temperature")


class TokenizerLike(Protocol):
    """The minimal tokenizer interface generation needs (e.g. ``BPETokenizer``)."""

    def encode(self, text: str) -> list[int]:
        """Convert a string to token IDs."""

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to a string."""


@dataclass
class GenerationResult:
    """The result of ``generate_with_probs``.

    Attributes:
        text:       The full decoded string (prompt + continuation).
        token_ids:  The generated token IDs (continuation only).
        probs:      Tensor of shape ``(n_tokens, vocab_size)`` — the softmax
            distribution the model produced at each generation step.
    """

    text: str
    token_ids: list[int]
    probs: Tensor


def _sample_next(
    logits: Tensor,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample the next token ID from a ``(vocab,)`` logit vector."""
    scaled = logits / temperature

    if strategy == "greedy":
        return int(scaled.argmax().item())

    if strategy == "top_k":
        k = min(top_k, scaled.shape[-1])
        values, indices = torch.topk(scaled, k)
        probs = F.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice].item())

    if strategy == "nucleus":
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep tokens whose *preceding* cumulative mass is below p (so the
        # token that crosses p is included); always keep at least one.
        keep = (cumulative - probs) < top_p
        keep[0] = True
        kept_probs = probs[keep]
        kept_idx = sorted_idx[keep]
        kept_probs = kept_probs / kept_probs.sum()
        choice = torch.multinomial(kept_probs, num_samples=1)
        return int(kept_idx[choice].item())

    if strategy == "temperature":
        probs = F.softmax(scaled, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(choice.item())

    raise ValueError(f"unknown strategy '{strategy}'; expected one of {_STRATEGIES}")


@torch.no_grad()
def generate(
    model: KAMUITransformer,
    tokenizer: TokenizerLike,
    prompt: str,
    max_new_tokens: int = 50,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    seed: int | None = None,
) -> str:
    """Generate a continuation of ``prompt``.

    Args:
        model:          A ``KAMUITransformer``.
        tokenizer:      An object with ``encode(str) -> list[int]`` and
            ``decode(list[int]) -> str`` (e.g. ``BPETokenizer``).
        prompt:         The (non-empty) prompt string.
        max_new_tokens: Number of tokens to generate.
        strategy:       ``"greedy"``, ``"top_k"``, ``"nucleus"``, or
            ``"temperature"``.
        temperature:    Logit temperature (> 0), applied before filtering.
        top_k:          k for ``top_k`` sampling (>= 1).
        top_p:          p for ``nucleus`` sampling (in (0, 1]).
        seed:           Optional RNG seed for reproducible sampling.

    Returns:
        The full decoded string (prompt + continuation).

    Raises:
        ValueError: If ``prompt`` is empty, ``strategy`` is unknown, or a
            hyperparameter is out of range.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if seed is not None:
        torch.manual_seed(seed)

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must be non-empty (encoded to zero tokens)")

    was_training = model.training
    model.eval()

    ctx = model.config.context_length
    tokens = torch.tensor([ids], dtype=torch.long)  # (1, S)
    for _ in range(max_new_tokens):
        window = tokens[:, -ctx:]
        logits = model(window)  # (1, s, V)
        next_id = _sample_next(logits[0, -1], strategy, temperature, top_k, top_p)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

    if was_training:
        model.train()
    return tokenizer.decode(tokens[0].tolist())


@torch.no_grad()
def generate_with_probs(
    model: KAMUITransformer,
    tokenizer: TokenizerLike,
    prompt: str,
    n_tokens: int = 20,
) -> GenerationResult:
    """Greedily generate ``n_tokens`` and capture per-step probabilities.

    Args:
        model:     A ``KAMUITransformer``.
        tokenizer: An object with ``encode`` / ``decode``.
        prompt:    The (non-empty) prompt string.
        n_tokens:  Number of tokens to generate.

    Returns:
        A ``GenerationResult`` with the full text, the generated token IDs,
        and a ``(n_tokens, vocab_size)`` tensor of softmax distributions.

    Raises:
        ValueError: If ``prompt`` is empty.
    """
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must be non-empty (encoded to zero tokens)")

    was_training = model.training
    model.eval()

    ctx = model.config.context_length
    tokens = torch.tensor([ids], dtype=torch.long)
    generated: list[int] = []
    step_probs: list[Tensor] = []
    for _ in range(n_tokens):
        window = tokens[:, -ctx:]
        logits = model(window)
        probs = F.softmax(logits[0, -1], dim=-1)
        step_probs.append(probs)
        next_id = int(probs.argmax().item())
        generated.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

    if was_training:
        model.train()
    return GenerationResult(
        text=tokenizer.decode(tokens[0].tolist()),
        token_ids=generated,
        probs=torch.stack(step_probs),
    )
