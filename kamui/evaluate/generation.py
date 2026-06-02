"""Text generation with configurable sampling strategies.

Responsibilities:
    - ``generate(model, tokenizer, prompt, strategy, **kwargs) -> str``:
        Generate a continuation of ``prompt`` using the specified sampling
        strategy.  Returns a decoded string.

    - Sampling strategies (all implemented from scratch):

        ``greedy``:
            At each step, select the token with the highest probability.
            Deterministic.  Tends to produce repetitive, safe text.

        ``top_k(k)``:
            Filter the vocabulary to the top-k tokens by probability,
            renormalise, and sample.  k=1 recovers greedy.
            k=50 is a common default.

        ``nucleus(p)`` (top-p sampling):
            Filter to the smallest set of tokens whose cumulative probability
            exceeds p, renormalise, and sample.
            Introduced by Holtzman et al. (2019).  p=0.9 is a common default.
            Adapts to the model's uncertainty: sharp distributions use few
            tokens, flat distributions use many.

        ``temperature(t)``:
            Divide logits by temperature t before softmax.
            t < 1: sharper (more confident), t > 1: flatter (more random).
            Applied before top-k or nucleus filtering.

    - ``generate_with_probs(model, tokenizer, prompt, n_tokens) -> GenerationResult``:
        Generate tokens and return both the generated string and the
        per-step probability distributions.  Used by interpretability tools
        to inspect generation dynamics.

References:
    Holtzman, A. et al. (2019). The Curious Case of Neural Text Degeneration.
    ICLR 2020. https://arxiv.org/abs/1904.09751

Implemented in: Phase 2, Week 11
"""

# Implementation begins in Phase 2.
