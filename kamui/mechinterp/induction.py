"""Induction head detection and scoring.

Induction heads are the canonical example of interpretable circuits in
transformers.  They implement in-context pattern matching: given a sequence
like [A][B]...[A], an induction head at the second [A] will attend to [B]
and boost the probability of [B] as the next token.

This is mechanistically achieved by two heads working together:
    1. A "previous token" head (typically layer 0): at each position t,
       attends to position t-1 and copies its token into the residual stream.
    2. An induction head (typically layer 1+): at position t, uses the
       previous-token information to attend to the last position that was
       followed by the current token.

Responsibilities:
    - ``InductionHeadDetector``:
        Score all (layer, head) pairs in the model for induction behaviour.

        ``InductionHeadDetector.score_all_heads() -> dict[(int,int), float]``:
            For each head, compute the induction score:
            - Construct sequences with a repeated prefix: [rand × 50][rand × 50]
            - Measure average attention weight on the "induction position"
              (position t - prefix_length) at each position in the second half
            - Score ∈ [0, 1]; score > 0.5 strongly indicates induction behaviour

        ``InductionHeadDetector.plot_scores(scores)``:
            Heatmap of (layer, head) induction scores.
            High-scoring cells are induction heads.

        ``InductionHeadDetector.ablate_and_measure(heads) -> float``:
            Zero-ablate the specified (layer, head) pairs and measure the
            drop in in-context learning performance.
            Used to causally confirm that induction heads drive ICL.

Expected finding:
    In a 6-layer model trained on TinyStories, induction heads typically
    emerge at layers 1–2.  Layer 0 contains "previous token" heads that
    feed into them.

References:
    Olsson, C. et al. (2022). In-context Learning and Induction Heads.
    Transformer Circuits Thread.
    https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/

Implemented in: Phase 4, Week 18
"""

# Implementation begins in Phase 4.
