"""Linear probing: train classifiers on cached activations to test what each layer knows.

Probing asks: does the residual stream at layer L contain enough information
to predict property P using a simple linear classifier?  If yes, the model
has encoded P by layer L.

Responsibilities:
    - ``LinearProbe``:
        Train a logistic regression on cached activations from a specific
        hook point to predict a categorical label.

        ``LinearProbe.train(hook_point, dataset, labels) -> ProbeResult``:
            1. Run the model on each example in ``dataset`` with a hook
               at ``hook_point``, cache the activation.
            2. Train a logistic regression (sklearn or from scratch) on the
               cached activations with ``labels`` as targets.
            3. Report train accuracy, val accuracy, and the probe weights.

        ``ProbeResult.plot_by_layer()``:
            Train probes at every layer and plot accuracy vs. layer depth.
            This reveals at which layer a property is encoded.

    - Useful properties to probe for:
        - Part-of-speech tags (syntactic — encoded early)
        - Named entity type (semantic — encoded mid-to-late)
        - Sentence length (trivially encoded early, used as sanity check)
        - Next-token category (encoded late — near final layer)
        - Coreference (encoded in specific attention heads)

Implementation note:
    Probes are intentionally simple (linear, not MLP).  A linear probe
    that succeeds tells us the property is *linearly decodable* from the
    representation — a strong claim.  A non-linear probe that succeeds
    tells us very little (a sufficiently powerful probe can decode anything
    from anything).

References:
    Ettinger, A. et al. (2016). Probing for semantic evidence of composition
    by means of simple classification tasks.
    ACL Workshop on Evaluating Vector Space Representations for NLP.

    Tenney, I. et al. (2019). BERT Rediscovers the Classical NLP Pipeline.
    ACL 2019. https://arxiv.org/abs/1905.05950

Implemented in: Phase 4, Week 16
"""

# Implementation begins in Phase 4.
