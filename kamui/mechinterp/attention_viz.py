"""Attention weight extraction and visualisation.

The simplest interpretability tool and the natural first step: extract
what each attention head is attending to and display it.

Responsibilities:
    - ``AttentionVisualizer``:
        Given a model and tokeniser, extract attention weight matrices
        for every (layer, head) pair for a given input sequence.

        ``AttentionVisualizer.run(token_ids) -> AttentionResult``:
            Run a forward pass with hooks on all attention weight points.
            Return an ``AttentionResult`` containing:
              - weights: shape (n_layers, n_heads, S, S)
              - tokens:  list of decoded token strings for the sequence

        ``AttentionResult.plot(layer, head)``:
            Render a single (layer, head) heatmap using matplotlib.
            Axes are labelled with token strings.

        ``AttentionResult.plot_all()``:
            Render a grid of heatmaps: rows = layers, columns = heads.
            Returns a matplotlib Figure.

        ``AttentionResult.plot_interactive()``:
            Render an interactive plotly heatmap with layer/head selectors.

    - ``head_summary_stats(result) -> DataFrame``:
        Compute per-head statistics useful for identifying head types:
        - mean entropy of attention distribution (low = sharp / interpretable)
        - fraction of attention on the diagonal (self-attention)
        - fraction of attention on position t-1 (previous-token heads)

What to look for:
    - Diagonal heads: attend to the current token (nearly useless)
    - Previous-token heads: attend strongly to position t-1
    - Induction heads: attend to position of the previous occurrence of the
      current token (see induction.py for formal detection)
    - Broad/diffuse heads: uniform attention, minimal information content

Implemented in: Phase 4, Week 14 (first tool, simplest)
"""

# Implementation begins in Phase 4.
