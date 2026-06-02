"""Logit lens: project the residual stream to vocabulary at every layer.

The logit lens (nostalgebraist, 2020) answers: at each layer, if the model
were to output now, what would it predict?

It works by applying the model's final LayerNorm and unembedding matrix
to the residual stream at each layer, converting the D-dimensional vector
at each (layer, position) to a probability distribution over the vocabulary.

Responsibilities:
    - ``LogitLens``:
        ``LogitLens.run(token_ids) -> LogitLensResult``:
            Capture residual stream at every layer via hooks.
            Apply final_ln + unembed to each.
            Return ``LogitLensResult`` containing:
              - probs:    (n_layers+1, S, V)  probability distributions
              - top_tokens: (n_layers+1, S, k) top-k predicted tokens per
                             layer per position
              - tokens:   list of input token strings

        ``LogitLensResult.plot()``:
            Render a heatmap where:
              - rows    = layers (0 = embedding, L = final)
              - columns = token positions
              - cell    = top-1 predicted token at that (layer, position)
            Cells are coloured by the probability assigned to the top-1 token.

        ``LogitLensResult.plot_position(pos)``:
            For a single token position, plot probability of each vocab
            token across layers.  Shows how the model's prediction evolves
            with depth.

Key insight this enables:
    Some facts are resolved early (low layers), others late (high layers).
    The logit lens reveals this structure.  For factual completion prompts
    like "The Eiffel Tower is in [Paris]", you can observe the correct
    answer emerging at a specific layer — identifying where factual
    associations are stored.

References:
    nostalgebraist (2020). Interpreting GPT: the logit lens.
    LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru

Implemented in: Phase 4, Week 15
"""

# Implementation begins in Phase 4.
