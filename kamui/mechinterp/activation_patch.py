"""Activation patching: causal interventions to localise model behaviours.

Activation patching is the core methodology of mechanistic interpretability.
It answers: which components are *causally responsible* for a behaviour,
not merely correlated with it?

Technique:
    1. Run model on a *clean* input. Cache activations at all hook points.
    2. Run model on a *corrupted* input. At a specific hook point, replace
       (patch) the activation with the cached clean activation.
    3. Measure how much the output changes back toward the clean output.
       A large change → that hook point is causally important.
       No change     → that hook point is not on the causal path.

Responsibilities:
    - ``ActivationPatcher``:
        ``ActivationPatcher.patch_single(
              clean_ids, corrupted_ids, hook_point
          ) -> float``:
            Patch one hook point. Return the logit difference recovery:
              0.0 = patching had no effect (hook point not causally relevant)
              1.0 = patching fully restored clean output (hook point is critical)

        ``ActivationPatcher.patch_all_layers(
              clean_ids, corrupted_ids, component="attn"
          ) -> PatchingResult``:
            Patch each layer's attention (or FFN) output one at a time.
            Return a ``PatchingResult`` with an effect per layer.

        ``PatchingResult.plot()``:
            Bar chart of causal effect by layer.
            The layer with the highest bar is where the behaviour is stored.

    - ``ActivationPatcher.patch_all_heads(
          clean_ids, corrupted_ids
      ) -> HeadPatchingResult``:
        Patch each (layer, head) pair one at a time.
        Returns a (n_layers, n_heads) matrix of causal effects.

Canonical experiment:
    clean     = "The Eiffel Tower is in Paris"
    corrupted = "The Eiffel Tower is in Berlin"
    Patch each layer back to the clean activations.
    The layer with the highest recovery is where "Paris" is stored.

References:
    Meng, K. et al. (2022). Locating and Editing Factual Associations in
    GPT. NeurIPS 2022 (ROME paper). https://arxiv.org/abs/2202.05262

    Wang, K. et al. (2022). Interpretability in the Wild: a Circuit for
    Indirect Object Identification in GPT-2 small.
    https://arxiv.org/abs/2211.00593

Implemented in: Phase 4, Week 17
"""

# Implementation begins in Phase 4.
