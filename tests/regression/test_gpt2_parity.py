"""Regression tests: verify KAMUI's interpretability tools against known GPT-2 results.

These tests use the gpt2_compatible config to load GPT-2 small weights and verify
that KAMUI's tools reproduce published findings.

References:
    nostalgebraist (2020): logit lens plots for GPT-2
    Olsson et al. (2022): induction head locations in GPT-2 small
"""

import pytest


@pytest.mark.slow
@pytest.mark.skip(reason="requires GPT-2 weight loading — planned for v0.2")
def test_logit_lens_gpt2_eiffel_tower() -> None:
    """LogitLens on GPT-2 small for 'The Eiffel Tower is located in the city of'
    must show 'Paris' emerging as the top prediction by layer 7 or earlier.

    This reproduces the canonical logit lens demonstration.
    """
    pass


@pytest.mark.slow
@pytest.mark.skip(reason="requires GPT-2 weight loading — planned for v0.2")
def test_induction_heads_in_gpt2_small() -> None:
    """InductionHeadDetector on GPT-2 small must find high-scoring induction
    heads at layers 1 and 2, consistent with Olsson et al. (2022).

    Specifically: heads (1,4), (1,10) are known induction heads in GPT-2 small.
    """
    pass
