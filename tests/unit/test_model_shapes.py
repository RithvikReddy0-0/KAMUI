"""Unit tests for tensor shape contracts across all model components.

Shape tests are the fastest and most valuable tests in an ML codebase.
A wrong shape silently produces wrong results; these tests catch that
immediately.

Every public-facing module in kamui.model must have shape tests here.
Tests are marked xfail until Phase 1 implementation is complete.

Shape contract reference (from kamui.model.config):
    B  = batch size
    S  = sequence length
    D  = d_model
    H  = n_heads
    Dh = d_head = D / H
    F  = d_ff = 4 * D
    V  = vocab_size
"""

import pytest


# ---------------------------------------------------------------------------
# Attention shape tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="MultiHeadAttention not yet implemented — Phase 1")
def test_attention_output_shape() -> None:
    """Attention output must be (B, S, D)."""
    # import torch
    # from kamui.model.attention import MultiHeadAttention
    # attn = MultiHeadAttention(d_model=64, n_heads=4)
    # x = torch.randn(2, 10, 64)  # B=2, S=10, D=64
    # out, weights = attn(x)
    # assert out.shape == (2, 10, 64)
    pass


@pytest.mark.xfail(reason="MultiHeadAttention not yet implemented — Phase 1")
def test_attention_weights_shape() -> None:
    """Attention weight matrix must be (B, H, S, S)."""
    # out, weights = attn(x)
    # assert weights.shape == (2, 4, 10, 10)  # B, H, S, S
    pass


@pytest.mark.xfail(reason="MultiHeadAttention not yet implemented — Phase 1")
def test_attention_weights_sum_to_one() -> None:
    """Attention weights must sum to 1.0 over the last dimension."""
    # import torch
    # assert weights.sum(dim=-1).allclose(torch.ones(2, 4, 10))
    pass


@pytest.mark.xfail(reason="MultiHeadAttention not yet implemented — Phase 1")
def test_causal_mask_no_future_leakage() -> None:
    """Changing future tokens must not change past token outputs.

    This is the most critical correctness test in the entire test suite.
    If this fails, the model is cheating during training.
    """
    # import torch
    # from kamui.model.transformer import KAMUITransformer
    # from kamui.model.config import ModelConfig
    # model = KAMUITransformer(ModelConfig(n_layers=2, d_model=64, n_heads=4,
    #                                      d_ff=256, vocab_size=100, context_length=32))
    # ids = torch.randint(0, 100, (1, 16))
    # out1 = model(ids)
    # ids_mod = ids.clone()
    # ids_mod[0, 8:] = torch.randint(0, 100, (8,))   # change future tokens
    # out2 = model(ids_mod)
    # assert out1[:, :8, :].allclose(out2[:, :8, :], atol=1e-5), \
    #     "Causal mask is broken: future tokens affect past positions"
    pass


# ---------------------------------------------------------------------------
# FFN shape tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="FeedForward not yet implemented — Phase 1")
def test_ffn_output_shape() -> None:
    """FFN output must be (B, S, D) — same as input."""
    pass


@pytest.mark.xfail(reason="FeedForward not yet implemented — Phase 1")
def test_ffn_hidden_shape() -> None:
    """FFN hidden activation must be (B, S, 4*D)."""
    pass


# ---------------------------------------------------------------------------
# Full model shape tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="KAMUITransformer not yet implemented — Phase 1")
def test_model_logits_shape() -> None:
    """Model logits must be (B, S, V)."""
    pass


@pytest.mark.xfail(reason="KAMUITransformer not yet implemented — Phase 1")
def test_model_loss_is_scalar() -> None:
    """When targets are provided, model output must be a scalar loss."""
    pass


@pytest.mark.xfail(reason="KAMUITransformer not yet implemented — Phase 1")
def test_model_loss_near_log_vocab_size_at_init() -> None:
    """At random initialisation, loss must be close to log(vocab_size).

    A randomly initialised model should predict all tokens with ~equal
    probability, so the cross-entropy should be approximately log(V).
    A large deviation indicates a broken initialisation strategy.
    """
    # import math
    # expected = math.log(config.vocab_size)
    # assert abs(initial_loss - expected) < 0.5, \
    #     f"Initial loss {initial_loss:.3f} is far from log(V)={expected:.3f}"
    pass


# ---------------------------------------------------------------------------
# Embedding shape tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Embedding not yet implemented — Phase 1")
def test_embedding_output_shape() -> None:
    """Embedding output must be (B, S, D)."""
    pass
