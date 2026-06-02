"""Unit tests for kamui.tokenizer.

Tests are written before implementation as a specification.
All tests are marked as expected-to-fail (xfail) until Phase 1 is complete.

Tests validate:
    - BPE training produces a vocabulary of the correct size
    - Encode/decode roundtrip is lossless for all ASCII inputs
    - Encode/decode roundtrip is lossless for UTF-8 inputs (emoji, CJK, etc.)
    - Special tokens are never split by merge operations
    - Vocabulary is stable across save/load cycles
    - Token IDs are always within [0, vocab_size)
    - Empty string encodes to empty list
    - Very long strings do not cause memory errors
"""

import pytest


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_encode_decode_roundtrip_ascii() -> None:
    """encode then decode must recover the original ASCII string exactly."""
    # from kamui.tokenizer import BPETokenizer
    # tokenizer = BPETokenizer.load("tests/fixtures/tokenizer_small.json")
    # texts = ["Hello, world!", "The quick brown fox.", "1 + 1 = 2"]
    # for text in texts:
    #     assert tokenizer.decode(tokenizer.encode(text)) == text
    pass


@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_encode_decode_roundtrip_unicode() -> None:
    """encode then decode must recover the original UTF-8 string exactly."""
    # from kamui.tokenizer import BPETokenizer
    # tokenizer = BPETokenizer.load("tests/fixtures/tokenizer_small.json")
    # texts = ["café", "北京", "🤗 transformers", "Ñoño"]
    # for text in texts:
    #     assert tokenizer.decode(tokenizer.encode(text)) == text
    pass


@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_encode_empty_string() -> None:
    """Empty string must encode to an empty list of token IDs."""
    # from kamui.tokenizer import BPETokenizer
    # tokenizer = BPETokenizer.load("tests/fixtures/tokenizer_small.json")
    # assert tokenizer.encode("") == []
    pass


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_vocab_size_matches_config() -> None:
    """A tokenizer trained with vocab_size=4096 must have exactly 4096 tokens."""
    pass


@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_token_ids_in_range() -> None:
    """All token IDs produced by encode() must be in [0, vocab_size)."""
    pass


@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_special_tokens_not_split() -> None:
    """The <|endoftext|> token must encode as a single token ID, never split."""
    pass


# ---------------------------------------------------------------------------
# Serialisation tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """A tokenizer saved to disk and reloaded must produce identical encodings."""
    pass


@pytest.mark.xfail(reason="BPETokenizer not yet implemented — Phase 1")
def test_vocabulary_stable_across_load() -> None:
    """Token IDs must be identical across save/load cycles (no random re-ordering)."""
    pass
