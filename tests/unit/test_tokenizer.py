"""Unit tests for kamui.tokenizer — utils and BPETokenizer.

Coverage targets:
    kamui/tokenizer/utils.py   — text_to_bytes, bytes_to_text, get_stats, merge_pair
    kamui/tokenizer/bpe.py     — BPETokenizer (train, encode, decode, save, load)

Test categories:
    - utils: each of the four helper functions, including all error paths
    - BPETokenizer.train: vocab_size contract, merge count, corpus variants
    - BPETokenizer.encode: roundtrip (ASCII, Unicode, emoji, CJK), special tokens,
      empty string, very long strings, token ID range
    - BPETokenizer.decode: roundtrip, special tokens, out-of-range ID errors
    - BPETokenizer.save / load: roundtrip, invalid file errors
    - BPETokenizer.__repr__
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kamui.tokenizer.bpe import BPETokenizer
from kamui.tokenizer.utils import bytes_to_text, get_stats, merge_pair, text_to_bytes

# ===========================================================================
# Fixtures
# ===========================================================================

# A small but meaningful corpus for fast training in tests
_SMALL_CORPUS = (
    "hello world hello world hello world "
    "the quick brown fox jumps over the lazy dog "
    "the cat sat on the mat "
    "abracadabra alakazam "
) * 10


@pytest.fixture(scope="module")
def tiny_tokenizer() -> BPETokenizer:
    """A BPETokenizer trained on a small in-memory corpus with vocab_size=300."""
    return BPETokenizer.train(_SMALL_CORPUS, vocab_size=300)


@pytest.fixture(scope="module")
def default_tokenizer() -> BPETokenizer:
    """A BPETokenizer with default special tokens (<|endoftext|>, <|pad|>)."""
    corpus = _SMALL_CORPUS + "<|endoftext|>" * 20
    return BPETokenizer.train(corpus, vocab_size=320)


# ===========================================================================
# utils — text_to_bytes
# ===========================================================================


class TestTextToBytes:
    def test_empty_string(self) -> None:
        assert text_to_bytes("") == []

    def test_ascii_single_char(self) -> None:
        assert text_to_bytes("A") == [65]

    def test_ascii_string(self) -> None:
        assert text_to_bytes("hi") == [104, 105]

    def test_ascii_string_known_bytes(self) -> None:
        result = text_to_bytes("Hello")
        assert result == [72, 101, 108, 108, 111]

    def test_unicode_two_byte(self) -> None:
        # é = U+00E9 → UTF-8: 0xC3 0xA9 = [195, 169]
        assert text_to_bytes("é") == [195, 169]

    def test_unicode_three_byte(self) -> None:
        # 中 = U+4E2D → UTF-8: 0xE4 0xB8 0xAD = [228, 184, 173]
        assert text_to_bytes("中") == [228, 184, 173]

    def test_emoji_four_byte(self) -> None:
        # 🤗 = U+1F917 → UTF-8: [240, 159, 164, 151]
        result = text_to_bytes("🤗")
        assert result == [240, 159, 164, 151]

    def test_roundtrip_with_bytes_to_text(self) -> None:
        for text in ["hello", "café", "北京", "🤗 hi", ""]:
            assert bytes_to_text(text_to_bytes(text)) == text

    def test_all_bytes_representable(self) -> None:
        # Every byte value 0–255 must appear somewhere in the output space
        result = text_to_bytes("hello world")
        assert all(0 <= b <= 255 for b in result)

    def test_type_error_on_non_string(self) -> None:
        with pytest.raises(TypeError):
            text_to_bytes(42)  # type: ignore[arg-type]

    def test_type_error_on_bytes_input(self) -> None:
        with pytest.raises(TypeError):
            text_to_bytes(b"hello")  # type: ignore[arg-type]

    def test_returns_list_of_ints(self) -> None:
        result = text_to_bytes("abc")
        assert isinstance(result, list)
        assert all(isinstance(b, int) for b in result)


# ===========================================================================
# utils — bytes_to_text
# ===========================================================================


class TestBytesToText:
    def test_empty_list(self) -> None:
        assert bytes_to_text([]) == ""

    def test_ascii_bytes(self) -> None:
        assert bytes_to_text([104, 105]) == "hi"

    def test_unicode_two_byte(self) -> None:
        assert bytes_to_text([195, 169]) == "é"

    def test_emoji(self) -> None:
        assert bytes_to_text([240, 159, 164, 151]) == "🤗"

    def test_type_error_on_non_list(self) -> None:
        with pytest.raises(TypeError):
            bytes_to_text("hello")  # type: ignore[arg-type]

    def test_type_error_on_float_element(self) -> None:
        with pytest.raises(TypeError):
            bytes_to_text([104, 1.5, 111])  # type: ignore[arg-type]

    def test_type_error_on_bool_element(self) -> None:
        # bool is a subclass of int; the function must reject it
        with pytest.raises(TypeError):
            bytes_to_text([True, False])  # type: ignore[arg-type]

    def test_value_error_on_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            bytes_to_text([256])

    def test_value_error_on_negative(self) -> None:
        with pytest.raises(ValueError):
            bytes_to_text([-1])

    def test_value_error_on_invalid_utf8(self) -> None:
        # 0xFF is not a valid UTF-8 start byte on its own
        with pytest.raises(ValueError):
            bytes_to_text([0xFF])

    def test_returns_string(self) -> None:
        assert isinstance(bytes_to_text([65]), str)


# ===========================================================================
# utils — get_stats
# ===========================================================================


class TestGetStats:
    def test_empty_vocab(self) -> None:
        assert get_stats([]) == {}

    def test_single_sequence_length_one(self) -> None:
        assert get_stats([[42]]) == {}

    def test_simple_pair_count(self) -> None:
        stats = get_stats([[1, 2, 3, 1, 2]])
        assert stats[(1, 2)] == 2
        assert stats[(2, 3)] == 1
        assert stats[(3, 1)] == 1

    def test_multiple_sequences(self) -> None:
        stats = get_stats([[1, 2], [1, 2], [3, 4]])
        assert stats[(1, 2)] == 2
        assert stats[(3, 4)] == 1

    def test_returns_dict(self) -> None:
        result = get_stats([[1, 2]])
        assert isinstance(result, dict)

    def test_keys_are_tuples_of_ints(self) -> None:
        stats = get_stats([[10, 20, 30]])
        for key in stats:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_no_cross_sequence_pairs(self) -> None:
        # Pair (2, 3) should NOT appear — they're in different sequences
        stats = get_stats([[1, 2], [3, 4]])
        assert (2, 3) not in stats

    def test_type_error_on_non_list(self) -> None:
        with pytest.raises(TypeError):
            get_stats("not a list")  # type: ignore[arg-type]

    def test_single_pair(self) -> None:
        stats = get_stats([[5, 6]])
        assert stats == {(5, 6): 1}

    def test_repeated_same_pair(self) -> None:
        # [1,1,1] has pairs (1,1) twice
        stats = get_stats([[1, 1, 1]])
        assert stats[(1, 1)] == 2


# ===========================================================================
# utils — merge_pair
# ===========================================================================


class TestMergePair:
    def test_basic_merge(self) -> None:
        assert merge_pair([1, 2, 3, 1, 2], (1, 2), 5) == [5, 3, 5]

    def test_all_occurrences_merged(self) -> None:
        assert merge_pair([1, 2, 1, 2, 1, 2], (1, 2), 9) == [9, 9, 9]

    def test_no_overlap_greedy(self) -> None:
        # [1,2,2,1]: merge (2,2)→7 → [1,7,1]
        assert merge_pair([1, 2, 2, 1], (2, 2), 7) == [1, 7, 1]

    def test_overlapping_not_double_counted(self) -> None:
        # [1,1,1]: merge (1,1)→2 greedily → [2, 1] not [1, 2]
        result = merge_pair([1, 1, 1], (1, 1), 2)
        assert result == [2, 1]

    def test_empty_list(self) -> None:
        assert merge_pair([], (1, 2), 5) == []

    def test_no_match(self) -> None:
        assert merge_pair([3, 4, 5], (1, 2), 99) == [3, 4, 5]

    def test_does_not_mutate_input(self) -> None:
        original = [1, 2, 3]
        merge_pair(original, (1, 2), 9)
        assert original == [1, 2, 3]

    def test_single_element_list(self) -> None:
        assert merge_pair([7], (7, 7), 8) == [7]

    def test_returns_new_list(self) -> None:
        ids = [1, 2, 3]
        result = merge_pair(ids, (1, 2), 5)
        assert result is not ids

    def test_type_error_on_ids_not_list(self) -> None:
        with pytest.raises(TypeError):
            merge_pair((1, 2, 3), (1, 2), 5)  # type: ignore[arg-type]

    def test_type_error_on_pair_not_tuple(self) -> None:
        with pytest.raises(TypeError):
            merge_pair([1, 2], [1, 2], 5)  # type: ignore[arg-type]

    def test_type_error_on_new_id_not_int(self) -> None:
        with pytest.raises(TypeError):
            merge_pair([1, 2], (1, 2), "5")  # type: ignore[arg-type]

    def test_type_error_on_bool_new_id(self) -> None:
        with pytest.raises(TypeError):
            merge_pair([1, 2], (1, 2), True)  # type: ignore[arg-type]


# ===========================================================================
# BPETokenizer — training
# ===========================================================================


class TestBPETokenizerTrain:
    def test_vocab_size_matches_request(self, tiny_tokenizer: BPETokenizer) -> None:
        assert tiny_tokenizer.vocab_size == 300

    def test_vocab_size_another_value(self) -> None:
        tok = BPETokenizer.train(_SMALL_CORPUS, vocab_size=270)
        assert tok.vocab_size == 270

    def test_minimum_vocab_size_no_merges(self) -> None:
        # 256 bytes + 2 specials = 258, zero merges
        tok = BPETokenizer.train(_SMALL_CORPUS, vocab_size=258)
        assert tok.vocab_size == 258
        assert len(tok._merges) == 0

    def test_vocab_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            BPETokenizer.train(_SMALL_CORPUS, vocab_size=257)

    def test_default_special_tokens(self, tiny_tokenizer: BPETokenizer) -> None:
        assert "<|endoftext|>" in tiny_tokenizer.special_tokens
        assert "<|pad|>" in tiny_tokenizer.special_tokens

    def test_custom_special_tokens(self) -> None:
        tok = BPETokenizer.train(
            _SMALL_CORPUS,
            vocab_size=260,
            special_tokens=("<|eos|>", "<|bos|>"),
        )
        assert "<|eos|>" in tok.special_tokens
        assert "<|bos|>" in tok.special_tokens

    def test_train_from_file(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text(_SMALL_CORPUS, encoding="utf-8")
        tok = BPETokenizer.train(corpus_file, vocab_size=280)
        assert tok.vocab_size == 280

    def test_train_produces_merges(self, tiny_tokenizer: BPETokenizer) -> None:
        # 300 - 258 = 42 merges expected
        assert len(tiny_tokenizer._merges) == 42

    def test_repr(self, tiny_tokenizer: BPETokenizer) -> None:
        r = repr(tiny_tokenizer)
        assert "BPETokenizer" in r
        assert "vocab_size=300" in r


# ===========================================================================
# BPETokenizer — encoding
# ===========================================================================


class TestBPETokenizerEncode:
    def test_empty_string_returns_empty_list(self, tiny_tokenizer: BPETokenizer) -> None:
        assert tiny_tokenizer.encode("") == []

    def test_returns_list(self, tiny_tokenizer: BPETokenizer) -> None:
        result = tiny_tokenizer.encode("hello")
        assert isinstance(result, list)

    def test_all_ids_in_range(self, tiny_tokenizer: BPETokenizer) -> None:
        texts = ["hello world", "the quick brown fox", "abracadabra"]
        for text in texts:
            ids = tiny_tokenizer.encode(text)
            for token_id in ids:
                assert (
                    0 <= token_id < tiny_tokenizer.vocab_size
                ), f"token_id {token_id} out of range for text={text!r}"

    def test_type_error_on_non_string(self, tiny_tokenizer: BPETokenizer) -> None:
        with pytest.raises(TypeError):
            tiny_tokenizer.encode(42)  # type: ignore[arg-type]

    def test_encode_nonempty_gives_nonempty(self, tiny_tokenizer: BPETokenizer) -> None:
        assert len(tiny_tokenizer.encode("a")) > 0

    def test_special_token_encodes_to_single_id(self, default_tokenizer: BPETokenizer) -> None:
        ids = default_tokenizer.encode("<|endoftext|>")
        assert len(ids) == 1
        assert ids[0] == default_tokenizer.token_to_id("<|endoftext|>")

    def test_special_token_not_split_when_embedded(self, default_tokenizer: BPETokenizer) -> None:
        text = "hello<|endoftext|>world"
        ids = default_tokenizer.encode(text)
        eos_id = default_tokenizer.token_to_id("<|endoftext|>")
        assert eos_id in ids

    def test_pad_token_encodes_to_single_id(self, default_tokenizer: BPETokenizer) -> None:
        ids = default_tokenizer.encode("<|pad|>")
        assert len(ids) == 1

    def test_long_string_no_error(self, tiny_tokenizer: BPETokenizer) -> None:
        long_text = "hello world " * 1000
        ids = tiny_tokenizer.encode(long_text)
        assert len(ids) > 0


# ===========================================================================
# BPETokenizer — decoding
# ===========================================================================


class TestBPETokenizerDecode:
    def test_empty_list_returns_empty_string(self, tiny_tokenizer: BPETokenizer) -> None:
        assert tiny_tokenizer.decode([]) == ""

    def test_returns_string(self, tiny_tokenizer: BPETokenizer) -> None:
        ids = tiny_tokenizer.encode("hello")
        result = tiny_tokenizer.decode(ids)
        assert isinstance(result, str)

    def test_type_error_on_non_list(self, tiny_tokenizer: BPETokenizer) -> None:
        with pytest.raises(TypeError):
            tiny_tokenizer.decode("hello")  # type: ignore[arg-type]

    def test_type_error_on_non_int_element(self, tiny_tokenizer: BPETokenizer) -> None:
        with pytest.raises(TypeError):
            tiny_tokenizer.decode([1.5])  # type: ignore[arg-type]

    def test_value_error_on_out_of_range_id(self, tiny_tokenizer: BPETokenizer) -> None:
        with pytest.raises(ValueError):
            tiny_tokenizer.decode([tiny_tokenizer.vocab_size])

    def test_value_error_on_negative_id(self, tiny_tokenizer: BPETokenizer) -> None:
        with pytest.raises(ValueError):
            tiny_tokenizer.decode([-1])

    def test_special_token_decoded_to_string(self, default_tokenizer: BPETokenizer) -> None:
        eos_id = default_tokenizer.token_to_id("<|endoftext|>")
        assert default_tokenizer.decode([eos_id]) == "<|endoftext|>"


# ===========================================================================
# BPETokenizer — encode/decode roundtrip
# ===========================================================================


class TestBPETokenizerRoundtrip:
    """encode(text) followed by decode must recover the original string exactly."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hello, world!",
            "The quick brown fox.",
            "1 + 1 = 2",
            "abracadabra",
            "the cat sat on the mat",
            "   leading and trailing spaces   ",
            "\nnewline\n",
            "tab\there",
        ],
    )
    def test_roundtrip_ascii(self, text: str, tiny_tokenizer: BPETokenizer) -> None:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text

    @pytest.mark.parametrize(
        "text",
        [
            "café",
            "naïve",
            "北京",
            "🤗 transformers",
            "Ñoño",
            "résumé",
        ],
    )
    def test_roundtrip_unicode(self, text: str, tiny_tokenizer: BPETokenizer) -> None:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text

    def test_roundtrip_empty_string(self, tiny_tokenizer: BPETokenizer) -> None:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode("")) == ""

    def test_roundtrip_long_string(self, tiny_tokenizer: BPETokenizer) -> None:
        text = "hello world " * 500
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text

    def test_roundtrip_with_special_tokens(self, default_tokenizer: BPETokenizer) -> None:
        text = "start<|endoftext|>end"
        assert default_tokenizer.decode(default_tokenizer.encode(text)) == text

    def test_roundtrip_only_special_token(self, default_tokenizer: BPETokenizer) -> None:
        text = "<|endoftext|>"
        assert default_tokenizer.decode(default_tokenizer.encode(text)) == text

    def test_roundtrip_multiple_special_tokens(self, default_tokenizer: BPETokenizer) -> None:
        text = "<|endoftext|><|pad|><|endoftext|>"
        assert default_tokenizer.decode(default_tokenizer.encode(text)) == text


# ===========================================================================
# BPETokenizer — save / load
# ===========================================================================


class TestBPETokenizerSaveLoad:
    def test_save_creates_file(self, tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tok.json"
        tiny_tokenizer.save(path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "tok.json"
        tiny_tokenizer.save(path)
        assert path.exists()

    def test_saved_file_is_valid_json(self, tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tok.json"
        tiny_tokenizer.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "special_tokens" in data
        assert "merges" in data
        assert "vocab" in data

    def test_load_roundtrip_vocab_size(self, tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tok.json"
        tiny_tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == tiny_tokenizer.vocab_size

    def test_load_roundtrip_special_tokens(
        self, tiny_tokenizer: BPETokenizer, tmp_path: Path
    ) -> None:
        path = tmp_path / "tok.json"
        tiny_tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.special_tokens == tiny_tokenizer.special_tokens

    def test_load_roundtrip_merges(self, tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tok.json"
        tiny_tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded._merges == tiny_tokenizer._merges

    def test_load_roundtrip_encoding_stable(
        self, tiny_tokenizer: BPETokenizer, tmp_path: Path
    ) -> None:
        """Loaded tokenizer must produce identical token IDs."""
        path = tmp_path / "tok.json"
        tiny_tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        for text in ["hello world", "the cat sat on the mat", "abracadabra"]:
            assert loaded.encode(text) == tiny_tokenizer.encode(text)

    def test_save_is_deterministic(self, tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        """Two saves of the same tokenizer produce identical bytes."""
        path1 = tmp_path / "tok1.json"
        path2 = tmp_path / "tok2.json"
        tiny_tokenizer.save(path1)
        tiny_tokenizer.save(path2)
        assert path1.read_bytes() == path2.read_bytes()

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BPETokenizer.load(tmp_path / "nonexistent.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{}", encoding="utf-8")
        with pytest.raises(ValueError):
            BPETokenizer.load(bad)

    def test_load_missing_key(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"special_tokens": [], "merges": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="vocab"):
            BPETokenizer.load(bad)
