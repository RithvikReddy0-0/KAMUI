"""Unit tests for kamui.tokenizer.vocab.Vocabulary.

Tests validate:
    - Initialization with default and custom special tokens
    - Token → id lookup (token_to_id and __getitem__)
    - Id → token lookup (id_to_token and __getitem__)
    - contains() named method and __contains__ operator
    - add_token() — valid insertion and all error paths
    - add_tokens() — batch insertion, atomicity, and error paths
    - Special token registration guarantees
    - save() / load() roundtrip (Path and str)
    - Serialization stability (identical bytes on repeated save)
    - Invalid access raises correct exceptions
    - __len__, __repr__, vocab_size, special_tokens properties
    - load() rejects malformed / inconsistent files
"""

import json
from pathlib import Path

import pytest

from kamui.tokenizer.vocab import Vocabulary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")


def make_vocab(extra_tokens: list[str] | None = None) -> Vocabulary:
    """Return a default Vocabulary, optionally with extra tokens pre-added."""
    v = Vocabulary()
    if extra_tokens:
        v.add_tokens(extra_tokens)
    return v


# ===========================================================================
# Initialization
# ===========================================================================


class TestInitialization:
    """Tests for Vocabulary.__init__."""

    def test_default_specials_assigned_in_order(self) -> None:
        """Default special tokens get IDs 0, 1, 2, 3 in declaration order."""
        v = Vocabulary()
        assert v.token_to_id("<pad>") == 0
        assert v.token_to_id("<bos>") == 1
        assert v.token_to_id("<eos>") == 2
        assert v.token_to_id("<unk>") == 3

    def test_default_vocab_size_is_four(self) -> None:
        """A freshly constructed default Vocabulary has exactly 4 tokens."""
        v = Vocabulary()
        assert v.vocab_size == 4

    def test_custom_special_tokens(self) -> None:
        """Custom special tokens receive IDs 0, 1, … in the supplied order."""
        v = Vocabulary(special_tokens=["<s>", "</s>"])
        assert v.token_to_id("<s>") == 0
        assert v.token_to_id("</s>") == 1
        assert v.vocab_size == 2

    def test_empty_special_token_raises(self) -> None:
        """An empty string in special_tokens must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Vocabulary(special_tokens=["<pad>", ""])

    def test_duplicate_special_token_raises(self) -> None:
        """Duplicate entries in special_tokens must raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate special token"):
            Vocabulary(special_tokens=["<pad>", "<pad>"])

    def test_non_string_special_token_raises(self) -> None:
        """Non-string entry in special_tokens must raise TypeError."""
        with pytest.raises(TypeError):
            Vocabulary(special_tokens=[123])  # type: ignore[list-item]

    def test_special_tokens_property_returns_set(self) -> None:
        """special_tokens property returns the correct set of special tokens."""
        v = Vocabulary()
        assert v.special_tokens == set(DEFAULT_SPECIALS)

    def test_special_tokens_property_is_copy(self) -> None:
        """Mutating the returned set must not affect the internal state."""
        v = Vocabulary()
        sp = v.special_tokens
        sp.add("intruder")
        assert "intruder" not in v.special_tokens

    def test_no_special_tokens(self) -> None:
        """An empty special_tokens iterable produces an empty vocabulary."""
        v = Vocabulary(special_tokens=[])
        assert v.vocab_size == 0
        assert v.special_tokens == set()


# ===========================================================================
# vocab_size and __len__
# ===========================================================================


class TestVocabSize:
    """Tests for vocab_size property and __len__."""

    def test_vocab_size_grows_with_add_token(self) -> None:
        """vocab_size increments by 1 for each add_token call."""
        v = Vocabulary()
        assert v.vocab_size == 4
        v.add_token("hello")
        assert v.vocab_size == 5
        v.add_token("world")
        assert v.vocab_size == 6

    def test_len_equals_vocab_size(self) -> None:
        """len(vocab) must equal vocab.vocab_size at all times."""
        v = make_vocab(["foo", "bar", "baz"])
        assert len(v) == v.vocab_size == 7


# ===========================================================================
# Token → ID lookup
# ===========================================================================


class TestTokenToId:
    """Tests for token_to_id() and __getitem__(str)."""

    def test_special_token_lookup(self) -> None:
        """Special tokens are accessible via token_to_id."""
        v = Vocabulary()
        assert v.token_to_id("<pad>") == 0

    def test_regular_token_lookup(self) -> None:
        """Tokens added via add_token are retrievable by name."""
        v = make_vocab(["hello"])
        assert v.token_to_id("hello") == 4

    def test_missing_token_raises_key_error(self) -> None:
        """Looking up an absent token must raise KeyError."""
        v = Vocabulary()
        with pytest.raises(KeyError):
            v.token_to_id("nonexistent")

    def test_non_string_raises_type_error(self) -> None:
        """Passing a non-string to token_to_id must raise TypeError."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v.token_to_id(42)  # type: ignore[arg-type]

    def test_getitem_with_string(self) -> None:
        """v['token'] must return the same result as v.token_to_id('token')."""
        v = make_vocab(["cat"])
        assert v["cat"] == v.token_to_id("cat")

    def test_getitem_wrong_type_raises(self) -> None:
        """__getitem__ with an unsupported type must raise TypeError."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v[3.14]  # type: ignore[index]


# ===========================================================================
# ID → Token lookup
# ===========================================================================


class TestIdToToken:
    """Tests for id_to_token() and __getitem__(int)."""

    def test_reverse_lookup_special_token(self) -> None:
        """id_to_token returns the correct string for special token IDs."""
        v = Vocabulary()
        assert v.id_to_token(0) == "<pad>"
        assert v.id_to_token(1) == "<bos>"
        assert v.id_to_token(2) == "<eos>"
        assert v.id_to_token(3) == "<unk>"

    def test_reverse_lookup_regular_token(self) -> None:
        """id_to_token returns the correct string for a regular token."""
        v = make_vocab(["hello"])
        assert v.id_to_token(4) == "hello"

    def test_negative_id_raises_value_error(self) -> None:
        """Negative IDs must raise ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="out of range"):
            v.id_to_token(-1)

    def test_out_of_bounds_id_raises_value_error(self) -> None:
        """An ID equal to vocab_size must raise ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="out of range"):
            v.id_to_token(v.vocab_size)

    def test_non_integer_id_raises_type_error(self) -> None:
        """Passing a float to id_to_token must raise TypeError."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v.id_to_token(0.0)  # type: ignore[arg-type]

    def test_bool_id_raises_type_error(self) -> None:
        """Booleans must not be accepted as token IDs (bool is subclass of int)."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v.id_to_token(True)  # type: ignore[arg-type]

    def test_getitem_with_int(self) -> None:
        """v[id] must return the same result as v.id_to_token(id)."""
        v = make_vocab(["dog"])
        assert v[4] == v.id_to_token(4) == "dog"


# ===========================================================================
# contains()
# ===========================================================================


class TestContains:
    """Tests for contains() named method and __contains__ operator."""

    def test_contains_returns_true_for_present_token(self) -> None:
        """contains() returns True for a token that exists."""
        v = make_vocab(["hello"])
        assert v.contains("hello") is True

    def test_contains_returns_false_for_absent_token(self) -> None:
        """contains() returns False for a token that is not in the vocabulary."""
        v = Vocabulary()
        assert v.contains("ghost") is False

    def test_contains_special_token(self) -> None:
        """contains() works for special tokens."""
        v = Vocabulary()
        assert v.contains("<pad>") is True

    def test_contains_non_string_raises(self) -> None:
        """contains() raises TypeError for non-string input."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v.contains(0)  # type: ignore[arg-type]

    def test_in_operator_with_string(self) -> None:
        """The ``in`` operator works for string membership checks."""
        v = make_vocab(["cat"])
        assert "cat" in v
        assert "dog" not in v

    def test_in_operator_with_int(self) -> None:
        """The ``in`` operator works for integer ID membership checks."""
        v = make_vocab(["cat"])
        assert 0 in v  # <pad>
        assert 4 in v  # cat
        assert 5 not in v  # beyond range

    def test_in_operator_negative_int(self) -> None:
        """Negative integers are not in the vocabulary."""
        v = Vocabulary()
        assert -1 not in v

    def test_in_operator_unsupported_type_returns_false(self) -> None:
        """__contains__ returns False for unsupported types (no crash)."""
        v = Vocabulary()
        assert (3.14 in v) is False  # type: ignore[operator]


# ===========================================================================
# add_token()
# ===========================================================================


class TestAddToken:
    """Tests for add_token()."""

    def test_add_token_returns_id(self) -> None:
        """add_token returns the newly assigned integer ID."""
        v = Vocabulary()
        token_id = v.add_token("hello")
        assert token_id == 4  # after 4 default specials

    def test_add_token_ids_are_consecutive(self) -> None:
        """Multiple calls to add_token assign consecutive IDs."""
        v = Vocabulary()
        for i, word in enumerate(["a", "b", "c"]):
            assert v.add_token(word) == 4 + i

    def test_add_token_empty_string_raises(self) -> None:
        """Adding an empty string must raise ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="empty"):
            v.add_token("")

    def test_add_token_duplicate_raises(self) -> None:
        """Re-adding an existing token must raise ValueError."""
        v = Vocabulary()
        v.add_token("hello")
        with pytest.raises(ValueError, match="already exists"):
            v.add_token("hello")

    def test_add_token_duplicate_special_raises(self) -> None:
        """Attempting to add a special token via add_token must raise ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="already exists"):
            v.add_token("<pad>")

    def test_add_token_non_string_raises(self) -> None:
        """add_token rejects non-string input with TypeError."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v.add_token(42)  # type: ignore[arg-type]

    def test_add_token_unicode(self) -> None:
        """Unicode tokens (emoji, CJK, etc.) are accepted without error."""
        v = Vocabulary()
        tid = v.add_token("北京")
        assert v.id_to_token(tid) == "北京"
        assert v.token_to_id("北京") == tid


# ===========================================================================
# add_tokens()
# ===========================================================================


class TestAddTokens:
    """Tests for add_tokens()."""

    def test_add_tokens_batch(self) -> None:
        """All tokens in a valid batch are inserted with correct IDs."""
        v = Vocabulary()
        v.add_tokens(["a", "b", "c"])
        assert v.token_to_id("a") == 4
        assert v.token_to_id("b") == 5
        assert v.token_to_id("c") == 6
        assert v.vocab_size == 7

    def test_add_tokens_empty_iterable(self) -> None:
        """An empty iterable is a no-op."""
        v = Vocabulary()
        v.add_tokens([])
        assert v.vocab_size == 4

    def test_add_tokens_empty_string_raises(self) -> None:
        """An empty string anywhere in the batch raises ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="empty"):
            v.add_tokens(["good", ""])

    def test_add_tokens_duplicate_in_batch_raises(self) -> None:
        """Duplicate within the input iterable raises ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="Duplicate token"):
            v.add_tokens(["a", "b", "a"])

    def test_add_tokens_duplicate_existing_raises(self) -> None:
        """A token already in the vocabulary raises ValueError."""
        v = make_vocab(["hello"])
        with pytest.raises(ValueError, match="already exists"):
            v.add_tokens(["world", "hello"])

    def test_add_tokens_is_atomic_on_error(self) -> None:
        """If add_tokens raises, no tokens from the batch should be added."""
        v = Vocabulary()
        initial_size = v.vocab_size
        with pytest.raises(ValueError):
            v.add_tokens(["alpha", "beta", "alpha"])
        # Vocabulary must be unchanged
        assert v.vocab_size == initial_size
        assert "alpha" not in v
        assert "beta" not in v

    def test_add_tokens_non_string_raises(self) -> None:
        """A non-string element in the batch raises TypeError."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v.add_tokens(["good", 999])  # type: ignore[list-item]

    def test_add_tokens_accepts_generator(self) -> None:
        """add_tokens accepts any iterable (generator, not just list)."""
        v = Vocabulary()
        v.add_tokens(x for x in ["x", "y", "z"])
        assert v.contains("x")
        assert v.contains("y")
        assert v.contains("z")


# ===========================================================================
# Special tokens
# ===========================================================================


class TestSpecialTokens:
    """Tests specific to special-token semantics."""

    def test_special_tokens_receive_lowest_ids(self) -> None:
        """Special tokens always occupy IDs 0…N-1."""
        v = Vocabulary(special_tokens=["<s>", "</s>"])
        v.add_tokens(["hello", "world"])
        assert v.token_to_id("<s>") == 0
        assert v.token_to_id("</s>") == 1
        assert v.token_to_id("hello") == 2
        assert v.token_to_id("world") == 3

    def test_special_tokens_returned_by_property(self) -> None:
        """Only the originally declared tokens appear in the special_tokens set."""
        v = Vocabulary(special_tokens=["<a>", "<b>"])
        v.add_token("regular")
        assert v.special_tokens == {"<a>", "<b>"}
        assert "regular" not in v.special_tokens

    def test_all_default_specials_present(self) -> None:
        """All four default special tokens are present."""
        v = Vocabulary()
        for tok in DEFAULT_SPECIALS:
            assert v.contains(tok)


# ===========================================================================
# save() / load() roundtrip
# ===========================================================================


class TestSaveLoad:
    """Tests for Vocabulary.save() and Vocabulary.load()."""

    def test_roundtrip_default_vocab(self, tmp_path: Path) -> None:
        """A default Vocabulary survives a save/load roundtrip."""
        v = Vocabulary()
        path = tmp_path / "vocab.json"
        v.save(path)
        loaded = Vocabulary.load(path)

        assert loaded.vocab_size == v.vocab_size
        assert loaded.special_tokens == v.special_tokens
        for tok in DEFAULT_SPECIALS:
            assert loaded.token_to_id(tok) == v.token_to_id(tok)

    def test_roundtrip_with_extra_tokens(self, tmp_path: Path) -> None:
        """Added tokens survive a save/load roundtrip with correct IDs."""
        v = make_vocab(["hello", "world", "北京", "🤗"])
        path = tmp_path / "vocab.json"
        v.save(path)
        loaded = Vocabulary.load(path)

        assert loaded.vocab_size == v.vocab_size
        for tok in ["hello", "world", "北京", "🤗"]:
            assert loaded.token_to_id(tok) == v.token_to_id(tok)

    def test_roundtrip_with_str_path(self, tmp_path: Path) -> None:
        """save/load accept str paths as well as Path objects."""
        v = make_vocab(["a"])
        path_str = str(tmp_path / "vocab_str.json")
        v.save(path_str)
        loaded = Vocabulary.load(path_str)
        assert loaded.vocab_size == v.vocab_size

    def test_roundtrip_custom_specials(self, tmp_path: Path) -> None:
        """Custom special tokens survive a save/load roundtrip."""
        v = Vocabulary(special_tokens=["<s>", "</s>"])
        v.add_tokens(["cat", "dog"])
        path = tmp_path / "custom.json"
        v.save(path)
        loaded = Vocabulary.load(path)

        assert loaded.special_tokens == {"<s>", "</s>"}
        assert loaded.token_to_id("<s>") == 0
        assert loaded.token_to_id("cat") == 2

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Vocabulary.load(tmp_path / "ghost.json")

    def test_load_malformed_json(self, tmp_path: Path) -> None:
        """A file with invalid JSON raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            Vocabulary.load(bad)

    def test_load_missing_vocab_key(self, tmp_path: Path) -> None:
        """A JSON file missing the 'vocab' key raises ValueError."""
        bad = tmp_path / "no_vocab.json"
        bad.write_text(json.dumps({"special_tokens": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            Vocabulary.load(bad)

    def test_load_missing_special_tokens_key(self, tmp_path: Path) -> None:
        """A JSON file missing 'special_tokens' raises ValueError."""
        bad = tmp_path / "no_specials.json"
        bad.write_text(json.dumps({"vocab": {}}), encoding="utf-8")
        with pytest.raises(ValueError):
            Vocabulary.load(bad)

    def test_load_top_level_not_dict(self, tmp_path: Path) -> None:
        """A JSON array at the top level raises ValueError."""
        bad = tmp_path / "array.json"
        bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError):
            Vocabulary.load(bad)

    def test_load_non_consecutive_ids(self, tmp_path: Path) -> None:
        """Vocabulary IDs that skip a number are rejected."""
        bad = tmp_path / "bad_ids.json"
        bad.write_text(
            json.dumps({"special_tokens": ["<a>"], "vocab": {"<a>": 0, "b": 2}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            Vocabulary.load(bad)

    def test_load_special_token_not_in_vocab_dict(self, tmp_path: Path) -> None:
        """A special token listed but absent from 'vocab' raises ValueError."""
        bad = tmp_path / "missing_special.json"
        bad.write_text(
            json.dumps({"special_tokens": ["<ghost>"], "vocab": {"<pad>": 0}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            Vocabulary.load(bad)

    def test_load_non_integer_id_raises(self, tmp_path: Path) -> None:
        """A string value in 'vocab' raises ValueError."""
        bad = tmp_path / "string_id.json"
        bad.write_text(
            json.dumps({"special_tokens": [], "vocab": {"hello": "zero"}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            Vocabulary.load(bad)


# ===========================================================================
# Serialization stability
# ===========================================================================


class TestSerializationStability:
    """Serialized output must be identical across multiple saves."""

    def test_identical_bytes_on_repeated_save(self, tmp_path: Path) -> None:
        """Saving the same Vocabulary twice produces byte-identical files."""
        v = make_vocab(["alpha", "beta", "gamma"])
        path1 = tmp_path / "vocab1.json"
        path2 = tmp_path / "vocab2.json"
        v.save(path1)
        v.save(path2)
        assert path1.read_bytes() == path2.read_bytes()

    def test_id_stability_after_load(self, tmp_path: Path) -> None:
        """Token IDs are identical before and after a save/load cycle."""
        tokens = ["alpha", "beta", "gamma", "delta"]
        v = make_vocab(tokens)
        original_ids = {tok: v.token_to_id(tok) for tok in tokens}

        path = tmp_path / "stable.json"
        v.save(path)
        loaded = Vocabulary.load(path)

        for tok, expected_id in original_ids.items():
            assert loaded.token_to_id(tok) == expected_id

    def test_special_token_id_stability(self, tmp_path: Path) -> None:
        """Special token IDs are stable across repeated save/load cycles."""
        v = Vocabulary()
        path = tmp_path / "specials.json"

        for _ in range(3):
            v.save(path)
            v = Vocabulary.load(path)

        for tok in DEFAULT_SPECIALS:
            assert v.token_to_id(tok) == DEFAULT_SPECIALS.index(tok)

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """save() creates missing parent directories automatically."""
        v = Vocabulary()
        path = tmp_path / "subdir" / "deep" / "vocab.json"
        v.save(path)
        assert path.exists()


# ===========================================================================
# __repr__
# ===========================================================================


class TestRepr:
    """Tests for Vocabulary.__repr__."""

    def test_repr_contains_vocab_size(self) -> None:
        """__repr__ mentions vocab_size."""
        v = Vocabulary()
        assert "vocab_size=4" in repr(v)

    def test_repr_contains_special_tokens(self) -> None:
        """__repr__ mentions special tokens."""
        v = Vocabulary()
        r = repr(v)
        assert "special_tokens" in r
        assert "<pad>" in r

    def test_repr_custom_specials(self) -> None:
        """__repr__ reflects custom special tokens."""
        v = Vocabulary(special_tokens=["<s>", "</s>"])
        assert "<s>" in repr(v)


# ===========================================================================
# Invalid access patterns
# ===========================================================================


class TestInvalidAccess:
    """Edge cases that must raise correct exceptions."""

    def test_getitem_bool_raises(self) -> None:
        """__getitem__ with a bool raises TypeError (bool is subclass of int)."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v[True]  # type: ignore[index]

    def test_getitem_float_raises(self) -> None:
        """__getitem__ with a float raises TypeError."""
        v = Vocabulary()
        with pytest.raises(TypeError):
            v[1.0]  # type: ignore[index]

    def test_id_to_token_large_id_raises(self) -> None:
        """id_to_token with a very large ID raises ValueError."""
        v = Vocabulary()
        with pytest.raises(ValueError, match="out of range"):
            v.id_to_token(9999)
