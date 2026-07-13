"""Vocabulary management for the BPE tokenizer.

Responsibilities:
    - Store the mapping from token string → integer ID (and reverse)
    - Manage special tokens: <pad>, <bos>, <eos>, <unk>
    - Enforce that special tokens are never split by merge operations
    - Provide deterministic ID assignment (vocabulary ordering is stable
      across save/load cycles)

The ``Vocabulary`` class is used internally by ``BPETokenizer`` and is not
part of the primary public API, but it is importable for users who need
direct vocabulary inspection.

Key design choice — why special tokens are explicit:
    Many tokenizers silently handle special tokens through ad-hoc string
    matching. KAMUI's Vocabulary makes special tokens first-class: they
    are registered at construction time and guaranteed to have reserved IDs
    (0, 1, 2, ...) that never collide with learned BPE tokens. This makes
    the tokenisation pipeline fully auditable.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class Vocabulary:
    """Vocabulary management for the KAMUI BPE tokenizer.

    Maintains bidirectional mapping between string tokens and integer IDs,
    ensuring deterministic ID assignment, duplicate prevention, and stable
    serialization.
    """

    def __init__(self, special_tokens: Iterable[str] | None = None) -> None:
        """Initialise a new Vocabulary.

        Args:
            special_tokens: Optional iterable of special tokens. If None,
                defaults to ("<pad>", "<bos>", "<eos>", "<unk>").

        Raises:
            ValueError: If empty strings are provided as special tokens,
                or if there are duplicate special tokens.
            TypeError: If special tokens contains non-string elements.
        """
        if special_tokens is None:
            specials: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
        else:
            specials = tuple(special_tokens)

        self._special_tokens: set[str] = set()
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

        # Register special tokens
        for token in specials:
            if not isinstance(token, str):
                raise TypeError(f"Special token must be a string, got type {type(token)}")
            if token == "":
                raise ValueError("Special token cannot be an empty string")
            if token in self._special_tokens:
                raise ValueError(f"Duplicate special token registered: '{token}'")

            token_id = len(self._id_to_token)
            self._token_to_id[token] = token_id
            self._id_to_token.append(token)
            self._special_tokens.add(token)

    @property
    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self._id_to_token)

    @property
    def special_tokens(self) -> set[str]:
        """Return a copy of the set of registered special tokens."""
        return self._special_tokens.copy()

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its assigned integer ID.

        Args:
            token: The string token to add.

        Returns:
            The newly assigned integer ID of the token.

        Raises:
            ValueError: If the token is empty, already exists, or is a duplicate
                special token.
            TypeError: If the token is not a string.
        """
        if not isinstance(token, str):
            raise TypeError(f"Token must be a string, got type {type(token)}")
        if token == "":
            raise ValueError("Token cannot be an empty string")
        if token in self._token_to_id:
            raise ValueError(f"Token '{token}' already exists in the vocabulary")

        token_id = len(self._id_to_token)
        self._token_to_id[token] = token_id
        self._id_to_token.append(token)
        return token_id

    def add_tokens(self, tokens: Iterable[str]) -> None:
        """Add multiple tokens to the vocabulary.

        This operation is atomic. If any token in the input iterable is invalid
        (e.g., empty or duplicate), no tokens will be added.

        Args:
            tokens: Iterable of string tokens to add.

        Raises:
            ValueError: If any token is empty, already exists in the vocabulary,
                or is a duplicate within the input iterable.
            TypeError: If any token is not a string.
        """
        tokens_list = list(tokens)
        seen: set[str] = set()

        for token in tokens_list:
            if not isinstance(token, str):
                raise TypeError(f"Token must be a string, got type {type(token)}")
            if token == "":
                raise ValueError("Token cannot be an empty string")
            if token in self._token_to_id:
                raise ValueError(f"Token '{token}' already exists in the vocabulary")
            if token in seen:
                raise ValueError(f"Duplicate token in input iterable: '{token}'")
            seen.add(token)

        for token in tokens_list:
            self.add_token(token)

    def token_to_id(self, token: str) -> int:
        """Look up the integer ID of a string token.

        Args:
            token: The string token to look up.

        Returns:
            The integer ID.

        Raises:
            KeyError: If the token is not in the vocabulary.
            TypeError: If the token is not a string.
        """
        if not isinstance(token, str):
            raise TypeError(f"Token must be a string, got type {type(token)}")
        if token not in self._token_to_id:
            raise KeyError(f"Token '{token}' not found in vocabulary")
        return self._token_to_id[token]

    def id_to_token(self, token_id: int) -> str:
        """Look up the string representation of a token ID.

        Args:
            token_id: The integer ID to look up.

        Returns:
            The string token.

        Raises:
            ValueError: If the token ID is out of range.
            TypeError: If the token ID is not an integer.
        """
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"Token ID must be an integer, got type {type(token_id)}")
        if token_id < 0 or token_id >= len(self._id_to_token):
            raise ValueError(f"Token ID {token_id} is out of range [0, {len(self._id_to_token)})")
        return self._id_to_token[token_id]

    def __getitem__(self, key: str | int) -> int | str:
        """Look up a token string to get its ID, or a token ID to get its string.

        Args:
            key: Either a string token or an integer token ID.

        Returns:
            Integer ID if key is a string, or string token if key is an integer.

        Raises:
            KeyError: If a string key is not found.
            ValueError: If an integer key is out of range.
            TypeError: If key is neither a string nor an integer.
        """
        if isinstance(key, str):
            return self.token_to_id(key)
        elif isinstance(key, int) and not isinstance(key, bool):
            return self.id_to_token(key)
        else:
            raise TypeError(f"Key must be a string or an integer, got type {type(key)}")

    def __contains__(self, item: str | int) -> bool:
        """Check if a token or ID is in the vocabulary.

        Args:
            item: Either a string token or an integer token ID.

        Returns:
            True if the token or ID exists in the vocabulary, False otherwise.
        """
        if isinstance(item, str):
            return item in self._token_to_id
        elif isinstance(item, int) and not isinstance(item, bool):
            return 0 <= item < len(self._id_to_token)
        return False

    def contains(self, token: str) -> bool:
        """Return True if *token* is present in the vocabulary.

        This is the named method form of ``token in vocabulary``.  It only
        accepts string tokens; use the ``in`` operator to check by integer ID.

        Args:
            token: The string token to look up.

        Returns:
            True if the token exists in the vocabulary, False otherwise.

        Raises:
            TypeError: If *token* is not a string.
        """
        if not isinstance(token, str):
            raise TypeError(f"Token must be a string, got type {type(token)}")
        return token in self._token_to_id

    def __len__(self) -> int:
        """Return the total number of tokens (same as ``vocab_size``)."""
        return len(self._id_to_token)

    def __repr__(self) -> str:
        """Return a human-readable summary of the vocabulary."""
        return (
            f"Vocabulary("
            f"vocab_size={self.vocab_size}, "
            f"special_tokens={sorted(self._special_tokens, key=lambda t: self._token_to_id[t])}"
            f")"
        )

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to a JSON file.

        Args:
            path: Path to the target JSON file.

        Raises:
            IOError: If writing to the file fails.
        """
        # Ensure special tokens are serialized in their ID order
        sorted_specials = sorted(list(self._special_tokens), key=lambda x: self._token_to_id[x])

        data = {
            "special_tokens": sorted_specials,
            "vocab": {token: token_id for token, token_id in self._token_to_id.items()},
        }

        path_obj = Path(path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise OSError(f"Failed to save vocabulary to {path}: {e}") from e

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":  # noqa: C901 — defensive JSON validation
        """Load a vocabulary from a JSON file.

        Args:
            path: Path to the JSON file to load.

        Returns:
            A new Vocabulary instance reconstructed from the file.

        Raises:
            ValueError: If the JSON format is invalid, IDs are not consecutive
                starting from 0, or special tokens are invalid or missing.
            FileNotFoundError: If the vocabulary file does not exist.
            IOError: If reading the file fails.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        try:
            with open(path_obj, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in vocabulary file {path}: {e}") from e
        except Exception as e:
            raise OSError(f"Failed to read vocabulary from {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Vocabulary file must contain a JSON object at the top level")

        if "vocab" not in data or "special_tokens" not in data:
            raise ValueError("Vocabulary file must contain 'vocab' and 'special_tokens' keys")

        special_tokens_list = data["special_tokens"]
        vocab_dict = data["vocab"]

        if not isinstance(special_tokens_list, list):
            raise ValueError("'special_tokens' must be a list of strings")
        for t in special_tokens_list:
            if not isinstance(t, str):
                raise ValueError("All elements in 'special_tokens' must be strings")

        if not isinstance(vocab_dict, dict):
            raise ValueError("'vocab' must be a dictionary")

        # Sort vocab items by ID to verify they are consecutive and start from 0
        sorted_vocab: list[tuple[Any, Any]] = sorted(vocab_dict.items(), key=lambda item: item[1])

        for idx, (token, token_id) in enumerate(sorted_vocab):
            if not isinstance(token, str):
                raise ValueError(f"Vocabulary token must be a string, got type {type(token)}")
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(
                    f"Vocabulary token ID must be an integer, got type {type(token_id)}"
                )
            if token_id != idx:
                raise ValueError(
                    f"Vocabulary IDs must be consecutive starting from 0. "
                    f"Expected {idx}, got {token_id}"
                )

        # Verify all listed special tokens are actually in the vocabulary
        for token in special_tokens_list:
            if token not in vocab_dict:
                raise ValueError(f"Special token '{token}' not found in vocabulary mapping")

        # Reconstruct Vocabulary
        instance = cls(special_tokens=special_tokens_list)

        for token, token_id in sorted_vocab:
            if token in instance._special_tokens:
                # Ensure the ID assigned at constructor matches the serialized ID
                if instance.token_to_id(token) != token_id:
                    raise ValueError(
                        f"Special token '{token}' ID mismatch: expected {token_id}, "
                        f"got {instance.token_to_id(token)}"
                    )
            else:
                assigned_id = instance.add_token(token)
                if assigned_id != token_id:
                    raise ValueError(
                        f"Failed to load token '{token}' at expected ID {token_id}, "
                        f"assigned ID {assigned_id}"
                    )

        return instance
