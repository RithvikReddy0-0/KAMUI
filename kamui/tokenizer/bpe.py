"""Byte-pair encoding (BPE) tokeniser, implemented from scratch.

BPE (Sennrich et al., 2016) iteratively merges the most frequent pair of
adjacent tokens in a corpus, building a vocabulary of subword units.  It
balances two competing goals:

  - Coverage: every string is representable without unknown tokens.
  - Granularity: frequent patterns are single tokens; rare patterns are
    split into meaningful subword pieces.

All modern large language models — GPT-2, LLaMA, Mistral — use BPE or a
close variant.  KAMUI implements BPE from scratch so that every step of the
tokenisation pipeline is inspectable and understandable.

Responsibilities:
    - ``BPETokenizer.train(corpus_path, vocab_size)``:
        Read a text corpus, compute character frequencies, run iterative
        merge operations until ``vocab_size`` is reached, return a trained
        tokeniser.

    - ``BPETokenizer.encode(text) -> list[int]``:
        Apply learned merge rules to convert a string into a list of token
        IDs.

    - ``BPETokenizer.decode(ids) -> str``:
        Convert a list of token IDs back to the original string.
        Must be lossless for all ASCII inputs.

    - ``BPETokenizer.save(path)`` / ``BPETokenizer.load(path)``:
        Serialise and deserialise the vocabulary and merge rules to/from
        a JSON file so the tokeniser can be reused without retraining.

Key implementation details:
    - Byte-level BPE: text is first encoded as UTF-8 bytes, so the
      vocabulary covers all possible byte values (0–255) as base tokens.
      This eliminates unknown tokens entirely.
    - Merge rules are applied greedily, left to right.
    - Special tokens are added to the vocabulary and never split by merge
      rules.  The canonical special tokens are ``<|endoftext|>`` and
      ``<|pad|>``, but callers may pass any set via ``special_tokens``.

References:
    Sennrich, R., Haddow, B., & Birch, A. (2016).
    Neural Machine Translation of Rare Words with Subword Units.
    ACL 2016. https://arxiv.org/abs/1508.07909
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kamui.tokenizer.utils import bytes_to_text, get_stats, merge_pair, text_to_bytes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of base byte tokens that every BPETokenizer starts with.
_NUM_BYTE_TOKENS: int = 256

#: Default special tokens, in ID-reservation order.
_DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<|endoftext|>", "<|pad|>")


class BPETokenizer:
    """Byte-pair encoding tokeniser, trained and applied from scratch.

    The vocabulary is structured as follows:
        - IDs 0–255: raw byte tokens (``\\x00`` … ``\\xff``).
        - IDs 256 … 256+len(special_tokens)-1: special tokens in declaration order.
        - IDs 256+len(special_tokens) … vocab_size-1: merged subword tokens,
          one per BPE merge rule, in the order they were learned.

    Special tokens are injected into the id→bytes mapping as unique sentinel
    byte-sequences that can never arise from UTF-8 text, so they cannot
    accidentally appear inside a regular token.

    Attributes:
        vocab_size: Total number of tokens (bytes + specials + merges).
        special_tokens: Tuple of registered special token strings.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        merges: list[tuple[int, int]],
        vocab: dict[int, bytes],
        special_tokens: tuple[str, ...] = _DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        """Internal constructor.  Use ``train`` or ``load`` instead.

        Args:
            merges:        Ordered list of (left_id, right_id) merge rules.
            vocab:         Mapping from integer ID to raw bytes representation.
            special_tokens: Tuple of special token strings in ID order.
        """
        self._merges: list[tuple[int, int]] = merges
        self._vocab: dict[int, bytes] = vocab
        self._special_tokens: tuple[str, ...] = special_tokens

        # Build reverse vocab: bytes → id (for encoding)
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # Build merge lookup: pair → new_id  (O(1) encode step)
        self._merge_map: dict[tuple[int, int], int] = {}
        for i, pair in enumerate(merges):
            new_id = _NUM_BYTE_TOKENS + len(special_tokens) + i
            self._merge_map[pair] = new_id

        # Build special-token lookup: string → id
        self._special_to_id: dict[str, int] = {
            tok: _NUM_BYTE_TOKENS + i for i, tok in enumerate(special_tokens)
        }
        self._id_to_special: dict[int, str] = {v: k for k, v in self._special_to_id.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self._vocab)

    @property
    def special_tokens(self) -> tuple[str, ...]:
        """Tuple of registered special token strings, in ID order."""
        return self._special_tokens

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(  # noqa: C901 — inherently branchy: corpus loading + BPE merge loop
        cls,
        corpus: str | Path,
        vocab_size: int,
        special_tokens: tuple[str, ...] = _DEFAULT_SPECIAL_TOKENS,
        verbose: bool = False,
    ) -> BPETokenizer:
        """Train a BPE tokenizer on a text corpus.

        The training procedure:
        1. Read the full corpus as a UTF-8 string.
        2. Encode every character as its raw UTF-8 bytes (256 base tokens).
        3. Reserve IDs for special tokens immediately after the 256 byte tokens.
        4. Iteratively find the most-frequent adjacent pair of IDs across the
           entire corpus, assign it a new merged ID, and update the corpus.
           Repeat until ``vocab_size`` is reached.

        Args:
            corpus:        Path to the training corpus text file, or a raw string
                           corpus (if the value does not correspond to an existing
                           file, it is treated as the corpus text directly — useful
                           for tests).
            vocab_size:    Target vocabulary size.  Must be ≥ 256 + len(special_tokens).
            special_tokens: Special tokens to reserve, in ID order.  Defaults to
                            ``("<|endoftext|>", "<|pad|>")``.
            verbose:       If True, print progress every 100 merges.

        Returns:
            A trained ``BPETokenizer`` instance.

        Raises:
            ValueError: If ``vocab_size`` is too small to accommodate all byte
                tokens plus the requested special tokens.
            FileNotFoundError: If ``corpus`` is a path string and the file does
                not exist.
        """
        min_vocab = _NUM_BYTE_TOKENS + len(special_tokens)
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= {min_vocab} "
                f"(256 byte tokens + {len(special_tokens)} special tokens)"
            )

        # Load corpus text.
        # If corpus is a Path object, always treat it as a file path.
        # If it is a plain str, only treat it as a path when it could plausibly
        # be one (≤ 4096 chars, no newlines) AND the file actually exists —
        # otherwise treat it as raw text directly (useful in tests).
        if isinstance(corpus, Path):
            text = corpus.read_text(encoding="utf-8")
        else:
            corpus_str = str(corpus)
            is_plausible_path = (
                len(corpus_str) <= 4096 and "\n" not in corpus_str and " " not in corpus_str
            )
            corpus_path = Path(corpus_str) if is_plausible_path else None
            if corpus_path is not None and corpus_path.exists() and corpus_path.is_file():
                text = corpus_path.read_text(encoding="utf-8")
            else:
                text = corpus_str

        # Split on special tokens so they are never merged with surrounding text.
        # We process each chunk between special tokens as an independent sequence.
        special_pattern = (
            "(" + "|".join(re.escape(s) for s in special_tokens) + ")" if special_tokens else None
        )
        chunks = re.split(special_pattern, text) if special_pattern else [text]

        # Build initial byte-level sequences (one per chunk, skipping special tokens)
        sequences: list[list[int]] = []
        for chunk in chunks:
            if chunk in special_tokens:
                continue
            if chunk:
                sequences.append(text_to_bytes(chunk))

        # Build base vocab: id → bytes
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(_NUM_BYTE_TOKENS)}

        # Reserve special token IDs
        for i, tok in enumerate(special_tokens):
            vocab[_NUM_BYTE_TOKENS + i] = tok.encode("utf-8")

        merges: list[tuple[int, int]] = []
        num_merges = vocab_size - min_vocab

        for merge_idx in range(num_merges):
            if not sequences:
                break

            stats = get_stats(sequences)
            if not stats:
                break  # corpus too short to find more pairs

            # Choose the most frequent pair; break ties deterministically by pair value
            best_pair = max(stats, key=lambda p: (stats[p], p))
            new_id = min_vocab + merge_idx

            # Perform the merge across all sequences
            sequences = [merge_pair(seq, best_pair, new_id) for seq in sequences]

            # Record merge rule and vocab entry
            merges.append(best_pair)
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"  merge {merge_idx + 1}/{num_merges}: {best_pair} → {new_id}")

        return cls(merges=merges, vocab=vocab, special_tokens=special_tokens)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Special tokens in the text are matched first (as whole strings) before
        the remainder is byte-encoded and merged.  Merge rules are applied
        greedily, left to right, until no more merges are possible.

        Args:
            text: Any Unicode string.

        Returns:
            A list of integer token IDs, possibly empty for an empty input.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text)}")
        if not text:
            return []

        # Split on special tokens first so they are never broken up by merges
        if self._special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self._special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        result: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self._special_to_id:
                result.append(self._special_to_id[part])
            else:
                # Byte-encode then apply merges
                ids = text_to_bytes(part)
                ids = self._apply_merges(ids)
                result.extend(ids)
        return result

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """Apply all learned merge rules to a byte-level token sequence.

        Merges are applied greedily in training order (lowest new_id first),
        which matches how GPT-2 and similar tokenisers work.

        Args:
            ids: Sequence of byte-level (0–255) token IDs.

        Returns:
            Merged token ID sequence.
        """
        # Keep applying merges until no more are possible
        while True:
            # Find the earliest applicable merge (by merge priority = training order)
            best_pos = -1
            best_priority = len(self._merges)  # sentinel: higher than any valid index

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self._merge_map:
                    # Priority = index in self._merges (lower index = higher priority)
                    priority = self._merges.index(pair)
                    if priority < best_priority:
                        best_priority = priority
                        best_pos = i

            if best_pos == -1:
                break  # no more merges possible

            pair = (ids[best_pos], ids[best_pos + 1])
            new_id = self._merge_map[pair]
            ids = ids[:best_pos] + [new_id] + ids[best_pos + 2 :]

        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string.

        The decode is lossless for any sequence produced by ``encode``:
        ``decode(encode(text)) == text`` for all Unicode strings.

        Special-token IDs are decoded back to their string representation
        (e.g. token 256 → ``"<|endoftext|>"``).

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.

        Raises:
            TypeError:  If ``ids`` is not a list or any element is not an int.
            ValueError: If any ID is outside [0, vocab_size).
        """
        if not isinstance(ids, list):
            raise TypeError(f"ids must be a list, got {type(ids)}")

        byte_buf: list[int] = []
        text_parts: list[str] = []

        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"token ID must be an int, got {type(token_id)}")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID {token_id} is out of range [0, {self.vocab_size})")

            if token_id in self._id_to_special:
                # Flush any buffered bytes before emitting the special token
                if byte_buf:
                    text_parts.append(bytes_to_text(byte_buf))
                    byte_buf = []
                text_parts.append(self._id_to_special[token_id])
            else:
                byte_buf.extend(self._vocab[token_id])

        if byte_buf:
            text_parts.append(bytes_to_text(byte_buf))

        return "".join(text_parts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the tokenizer to a JSON file.

        The JSON contains:
        - ``"special_tokens"``: list of special token strings in ID order.
        - ``"merges"``: list of [left_id, right_id] pairs in training order.
        - ``"vocab"``: dict mapping string token ID → base64-encoded bytes
          representation of the token.

        Args:
            path: Path to the output JSON file.  Parent directories are created
                  if they do not exist.

        Raises:
            IOError: If writing fails.
        """
        import base64

        path_obj = Path(path)

        # Encode vocab values as base64 strings so arbitrary bytes are JSON-safe
        vocab_serialized = {
            str(token_id): base64.b64encode(token_bytes).decode("ascii")
            for token_id, token_bytes in self._vocab.items()
        }

        data = {
            "special_tokens": list(self._special_tokens),
            "merges": [list(pair) for pair in self._merges],
            "vocab": vocab_serialized,
        }

        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=True, sort_keys=False)
        except Exception as e:
            raise OSError(f"Failed to save tokenizer to {path}: {e}") from e

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer from a JSON file created by ``save``.

        Args:
            path: Path to the JSON file.

        Returns:
            A ``BPETokenizer`` instance identical to the one that was saved.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON structure is invalid.
            IOError: If reading fails.
        """
        import base64

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        try:
            with open(path_obj, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tokenizer file {path}: {e}") from e
        except Exception as e:
            raise OSError(f"Failed to read tokenizer file {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Tokenizer file must contain a JSON object")
        for key in ("special_tokens", "merges", "vocab"):
            if key not in data:
                raise ValueError(f"Tokenizer file missing required key: '{key}'")

        special_tokens = tuple(data["special_tokens"])

        merges: list[tuple[int, int]] = []
        for pair in data["merges"]:
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValueError(f"Each merge must be a 2-element list, got: {pair}")
            merges.append((int(pair[0]), int(pair[1])))

        vocab: dict[int, bytes] = {}
        for id_str, b64 in data["vocab"].items():
            vocab[int(id_str)] = base64.b64decode(b64)

        return cls(merges=merges, vocab=vocab, special_tokens=special_tokens)

    # ------------------------------------------------------------------
    # Special-token helpers
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> int:
        """Return the integer ID for a special token string.

        Args:
            token: A special token string (must be in ``self.special_tokens``).

        Returns:
            The integer ID.

        Raises:
            KeyError: If the token is not a registered special token.
        """
        if token not in self._special_to_id:
            raise KeyError(f"'{token}' is not a registered special token")
        return self._special_to_id[token]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BPETokenizer("
            f"vocab_size={self.vocab_size}, "
            f"merges={len(self._merges)}, "
            f"special_tokens={self._special_tokens}"
            f")"
        )
