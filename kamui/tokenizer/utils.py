"""Encoding and decoding utility functions for the tokeniser.

Responsibilities:
    - ``text_to_bytes(text) -> list[int]``:
        Convert a Unicode string to a list of raw byte values (0–255).
        This is the first step of byte-level BPE.

    - ``bytes_to_text(byte_values) -> str``:
        Inverse of text_to_bytes.  Converts a list of byte values back to
        a Unicode string.

    - ``get_stats(vocab) -> dict[tuple[int, int], int]``:
        Count the frequency of every adjacent pair of tokens in the current
        vocabulary.  Core subroutine of the BPE merge loop.

    - ``merge_pair(ids, pair, new_id) -> list[int]``:
        Replace all occurrences of ``pair`` in ``ids`` with ``new_id``.
        Single merge step of BPE.

These functions are kept separate from ``bpe.py`` so they can be unit-tested
independently without constructing a full ``BPETokenizer`` instance.
"""

from __future__ import annotations


def text_to_bytes(text: str) -> list[int]:
    """Convert a Unicode string to a list of raw UTF-8 byte values (0–255).

    This is the entry point for byte-level BPE.  Every possible Unicode string
    maps to a finite sequence of bytes, so no unknown tokens are ever possible.

    Args:
        text: Any Unicode string.

    Returns:
        A list of integers in the range [0, 255], one per UTF-8 byte.

    Raises:
        TypeError: If ``text`` is not a string.

    Examples:
        >>> text_to_bytes("hi")
        [104, 105]
        >>> text_to_bytes("é")   # U+00E9, encoded as two bytes in UTF-8
        [195, 169]
        >>> text_to_bytes("")
        []
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text)}")
    return list(text.encode("utf-8"))


def bytes_to_text(byte_values: list[int]) -> str:
    """Convert a list of UTF-8 byte values back to a Unicode string.

    This is the inverse of ``text_to_bytes``.  The round-trip
    ``bytes_to_text(text_to_bytes(s)) == s`` holds for all Unicode strings.

    Args:
        byte_values: A list of integers in [0, 255] representing UTF-8 bytes.

    Returns:
        The decoded Unicode string.

    Raises:
        TypeError: If ``byte_values`` is not a list, or any element is not an int.
        ValueError: If any element is outside [0, 255], or the byte sequence is
            not valid UTF-8.

    Examples:
        >>> bytes_to_text([104, 105])
        'hi'
        >>> bytes_to_text([195, 169])
        'é'
        >>> bytes_to_text([])
        ''
    """
    if not isinstance(byte_values, list):
        raise TypeError(f"byte_values must be a list, got {type(byte_values)}")
    for i, b in enumerate(byte_values):
        if not isinstance(b, int) or isinstance(b, bool):
            raise TypeError(f"byte_values[{i}] must be an int, got {type(b)}")
        if not (0 <= b <= 255):
            raise ValueError(f"byte_values[{i}] = {b} is out of range [0, 255]")
    try:
        return bytes(byte_values).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"byte sequence is not valid UTF-8: {e}") from e


def get_stats(vocab: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count the frequency of every adjacent pair of token IDs across all sequences.

    This is the core subroutine of the BPE training loop.  On each iteration,
    the most-frequent pair returned by this function is chosen as the next merge.

    Args:
        vocab: A list of token-ID sequences.  Typically this is the entire
            training corpus represented as a list of word-level (or document-level)
            token-ID lists.  An empty outer list or sequences of length < 2
            contribute no pairs.

    Returns:
        A dict mapping ``(left_id, right_id)`` pairs to their total occurrence
        count across all sequences.  Returns an empty dict if ``vocab`` is
        empty or all sequences have length < 2.

    Raises:
        TypeError: If ``vocab`` is not a list.

    Examples:
        >>> get_stats([[1, 2, 3, 1, 2]])
        {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        >>> get_stats([])
        {}
        >>> get_stats([[1]])
        {}
    """
    if not isinstance(vocab, list):
        raise TypeError(f"vocab must be a list, got {type(vocab)}")
    counts: dict[tuple[int, int], int] = {}
    for seq in vocab:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``.

    Merges are applied greedily from left to right.  Overlapping occurrences
    are not merged — once a pair is consumed, the cursor advances past it.

    Args:
        ids:    The current sequence of token IDs.
        pair:   The ``(left_id, right_id)`` pair to replace.
        new_id: The integer ID to substitute for each matched pair.

    Returns:
        A new list with every non-overlapping occurrence of ``pair`` replaced
        by ``new_id``.  The original ``ids`` list is not mutated.

    Raises:
        TypeError: If ``ids`` is not a list, ``pair`` is not a tuple of two ints,
            or ``new_id`` is not an int.

    Examples:
        >>> merge_pair([1, 2, 3, 1, 2], (1, 2), 5)
        [5, 3, 5]
        >>> merge_pair([1, 2, 1, 2, 1, 2], (1, 2), 9)
        [9, 9, 9]
        >>> merge_pair([1, 2, 2, 1], (2, 2), 7)
        [1, 7, 1]
        >>> merge_pair([], (1, 2), 5)
        []
    """
    if not isinstance(ids, list):
        raise TypeError(f"ids must be a list, got {type(ids)}")
    if not (isinstance(pair, tuple) and len(pair) == 2):
        raise TypeError(f"pair must be a 2-tuple, got {type(pair)}")
    if not isinstance(new_id, int) or isinstance(new_id, bool):
        raise TypeError(f"new_id must be an int, got {type(new_id)}")

    result: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result
