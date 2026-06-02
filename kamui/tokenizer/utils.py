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

Implemented in: Phase 1, Week 3
"""

# Implementation begins in Phase 1.
