"""Vocabulary management for the BPE tokeniser.

Responsibilities:
    - Store the mapping from token string → integer ID (and reverse)
    - Manage special tokens: <|endoftext|>, <|pad|>, <|bos|>, <|eos|>
    - Enforce that special tokens are never split by merge operations
    - Provide deterministic ID assignment (vocabulary ordering is stable
      across save/load cycles)

The ``Vocabulary`` class is used internally by ``BPETokenizer`` and is not
part of the primary public API, but it is importable for users who need
direct vocabulary inspection.

Key design choice — why special tokens are explicit:
    Many tokenisers silently handle special tokens through ad-hoc string
    matching.  KAMUI's Vocabulary makes special tokens first-class: they
    are registered at construction time and guaranteed to have reserved IDs
    (0, 1, 2, ...) that never collide with learned BPE tokens.  This makes
    the tokenisation pipeline fully auditable.

Implemented in: Phase 1, Week 3
"""

# Implementation begins in Phase 1.
