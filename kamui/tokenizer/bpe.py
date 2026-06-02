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

Key implementation details (to be verified against tests):
    - Byte-level BPE: text is first encoded as UTF-8 bytes, so the
      vocabulary covers all possible byte values (0–255) as base tokens.
      This eliminates unknown tokens entirely.
    - Merge rules are applied greedily, left to right.
    - Special tokens (``<|endoftext|>``, ``<|pad|>``) are added to the
      vocabulary and never split by merge rules.

References:
    Sennrich, R., Haddow, B., & Birch, A. (2016).
    Neural Machine Translation of Rare Words with Subword Units.
    ACL 2016. https://arxiv.org/abs/1508.07909

Implemented in: Phase 1, Week 3
"""

# Implementation begins in Phase 1.
# Do not add code here until the test suite in tests/unit/test_tokenizer.py
# has been written and the interface contract has been finalised.
