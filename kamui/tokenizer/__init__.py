"""Tokeniser subpackage for KAMUI.

Responsibilities:
    - Convert raw text to integer token IDs (encoding)
    - Convert integer token IDs back to text (decoding)
    - Train a BPE vocabulary on a text corpus
    - Save and load a trained tokeniser to/from disk

This package contains a full byte-pair encoding (BPE) tokeniser implemented
from scratch, without any dependency on tiktoken, sentencepiece, or the
HuggingFace tokenizers library.

Public API:
    BPETokenizer    — the main tokeniser class
    Vocabulary      — vocabulary management

Design constraint:
    This package must not import from kamui.model, kamui.training,
    kamui.hooks, or kamui.mechinterp. The tokeniser is a standalone
    text-processing utility with no knowledge of model architecture.
"""

from kamui.tokenizer.bpe import BPETokenizer
from kamui.tokenizer.vocab import Vocabulary

__all__: list[str] = ["BPETokenizer", "Vocabulary"]
