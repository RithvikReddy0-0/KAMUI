"""Pre-tokenise a text corpus: train a BPE tokenizer and save the token array.

Usage:
    python scripts/tokenize_corpus.py \
        --input data/corpus.txt \
        --vocab-size 4096 \
        --tokenizer-out checkpoints/tokenizer.json \
        --tokens-out data/corpus_tokens

Pre-tokenisation is done once; training runs then load the ``.npy`` array
directly via ``kamui.training.data.load_tokens``.
"""

from __future__ import annotations

import argparse

from kamui.tokenizer.bpe import BPETokenizer
from kamui.training.data import tokenise_corpus
from kamui.utils.logging import get_logger

logger = get_logger("kamui.tokenize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tokenizer and pre-tokenise a corpus")
    parser.add_argument("--input", type=str, required=True, help="UTF-8 text corpus")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-out", type=str, required=True, help="tokenizer.json path")
    parser.add_argument("--tokens-out", type=str, required=True, help=".npy output (no suffix)")
    args = parser.parse_args()

    logger.info("training BPE tokenizer (vocab_size=%d)", args.vocab_size)
    from pathlib import Path

    tokenizer = BPETokenizer.train(Path(args.input), vocab_size=args.vocab_size)
    tokenizer.save(args.tokenizer_out)
    arr = tokenise_corpus(args.input, tokenizer, output_path=args.tokens_out)
    logger.info("wrote %d tokens to %s.npy", len(arr), args.tokens_out)


if __name__ == "__main__":
    main()
