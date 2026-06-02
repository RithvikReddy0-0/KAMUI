"""Pre-tokenise a text corpus and save as a memory-mapped binary file.

Usage:
    python scripts/tokenize_corpus.py \\
        --input data/tinystories.txt \\
        --tokenizer checkpoints/tokenizer.json \\
        --output data/tinystories_tokens.bin

Pre-tokenisation is done once; subsequent training runs load directly
from the binary file, avoiding repeated tokenisation overhead.

Implemented in: Phase 2
"""

# Implementation begins in Phase 2.
