"""Data loading, tokenisation, and batch construction.

Responsibilities:
    - ``TextDataset``:
        Loads a pre-tokenised binary file (or tokenises on-the-fly from a
        text file) and provides sequences of fixed length ``context_length``
        for training.

        Sequence packing: tokens are laid out as one long 1-D array and
        sliced into (context_length + 1)-length chunks.  The +1 allows
        constructing the input/target pair: input = chunk[:-1],
        target = chunk[1:].  This packs the dataset with zero padding.

    - ``DataLoader``:
        A minimal DataLoader that wraps ``TextDataset``, shuffles chunk
        start offsets, and returns (input_ids, target_ids) batches of
        shape (B, S).

    - ``tokenise_corpus(corpus_path, tokenizer, output_path)``:
        Pre-tokenise a large text file and save the resulting token ID
        array as a memory-mapped numpy array.  Pre-tokenisation is done
        once; subsequent training runs load directly from the binary file,
        avoiding repeated tokenisation overhead.

Design notes:
    Memory mapping (``np.memmap``):
        Large datasets (billions of tokens) do not fit in RAM.  Using a
        memory-mapped file allows the OS to page data in and out as needed,
        making datasets of arbitrary size trainable on machines with limited
        RAM.  This is the approach used by nanoGPT.

    Train / validation split:
        The first 95% of tokens are used for training; the last 5% for
        validation.  The split is deterministic (not shuffled at the token
        level) to ensure validation loss is always computed on the same
        held-out data.

Implemented in: Phase 2, Week 10
"""

# Implementation begins in Phase 2.
