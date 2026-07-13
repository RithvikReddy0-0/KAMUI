"""Data loading, tokenisation, and batch construction.

Responsibilities:
    - ``TextDataset``:
        Wraps a 1-D token array and yields ``(input, target)`` pairs by
        slicing ``(context_length + 1)``-length chunks: input = chunk[:-1],
        target = chunk[1:].  This packs the corpus with zero padding.

    - ``DataLoader``:
        A minimal loader that shuffles chunk start offsets and returns
        ``(input_ids, target_ids)`` batches of shape ``(B, context_length)``.

    - ``tokenise_corpus(corpus_path, tokenizer, output_path=None)``:
        Tokenise a text file and (optionally) save the token-ID array to a
        ``.npy`` file for reuse.

    - ``train_val_split(tokens, val_fraction)``:
        Deterministic split — the first ``1 - val_fraction`` of tokens is
        training data, the tail is validation.

Design notes:
    Sequence packing means every token participates in training with no
    padding; a chunk of ``context_length + 1`` tokens yields one input/target
    pair.  The number of available chunks is ``len(tokens) - context_length``.

Implemented in: Phase 5.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class TextDataset:
    """A packed next-token dataset over a 1-D token array.

    Attributes:
        tokens:         The underlying 1-D token sequence.
        context_length: Sequence length of each ``(input, target)`` pair.
    """

    def __init__(self, tokens: Sequence[int] | np.ndarray, context_length: int) -> None:
        """Wrap ``tokens`` as a packed dataset.

        Args:
            tokens:         A 1-D array/sequence of token IDs.
            context_length: Length of each input/target sequence (>= 1).

        Raises:
            ValueError: If ``context_length < 1`` or there are too few tokens
                to form a single ``context_length + 1`` chunk.
        """
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")
        self.tokens = np.asarray(tokens)
        if self.tokens.ndim != 1:
            raise ValueError(f"tokens must be 1-D, got shape {self.tokens.shape}")
        self.context_length = context_length
        if len(self.tokens) < context_length + 1:
            raise ValueError(
                f"need at least context_length + 1 = {context_length + 1} tokens, "
                f"got {len(self.tokens)}"
            )

    def __len__(self) -> int:
        """Number of available ``context_length + 1`` chunks."""
        return len(self.tokens) - self.context_length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the ``(input, target)`` pair starting at ``idx``.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        chunk = self.tokens[idx : idx + self.context_length + 1]
        chunk_t = torch.as_tensor(np.asarray(chunk), dtype=torch.long)
        return chunk_t[:-1], chunk_t[1:]


class DataLoader:
    """A minimal batching loader over a ``TextDataset``.

    Attributes:
        dataset:    The wrapped dataset.
        batch_size: Number of sequences per batch.
        shuffle:    Whether to shuffle chunk offsets each epoch.
        drop_last:  Whether to drop a trailing partial batch.
    """

    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = True,
    ) -> None:
        """Create a loader.

        Args:
            dataset:    A ``TextDataset``.
            batch_size: Sequences per batch (>= 1).
            shuffle:    Shuffle chunk offsets each iteration.
            seed:       Optional RNG seed for reproducible shuffling.
            drop_last:  Drop a trailing partial batch if True.

        Raises:
            ValueError: If ``batch_size < 1``.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        """Number of batches per iteration."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(inputs, targets)`` batches of shape ``(B, context_length)``."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            pairs = [self.dataset[i] for i in batch_idx]
            inputs = torch.stack([x for x, _ in pairs])
            targets = torch.stack([y for _, y in pairs])
            yield inputs, targets


def tokenise_corpus(
    corpus_path: str | Path,
    tokenizer: object,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Tokenise a text file into a token-ID array, optionally saving it.

    Args:
        corpus_path: Path to a UTF-8 text file.
        tokenizer:   An object with ``encode(str) -> list[int]``.
        output_path: If given, the array is saved there via ``np.save``.

    Returns:
        A 1-D ``np.int32`` array of token IDs.
    """
    text = Path(corpus_path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)  # type: ignore[attr-defined]
    arr = np.asarray(ids, dtype=np.int32)
    if output_path is not None:
        np.save(output_path, arr)
    return arr


def load_tokens(path: str | Path) -> np.ndarray:
    """Memory-map a token array saved by ``tokenise_corpus`` (``np.save``)."""
    return np.load(path, mmap_mode="r")


def train_val_split(
    tokens: Sequence[int] | np.ndarray, val_fraction: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Split a token array into (train, val) by a deterministic tail fraction.

    Args:
        tokens:       A 1-D token array.
        val_fraction: Fraction of tokens held out at the end for validation
            (in ``(0, 1)``).

    Returns:
        ``(train_tokens, val_tokens)``.

    Raises:
        ValueError: If ``val_fraction`` is not in ``(0, 1)``.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    arr = np.asarray(tokens)
    split = int(len(arr) * (1.0 - val_fraction))
    return arr[:split], arr[split:]
