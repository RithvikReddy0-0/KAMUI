"""The fast BPE functions must match the textbook algorithm exactly.

``learn_merges`` and ``apply_merges`` exist only to be faster.  The textbook
versions below are the loops ``BPETokenizer`` used before them, built from
``get_stats`` and ``merge_pair``.  Every test compares the two, on corpora
chosen to break an incremental algorithm: overlapping runs, ties, sequence
boundaries, and random strings over tiny alphabets.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from kamui.tokenizer.bpe import BPETokenizer
from kamui.tokenizer.merges import apply_merges, learn_merges
from kamui.tokenizer.utils import get_stats, merge_pair, text_to_bytes

ROOT = Path(__file__).resolve().parents[2]
FIRST_ID = 258


def textbook_learn(sequences: list[list[int]], num_merges: int, first_id: int) -> list[tuple]:
    merges = []
    for index in range(num_merges):
        stats = get_stats(sequences)
        if not stats:
            break
        best = max(stats, key=lambda pair: (stats[pair], pair))
        if stats[best] < 2:
            break
        sequences = [merge_pair(seq, best, first_id + index) for seq in sequences]
        merges.append(best)
    return merges


def textbook_apply(ids: list[int], merge_ids: dict[tuple[int, int], int]) -> list[int]:
    for pair, new_id in merge_ids.items():
        ids = merge_pair(ids, pair, new_id)
    return ids


def _merge_ids(merges: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {pair: FIRST_ID + index for index, pair in enumerate(merges)}


def _check(sequences: list[list[int]], num_merges: int = 50) -> list[tuple[int, int]]:
    merges = learn_merges(sequences, num_merges, FIRST_ID)
    assert merges == textbook_learn(sequences, num_merges, FIRST_ID)
    merge_ids = _merge_ids(merges)
    for sequence in sequences:
        assert apply_merges(sequence, merge_ids) == textbook_apply(sequence, merge_ids)
    return merges


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "aa",
        "aaa",
        "aaaa",
        "aaaaa",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "abababababab",
        "aabaabaabaab",
        "abcabcabcabc",
        "aaabaaabaaab",
        "abba abba abba",
        "the cat sat on the mat . the dog sat on the log .",
        "héllo wörld, héllo wörld 🙂🙂🙂",
    ],
)
def test_hand_picked_corpora(text: str) -> None:
    _check([list(text.encode("utf-8"))])


def test_pairs_never_span_two_sequences() -> None:
    # Joined up, (1, 2) would occur three times; split, no pair occurs at all.
    assert _check([[1], [2], [1], [2], [1], [2]]) == []
    _check([[1, 2], [1, 2], [2, 1], [1], [], [2, 1, 2]])


def test_input_is_left_unchanged() -> None:
    sequences = [[1, 1, 1, 2], [2, 1, 1]]
    before = [list(seq) for seq in sequences]
    learn_merges(sequences, 10, FIRST_ID)
    apply_merges(sequences[0], {(1, 1): FIRST_ID})
    assert sequences == before


@pytest.mark.parametrize("seed", range(150))
def test_random_corpora_over_tiny_alphabets(seed: int) -> None:
    rng = random.Random(seed)
    alphabet = list(range(rng.randint(1, 4)))
    sequences = [
        [rng.choice(alphabet) for _ in range(rng.randint(0, 120))] for _ in range(rng.randint(1, 4))
    ]
    _check(sequences, num_merges=rng.randint(1, 60))


def test_real_text() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")[:30_000]
    merges = _check([text_to_bytes(text)], num_merges=400)
    assert len(merges) == 400


def test_stops_when_no_pair_repeats() -> None:
    assert learn_merges([[1, 2, 3, 4]], 10, FIRST_ID) == []
    assert learn_merges([[1, 2, 1, 2]], 10, FIRST_ID) == [(1, 2)]


def test_verbose_reports_progress(capsys: pytest.CaptureFixture[str]) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")[:20_000]
    learn_merges([text_to_bytes(text)], 120, FIRST_ID, verbose=True)
    assert "merge 100/120" in capsys.readouterr().out


def test_tokenizer_matches_the_textbook_end_to_end() -> None:
    corpus = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train(corpus, vocab_size=600)
    first_id = 256 + len(tokenizer.special_tokens)
    assert tokenizer._merges == textbook_learn([text_to_bytes(corpus)], 600 - first_id, first_id)

    other = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")[:20_000]
    merge_ids = {pair: first_id + i for i, pair in enumerate(tokenizer._merges)}
    assert tokenizer.encode(other) == textbook_apply(text_to_bytes(other), merge_ids)
    assert tokenizer.decode(tokenizer.encode(other)) == other
