"""Fast BPE training and encoding, with the same results as the simple versions.

``get_stats`` and ``merge_pair`` in ``kamui.tokenizer.utils`` are the textbook
algorithm: after each merge, recount every pair and rewrite the whole corpus.
That is easy to follow but slow, because every one of thousands of merges
touches every token.

The functions here keep the corpus as a linked list and change only what each
merge touches, so the work grows with the number of replacements rather than
with merges × corpus size.  They return exactly what the textbook loop returns,
and ``tests/unit/test_merges.py`` checks that against it.

    - ``learn_merges``: tracks every pair's count and where it occurs, and takes
      the most frequent pair from a heap.
    - ``apply_merges``: keeps a heap of the merges that apply, earliest first.

Why the results match: merging a pair creates a new token, so every pair it
creates contains that token.  A pair therefore never reappears once merged,
and all the places a pair occurs exist before its turn comes.  Within one
merge, places are handled left to right and rechecked first, which is what
``merge_pair`` does for overlapping runs such as ``a a a``.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable, Iterator

Pair = tuple[int, int]

#: Told about every pair that appears (+1) or disappears (-1) during a merge.
OnChange = Callable[[int, Pair, int], None]


class _LinkedTokens:
    """Token sequences as one doubly linked list, so a merge is a local edit.

    Node ``i`` holds ``tokens[i]``.  ``after[i]`` and ``before[i]`` are its
    neighbours, or -1 at either end of a sequence, so no pair spans two
    sequences.  A merge keeps the left node and drops the right one, which
    leaves the surviving nodes in their original left-to-right order.
    """

    def __init__(self, sequences: list[list[int]]) -> None:
        self.tokens: list[int] = []
        self.before: list[int] = []
        self.after: list[int] = []
        for sequence in sequences:
            if not sequence:
                continue
            start = len(self.tokens)
            self.tokens.extend(sequence)
            self.before.extend(range(start - 1, start + len(sequence) - 1))
            self.after.extend(range(start + 1, start + len(sequence) + 1))
            self.before[start] = -1
            self.after[-1] = -1
        self.alive = [True] * len(self.tokens)

    def pairs(self) -> Iterator[tuple[int, Pair]]:
        """Yield ``(node, pair)`` for every adjacent pair, left to right."""
        for node, following in enumerate(self.after):
            if following != -1:
                yield node, (self.tokens[node], self.tokens[following])

    def pair_at(self, node: int) -> Pair | None:
        """Return the pair that starts at ``node``, if one still does."""
        following = self.after[node]
        if not self.alive[node] or following == -1:
            return None
        return self.tokens[node], self.tokens[following]

    def merge(self, node: int, new_id: int, on_change: OnChange) -> None:
        """Replace ``node`` and the node after it with ``new_id``."""
        right = self.after[node]
        left, beyond = self.before[node], self.after[right]
        tokens = self.tokens

        if left != -1:
            on_change(left, (tokens[left], tokens[node]), -1)
        on_change(node, (tokens[node], tokens[right]), -1)
        if beyond != -1:
            on_change(right, (tokens[right], tokens[beyond]), -1)

        tokens[node] = new_id
        self.alive[right] = False
        self.after[node] = beyond
        if beyond != -1:
            self.before[beyond] = node

        if left != -1:
            on_change(left, (tokens[left], new_id), +1)
        if beyond != -1:
            on_change(node, (new_id, tokens[beyond]), +1)

    def survivors(self) -> list[int]:
        """Return the tokens that remain, in order."""
        return [token for token, alive in zip(self.tokens, self.alive, strict=True) if alive]


def learn_merges(
    sequences: list[list[int]],
    num_merges: int,
    first_id: int,
    verbose: bool = False,
) -> list[Pair]:
    """Learn up to ``num_merges`` BPE merges, exactly as the textbook loop does.

    Each round merges the most frequent adjacent pair, breaking ties toward the
    larger pair, and training stops early once no pair occurs twice.

    Args:
        sequences:  Token sequences to learn from (left unchanged).  Pairs never
            span two sequences.
        num_merges: The most merges to learn.
        first_id:   The ID the first merge creates; each later merge takes the next.
        verbose:    Print progress every 100 merges.

    Returns:
        The merged pairs, in the order they were learned.
    """
    corpus = _LinkedTokens(sequences)
    counts: defaultdict[Pair, int] = defaultdict(int)
    places: defaultdict[Pair, list[int]] = defaultdict(list)
    for node, pair in corpus.pairs():
        counts[pair] += 1
        places[pair].append(node)

    # A max-heap on (count, pair), by negating both.  An entry goes stale when its
    # pair's count changes, and a fresh one is pushed, so stale entries are skipped.
    heap = [(-count, -left, -right) for (left, right), count in counts.items()]
    heapq.heapify(heap)

    def on_change(node: int, pair: Pair, delta: int) -> None:
        counts[pair] += delta
        if delta > 0:
            places[pair].append(node)
        heapq.heappush(heap, (-counts[pair], -pair[0], -pair[1]))

    merges: list[Pair] = []
    while heap and len(merges) < num_merges:
        negative_count, negative_left, negative_right = heapq.heappop(heap)
        pair = (-negative_left, -negative_right)
        if counts[pair] != -negative_count:
            continue  # stale
        if counts[pair] < 2:
            # Only unique pairs remain.  Merging them would just memorise the corpus.
            break

        new_id = first_id + len(merges)
        for node in sorted(set(places.pop(pair))):
            if corpus.pair_at(node) == pair:  # an earlier merge may have used it
                corpus.merge(node, new_id, on_change)
        merges.append(pair)

        if verbose and len(merges) % 100 == 0:
            print(f"  merge {len(merges)}/{num_merges}: {pair} → {new_id}")
    return merges


def apply_merges(ids: list[int], merge_ids: dict[Pair, int]) -> list[int]:
    """Merge a token sequence, exactly as applying each merge in order would.

    Args:
        ids:       The tokens to merge, usually raw bytes.
        merge_ids: Each merge pair's new ID.  A lower ID was learned earlier, so
            it is also the merge's priority.

    Returns:
        The merged tokens.
    """
    if len(ids) < 2:
        return list(ids)
    corpus = _LinkedTokens([ids])
    heap = [(merge_ids[pair], node) for node, pair in corpus.pairs() if pair in merge_ids]
    heapq.heapify(heap)

    def on_change(node: int, pair: Pair, delta: int) -> None:
        if delta > 0 and pair in merge_ids:
            heapq.heappush(heap, (merge_ids[pair], node))

    while heap:
        new_id, node = heapq.heappop(heap)
        pair = corpus.pair_at(node)
        if pair is not None and merge_ids.get(pair) == new_id:
            corpus.merge(node, new_id, on_change)
    return corpus.survivors()
