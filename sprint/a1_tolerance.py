"""Part (a), step 1: does induction tolerate surface variants of the query token?

Tests prediction P1 of PREREG.md.  For each word with four single-token variants
(" w", "w", " W", "W"), builds repeated random sequences in which one token of the
first half is variant a1 and the matching token of the second half is a2, then
measures the log-probability of the correct next token B and the attention of each
induction head to B.

    python sprint/a1_tolerance.py --model gpt2
    python sprint/a1_tolerance.py --model pythia-160m

Writes sprint/results/a1_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

HALF = 20  # tokens in each half of a sequence
FORMS = ("sp_lower", "lower", "sp_title", "title")
RESULTS = Path(__file__).parent / "results"


def variants(word: str) -> dict[str, str]:
    return {
        "sp_lower": " " + word,
        "lower": word,
        "sp_title": " " + word.capitalize(),
        "title": word.capitalize(),
    }


def kind(form1: str, form2: str) -> str:
    space = form1.startswith("sp_") != form2.startswith("sp_")
    case = form1.endswith("title") != form2.endswith("title")
    return "both" if space and case else "space" if space else "case"


def find_words(model: HookedTransformer) -> dict[str, dict[str, int]]:
    """Return {word: {form: token id}} for words whose four variants are single tokens."""
    tokenizer = model.tokenizer
    words: dict[str, dict[str, int]] = {}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id])
        word = text[1:]
        if not (text.startswith(" ") and word.isascii() and word.isalpha() and word.islower()):
            continue
        if len(word) < 3:
            continue
        ids = {}
        for form, string in variants(word).items():
            encoded = tokenizer.encode(string, add_special_tokens=False)
            if len(encoded) != 1 or tokenizer.decode(encoded) != string:
                break
            ids[form] = encoded[0]
        else:
            if len(set(ids.values())) == 4:
                words[word] = ids
    return words


def ordinary_tokens(model: HookedTransformer, exclude: set[int]) -> list[int]:
    """Token IDs safe to use as random filler and targets.

    Keeps tokens the tokenizer actually produces for their own text, and drops
    whitespace runs and raw byte fragments.  Pythia gives some of those a
    log-probability near -800, which swamps any mean they enter.
    """
    tokenizer = model.tokenizer
    pool = []
    for token_id in range(len(tokenizer)):
        if token_id in exclude:
            continue
        text = tokenizer.decode([token_id])
        if not text.strip() or "�" in text:
            continue
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            continue
        pool.append(token_id)
    return pool


def induction_scores(model: HookedTransformer, pool: list[int], bos: int, n: int = 50) -> np.ndarray:
    """Mean attention from each second-half token to the token after its first copy."""
    rng = random.Random(1234)
    rows = [[bos] + (seq := rng.sample(pool, HALF)) + seq for _ in range(n)]
    tokens = torch.tensor(rows)
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith("pattern"))
    queries = torch.arange(HALF + 1, 2 * HALF + 1)
    keys = queries - HALF + 1
    scores = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        pattern = cache["pattern", layer]  # (batch, head, query, key)
        scores[layer] = pattern[:, :, queries, keys].mean(dim=(0, 2)).numpy()
    return scores


def build_trials(words, chosen, pool, bos, n_contexts):
    """Return (token rows, metadata rows)."""
    rows, meta = [], []
    for w_index, word in enumerate(chosen):
        others = [w for w in chosen if w != word]
        for context in range(n_contexts):
            rng = random.Random(f"{word}-{context}")
            base = rng.sample(pool, HALF)
            slot = rng.randrange(2, HALF - 3)  # where a1 sits; B is the next token
            other = rng.choice(others)
            stranger = rng.choice(pool)
            for form1 in FORMS:
                first = list(base)
                first[slot] = words[word][form1]
                target = first[slot + 1]

                def add(condition, a2, form2, first=first, target=target, form1=form1):
                    second = list(first)
                    second[slot] = a2
                    rows.append([bos] + first + second)
                    meta.append(
                        dict(word=w_index, context=context, form1=form1, form2=form2,
                             condition=condition, slot=slot, target=target)
                    )

                add("same", words[word][form1], form1)
                add("random", stranger, "random")
                for form2 in FORMS:
                    if form2 != form1:
                        add("variant", words[word][form2], form2)
                        add("control", words[other][form2], form2)
    return torch.tensor(rows), meta


def tolerance(values: np.ndarray, meta, n_words: int, which, seed: int = 0, n_boot: int = 2000):
    """Ratio of means (variant - control) / (same - control), bootstrapped over words.

    ``which`` selects the variant/control trials to include, by their metadata.
    """
    sums = {c: np.zeros(n_words) for c in ("variant", "control", "same")}
    counts = {c: np.zeros(n_words) for c in sums}
    for value, m in zip(values, meta, strict=True):
        c = m["condition"]
        if c == "same" or (c in ("variant", "control") and which(m)):
            sums[c][m["word"]] += value
            counts[c][m["word"]] += 1
    means = {c: sums[c] / np.maximum(counts[c], 1) for c in sums}
    keep = (counts["variant"] > 0) & (counts["same"] > 0)

    def ratio(index):
        v, c, s = (means[k][keep][index].mean() for k in ("variant", "control", "same"))
        return (v - c) / (s - c)

    n = int(keep.sum())
    rng = np.random.default_rng(seed)
    boots = [ratio(rng.integers(0, n, n)) for _ in range(n_boot)]
    point = ratio(np.arange(n))
    low, high = np.percentile(boots, [2.5, 97.5])
    return dict(T=float(point), low=float(low), high=float(high), n_words=n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n-words", type=int, default=200)
    parser.add_argument("--n-contexts", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device="cpu")
    model.eval()
    bos = model.tokenizer.bos_token_id

    words = find_words(model)
    chosen = sorted(words)
    random.Random(0).shuffle(chosen)
    chosen = chosen[: args.n_words]
    variant_ids = {i for w in words.values() for i in w.values()}
    special = set(model.tokenizer.all_special_ids)
    pool = ordinary_tokens(model, variant_ids | special)
    print(f"{args.model}: {len(words)} words with four single-token variants, using {len(chosen)}; "
          f"{len(pool)} filler tokens")

    scores = induction_scores(model, pool, bos)
    heads = [(int(l), int(h)) for l, h in zip(*np.where(scores > 0.4), strict=True)]
    rng = random.Random(0)
    non_induction = [
        (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
        if scores[l, h] < 0.05
    ]
    control_heads = rng.sample(non_induction, 5)
    print("induction heads (score > 0.4):", [(l, h, round(float(scores[l, h]), 2)) for l, h in heads])

    tokens, meta = build_trials(words, chosen, pool, bos, args.n_contexts)
    layers = sorted({l for l, _ in heads + control_heads})
    names = {f"blocks.{l}.attn.hook_pattern" for l in layers}
    logprob = np.zeros(len(meta))
    attention = {head: np.zeros(len(meta)) for head in heads + control_heads}
    for start in range(0, len(meta), args.batch):
        batch = tokens[start : start + args.batch]
        logits, cache = model.run_with_cache(batch, names_filter=lambda n: n in names)
        for offset, m in enumerate(meta[start : start + args.batch]):
            query = HALF + 1 + m["slot"]
            key = 2 + m["slot"]
            logprob[start + offset] = logits[offset, query].log_softmax(-1)[m["target"]].item()
            for layer, head in attention:
                attention[(layer, head)][start + offset] = cache["pattern", layer][
                    offset, head, query, key
                ].item()
        print(f"  {min(start + args.batch, len(meta))}/{len(meta)}", flush=True)

    def mean_of(values, condition):
        return float(np.mean([v for v, m in zip(values, meta, strict=True) if m["condition"] == condition]))

    selectors = {
        "all": lambda m: True,
        "space": lambda m: kind(m["form1"], m["form2"]) == "space",
        "case": lambda m: kind(m["form1"], m["form2"]) == "case",
        "both": lambda m: kind(m["form1"], m["form2"]) == "both",
    }
    n = len(chosen)
    out = {
        "model": args.model,
        "n_words_available": len(words),
        "n_words": n,
        "n_trials": len(meta),
        "induction_heads": [[l, h, float(scores[l, h])] for l, h in heads],
        "control_heads": control_heads,
        "logprob_means": {c: mean_of(logprob, c) for c in ("same", "variant", "control", "random")},
        "T_logit": {name: tolerance(logprob, meta, n, sel) for name, sel in selectors.items()},
        "T_attn": {
            f"{l}.{h}": dict(
                induction=(l, h) in heads,
                means={c: mean_of(attention[(l, h)], c) for c in ("same", "variant", "control", "random")},
                **tolerance(attention[(l, h)], meta, n, selectors["all"]),
            )
            for l, h in attention
        },
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"a1_{args.model}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez_compressed(
        RESULTS / f"a1_{args.model}_trials.npz",
        logprob=logprob,
        meta=json.dumps(meta),
        words=json.dumps(chosen),
        **{f"attn_{l}_{h}": v for (l, h), v in attention.items()},
    )

    print("log-prob of B:", {k: round(v, 2) for k, v in out["logprob_means"].items()})
    for name, t in out["T_logit"].items():
        print(f"T_logit[{name}] = {t['T']:.3f}  95% CI [{t['low']:.3f}, {t['high']:.3f}]")
    for head, t in out["T_attn"].items():
        tag = "induction" if t["induction"] else "control  "
        m = t["means"]
        print(f"head {head:5} {tag} same={m['same']:.3f} variant={m['variant']:.3f} "
              f"control={m['control']:.3f}  T={t['T']:.3f} [{t['low']:.3f}, {t['high']:.3f}]")
    print("wrote", path)


if __name__ == "__main__":
    main()
