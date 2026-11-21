"""Part (a), post-hoc control: is tolerance just raw embedding similarity?

NOT PRE-REGISTERED.  Added after seeing a1's results, because its CONTROL condition
(an unrelated word) cannot separate "same word" from "any token with a similar
embedding".

For each variant trial, picks a NEIGHBOUR: a token of a different word whose raw
embedding is as similar to a1 as the variant a2 is (mean-centred cosine, matched
within 0.02).  Tokens sharing the word's first four letters, in any case, are
excluded, and so are tokens that contain the word or are contained in it, so
morphological relatives (" invest" / " reinvest") don't count as different words.  If tolerance only tracked raw embedding similarity, NEIGHBOUR
would score about as high as VARIANT.

    python sprint/a2_neighbour.py --model gpt2

Writes sprint/results/a2_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from transformer_lens import HookedTransformer

from a1_tolerance import FORMS, HALF, RESULTS, find_words, ordinary_tokens, tolerance

MATCH = 0.02  # how close the neighbour's similarity must be to the variant's


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n-words", type=int, default=200)
    parser.add_argument("--n-contexts", type=int, default=3)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device="cpu")
    tok = model.tokenizer
    bos = tok.bos_token_id
    n_vocab = len(tok)

    words = find_words(model)
    chosen = sorted(words)
    random.Random(0).shuffle(chosen)
    chosen = chosen[: args.n_words]
    variant_ids = {i for w in words.values() for i in w.values()}
    special = set(tok.all_special_ids)
    pool = ordinary_tokens(model, variant_ids | special)

    embed = model.W_E[:n_vocab].float()
    embed = embed - embed.mean(0)
    embed = embed / embed.norm(dim=1, keepdim=True)
    texts = [tok.decode([i]) for i in range(n_vocab)]
    lowered = [t.strip().lower() for t in texts]
    wordlike = np.array([t.strip().isalpha() and t.isascii() for t in texts])

    rows, meta, sims = [], [], []
    unmatched = 0
    for w_index, word in enumerate(chosen):
        ids = words[word]
        excluded = np.array(
            [s[:4] == word[:4] or word in s or (len(s) >= 3 and s in word) for s in lowered]
        ) | ~wordlike
        excluded[list(special)] = True
        for context in range(args.n_contexts):
            rng = random.Random(f"{word}-{context}")  # same contexts as a1
            base = rng.sample(pool, HALF)
            slot = rng.randrange(2, HALF - 3)
            for form1 in FORMS:
                first = list(base)
                first[slot] = ids[form1]
                target = first[slot + 1]
                cos = embed @ embed[ids[form1]]

                def add(condition, a2, form2, sim, first=first, target=target, form1=form1):
                    second = list(first)
                    second[slot] = a2
                    rows.append([bos] + first + second)
                    meta.append(dict(word=w_index, context=context, form1=form1, form2=form2,
                                     condition=condition, slot=slot, target=target))
                    sims.append(sim)

                add("same", ids[form1], form1, 1.0)
                for form2 in FORMS:
                    if form2 == form1:
                        continue
                    target_sim = float(cos[ids[form2]])
                    gap = (cos - target_sim).abs()
                    gap[torch.from_numpy(excluded)] = float("inf")
                    best = int(gap.argmin())
                    if float(gap[best]) > MATCH:
                        unmatched += 1
                        continue
                    add("variant", ids[form2], form2, target_sim)
                    add("neighbour", best, form2, float(cos[best]))
                    other = rng.choice([w for w in chosen if w != word])
                    add("control", words[other][form2], form2, float(cos[words[other][form2]]))

    tokens = torch.tensor(rows)
    logprob = np.zeros(len(meta))
    for start in range(0, len(meta), args.batch):
        logits = model(tokens[start : start + args.batch])
        for offset, m in enumerate(meta[start : start + args.batch]):
            query = HALF + 1 + m["slot"]
            logprob[start + offset] = logits[offset, query].log_softmax(-1)[m["target"]].item()

    # T for neighbours: treat them as the "variant" condition in the same ratio.
    as_neighbour = [dict(m, condition="variant" if m["condition"] == "neighbour" else
                         ("skip" if m["condition"] == "variant" else m["condition"])) for m in meta]
    as_variant = [dict(m, condition="skip" if m["condition"] == "neighbour" else m["condition"])
                  for m in meta]
    n = len(chosen)
    sim_of = lambda c: float(np.mean([s for s, m in zip(sims, meta) if m["condition"] == c]))  # noqa: E731
    out = {
        "model": args.model,
        "note": "post-hoc control, not pre-registered",
        "unmatched_pairs": unmatched,
        "mean_similarity": {c: sim_of(c) for c in ("variant", "neighbour", "control")},
        "T_variant": tolerance(logprob, as_variant, n, lambda m: True),
        "T_neighbour": tolerance(logprob, as_neighbour, n, lambda m: True),
        "examples": [
            [tok.decode([words[chosen[m["word"]]][m["form1"]]]),
             tok.decode([rows[i][HALF + 1 + m["slot"]]])]
            for i, m in enumerate(meta) if m["condition"] == "neighbour"
        ][:20],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"a2_{args.model}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
    print("examples (a1 -> neighbour):", out["examples"][:12])


if __name__ == "__main__":
    main()
