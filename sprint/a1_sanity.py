"""Sanity checks for a1_tolerance.py: look at single examples by hand.

Prints, for a few words, the tokens involved and the model's top predictions at the
query position in each condition, so a bug in indexing or token choice is visible.

    python sprint/a1_sanity.py --model gpt2
"""

from __future__ import annotations

import argparse
import random

import torch
from transformer_lens import HookedTransformer

from a1_tolerance import HALF, build_trials, find_words, ordinary_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device="cpu")
    tok = model.tokenizer
    bos = tok.bos_token_id

    words = find_words(model)
    chosen = sorted(words)
    random.Random(0).shuffle(chosen)
    chosen = chosen[:200]
    variant_ids = {i for w in words.values() for i in w.values()}
    pool = ordinary_tokens(model, variant_ids | set(tok.all_special_ids))
    tokens, meta = build_trials(words, chosen, pool, bos, 1)

    shown = 0
    for row, m in zip(tokens, meta, strict=True):
        if m["form1"] != "sp_lower" or m["word"] > 2:
            continue
        if m["condition"] == "variant" and m["form2"] != "title":
            continue
        if m["condition"] == "control" and m["form2"] != "title":
            continue
        slot = m["slot"]
        a1, a2 = row[1 + slot].item(), row[HALF + 1 + slot].item()
        assert row[2 + slot].item() == m["target"]
        differ = [i for i in range(1, HALF + 1) if row[i] != row[i + HALF]]
        logits = model(row.unsqueeze(0))[0, HALF + 1 + slot]
        logp = logits.log_softmax(-1)
        top = [repr(tok.decode([i])) for i in logp.topk(3).indices.tolist()]
        rank = int((logp > logp[m["target"]]).sum())
        print(
            f"{chosen[m['word']]:10} {m['condition']:8} a1={tok.decode([a1])!r:10} a2={tok.decode([a2])!r:10} "
            f"B={tok.decode([m['target']])!r:12} logp(B)={logp[m['target']]:.2f} rank={rank:5} top3={top} "
            f"differing positions={differ} (slot+1={slot + 1})"
        )
        shown += 1
    print("rows shown:", shown)


if __name__ == "__main__":
    main()
