# Sprint log

One small, original mechanistic interpretability finding. Newest entries at the bottom.

## 2026-11-10: Phase 1, picking the question

**Setup facts**
- `RESEARCH_CHARTER.md` does not exist in the repo. `research/RESEARCH_LOG.md` is an empty template.
- KAMUI is further along than the brief says: model, training, hooks, patching, SAEs and steering all work (v0.4 plus GPU support and a fast tokenizer).
- Compute: laptop CPU only (Intel integrated graphics), plus free Colab T4. TransformerLens, nnsight and transformers are not installed locally, and Python here is 3.14, so they may not install. Colab is the fallback.

**Literature searches run (12)** and the closest prior work found
- Tokenization granularity vs attention/MLP circuits: nothing direct.
- Induction across token variants (space, case): nothing quantitative in small models. Closest: Olsson et al. 2022 (fuzzy induction, qualitative, large models), Feucht et al. 2025 dual-route model (token vs concept induction heads, 7B models).
- Same word under different segmentations: largely done (Kaplan et al. 2024 inner lexicon; Feucht et al. 2024 token erasure).
- Positional encoding vs induction formation: crowded and contradictory (APE theory paper, multimodal ICL paper, 2026 spectral paper).
- Weight tying vs logit lens: done (Lopardo et al. 2026).
- Seed universality: done at larger scale (Bali et al. 2026; modular addition line).
- Ablation-method disagreement: done (Miller et al. 2024; Heimersheim and Nanda 2024).
- Data statistics vs induction emergence: done (Chan, Singh, Reddy, Aoyama, Zucchet).
- SAE features vs token variants: nothing direct; adjacent work on token-level features and feature sensitivity.

**Candidates considered:** 10 (see the Phase 1 message). Killed as already done: 3, 5, 6, 7, 9.

**Recommended top 3**
1. Variant-tolerant induction in GPT-2 small and Pythia.
2. SAE feature invariance across token variants, by layer.
3. Tokenization granularity as a knob that moves computation from attention to MLPs (toy, KAMUI).

**Status:** waiting for the user's choice. Novelty of the top pick needs one deeper check (full read of Olsson et al. and the dual-route paper) before pre-registration.

**Dead ends from earlier KAMUI work, relevant here**
- Could not train a trustworthy induction head in KAMUI's toy models, so any question needing one must use a pretrained model.

## 2026-11-10: Phase 2 setup and pre-registration

- Read Olsson et al. 2022 and the dual-route paper in full text, and searched for MLP0 work. Neither measures surface variants; details are in `PREREG.md`.
- Environment: `sprint/.venv`, Python 3.11 via uv. Windows Application Control blocks recently released compiled packages, so the stack is pinned to 2024 releases (`sprint/requirements.txt`): torch 2.12.0, transformer-lens 2.7.0, transformers 4.44.2.
- `PREREG.md` written before any experiment. SHA-256: `28b654a3cea2cf98e2d1737738f5cefeafa03f14e7bdc3a4689335c8252da001`. Not committed yet.
- Analysis detail fixed before seeing data: tolerance is a ratio of means, bootstrapped over words.

## 2026-11-10: Part (a) step 1, P1 (does tolerance exist?)

Script: `a1_tolerance.py`. 200 words, 3 contexts, 9,600 sequences per model.

| Model | T_logit, all variants | space only | case only | both |
|---|---|---|---|---|
| GPT-2 small | 0.926 [0.917, 0.934] | 0.956 | 0.950 | 0.875 |
| Pythia-160m | 0.856 [0.831, 0.881] | 0.923 | 0.880 | 0.771 |

- **P1 supported in both models.** My own expectation (0.2 to 0.6) was wrong: tolerance is close to complete.
- Surprise check (`a1_sanity.py`): looked at single examples by hand. Exactly one position differs, the target index is right, and the variant run ranks B first. Not a bug as far as I can tell.
- Heads differ a lot in GPT-2: 5.0 has T = 0.22 and 5.1 has 0.56, while 5.5, 6.9 and 7.10 are at 0.84 to 0.91. Exact-match and tolerant heads sit in the same layer.
- The output side looks fuzzy too: top predictions include variants of B (" GU", "Gu", " Gu"). Not pre-registered.

**Open problems**
- Pythia-160m's log-probability of B is -9.5 even in the SAME condition, so it barely does induction on these sequences. Its T is a ratio over a weak effect. Needs checking (BOS handling, sequence length) before the Pythia number is trusted.
- CONTROL uses an unrelated word. It can't separate "same word" from "any token with a similar embedding". A near-neighbour control (different word, similar embedding) is needed. This would be an addition after seeing data, and must be labelled as such.
- T for non-induction heads is meaningless (attention about 0), so that control is uninformative as designed.

## 2026-11-14 to 21: Fixes, a rerun, and the post-hoc neighbour control

**Two bugs in the random-token pool, both fixed**
1. Pythia's embedding has 50,304 rows but its tokenizer uses only 50,277. The pool included the untrained padding rows.
2. The bigger problem: Pythia gives some tokens a log-probability near -800 (whitespace runs, raw byte fragments). They were 2.7% of SAME trials but dominated every mean. The pool now keeps only tokens that round-trip through the tokenizer and aren't whitespace or byte fragments (`ordinary_tokens`). This changes the pre-registered stimuli, so it's a deviation. It applies to both models and was decided before rerunning.

Correction to the earlier entry: each model runs 19,200 sequences, not 9,600.

**P1 after the fix** (`a1_tolerance.py`, 200 words, 3 contexts)

| Model | T_logit all | space | case | both | SAME log-prob of B |
|---|---|---|---|---|---|
| GPT-2 small | 0.931 [0.924, 0.938] | 0.961 | 0.955 | 0.880 | -0.37 |
| Pythia-160m | 0.905 [0.883, 0.924] | 0.940 | 0.927 | 0.851 | -5.6 |

- **P1 holds in both models.**
- Pythia's SAME mean is still -5.6, so some tail remains. Its median was about -1.0 before the fix. The remaining tail needs a look before Pythia numbers go in a paper.

**Neighbour control** (`a2_neighbour.py`, NOT pre-registered). Each neighbour is a different word whose raw embedding matches a1 as closely as the variant does (mean similarity 0.43 for both).

| Model | Words matched | T variant | T neighbour |
|---|---|---|---|
| GPT-2 small | 98 | 0.905 [0.880, 0.927] | 0.611 [0.553, 0.666] |
| Pythia-160m | 34 | 0.881 [0.835, 0.931] | 0.863 [0.738, 0.928] |

- **GPT-2:** variants are tolerated more than equally similar different words, so raw embedding similarity does not explain all of GPT-2's tolerance. This points to something after the embedding, which is what P2 will test.
- **Pythia:** no difference. It's underpowered (34 words), but it's also not a replication. Don't claim a cross-model effect.
- **Confounds:**
  - Most variant pairs (6,400+) have no neighbour within 0.02, so the matched words are a selected subset.
  - Neighbours are often frequent function words (" over" -> " of", " on"), so frequency is not controlled.
  - Some Pythia "words" are fragments ("ast").

**Figures:** `figures/a1_tolerance.png` and `figures/a1_heads.png`, regenerated by `fig_a.py`.

**Next:** P2 (patching the embedding path vs MLP0 output); a frequency-matched neighbour control; the remaining Pythia tail.
