# Pre-registration: variant-tolerant induction

Written 2026-11-10, before any experiment in this sprint was run. The only things
checked beforehand were that the models load and that word variants are single tokens.

## Question

An induction head completes `A B ... A -> B`. In GPT-2, " cat", "cat", " Cat" and "Cat"
are four unrelated token IDs. If the second `A` is a different variant of the same
word, does induction still work, and if so, what makes it work?

## What is already known, and what would be new

| Source | What it covers | What it leaves open |
|---|---|---|
| Olsson et al. 2022 | Proposes fuzzy induction, `A* B* ... A -> B` with `A*` "similar in some embedding space". Examples such as translation, in large models. | No measurement for surface variants, no small-model result, no mechanism for where the similarity comes from. |
| Feucht et al. 2025 (dual-route) | Token vs concept induction heads in 7B models, found by patching. Capitalization appears only as a task type. | Never tests what token heads match on (exact token or similar tokens). No early-layer analysis. No small models. |
| MLP0 as "extended embedding" (Nanda; Wang et al. 2022; McDougall et al. 2023; Lad et al. 2024) | Later layers read MLP0's output more than the raw embedding. | Not tested on variants, and not linked to induction matching. |
| Detokenization (Kamoda et al. 2025; "Inside the LLM Word Factory" 2026) | Early layers join sub-word pieces into words. | About multi-token words, not single-token variants of one word. |

**Our claim, if the results support it:** a quantitative measure of how much induction in
GPT-2 small and Pythia-160m tolerates surface variants of the query token, a localization
of where that tolerance is built, and a training experiment showing that shared
distributional statistics alone are enough to create it.

Caveat: Olsson et al. is mostly figures. I read all of its text and found nothing on
case or space variants, but I can't rule out a figure that shows one.

## Part (a): observation in GPT-2 small and Pythia-160m

### Stimuli
- **Words:** every word whose four variants (" w", "w", " W", "W") are each one token, at
  least 3 letters, alphabetic. Sample 200 with seed 0, per model.
- **Sequence:** BOS, then 20 random tokens, then the same 20 again. One position in the
  first half holds variant `a1`, followed by a random token `B`. The matching position in
  the second half holds `a2`. Nothing else differs between conditions.
- **Conditions for `a2`:**
  - SAME: `a2 = a1`.
  - VARIANT: same word, different form. Three kinds: space only, case only, both.
  - CONTROL: a different word in the same form as the variant. This is the floor.
- 3 random contexts per word (seeds 0 to 2).

### Metrics
- **Behaviour:** log-probability of `B` at the `a2` position.
- **Attention:** attention from `a2` to `B`'s position, for each induction head. Induction
  heads are chosen beforehand on separate plain repeated sequences (score > 0.4).
- **Tolerance:** `T = (VARIANT - CONTROL) / (SAME - CONTROL)`. 0 means a variant is treated
  like a stranger, 1 means it is treated as the same token. Reported for behaviour
  (`T_logit`) and per head (`T_attn`), with bootstrap 95% intervals over words.
- **Variant similarity at layer L:** cosine between mean-centred vectors of the two
  variants, minus the same quantity for different words in the same forms. Measured on the
  raw embedding and on the residual stream after each of layers 0 to 5.

### Predictions
- **P1, tolerance exists.** Mean `T_logit` over variant pairs is above 0.1 and its interval
  excludes 0, in both models. *Falsified* if either interval includes 0.
- **P2, mechanism.** Two accounts, and what each predicts:

  | Test | "Tracks embedding similarity" | "MLP0 builds the invariance" |
  |---|---|---|
  | Similarity gain from embedding to after layer 0 | about 0 or negative | positive, interval excludes 0 |
  | Per-pair correlation of `T` with similarity | raw embedding predicts `T` at least as well as layer 0 | layer 0 predicts `T` better |
  | Patching, in the VARIANT run at the query position: correct only the raw embedding path, or only MLP0's output, to the SAME run's value | fixing MLP0's output recovers at least as much as fixing the embedding | fixing the embedding recovers more, because MLP0's output was already similar |

  A verdict needs at least two of three tests to agree. Otherwise the result is "mixed" and
  is reported that way. If similarity keeps rising through layers 1 to 4, the answer is
  "another early component", and I will localize it by the same patching.
- **My expectation:** P1 holds with `T_logit` between 0.2 and 0.6. For P2, I lean towards
  MLP0, at about 60% confidence.

### Controls
- CONTROL condition as the floor, and a random-token floor as a check on it.
- `T_attn` for 5 randomly chosen non-induction heads.
- Two models, three context seeds.
- Exploratory, not confirmatory: differences between the three variant kinds.

## Part (b): causal test in KAMUI

### Idea
Train twin models that differ in one thing: whether the two forms of a word share
distributional statistics. Cross-form lookups never appear in training, so any
tolerance is a by-product, not something the model was taught.

### Setup
- **Vocabulary:** K word identities, each with two forms (separate token IDs), plus value
  tokens.
- **Training sequences** mix two things:
  - Statistics segments: a word token is followed by tokens drawn from that word's own
    successor distribution.
  - Lookup segments: random key-value pairs, then a key is repeated and the model must
    produce its value. The repeated key is always in the form it was defined in.
- **Twins**, same architecture and initial seed:
  - SHARED: both forms of a word have the same successor distribution.
  - UNRELATED: each form has its own.
  - MIXED: half the words shared, half unrelated (dose check).
  - NORMALIZED: both forms map to one token ID. Ceiling, `T = 1` by construction.
- **Test:** lookups where the repeated key is in the other form. Same `T` as part (a).
- At least 5 seeds per twin.

### Predictions
- **P3.** `T(SHARED) - T(UNRELATED) > 0.2`, interval over seeds excludes 0. *Falsified*
  otherwise.
- **P4.** `T` rises in the order UNRELATED, MIXED, SHARED.
- **P5, exploratory.** Whether removing the MLPs from the twins changes `T`, in the
  direction part (a) suggests.

### Gate
I failed to train trustworthy induction heads in KAMUI before. If the twins don't reach
90% accuracy on same-form lookups, part (b) is reported as a failed setup, not as
evidence. The toy may be tuned only on same-form accuracy, before any cross-form `T` is
looked at.

## Optional supporting evidence
SAE feature overlap between variants, by layer, using public GPT-2 SAEs. Only if it
takes under an hour. Exploratory.

## What would make this not worth writing up
- P1 fails in both models: one-paragraph negative result.
- P1 holds but P2 is "mixed" and part (b) fails its gate: a measurement without a
  mechanism, which is a short post at most.
