"""Keep the README honest: run its Quickstart and check each chapter's claims.

The Quickstart's code blocks are executed exactly as written, in order, in one
namespace.  The tests then check what the story says happened to that model.
They assert the claims rather than exact decimals, because the README's numbers
may vary slightly by machine while the claims must hold everywhere.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pytest
import torch

import kamui
from kamui.hooks.manager import HookManager
from kamui.mechinterp import (
    CircuitAblator,
    FeatureSteerer,
    InductionHeadDetector,
    collect_activations,
    sae_feature_metrics,
)
from kamui.training import DataLoader, TextDataset, TrainingConfig

ROOT = Path(__file__).resolve().parents[2]


def _first_word(text: str) -> str:
    return (text.split() or [""])[0]


@pytest.fixture(scope="module")
def story() -> dict[str, Any]:
    """Run every Python block of the README's Quickstart in one namespace."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Quickstart") : text.index("## The toolkit")]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) >= 6, "the Quickstart should hold the story's code blocks"

    namespace: dict[str, Any] = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)  # the blocks use repository-relative paths
        for index, block in enumerate(blocks, start=1):
            exec(compile(block, f"README.md Quickstart block {index}", "exec"), namespace)
    return namespace


def test_chapter_3_windows_do_not_change_the_model(story: dict[str, Any]) -> None:
    model, ids = story["model"], story["ids"]
    with torch.no_grad():
        plain = model(ids.unsqueeze(0))
        with HookManager(model) as hooks:
            hooks.attach("blocks.0.ffn", "output")
            hooked = model(ids.unsqueeze(0))
    assert torch.equal(plain, hooked)


def test_chapter_4_random_words_leave_nothing_to_see() -> None:
    torch.manual_seed(0)
    rng = random.Random(0)
    words = ["the", "cat", "dog", "sat", "ran", "on", "to", "mat", "log", "sun"]
    corpus = " ".join(rng.choice(words) for _ in range(4000))
    config = kamui.ModelConfig(
        n_layers=2, d_model=64, n_heads=4, d_ff=128, vocab_size=300, context_length=32
    )
    tokenizer = kamui.BPETokenizer.train(corpus, vocab_size=config.vocab_size)
    tokens = tokenizer.encode(corpus)
    model = kamui.KAMUITransformer(config)
    trainer = kamui.Trainer(
        model,
        DataLoader(TextDataset(tokens, config.context_length), batch_size=8, seed=0),
        config=TrainingConfig(max_lr=3e-3, warmup_steps=10, max_steps=1000),
    )
    records = trainer.train(150)
    model.eval()

    seqs = [torch.tensor(tokens[i : i + 32]) for i in range(0, 60 * 32, 32)]
    acts = collect_activations(model, "blocks.1.ffn.output", seqs)
    mean = acts.mean(dim=0)
    assert (acts - mean).norm(dim=1).mean() < 0.05 * mean.norm()  # same reading for every word
    assert records[-1]["train_loss"] > 2.5  # the error stalls


def test_chapter_4_real_sentences_lower_the_error(story: dict[str, Any]) -> None:
    history = story["trainer"].history
    assert history[0]["train_loss"] > 5.0
    assert history[-1]["train_loss"] < 2.5


def test_chapter_5_the_thought_forms_in_the_first_block(story: dict[str, Any]) -> None:
    result, tokenizer = story["result"], story["tokenizer"]
    guesses = [
        _first_word(tokenizer.decode([result.probs[layer, -1].argmax().item()]))
        for layer in range(result.probs.shape[0])
    ]
    assert guesses[0] in story["verbs"]  # before any block: a wrong kind of word
    assert guesses[1] in story["objects"]  # after the first block: an object
    assert guesses[2] in story["objects"]


def test_chapter_6_patching_finds_the_first_feed_forward(story: dict[str, Any]) -> None:
    patcher, ids, corrupted = story["patcher"], story["ids"], story["corrupted"]
    attn = patcher.patch_all_layers(ids, corrupted, component="attn").effects
    ffn = patcher.patch_all_layers(ids, corrupted, component="ffn").effects
    assert attn.abs().max() < 0.1
    assert ffn[0] > 0.9
    assert ffn[1] < 0.3


def test_chapter_6_ablation_agrees(story: dict[str, Any]) -> None:
    model, ids, tokenizer = story["model"], story["ids"], story["tokenizer"]
    target = tokenizer.encode("mat . the ")[0]

    def metric(logits: torch.Tensor) -> float:
        return logits[0, -1, target].item()

    ablator = CircuitAblator(model)
    deltas = {
        name: ablator.ablate([name], ids, metric).delta
        for name in (
            "blocks.0.attn.output",
            "blocks.0.ffn.output",
            "blocks.1.attn.output",
            "blocks.1.ffn.output",
        )
    }
    culprit = deltas.pop("blocks.0.ffn.output")
    assert culprit < -5.0
    assert all(abs(delta) < 1.0 for delta in deltas.values())


def test_chapter_7_specialists_are_sparse_and_readable(story: dict[str, Any]) -> None:
    sae, acts, tokenizer = story["sae"], story["acts"], story["tokenizer"]
    metrics = sae_feature_metrics(sae, acts)
    assert 1 - metrics.reconstruction_mse / acts.var(dim=0).mean().item() > 0.95
    assert metrics.mean_l0 < 16  # a handful of the 256 on any word

    groups = [
        {_first_word(tokenizer.decode([i])) for i in p.top_token_ids} for p in story["selective"]
    ]
    assert any(group <= set(story["animals"]) for group in groups)
    assert any(group <= set(story["verbs"]) for group in groups)
    assert any(group <= set(story["objects"]) for group in groups)


def test_chapter_8_the_dial_works(story: dict[str, Any]) -> None:
    model, tokenizer = story["model"], story["tokenizer"]
    words, starts = story["words"], story["starts"]
    animal_ids = [t for t in words if starts(t, story["animals"])]
    object_ids = [t for t in words if starts(t, story["objects"])]

    slot = torch.tensor(tokenizer.encode("the bird hid near the box . the "))
    with torch.no_grad():
        before = model(slot.unsqueeze(0))[0, -1].softmax(-1)
    steered = FeatureSteerer(model).steer(slot, story["hook"], story["direction"], coefficient=4.0)
    after = steered.steered_logits[0, -1].softmax(-1)

    assert before[animal_ids].sum() > 0.9
    assert after[animal_ids].sum() < 0.3
    assert after[object_ids].sum() > 0.7


def test_chapter_9_no_strong_induction_heads(story: dict[str, Any]) -> None:
    detector = InductionHeadDetector(story["model"], token_ids=set(story["tokens"]))
    scores = detector.score_all_heads(prefix_len=12, n_samples=4)
    assert max(scores.values()) < 0.5
