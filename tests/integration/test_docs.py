"""Keep the documentation honest: run its code and check its commands.

Every ``python`` block under ``docs/`` runs here, so a renamed argument or an
invented method fails CI instead of reaching a reader.  Pseudocode belongs in a
``text`` block.

    - The Quickstart and the interpretability tutorial run as one session.
    - The training tutorial really runs, in a temporary folder: its Python
      blocks and its ``python scripts/...`` commands, in page order.
    - Reference pages run against a shared tiny model.  It's untrained because
      these pages show how to call the API, and quote results from the tutorials.
    - Every ``python scripts/...`` command in any guide is parsed by the real script.
"""

from __future__ import annotations

import argparse
import re
import runpy
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pytest
import torch

import kamui
from kamui.hooks import HookRegistry

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
TUTORIALS = DOCS / "tutorials"
GUIDES = [
    *sorted(DOCS.rglob("*.md")),
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "data" / "README.md",
    ROOT / "research" / "experiments" / "README.md",
]


def _fences(text: str) -> list[tuple[str, str]]:
    """Return ``(language, body)`` for each top-level fenced block, in order."""
    return re.findall(r"^```(\w*)\n(.*?)^```", text, re.M | re.S)


def _script_commands(body: str) -> list[str]:
    """Return the ``python scripts/...`` commands in a shell block."""
    lines = re.sub(r"\\\n\s*", " ", body).splitlines()
    return [line.strip() for line in lines if line.strip().startswith("python scripts/")]


def _read(page: Path) -> str:
    return page.read_text(encoding="utf-8")


def _exec_python(blocks: list[str], page: Path, namespace: dict[str, Any]) -> None:
    for index, block in enumerate(blocks, start=1):
        exec(compile(block, f"{page.name} block {index}", "exec"), namespace)


def _python_blocks(page: Path) -> list[str]:
    return [body for language, body in _fences(_read(page)) if language == "python"]


REFERENCE_PAGES = [
    p for p in DOCS.rglob("*.md") if TUTORIALS not in p.parents and _python_blocks(p)
]
COMMANDS = sorted(
    {
        command
        for page in GUIDES
        for language, body in _fences(_read(page))
        if language == "bash"
        for command in _script_commands(body)
    }
)


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """The names that reference pages say they use."""
    torch.manual_seed(0)
    corpus = "the cat sat on the mat . the dog ran to the log . " * 40
    config = kamui.ModelConfig(
        n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=300, context_length=32
    )
    tokenizer = kamui.BPETokenizer.train(corpus, vocab_size=config.vocab_size)
    tokens = tokenizer.encode(corpus)
    model = kamui.KAMUITransformer(config).eval()
    clean_ids = torch.tensor(tokens[:6])
    return {
        "__name__": "__main__",
        "model": model,
        "tokenizer": tokenizer,
        "tokens": tokens,
        "token_ids": clean_ids.unsqueeze(0),
        "clean_ids": clean_ids,
        "corrupted_ids": clean_ids.flip(0),
        "baseline_ids": torch.tensor(tokens[6:12]).unsqueeze(0),
    }


def test_reference_pages_are_discovered() -> None:
    assert len(REFERENCE_PAGES) >= 6
    assert all(p.is_relative_to(DOCS) for p in REFERENCE_PAGES)


@pytest.mark.parametrize("page", REFERENCE_PAGES, ids=lambda p: p.stem)
def test_reference_page_runs(
    page: Path, reference: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)  # pages use repository-relative paths
    try:
        _exec_python(_python_blocks(page), page, dict(reference))
    finally:
        plt.close("all")


def test_tutorials_run_as_one_session_and_reach_their_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for name in ("quickstart.md", "your_first_interpretability_experiment.md"):
        _exec_python(_python_blocks(TUTORIALS / name), TUTORIALS / name, namespace)

    ids, corrupted, patcher = namespace["ids"], namespace["corrupted"], namespace["patcher"]
    assert patcher.patch_all_layers(ids, corrupted, component="ffn").effects[0] > 0.9
    assert patcher.patch_all_layers(ids, corrupted, component="attn").effects.abs().max() < 0.1
    assert namespace["heads"].effects.abs().max() < 0.05
    assert namespace["circuit"] == ["blocks.0.ffn.output"]
    assert max(namespace["scores"].values()) < 0.5


def _run_script(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = shlex.split(command)[1:]
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        runpy.run_path(str(ROOT / argv[0]), run_name="__main__")


def test_training_tutorial_runs_for_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    page = TUTORIALS / "training_your_first_model.md"
    steps = _read(page).split("## Train on real text")[0]

    namespace: dict[str, Any] = {"__name__": "__main__"}
    for language, body in _fences(steps):
        if language == "python":
            _exec_python([body], page, namespace)
        elif language == "bash":
            for command in _script_commands(body):
                _run_script(command, monkeypatch)

    assert (tmp_path / "checkpoints" / "tiny" / "tokenizer.json").exists()
    assert (tmp_path / "checkpoints" / "tiny" / "step_0000200.pt").exists()
    assert "val_loss" in capsys.readouterr().out  # printed by inspect_checkpoint
    assert namespace["model"].config.n_layers == 2


class _ParsedError(Exception):
    """Raised in place of running a script, once its arguments have parsed."""


def test_commands_are_discovered() -> None:
    scripts = {shlex.split(command)[1] for command in COMMANDS}
    assert {"scripts/train.py", "scripts/evaluate.py", "scripts/tokenize_corpus.py"} <= scripts


@pytest.mark.parametrize("command", COMMANDS)
def test_documented_command_is_accepted(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    real_parse = argparse.ArgumentParser.parse_args

    def parse_then_stop(self: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> None:
        real_parse(self, *args, **kwargs)  # exits on an unknown or missing flag
        raise _ParsedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    with pytest.raises(_ParsedError):
        _run_script(command, monkeypatch)


@pytest.mark.parametrize("page", GUIDES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_page_installs_the_unrelated_pypi_package(page: Path) -> None:
    assert not re.search(r"^\s*pip install kamui\b", _read(page), re.M)


@pytest.mark.parametrize("page", sorted(DOCS.rglob("*.md")), ids=lambda p: p.stem)
def test_every_hook_point_in_the_docs_exists(page: Path) -> None:
    valid = set(HookRegistry.all_points(kamui.ModelConfig.from_yaml(ROOT / "configs/small.yaml")))
    named = re.findall(r"\b(?:un)?embed\.\w+|\bblocks\.\d+\.(?:attn|ffn)\.\w+", _read(page))
    assert set(named) <= valid
