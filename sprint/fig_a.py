"""Regenerate the part (a) figures from sprint/results/*.json.

    python sprint/fig_a.py

Writes sprint/figures/a1_tolerance.png and sprint/figures/a1_heads.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
MODELS = ("gpt2", "pythia-160m")


def load(prefix: str, model: str) -> dict | None:
    path = RESULTS / f"{prefix}_{model}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def tolerance_figure() -> None:
    """Tolerance by variant kind, with the post-hoc neighbour control beside it."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 3.6), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        a1, a2 = load("a1", model), load("a2", model)
        if a1 is None:
            continue
        bars = [(k, a1["T_logit"][k]) for k in ("space", "case", "both", "all")]
        if a2 is not None:
            bars += [("variant\n(matched)", a2["T_variant"]), ("neighbour\n(post-hoc)", a2["T_neighbour"])]
        labels = [label for label, _ in bars]
        values = [t["T"] for _, t in bars]
        errors = [[t["T"] - t["low"] for _, t in bars], [t["high"] - t["T"] for _, t in bars]]
        colors = ["#4c72b0"] * 4 + ["#4c72b0", "#dd8452"][: len(bars) - 4]
        ax.bar(range(len(bars)), values, yerr=errors, color=colors, capsize=3)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(model)
    axes[0].set_ylabel("tolerance T (0 = stranger, 1 = same token)")
    fig.suptitle("Induction tolerates surface variants of the query token (95% CI over words)")
    fig.tight_layout()
    fig.savefig(FIGURES / "a1_tolerance.png", dpi=150)


def heads_figure() -> None:
    """Per-head attention to B: same token vs variant vs control."""
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(9, 6))
    for ax, model in zip(axes, MODELS, strict=True):
        a1 = load("a1", model)
        if a1 is None:
            continue
        heads = [(name, t) for name, t in a1["T_attn"].items() if t["induction"]]
        x = range(len(heads))
        for offset, condition, color in ((-0.25, "same", "#4c72b0"), (0, "variant", "#55a868"),
                                         (0.25, "control", "#c44e52")):
            ax.bar([i + offset for i in x], [t["means"][condition] for _, t in heads],
                   width=0.25, color=color, label=condition)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{name}\nT={t['T']:.2f}" for name, t in heads], fontsize=7)
        ax.set_ylabel("attention to B")
        ax.set_title(f"{model}: induction heads")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "a1_heads.png", dpi=150)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    tolerance_figure()
    heads_figure()
    print("wrote", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
