"""Check that KAMUI really runs on a device (``kamui-check-device``).

Usage:
    kamui-check-device
    python scripts/check_device.py --device cuda

Trains a small model on the device, then runs every tool in the library on it
and prints PASS or FAIL for each.  Run this once on any new kind of hardware.
CI has no GPU, so this is how GPU support gets checked for real; the test suite
runs these same checks on a simulated GPU (``tests/simulated_gpu.py``).

Needs the plotting extra: ``pip install -e ".[all]"``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import tempfile
import traceback
from collections.abc import Callable, Collection
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from torch import Tensor

from kamui.evaluate.accuracy import compute_accuracy
from kamui.evaluate.calibration import (
    expected_calibration_error,
    reliability_diagram,
    temperature_scaling,
)
from kamui.evaluate.generation import generate, generate_with_probs
from kamui.evaluate.perplexity import (
    compute_perplexity,
    compute_sequence_perplexity,
    compute_token_loss,
)
from kamui.hooks.manager import HookManager
from kamui.mechinterp import (
    ActivationPatcher,
    AttentionVisualizer,
    CircuitAblator,
    FeatureSteerer,
    GradientAttribution,
    InductionHeadDetector,
    LinearProbe,
    LogitLens,
    SparseAutoencoder,
    build_steering_vector,
    collect_activations,
    feature_cooccurrence,
    feature_similarity,
    head_summary_stats,
    interpret_features,
    max_activating_examples,
    sae_feature_metrics,
    train_sae,
)
from kamui.mechinterp.circuits import find_minimal_circuit
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.tokenizer.bpe import BPETokenizer
from kamui.training import (
    EMA,
    DataLoader,
    TextDataset,
    Trainer,
    TrainingConfig,
    all_reduce_mean,
    destroy_process_group,
    init_process_group,
    load_checkpoint,
    save_checkpoint,
    wrap_ddp,
)
from kamui.utils.reproducibility import module_device, resolve_device

_HOOK = "blocks.1.ffn.output"
_PROMPT = "the cat sat on the "


@dataclass
class _Lab:
    """A small trained model, and what every check shares.

    ``ids`` and ``corrupted`` stay on the CPU on purpose: that's how they arrive
    from a tokenizer, so every check also tests that tools accept them.
    """

    device: torch.device
    model: KAMUITransformer
    tokenizer: BPETokenizer
    tokens: list[int]
    ids: Tensor
    corrupted: Tensor
    batches: list[tuple[Tensor, Tensor]]

    def fresh_model(self) -> KAMUITransformer:
        """Return an untrained model on the device, for checks that change weights."""
        return KAMUITransformer(self.model.config).to(self.device)

    def sequences(self) -> list[Tensor]:
        """Return a few windows of the corpus, as CPU tensors."""
        size = self.model.config.context_length
        return [torch.tensor(self.tokens[i : i + size]) for i in range(0, 4 * size, size)]

    def metric(self, logits: Tensor) -> float:
        """Score the model's top answer to ``ids``, for the ablation tools."""
        return float(logits[0, -1].max().item())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _short_training(steps: int = 4, **options: int) -> TrainingConfig:
    return TrainingConfig(max_lr=1e-3, warmup_steps=1, max_steps=steps + 1, **options)


def _build_lab(device: torch.device) -> _Lab:
    torch.manual_seed(0)
    rng = random.Random(0)
    animals, verbs = ["cat", "dog", "bird", "fish"], ["sat", "ran", "slept", "hid"]
    preps, objects = ["on", "under", "near"], ["mat", "log", "rug", "box"]
    corpus = " ".join(
        f"the {rng.choice(animals)} {rng.choice(verbs)} {rng.choice(preps)} "
        f"the {rng.choice(objects)} ."
        for _ in range(300)
    )
    config = ModelConfig(
        n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=320, context_length=32
    )
    tokenizer = BPETokenizer.train(corpus, vocab_size=config.vocab_size)
    tokens = tokenizer.encode(corpus)
    loader = DataLoader(TextDataset(tokens, config.context_length), batch_size=8, seed=0)

    model = KAMUITransformer(config).to(device)
    records = Trainer(model, loader, config=_short_training(steps=30)).train(30)
    _require(records[-1]["train_loss"] < records[0]["train_loss"], "training didn't lower the loss")
    model.eval()

    ids = torch.tensor(tokens[:8])
    return _Lab(
        device=device,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        ids=ids,
        corrupted=ids.flip(0),
        batches=list(islice(loader, 2)),
    )


def _check_placement(lab: _Lab) -> None:
    found = module_device(lab.model)
    _require(found.type == lab.device.type, f"the model is on {found}, not {lab.device}")
    ids_on_device = lab.ids.to(lab.device)
    _require(lab.model(ids_on_device.unsqueeze(0)).device.type == lab.device.type, "wrong output")


@torch.no_grad()
def _check_matches_cpu(lab: _Lab) -> None:
    cpu_model = KAMUITransformer(lab.model.config).eval()
    cpu_model.load_state_dict({k: v.cpu() for k, v in lab.model.state_dict().items()})
    batch = lab.ids.unsqueeze(0)
    difference = (lab.model(batch).cpu() - cpu_model(batch)).abs().max().item()
    _require(difference < 1e-3, f"logits differ from the CPU's by {difference}")


def _check_training_options(lab: _Lab) -> None:
    model = lab.fresh_model()
    config = _short_training(grad_accum_steps=2, eval_interval=2)
    trainer = Trainer(model, lab.batches, val_loader=lab.batches, config=config)
    records = trainer.train(2)
    _require(math.isfinite(records[-1]["val_loss"]), "validation loss isn't finite")


def _check_model_variants(lab: _Lab) -> None:
    for position in ("sinusoidal", "rope"):
        config = dataclasses.replace(
            lab.model.config, positional_encoding=position, normalization="rmsnorm"
        )
        model = KAMUITransformer(config).to(lab.device)
        Trainer(model, lab.batches, config=_short_training()).train(1)
        model.eval()

        batch, cache = lab.ids.unsqueeze(0), model.new_cache()
        whole = model(batch)[0, -1]
        model(batch[:, :-1], cache=cache)
        cached = model(batch[:, -1:], cache=cache)[0, -1]
        difference = (whole - cached).abs().max().item()
        _require(difference < 1e-3, f"the KV-cache changed {position} by {difference}")


def _check_gradient_checkpointing(lab: _Lab) -> None:
    model = lab.fresh_model()
    model.set_gradient_checkpointing(True)
    Trainer(model, lab.batches, config=_short_training()).train(2)


def _check_checkpoints(lab: _Lab) -> None:
    model = lab.fresh_model()
    trainer = Trainer(model, lab.batches, config=_short_training())
    trainer.train(1)
    ema = EMA(model, decay=0.9)
    ema.update(model)

    with tempfile.TemporaryDirectory() as folder:
        model_path, ema_path = Path(folder, "model.pt"), Path(folder, "ema.pt")
        save_checkpoint(model_path, model, trainer.optimizer, trainer.scheduler, step=trainer.step)
        torch.save(ema.state_dict(), ema_path)

        resumed = lab.fresh_model()
        resumed_trainer = Trainer(resumed, lab.batches, config=_short_training())
        load_checkpoint(model_path, resumed, resumed_trainer.optimizer, resumed_trainer.scheduler)
        resumed_ema = EMA(resumed, decay=0.9)
        resumed_ema.load_state_dict(torch.load(ema_path, map_location="cpu"))

    resumed_trainer.train(1)
    resumed_ema.update(resumed)
    with resumed_ema.average_parameters(resumed):
        resumed(lab.ids.unsqueeze(0))


def _check_generation(lab: _Lab) -> None:
    for strategy in ("greedy", "top_k", "nucleus", "temperature"):
        generate(lab.model, lab.tokenizer, _PROMPT, max_new_tokens=3, strategy=strategy, seed=0)
    generate(lab.model, lab.tokenizer, _PROMPT, max_new_tokens=3, repetition_penalty=1.3)
    generate_with_probs(lab.model, lab.tokenizer, _PROMPT, n_tokens=3)

    plain = generate(lab.model, lab.tokenizer, _PROMPT, max_new_tokens=6)
    cached = generate(lab.model, lab.tokenizer, _PROMPT, max_new_tokens=6, use_cache=True)
    _require(plain == cached, "the KV-cache changed the output")


def _check_metrics(lab: _Lab) -> None:
    compute_perplexity(lab.model, lab.batches)
    compute_accuracy(lab.model, lab.batches, k=2)
    compute_token_loss(lab.model, lab.ids)
    compute_sequence_perplexity(lab.model, torch.tensor(lab.tokens[:80]))
    temperature_scaling(lab.model, lab.batches, temperatures=torch.tensor([0.5, 1.0, 2.0]))

    inputs, targets = lab.batches[0]
    with torch.no_grad():
        logits = lab.model(inputs)
    probs, labels = logits.softmax(-1).reshape(-1, logits.shape[-1]), targets.reshape(-1)
    expected_calibration_error(probs, labels)
    reliability_diagram(probs, labels)


def _check_hooks(lab: _Lab) -> None:
    with HookManager(lab.model) as hooks:
        hooks.attach("blocks.0.attn", "weights")
        hooks.attach("blocks.0.ffn", "mid")
        hooks.attach("unembed", "input")
        lab.model(lab.ids.unsqueeze(0))
        _require(len(hooks.get_all()) == 3, "hooks missed an activation")


def _check_logit_lens(lab: _Lab) -> None:
    result = LogitLens(lab.model, lab.tokenizer).run(lab.ids)
    result.plot()
    result.plot_position(pos=len(result.tokens) - 1)


def _check_attention(lab: _Lab) -> None:
    result = AttentionVisualizer(lab.model, lab.tokenizer).run(lab.ids)
    result.plot(0, 0)
    result.plot_all()
    head_summary_stats(result)


def _check_patching(lab: _Lab) -> None:
    patcher = ActivationPatcher(lab.model)
    patcher.patch_single(lab.ids, lab.corrupted, "blocks.0.ffn.output")
    patcher.patch_all_layers(lab.ids, lab.corrupted, component="ffn").plot()
    patcher.patch_all_heads(lab.ids, lab.corrupted).plot()
    patcher.patch_all_layers(lab.ids.to(lab.device), lab.corrupted.to(lab.device))


def _check_ablation(lab: _Lab) -> None:
    ablator = CircuitAblator(lab.model)
    ablator.ablate(["blocks.0.attn.output"], lab.ids, lab.metric)
    ablator.mean_ablate(["blocks.0.ffn.output"], lab.ids, lab.corrupted, lab.metric)
    find_minimal_circuit(lab.model, lab.ids, lab.metric, threshold=0.5)


def _check_induction(lab: _Lab) -> None:
    detector = InductionHeadDetector(lab.model, token_ids=set(lab.tokens))
    scores = detector.score_all_heads(prefix_len=12, n_samples=2)
    detector.plot_scores(scores)
    detector.ablate_and_measure(heads=[(1, 0)], prefix_len=12)


def _check_probing(lab: _Lab) -> None:
    sequences = lab.sequences() * 2
    labels = [i % 2 for i in range(len(sequences))]
    probe = LinearProbe(lab.model)
    probe.train(_HOOK, sequences, labels, epochs=2)
    probe.probe_all_layers(sequences, labels, epochs=2).plot()


def _check_attribution(lab: _Lab) -> None:
    attribution = GradientAttribution(lab.model, lab.tokenizer)
    attribution.token_attribution(lab.ids).plot()
    attribution.token_attribution(lab.ids, method="integrated_gradients", steps=2)


def _check_sae_tools(lab: _Lab, sae_device: torch.device) -> None:
    sequences = lab.sequences()
    activations = collect_activations(lab.model, _HOOK, sequences)
    _require(activations.device.type == "cpu", "activations should stay on the CPU")
    d_model = lab.model.config.d_model
    sae = SparseAutoencoder(d_model=d_model, n_features=64, l1_coeff=0.1).to(sae_device)
    train_sae(sae, activations, epochs=2, batch_size=64)

    sae_feature_metrics(sae, activations)
    interpret_features(sae, activations, torch.cat(sequences), top_k=3)
    feature_cooccurrence(sae, activations)
    feature_similarity(sae)
    max_activating_examples(lab.model, sae, _HOOK, sequences[:2], feature=0, top_k=2)

    steerer = FeatureSteerer(lab.model, sae)
    steerer.steer_with_feature(lab.ids, _HOOK, feature=0, coefficient=2.0)
    steerer.ablate_features(lab.ids, _HOOK, [0, 1])
    steerer.ablate_features(lab.ids, _HOOK, [0], position=2)
    steerer.generate_steered_with_feature(
        lab.tokenizer, _PROMPT, _HOOK, feature=0, max_new_tokens=2
    )

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder, "sae.pt")
        sae.save(path)
        SparseAutoencoder.load(path)


def _check_sae_on_device(lab: _Lab) -> None:
    _check_sae_tools(lab, lab.device)


def _check_sae_on_cpu(lab: _Lab) -> None:
    _check_sae_tools(lab, torch.device("cpu"))


def _check_steering(lab: _Lab) -> None:
    steerer = FeatureSteerer(lab.model)
    direction = build_steering_vector(lab.model, _HOOK, [lab.ids], [lab.corrupted], normalize=True)
    result = steerer.steer(lab.ids, _HOOK, direction, coefficient=2.0)
    result.top_promoted(3)
    result.top_suppressed(3)
    result.plot()
    steerer.steer(lab.ids, _HOOK, direction, coefficient=2.0, position=1)
    steerer.steer(lab.ids, _HOOK, direction.to(lab.device), coefficient=2.0)
    steerer.generate_steered(lab.tokenizer, _PROMPT, _HOOK, direction, max_new_tokens=2)


def _check_distributed(lab: _Lab) -> None:
    on_cuda = lab.device.type == "cuda"
    with tempfile.TemporaryDirectory() as folder:
        init_process_group(
            0,
            1,
            backend="nccl" if on_cuda else "gloo",
            init_method=Path(folder, "rendezvous").resolve().as_uri(),
        )
        try:
            model = wrap_ddp(
                lab.fresh_model(), device_ids=[lab.device.index or 0] if on_cuda else None
            )
            Trainer(model, lab.batches, config=_short_training()).train(2)
            _require(all_reduce_mean(2.0) == 2.0, "all_reduce_mean changed a single value")
        finally:
            destroy_process_group()


#: Every check, in the order they run.
CHECKS: dict[str, Callable[[_Lab], None]] = {
    "model and outputs are on the device": _check_placement,
    "results match the CPU": _check_matches_cpu,
    "training options": _check_training_options,
    "model variants": _check_model_variants,
    "gradient checkpointing": _check_gradient_checkpointing,
    "checkpoints and EMA": _check_checkpoints,
    "generation": _check_generation,
    "metrics": _check_metrics,
    "hooks": _check_hooks,
    "logit lens": _check_logit_lens,
    "attention": _check_attention,
    "activation patching": _check_patching,
    "circuit ablation": _check_ablation,
    "induction heads": _check_induction,
    "linear probing": _check_probing,
    "gradient attribution": _check_attribution,
    "SAE on the device": _check_sae_on_device,
    "SAE on the CPU": _check_sae_on_cpu,
    "steering": _check_steering,
    "distributed training": _check_distributed,
}


@dataclass
class CheckResult:
    """The outcome of one check.

    Attributes:
        name:  The check's name.
        error: The traceback if it failed, or ``None`` if it passed.
    """

    name: str
    error: str | None = None


def run_checks(device: torch.device, skip: Collection[str] = ()) -> list[CheckResult]:
    """Train a small model on ``device`` and run every check against it.

    Args:
        device: The device to check.
        skip:   Names of checks to leave out.

    Returns:
        One ``CheckResult`` for training, then one per check that ran.  If
        training fails, nothing else can run and it's the only result.
    """
    import matplotlib.pyplot as plt

    try:
        lab = _build_lab(device)
    except Exception:
        return [CheckResult("training", traceback.format_exc())]

    results = [CheckResult("training")]
    for name, check in CHECKS.items():
        if name in skip:
            continue
        try:
            check(lab)
            results.append(CheckResult(name))
        except Exception:
            results.append(CheckResult(name, traceback.format_exc()))
        finally:
            plt.close("all")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check that KAMUI runs on a device")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', ...")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run every check, print the results, and exit non-zero if any failed."""
    import matplotlib

    args = parse_args(argv)
    matplotlib.use("Agg")
    device = resolve_device(args.device)
    print(f"Checking KAMUI on {device}", flush=True)

    results = run_checks(device)
    for result in results:
        print(f"{'FAIL' if result.error else 'PASS'}  {result.name}")
    failures = [result for result in results if result.error]
    for result in failures:
        print(f"\n--- {result.name} ---\n{result.error}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(results)} checks failed on {device}")
    print(f"All {len(results)} checks passed on {device}")


if __name__ == "__main__":
    main()
