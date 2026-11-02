"""Keep every tool working on a GPU, on machines that have no GPU.

``kamui-check-device`` runs every tool on a device.  Here the same checks run on
the CPU and on ``SimulatedGPU``, which fails the way real CUDA does when tensors
are on the wrong device.  Real hardware has the final word, which is why the
checks are a script that anyone can run on a GPU.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import kamui.scripts.evaluate as evaluate_script
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.scripts import check_device
from kamui.scripts.check_device import CHECKS, CheckResult, run_checks
from kamui.scripts.train import main as train_main
from kamui.training import Trainer
from kamui.utils.reproducibility import module_device
from tests.simulated_gpu import CPU, GPU, SimulatedGPU

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]

#: These need real CUDA state (its RNG, and a process group), so the simulator can't run them.
NEEDS_REAL_CUDA = {"gradient checkpointing", "distributed training"}


def _as_dict(results: list[CheckResult]) -> dict[str, str | None]:
    return {result.name: result.error for result in results}


@pytest.fixture(scope="module")
def gpu_results() -> dict[str, str | None]:
    with SimulatedGPU():
        return _as_dict(run_checks(GPU, skip=NEEDS_REAL_CUDA))


@pytest.fixture(scope="module")
def cpu_results() -> dict[str, str | None]:
    return _as_dict(run_checks(CPU))


@pytest.mark.parametrize("name", ["training", *CHECKS])
def test_check_passes_on_the_cpu(name: str, cpu_results: dict[str, str | None]) -> None:
    assert name in cpu_results
    assert cpu_results[name] is None, cpu_results[name]


@pytest.mark.parametrize("name", ["training", *(n for n in CHECKS if n not in NEEDS_REAL_CUDA)])
def test_check_passes_on_a_simulated_gpu(name: str, gpu_results: dict[str, str | None]) -> None:
    assert name in gpu_results
    assert gpu_results[name] is None, gpu_results[name]


def test_only_named_checks_are_skipped_on_the_simulator() -> None:
    assert set(CHECKS) >= NEEDS_REAL_CUDA


class TestSimulatedGPU:
    """The simulator is only worth trusting if it fails when CUDA would."""

    def test_mixing_devices_raises(self) -> None:
        with SimulatedGPU():
            gpu = torch.ones(3).to("cuda")
            with pytest.raises(RuntimeError, match="same device"):
                gpu + torch.ones(3)
            with pytest.raises(RuntimeError, match="same device"):
                torch.cat([gpu, torch.ones(3)])

    def test_tensors_report_their_device(self) -> None:
        with SimulatedGPU():
            cpu = torch.ones(3)
            gpu = cpu.to("cuda")
            assert cpu.device == CPU
            assert gpu.device == GPU
            assert (gpu + gpu).device == GPU
            assert gpu.cpu().device == CPU
            assert cpu.cuda().device == GPU
            assert cpu.to(gpu).device == GPU

    def test_moving_copies_like_a_real_transfer(self) -> None:
        with SimulatedGPU():
            cpu = torch.ones(3)
            gpu = cpu.to("cuda")
            assert cpu.device == CPU  # the original stays behind
            assert gpu.to(GPU) is gpu  # no copy when it's already there

    def test_a_dtype_change_stays_on_the_device(self) -> None:
        with SimulatedGPU():
            gpu = torch.ones(3).to("cuda")
            assert gpu.to(torch.float64).device == GPU
            assert gpu.to(torch.float64).dtype == torch.float64

    def test_scalars_may_mix(self) -> None:
        with SimulatedGPU():
            assert (torch.ones(3).to("cuda") * torch.tensor(2.0)).device == GPU

    def test_a_gpu_tensor_takes_cpu_indices(self) -> None:
        with SimulatedGPU():
            gpu = torch.arange(4).to("cuda")
            assert gpu[torch.tensor([0, 1])].device == GPU

    def test_a_cpu_tensor_rejects_gpu_indices(self) -> None:
        with SimulatedGPU(), pytest.raises(RuntimeError, match="indices should be"):
            torch.arange(4)[torch.tensor([0, 1]).to("cuda")]

    def test_assignment_and_copy_may_cross_devices(self) -> None:
        with SimulatedGPU():
            gpu = torch.zeros(3).to("cuda")
            gpu[0:2] = torch.ones(2)
            gpu.copy_(torch.ones(3))
            assert gpu.device == GPU

    def test_numpy_needs_the_cpu(self) -> None:
        with SimulatedGPU(), pytest.raises(TypeError, match="numpy"):
            torch.ones(3).to("cuda").numpy()

    def test_factories_honour_device(self) -> None:
        with SimulatedGPU():
            assert torch.zeros(3, device="cuda").device == GPU
            assert torch.arange(3, device=GPU).device == GPU
            assert torch.zeros_like(torch.ones(3).to("cuda")).device == GPU

    def test_a_cpu_generator_cant_fill_a_gpu_tensor(self) -> None:
        with SimulatedGPU(), pytest.raises(RuntimeError, match="generator"):
            torch.randint(0, 9, (3,), generator=torch.Generator(), device="cuda")

    def test_modules_move_with_their_buffers(self) -> None:
        with SimulatedGPU():
            layer = nn.BatchNorm1d(3).to("cuda")
            assert layer.weight.device == GPU
            assert layer.running_mean.device == GPU
            assert module_device(layer) == GPU
            assert layer(torch.ones(2, 3).to("cuda")).device == GPU

    def test_a_loaded_tensor_lands_on_the_cpu(self) -> None:
        with SimulatedGPU():
            restored = pickle.loads(pickle.dumps(torch.ones(3).to("cuda")))
            assert restored.device == CPU

    def test_a_loss_needs_its_targets_on_the_device(self) -> None:
        with SimulatedGPU(), pytest.raises(RuntimeError, match="same device"):
            F.cross_entropy(torch.zeros(2, 4).to("cuda"), torch.tensor([0, 1]))

    def test_the_model_moves_its_inputs_to_its_own_device(self) -> None:
        config = ModelConfig(
            n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=32, context_length=8
        )
        seen: list[torch.device] = []
        with SimulatedGPU():
            model = KAMUITransformer(config).to("cuda")
            model.embed.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].device))
            ids = torch.zeros(1, 4, dtype=torch.long)
            assert model(ids).device == GPU
            assert model(ids, targets=ids).device == GPU
        assert seen == [GPU, GPU]


class TestCommands:
    def test_training_and_evaluation_use_the_requested_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus = tmp_path / "corpus.txt"
        rng = random.Random(0)
        words = ["cat", "dog", "bird", "sat", "ran", "hid", "mat", "log", "rug"]
        corpus.write_text(" ".join(rng.choice(words) for _ in range(3000)), "utf-8")
        monkeypatch.chdir(ROOT)

        used: dict[str, torch.device] = {}
        real_train = Trainer.train

        def spy_on_training(self: Trainer, n_steps: int) -> list[dict[str, float]]:
            used["train"] = module_device(self.model)
            return real_train(self, n_steps)

        real_generate = evaluate_script.generate

        def spy_on_generation(model: nn.Module, *args: Any, **kwargs: Any) -> str:
            used["evaluate"] = module_device(model)
            return real_generate(model, *args, **kwargs)

        monkeypatch.setattr(Trainer, "train", spy_on_training)
        monkeypatch.setattr(evaluate_script, "generate", spy_on_generation)

        config = str(ROOT / "configs" / "tiny.yaml")
        with SimulatedGPU():
            train_main(
                ["--config", config, "--corpus", str(corpus), "--steps", "2"]
                + ["--out", str(tmp_path / "run"), "--device", "cuda"]
            )
            evaluate_script.main(
                ["--checkpoint", str(tmp_path / "run" / "step_0000002.pt"), "--config", config]
                + ["--tokenizer", str(tmp_path / "run" / "tokenizer.json")]
                + ["--generate", "the cat", "--max-new-tokens", "2", "--device", "cuda"]
            )

        assert used == {"train": GPU, "evaluate": GPU}

    def test_a_clean_run_reports_every_check(self, capsys: pytest.CaptureFixture[str]) -> None:
        check_device.main(["--device", "cpu"])
        output = capsys.readouterr().out
        assert "Checking KAMUI on cpu" in output
        assert "PASS  distributed training" in output
        assert f"All {len(CHECKS) + 1} checks passed on cpu" in output

    def test_a_failure_exits_nonzero_and_shows_why(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken(_lab: Any) -> None:
            raise RuntimeError("this device is on fire")

        monkeypatch.setattr(check_device, "CHECKS", {"broken": broken})
        with pytest.raises(SystemExit, match="1 of 2 checks failed on cpu"):
            check_device.main(["--device", "cpu"])
        output = capsys.readouterr().out
        assert "FAIL  broken" in output
        assert "this device is on fire" in output

    def test_nothing_else_runs_if_training_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(_device: torch.device) -> None:
            raise RuntimeError("no memory left")

        monkeypatch.setattr(check_device, "_build_lab", broken)
        results = run_checks(CPU)
        assert [result.name for result in results] == ["training"]
        assert "no memory left" in (results[0].error or "")
