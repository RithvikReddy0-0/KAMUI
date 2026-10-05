"""Unit tests for gradient (activation) checkpointing on KAMUITransformer.

Checkpointing must be invisible to the maths: identical losses, gradients and
training trajectories, with or without dropout.  It must only engage while
training with gradients enabled, and it must actually cut the activation memory
held for the backward pass.
"""

from __future__ import annotations

import copy

import torch

from kamui.evaluate.generation import generate
from kamui.hooks.manager import HookManager
from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.training.trainer import Trainer, TrainingConfig


def _model(dropout: float = 0.0) -> KAMUITransformer:
    torch.manual_seed(0)
    config = ModelConfig(
        n_layers=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=50,
        context_length=16,
        dropout=dropout,
    )
    return KAMUITransformer(config)


def _pair(dropout: float = 0.0) -> tuple[KAMUITransformer, KAMUITransformer]:
    plain = _model(dropout)
    checkpointed = copy.deepcopy(plain)
    checkpointed.set_gradient_checkpointing(True)
    return plain, checkpointed


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.randint(0, 50, (4, 16)), torch.randint(0, 50, (4, 16))


def _loss_and_grads(model: KAMUITransformer, seed: int = 7) -> tuple[torch.Tensor, list]:
    ids, targets = _batch()
    torch.manual_seed(seed)  # same dropout masks for both models
    loss = model(ids, targets=targets)
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in model.parameters()]


def _saved_bytes(model: KAMUITransformer) -> int:
    """Bytes autograd stores for the backward pass (outside checkpointed regions)."""
    total = 0

    def pack(t: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += t.numel() * t.element_size()
        return t

    ids, targets = _batch()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(ids, targets=targets)
    loss.backward()
    return total


class TestSwitch:
    def test_off_by_default(self) -> None:
        assert _model().gradient_checkpointing is False

    def test_set_and_unset(self) -> None:
        model = _model()
        model.set_gradient_checkpointing()
        assert model.gradient_checkpointing is True
        model.set_gradient_checkpointing(False)
        assert model.gradient_checkpointing is False


class TestNumericalEquivalence:
    def test_same_loss_and_gradients(self) -> None:
        plain, checkpointed = _pair()
        plain.train()
        checkpointed.train()
        loss_a, grads_a = _loss_and_grads(plain)
        loss_b, grads_b = _loss_and_grads(checkpointed)
        assert torch.allclose(loss_a, loss_b, atol=1e-6)
        for ga, gb in zip(grads_a, grads_b, strict=True):
            assert torch.allclose(ga, gb, atol=1e-6)

    def test_same_gradients_with_dropout(self) -> None:
        # Recompute must replay the same dropout masks as the original forward.
        plain, checkpointed = _pair(dropout=0.3)
        plain.train()
        checkpointed.train()
        loss_a, grads_a = _loss_and_grads(plain)
        loss_b, grads_b = _loss_and_grads(checkpointed)
        assert torch.allclose(loss_a, loss_b, atol=1e-6)
        for ga, gb in zip(grads_a, grads_b, strict=True):
            assert torch.allclose(ga, gb, atol=1e-6)

    def test_training_trajectory_is_unchanged(self) -> None:
        plain, checkpointed = _pair()
        loader = [_batch() for _ in range(2)]
        config = TrainingConfig(max_steps=10, warmup_steps=1, max_lr=1e-2)
        Trainer(plain, loader, config=config).train(4)
        Trainer(checkpointed, loader, config=config).train(4)
        for pa, pb in zip(plain.parameters(), checkpointed.parameters(), strict=True):
            assert torch.allclose(pa, pb, atol=1e-5)


class TestMemory:
    def test_cuts_activation_memory(self) -> None:
        plain, checkpointed = _pair()
        plain.train()
        checkpointed.train()
        assert _saved_bytes(checkpointed) < 0.5 * _saved_bytes(plain)

    def test_inactive_in_eval_mode(self) -> None:
        # Gradients still flow in eval mode, but checkpointing must not engage.
        plain, checkpointed = _pair()
        plain.eval()
        checkpointed.eval()
        assert _saved_bytes(checkpointed) == _saved_bytes(plain)


class TestLeavesOtherPathsAlone:
    def test_no_grad_forward_is_identical(self) -> None:
        plain, checkpointed = _pair()
        plain.train()
        checkpointed.train()
        ids, _ = _batch()
        with torch.no_grad():
            assert torch.equal(plain(ids), checkpointed(ids))

    def test_cached_generation_is_identical(self) -> None:
        plain, checkpointed = _pair()

        class Tok:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 50 for c in text]

            def decode(self, ids: list[int]) -> str:
                return ",".join(map(str, ids))

        a = generate(plain, Tok(), "hello", max_new_tokens=6, use_cache=True)
        b = generate(checkpointed, Tok(), "hello", max_new_tokens=6, use_cache=True)
        assert a == b

    def test_hooks_capture_during_checkpointed_training(self) -> None:
        plain, checkpointed = _pair()
        plain.train()
        checkpointed.train()
        ids, targets = _batch()
        captured = []
        for model in (plain, checkpointed):
            with HookManager(model) as hooks:
                hooks.attach("blocks.1.ffn", "output")
                model(ids, targets=targets).backward()
                captured.append(hooks.get("blocks.1.ffn.output"))
        assert torch.allclose(captured[0], captured[1], atol=1e-6)
