"""Unit tests for training-loop mechanics.

These verify the *mechanics* of training — overfitting a fixed batch,
gradient accumulation equivalence, LR schedule behaviour inside the loop,
and weight-decay group assignment — complementing tests/unit/test_training.py.
"""

from __future__ import annotations

import pytest
import torch

from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.training.trainer import Trainer, TrainingConfig


def _config() -> ModelConfig:
    return ModelConfig(
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=32,
        context_length=8,
        dropout=0.0,
    )


def _fixed_batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    ids = torch.randint(0, 32, (8, 8))
    return ids, torch.randint(0, 32, (8, 8))


def test_loss_decreases_on_single_fixed_batch() -> None:
    """The model must overfit one fixed batch — the most basic sanity check.

    If loss does not decrease here, the training loop has a fundamental bug.
    """
    torch.manual_seed(0)
    batch = _fixed_batch()
    trainer = Trainer(
        KAMUITransformer(_config()),
        train_loader=[batch],
        config=TrainingConfig(max_lr=1e-2, warmup_steps=2, max_steps=1000),
    )
    records = trainer.train(40)
    assert records[-1]["train_loss"] < records[0]["train_loss"] * 0.5


def test_gradient_accumulation_matches_large_batch() -> None:
    """2 accumulation steps over two half-batches must equal 1 full-batch step.

    Same data, same init: the accumulated *gradients* must match the
    full-batch gradients to numerical precision.  (Gradients, not post-step
    parameters: Adam's first update is ~``lr * sign(g)``, so a float-rounding
    difference in a near-zero gradient would flip an entire update — that
    comparison is flaky by construction across platforms.)
    """
    ids, targets = _fixed_batch()
    half_a = (ids[:4], targets[:4])
    half_b = (ids[4:], targets[4:])

    torch.manual_seed(7)
    model_accum = KAMUITransformer(_config())
    torch.manual_seed(7)
    model_full = KAMUITransformer(_config())

    config = TrainingConfig(max_lr=1e-3, warmup_steps=0, max_steps=100, max_grad_norm=0.0)
    trainer_accum = Trainer(
        model_accum,
        train_loader=[half_a, half_b],
        config=TrainingConfig(
            max_lr=1e-3, warmup_steps=0, max_steps=100, max_grad_norm=0.0, grad_accum_steps=2
        ),
    )
    trainer_full = Trainer(model_full, train_loader=[(ids, targets)], config=config)

    # train(1) computes gradients on the identical initial parameters; .grad
    # is populated before the optimiser step and not cleared until the next
    # update, so it can be compared directly afterwards.
    trainer_accum.train(1)
    trainer_full.train(1)

    for p_a, p_f in zip(model_accum.parameters(), model_full.parameters(), strict=True):
        assert p_a.grad is not None and p_f.grad is not None
        assert torch.allclose(p_a.grad, p_f.grad, atol=1e-6)


def test_lr_warmup_ramps_inside_loop() -> None:
    """The optimiser LR must follow the linear warmup during training."""
    trainer = Trainer(
        KAMUITransformer(_config()),
        train_loader=[_fixed_batch()],
        config=TrainingConfig(max_lr=1.0, min_lr=0.0, warmup_steps=10, max_steps=100),
    )
    records = trainer.train(5)
    # Update k (1-indexed) applies the schedule at step k-1: lr = max_lr * (k-1)/10.
    for k, record in enumerate(records):
        assert record["lr"] == pytest.approx(k / 10)


def test_lr_cosine_decay_reaches_min_lr() -> None:
    """The schedule must floor at min_lr at and beyond max_steps."""
    trainer = Trainer(
        KAMUITransformer(_config()),
        train_loader=[_fixed_batch()],
        config=TrainingConfig(max_lr=1.0, min_lr=0.1, warmup_steps=0, max_steps=3),
    )
    records = trainer.train(6)
    assert records[-1]["lr"] == pytest.approx(0.1)
    assert min(r["lr"] for r in records) >= 0.1


def test_weight_decay_not_applied_to_biases_or_layernorm() -> None:
    """Biases and LayerNorm parameters must sit in the zero-decay group."""
    trainer = Trainer(KAMUITransformer(_config()), train_loader=[_fixed_batch()])
    decay_group, no_decay_group = trainer.optimizer.param_groups
    assert decay_group["weight_decay"] > 0.0
    assert no_decay_group["weight_decay"] == 0.0
    # Every 1-D parameter (biases, LayerNorm gamma/beta) is in the no-decay group.
    assert all(p.dim() < 2 for p in no_decay_group["params"])
    assert all(p.dim() >= 2 for p in decay_group["params"])
