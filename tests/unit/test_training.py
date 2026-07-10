"""Unit tests for kamui.training (scheduler, optimizer, data, checkpointing, trainer).

Coverage target:
    kamui/training/*.py — 100%
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.training.checkpointing import (
    load_checkpoint,
    load_model_only,
    save_checkpoint,
)
from kamui.training.data import (
    DataLoader,
    TextDataset,
    load_tokens,
    tokenise_corpus,
    train_val_split,
)
from kamui.training.optimizer import build_optimizer
from kamui.training.scheduler import CosineWithWarmup
from kamui.training.trainer import Trainer, TrainingConfig


def _config(**overrides: object) -> ModelConfig:
    base = dict(
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ff=32,
        vocab_size=32,
        context_length=8,
        positional_encoding="learned",
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _model(**overrides: object) -> KAMUITransformer:
    return KAMUITransformer(_config(**overrides))


# ===========================================================================
# Scheduler
# ===========================================================================

class TestCosineWithWarmup:
    def test_warmup_linear(self) -> None:
        s = CosineWithWarmup(max_lr=1.0, min_lr=0.0, warmup_steps=10, max_steps=110)
        assert s.get_lr(0) == pytest.approx(0.0)
        assert s.get_lr(5) == pytest.approx(0.5)
        assert s.get_lr(10) == pytest.approx(1.0)  # peak at end of warmup

    def test_cosine_midpoint(self) -> None:
        s = CosineWithWarmup(max_lr=1.0, min_lr=0.0, warmup_steps=0, max_steps=100)
        # Halfway through decay → (max+min)/2.
        assert s.get_lr(50) == pytest.approx(0.5, abs=1e-6)

    def test_cosine_endpoint_and_floor(self) -> None:
        s = CosineWithWarmup(max_lr=1.0, min_lr=0.1, warmup_steps=0, max_steps=100)
        assert s.get_lr(100) == pytest.approx(0.1)   # reaches min_lr
        assert s.get_lr(500) == pytest.approx(0.1)   # floors after max_steps

    def test_peak_at_warmup_end(self) -> None:
        s = CosineWithWarmup(max_lr=3e-4, min_lr=3e-5, warmup_steps=100, max_steps=5000)
        assert s.get_lr(100) == pytest.approx(3e-4)

    def test_invalid_max_lr(self) -> None:
        with pytest.raises(ValueError, match="max_lr must be > 0"):
            CosineWithWarmup(max_lr=0.0, min_lr=0.0, warmup_steps=1, max_steps=10)

    def test_invalid_min_lr(self) -> None:
        with pytest.raises(ValueError, match="min_lr must be"):
            CosineWithWarmup(max_lr=1.0, min_lr=2.0, warmup_steps=1, max_steps=10)

    def test_invalid_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
            CosineWithWarmup(max_lr=1.0, min_lr=0.0, warmup_steps=-1, max_steps=10)

    def test_invalid_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            CosineWithWarmup(max_lr=1.0, min_lr=0.0, warmup_steps=10, max_steps=10)

    def test_state_dict_roundtrip(self) -> None:
        s = CosineWithWarmup(max_lr=1.0, min_lr=0.1, warmup_steps=5, max_steps=50)
        s2 = CosineWithWarmup(max_lr=9.0, min_lr=9.0, warmup_steps=1, max_steps=2)
        s2.load_state_dict(s.state_dict())
        assert s2.get_lr(3) == pytest.approx(s.get_lr(3))

    def test_repr(self) -> None:
        s = CosineWithWarmup(max_lr=1.0, min_lr=0.1, warmup_steps=5, max_steps=50)
        assert "CosineWithWarmup" in repr(s)


# ===========================================================================
# Optimizer
# ===========================================================================

class TestBuildOptimizer:
    def test_returns_adamw(self) -> None:
        opt = build_optimizer(_model())
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups_with_correct_decay(self) -> None:
        opt = build_optimizer(_model(), weight_decay=0.1)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 0.1
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_decay_group_is_matrices_only(self) -> None:
        opt = build_optimizer(_model())
        assert all(p.dim() >= 2 for p in opt.param_groups[0]["params"])
        assert all(p.dim() < 2 for p in opt.param_groups[1]["params"])

    def test_biases_and_norms_not_decayed(self) -> None:
        # There is at least one 1-D parameter (LayerNorm/bias) in the no-decay group.
        opt = build_optimizer(_model())
        assert len(opt.param_groups[1]["params"]) > 0

    def test_tied_weight_counted_once(self) -> None:
        model = _model()
        opt = build_optimizer(model)
        total = sum(len(g["params"]) for g in opt.param_groups)
        distinct = len({id(p) for p in model.parameters()})
        assert total == distinct

    def test_invalid_lr(self) -> None:
        with pytest.raises(ValueError, match="lr must be > 0"):
            build_optimizer(_model(), lr=0.0)

    def test_invalid_weight_decay(self) -> None:
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            build_optimizer(_model(), weight_decay=-0.1)

    def test_frozen_params_excluded(self) -> None:
        model = _model()
        # Freeze the final LayerNorm; its params must not appear in any group.
        for p in model.final_ln.parameters():
            p.requires_grad_(False)
        opt = build_optimizer(model)
        grouped = {id(p) for g in opt.param_groups for p in g["params"]}
        assert all(id(p) not in grouped for p in model.final_ln.parameters())


# ===========================================================================
# Data
# ===========================================================================

class TestData:
    def test_dataset_len(self) -> None:
        ds = TextDataset(list(range(20)), context_length=8)
        assert len(ds) == 20 - 8

    def test_dataset_getitem_shift(self) -> None:
        ds = TextDataset(list(range(20)), context_length=4)
        x, y = ds[0]
        assert x.tolist() == [0, 1, 2, 3]
        assert y.tolist() == [1, 2, 3, 4]  # target = input shifted by one

    def test_dataset_context_too_small(self) -> None:
        with pytest.raises(ValueError, match="context_length must be >= 1"):
            TextDataset(list(range(20)), context_length=0)

    def test_dataset_non_1d(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            TextDataset(np.zeros((3, 3)), context_length=2)

    def test_dataset_too_few_tokens(self) -> None:
        with pytest.raises(ValueError, match="at least context_length"):
            TextDataset([1, 2, 3], context_length=8)

    def test_dataset_index_out_of_range(self) -> None:
        ds = TextDataset(list(range(20)), context_length=8)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_dataloader_batches(self) -> None:
        ds = TextDataset(list(range(100)), context_length=8)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        x, y = batches[0]
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)

    def test_dataloader_len_drop_last(self) -> None:
        ds = TextDataset(list(range(100)), context_length=8)  # 92 chunks
        loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True)
        assert len(loader) == 92 // 10

    def test_dataloader_len_keep_last(self) -> None:
        ds = TextDataset(list(range(100)), context_length=8)  # 92 chunks
        loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=False)
        assert len(loader) == math.ceil(92 / 10)
        # The trailing partial batch is yielded.
        assert list(loader)[-1][0].shape[0] == 92 % 10

    def test_dataloader_drop_last_partial(self) -> None:
        ds = TextDataset(list(range(100)), context_length=8)  # 92 chunks
        loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True)
        batches = list(loader)  # exercises the partial-batch break
        assert len(batches) == 9
        assert all(x.shape[0] == 10 for x, _ in batches)

    def test_dataloader_shuffle_reproducible(self) -> None:
        ds = TextDataset(list(range(100)), context_length=8)
        a = list(DataLoader(ds, batch_size=4, shuffle=True, seed=0))
        b = list(DataLoader(ds, batch_size=4, shuffle=True, seed=0))
        assert torch.equal(a[0][0], b[0][0])

    def test_dataloader_bad_batch_size(self) -> None:
        ds = TextDataset(list(range(20)), context_length=8)
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            DataLoader(ds, batch_size=0)

    def test_tokenise_corpus_and_load(self, tmp_path) -> None:
        class _Tok:
            def encode(self, text: str) -> list[int]:
                return [ord(c) % 32 for c in text]

        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello world", encoding="utf-8")
        out = tmp_path / "toks"
        arr = tokenise_corpus(corpus, _Tok(), output_path=out)
        assert arr.dtype == np.int32
        loaded = load_tokens(str(out) + ".npy")
        assert np.array_equal(np.asarray(loaded), arr)

    def test_tokenise_corpus_no_save(self, tmp_path) -> None:
        class _Tok:
            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

        corpus = tmp_path / "c.txt"
        corpus.write_text("x", encoding="utf-8")
        arr = tokenise_corpus(corpus, _Tok())
        assert arr.tolist() == [1, 2, 3]

    def test_train_val_split(self) -> None:
        train, val = train_val_split(np.arange(100), val_fraction=0.05)
        assert len(train) == 95
        assert len(val) == 5
        assert val[0] == 95  # deterministic tail

    def test_train_val_split_bad_fraction(self) -> None:
        with pytest.raises(ValueError, match="val_fraction must be"):
            train_val_split(np.arange(100), val_fraction=1.5)


# ===========================================================================
# Checkpointing
# ===========================================================================

def _make_trainer(**cfg_overrides: object) -> Trainer:
    model = _model()
    # Tokens must stay within [0, vocab_size=32).
    ds = TextDataset(list(range(32)) * 20, context_length=8)
    loader = DataLoader(ds, batch_size=4, shuffle=True, seed=0)
    cfg_kwargs: dict[str, object] = {"warmup_steps": 2, "max_steps": 1000}
    cfg_kwargs.update(cfg_overrides)
    config = TrainingConfig(**cfg_kwargs)  # type: ignore[arg-type]
    return Trainer(model, loader, val_loader=loader, config=config)


class TestCheckpointing:
    def test_save_creates_file(self, tmp_path) -> None:
        t = _make_trainer()
        path = tmp_path / "ckpts" / "step.pt"
        save_checkpoint(path, t.model, t.optimizer, t.scheduler, step=5)
        assert path.exists()

    def test_load_checkpoint_restores_step(self, tmp_path) -> None:
        t = _make_trainer()
        t.train(3)
        path = tmp_path / "c.pt"
        save_checkpoint(path, t.model, t.optimizer, t.scheduler, step=t.step)
        fresh = _model()
        opt = build_optimizer(fresh)
        sched = CosineWithWarmup(3e-4, 3e-5, 2, 1000)
        step = load_checkpoint(path, fresh, opt, sched)
        assert step == t.step

    def test_load_restores_weights(self, tmp_path) -> None:
        t = _make_trainer()
        t.train(3)
        path = tmp_path / "c.pt"
        save_checkpoint(path, t.model, t.optimizer, t.scheduler, step=t.step)
        fresh = _model()
        load_model_only(path, fresh)
        for (n1, p1), (n2, p2) in zip(
            t.model.named_parameters(), fresh.named_parameters()
        ):
            assert torch.equal(p1, p2)

    def test_save_stores_config_and_losses(self, tmp_path) -> None:
        t = _make_trainer()
        path = tmp_path / "c.pt"
        save_checkpoint(
            path, t.model, t.optimizer, t.scheduler, step=1,
            config=t.config, train_loss=2.5, val_loss=2.7,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["train_loss"] == 2.5
        assert ckpt["config"]["warmup_steps"] == 2

    def test_save_verification_failure(self, tmp_path, monkeypatch) -> None:
        # Simulate a corrupt write: the read-back returns mismatched keys.
        t = _make_trainer()
        path = tmp_path / "c.pt"
        monkeypatch.setattr(
            torch, "load", lambda *a, **k: {"model_state": {"corrupt": 1}}
        )
        with pytest.raises(IOError, match="verification failed"):
            save_checkpoint(path, t.model, t.optimizer, t.scheduler, step=1)

    def test_load_checkpoint_missing(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nope.pt", _model())

    def test_load_model_only_missing(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_only(tmp_path / "nope.pt", _model())

    def test_load_checkpoint_without_optimizer(self, tmp_path) -> None:
        t = _make_trainer()
        path = tmp_path / "c.pt"
        save_checkpoint(path, t.model, t.optimizer, t.scheduler, step=7)
        # optimizer/scheduler optional.
        assert load_checkpoint(path, _model()) == 7


# ===========================================================================
# Trainer
# ===========================================================================

class TestTrainingConfig:
    def test_defaults(self) -> None:
        c = TrainingConfig()
        assert c.grad_accum_steps == 1

    def test_bad_grad_accum(self) -> None:
        with pytest.raises(ValueError, match="grad_accum_steps must be"):
            TrainingConfig(grad_accum_steps=0)

    def test_bad_eval_interval(self) -> None:
        with pytest.raises(ValueError, match="eval_interval must be"):
            TrainingConfig(eval_interval=-1)


class TestTrainer:
    def test_builds_optimizer_and_scheduler(self) -> None:
        t = _make_trainer()
        assert isinstance(t.optimizer, torch.optim.AdamW)
        assert isinstance(t.scheduler, CosineWithWarmup)

    def test_train_returns_records_and_advances_step(self) -> None:
        t = _make_trainer()
        records = t.train(5)
        assert len(records) == 5
        assert t.step == 5
        assert all("train_loss" in r and "lr" in r for r in records)

    def test_train_reduces_loss_on_tiny_data(self) -> None:
        # Overfit a small, repeating corpus: loss should drop substantially.
        torch.manual_seed(0)
        model = _model()
        ds = TextDataset(list(range(16)) * 20, context_length=8)
        loader = DataLoader(ds, batch_size=8, shuffle=True, seed=0)
        config = TrainingConfig(max_lr=3e-3, warmup_steps=5, max_steps=500)
        trainer = Trainer(model, loader, config=config)
        records = trainer.train(80)
        assert records[-1]["train_loss"] < records[0]["train_loss"] * 0.7

    def test_grad_accumulation(self) -> None:
        t = _make_trainer(grad_accum_steps=3)
        records = t.train(2)
        assert len(records) == 2
        assert t.step == 2

    def test_lr_follows_schedule(self) -> None:
        t = _make_trainer(max_lr=1.0, min_lr=0.0, warmup_steps=4, max_steps=100)
        t.train(1)  # step 0 → lr 0 during warmup
        # After the first update, the optimiser LR equals the schedule at step 0.
        assert t.optimizer.param_groups[0]["lr"] == pytest.approx(
            t.scheduler.get_lr(0)
        )

    def test_evaluate_returns_float(self) -> None:
        t = _make_trainer()
        val = t.evaluate()
        assert isinstance(val, float) and math.isfinite(val)

    def test_evaluate_no_val_loader_raises(self) -> None:
        model = _model()
        ds = TextDataset(list(range(100)), context_length=8)
        loader = DataLoader(ds, batch_size=4, seed=0)
        trainer = Trainer(model, loader, val_loader=None)
        with pytest.raises(ValueError, match="no val_loader"):
            trainer.evaluate()

    def test_evaluate_empty_val_loader_raises(self) -> None:
        model = _model()
        ds = TextDataset(list(range(32)) * 4, context_length=8)
        loader = DataLoader(ds, batch_size=4, seed=0)
        trainer = Trainer(model, loader, val_loader=[])  # non-None but empty
        with pytest.raises(ValueError, match="no tokens"):
            trainer.evaluate()

    def test_eval_interval_records_val_loss(self) -> None:
        t = _make_trainer(eval_interval=2)
        records = t.train(4)
        assert "val_loss" in records[1]   # step 2
        assert "val_loss" not in records[0]

    def test_no_clip_uses_grad_norm_path(self) -> None:
        t = _make_trainer(max_grad_norm=0.0)  # disables clipping
        records = t.train(2)
        assert all(r["grad_norm"] >= 0 for r in records)

    def test_log_prints(self, capsys) -> None:
        t = _make_trainer(log=True)
        t.train(1)
        out = capsys.readouterr().out
        assert "step=1" in out

    def test_empty_train_loader_raises(self) -> None:
        model = _model()
        trainer = Trainer(model, train_loader=[])
        with pytest.raises(ValueError, match="empty"):
            trainer.train(1)
