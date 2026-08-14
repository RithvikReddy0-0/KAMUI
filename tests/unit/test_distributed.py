"""Unit tests for kamui.training.distributed (DDP / multi-GPU training).

Coverage target:
    kamui/training/distributed.py — 100%

Strategy:
    - Pure-logic helpers (rank/world queries, sharding, validation) are tested
      in-process with no process group.
    - The real ``torch.distributed`` code paths (init, wrap_ddp, all_reduce,
      barrier, destroy, the ``_worker_entry`` bootstrap) are exercised with a
      single-process ``gloo`` group over a portable ``FileStore``.
    - The correctness anchor spawns **two real gloo processes** on CPU and
      asserts DDP's averaged gradient equals the single-process full-batch
      gradient — the defining guarantee of data-parallel training.
"""

from __future__ import annotations

import os
import tempfile
import uuid

import pytest
import torch
from torch import nn

from kamui.model.config import ModelConfig
from kamui.model.transformer import KAMUITransformer
from kamui.training.data import TextDataset
from kamui.training.distributed import (
    DistributedDataLoader,
    _worker_entry,
    all_reduce_mean,
    barrier,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
    is_dist_available_and_initialized,
    is_main_process,
    shard_indices,
    spawn_workers,
    unwrap_model,
    wrap_ddp,
)


def _file_init_method() -> str:
    """A private FileStore rendezvous URL (portable, no TCP port needed)."""
    store = os.path.join(tempfile.gettempdir(), f"kamui_ddp_test_{uuid.uuid4().hex}")
    return "file:///" + store.replace("\\", "/")


def _tiny_config() -> ModelConfig:
    """A minimal, dropout-free config so forward/backward is deterministic."""
    return ModelConfig(
        n_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        vocab_size=16,
        context_length=4,
        dropout=0.0,
    )


# Module-level so torch.multiprocessing.spawn can re-import it in each child.
def _ddp_grad_check_worker(rank: int, world_size: int, work_dir: str, seed: int) -> None:
    """Each rank trains on its shard; rank 0 saves the DDP-averaged gradients."""
    torch.manual_seed(seed)
    model = KAMUITransformer(_tiny_config())
    model.train()
    ddp = wrap_ddp(model)

    batch = torch.load(os.path.join(work_dir, "batch.pt"))
    inputs, targets = batch["inputs"], batch["targets"]
    n = inputs.shape[0] // world_size
    lo, hi = rank * n, (rank + 1) * n

    loss = ddp(inputs[lo:hi], targets=targets[lo:hi])
    loss.backward()

    if rank == 0:
        grads = {name: p.grad.detach().clone() for name, p in unwrap_model(ddp).named_parameters()}
        torch.save(grads, os.path.join(work_dir, "ddp_grads.pt"))
    barrier()


# ===========================================================================
# Queries with no active process group
# ===========================================================================


class TestQueriesWithoutGroup:
    def test_not_initialized_defaults(self) -> None:
        assert not is_dist_available_and_initialized()
        assert get_rank() == 0
        assert get_world_size() == 1
        assert is_main_process()

    def test_barrier_is_noop_without_group(self) -> None:
        barrier()  # must not raise

    def test_destroy_is_noop_without_group(self) -> None:
        destroy_process_group()  # must not raise

    def test_all_reduce_mean_identity_without_group(self) -> None:
        assert all_reduce_mean(3.5) == 3.5
        t = torch.tensor([1.0, 2.0])
        assert torch.equal(all_reduce_mean(t), t)

    def test_unwrap_plain_model_returns_itself(self) -> None:
        model = nn.Linear(3, 3)
        assert unwrap_model(model) is model

    def test_wrap_ddp_without_group_raises(self) -> None:
        with pytest.raises(RuntimeError, match="init_process_group"):
            wrap_ddp(nn.Linear(2, 2))


# ===========================================================================
# Sharding logic
# ===========================================================================


class TestShardIndices:
    def test_strided_partition_is_disjoint_and_complete(self) -> None:
        r0 = shard_indices(10, rank=0, world_size=2)
        r1 = shard_indices(10, rank=1, world_size=2)
        assert r0 == [0, 2, 4, 6, 8]
        assert r1 == [1, 3, 5, 7, 9]
        assert set(r0).isdisjoint(r1)
        assert sorted(r0 + r1) == list(range(10))

    def test_drop_remainder_gives_equal_counts(self) -> None:
        r0 = shard_indices(5, rank=0, world_size=2, drop_remainder=True)
        r1 = shard_indices(5, rank=1, world_size=2, drop_remainder=True)
        assert len(r0) == len(r1) == 2  # index 4 dropped

    def test_keep_remainder(self) -> None:
        assert shard_indices(5, rank=0, world_size=2, drop_remainder=False) == [0, 2, 4]

    def test_defaults_to_current_rank_and_world(self) -> None:
        assert shard_indices(4) == [0, 1, 2, 3]  # rank 0 of world 1

    @pytest.mark.parametrize(
        ("n", "rank", "world_size"),
        [(-1, 0, 1), (10, 0, 0), (10, 2, 2), (10, -1, 2)],
    )
    def test_invalid_args_raise(self, n: int, rank: int, world_size: int) -> None:
        with pytest.raises(ValueError):
            shard_indices(n, rank=rank, world_size=world_size)


# ===========================================================================
# DistributedDataLoader
# ===========================================================================


class TestDistributedDataLoader:
    def _dataset(self, n_tokens: int = 100) -> TextDataset:
        return TextDataset(list(range(n_tokens)), context_length=4)

    def test_ranks_see_disjoint_equal_shards(self) -> None:
        ds = self._dataset()  # len == 96 chunks
        l0 = DistributedDataLoader(ds, batch_size=8, rank=0, world_size=2, seed=0)
        l1 = DistributedDataLoader(ds, batch_size=8, rank=1, world_size=2, seed=0)
        idx0, idx1 = l0._shard(), l1._shard()
        assert len(idx0) == len(idx1) == 48
        assert set(idx0).isdisjoint(idx1)
        assert sorted(idx0 + idx1) == list(range(96))

    def test_iteration_shapes_and_count(self) -> None:
        ds = self._dataset()
        loader = DistributedDataLoader(ds, batch_size=8, rank=0, world_size=2)
        batches = list(loader)
        assert len(loader) == 48 // 8 == 6
        assert len(batches) == 6
        for inputs, targets in batches:
            assert inputs.shape == (8, 4)
            assert targets.shape == (8, 4)

    def test_default_rank_world_is_single_process(self) -> None:
        ds = self._dataset()
        loader = DistributedDataLoader(ds, batch_size=8)  # rank 0, world 1
        assert loader.rank == 0 and loader.world_size == 1
        assert len(loader._shard()) == 96

    def test_set_epoch_changes_order(self) -> None:
        ds = self._dataset()
        loader = DistributedDataLoader(ds, batch_size=8, rank=0, world_size=2, seed=0)
        first = loader._shard()
        loader.set_epoch(1)
        assert loader._shard() != first

    def test_no_shuffle_is_contiguous(self) -> None:
        ds = self._dataset()
        loader = DistributedDataLoader(ds, batch_size=8, rank=0, world_size=2, shuffle=False)
        assert loader._shard() == list(range(48))

    def test_iteration_advances_epoch(self) -> None:
        ds = self._dataset()
        loader = DistributedDataLoader(ds, batch_size=8, rank=0, world_size=2, seed=0)
        shard_epoch0 = loader._shard()
        list(loader)  # one pass bumps the internal epoch
        assert loader._epoch == 1
        assert loader._shard() != shard_epoch0

    def test_drop_last_false_len(self) -> None:
        ds = self._dataset()
        loader = DistributedDataLoader(ds, batch_size=7, rank=0, world_size=2, drop_last=False)
        assert len(loader) == (48 + 6) // 7  # ceil(48 / 7) == 7

    def test_drop_last_true_drops_partial_batch(self) -> None:
        ds = self._dataset()  # per_rank == 48
        loader = DistributedDataLoader(ds, batch_size=9, rank=0, world_size=2, drop_last=True)
        assert len(loader) == 48 // 9 == 5
        assert len(list(loader)) == 5  # trailing 3-sequence batch is dropped

    @pytest.mark.parametrize(
        ("batch_size", "rank", "world_size"),
        [(0, 0, 1), (8, 0, 0), (8, 3, 2)],
    )
    def test_invalid_args_raise(self, batch_size: int, rank: int, world_size: int) -> None:
        ds = self._dataset()
        with pytest.raises(ValueError):
            DistributedDataLoader(ds, batch_size=batch_size, rank=rank, world_size=world_size)


# ===========================================================================
# init_process_group validation
# ===========================================================================


class TestInitValidation:
    @pytest.mark.parametrize(("rank", "world_size"), [(0, 0), (2, 2), (-1, 2)])
    def test_bad_rank_or_world_raises(self, rank: int, world_size: int) -> None:
        with pytest.raises(ValueError):
            init_process_group(rank, world_size)

    def test_defaults_to_env_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_init(**kwargs: object) -> None:
            captured.update(kwargs)

        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)
        monkeypatch.setattr("kamui.training.distributed.dist.init_process_group", fake_init)

        init_process_group(0, 1)  # no real group is created by the stub
        assert captured["init_method"] == "env://"
        assert os.environ["MASTER_ADDR"] == "127.0.0.1"
        assert os.environ["MASTER_PORT"] == "29500"

    def test_spawn_workers_rejects_bad_world_size(self) -> None:
        with pytest.raises(ValueError):
            spawn_workers(_ddp_grad_check_worker, world_size=0)

    def test_unavailable_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("kamui.training.distributed.dist.is_available", lambda: False)
        with pytest.raises(RuntimeError, match="not available"):
            init_process_group(0, 1)

    def test_double_init_raises(self) -> None:
        init_process_group(0, 1, backend="gloo", init_method=_file_init_method())
        try:
            with pytest.raises(RuntimeError, match="already initialised"):
                init_process_group(0, 1, backend="gloo", init_method=_file_init_method())
        finally:
            destroy_process_group()


# ===========================================================================
# Real single-process gloo group (exercises the torch.distributed paths)
# ===========================================================================


class TestSingleProcessGroup:
    def test_worker_entry_runs_full_stack(self) -> None:
        """_worker_entry: init -> run worker (wrap/all_reduce/barrier) -> destroy."""
        seen: dict[str, object] = {}

        def worker(rank: int, world_size: int) -> None:
            seen["rank"] = rank
            seen["world_size"] = world_size
            assert is_dist_available_and_initialized()
            assert get_rank() == 0
            assert get_world_size() == 1
            assert is_main_process()

            model = nn.Linear(4, 2)
            ddp = wrap_ddp(model)
            out = ddp(torch.randn(3, 4))
            out.sum().backward()
            assert unwrap_model(ddp) is model

            assert all_reduce_mean(2.0) == pytest.approx(2.0)  # 1 rank -> identity
            reduced = all_reduce_mean(torch.tensor([4.0, 6.0]))
            assert torch.allclose(reduced, torch.tensor([4.0, 6.0]))
            barrier()

        try:
            _worker_entry(0, 1, _file_init_method(), "gloo", worker, ())
        finally:
            destroy_process_group()  # safety net if the worker raised mid-way

        assert seen == {"rank": 0, "world_size": 1}
        assert not is_dist_available_and_initialized()  # destroyed on exit


# ===========================================================================
# Correctness anchor: 2 real gloo processes, DDP grad == full-batch grad
# ===========================================================================


class TestDDPGradientAveraging:
    def test_averaged_gradient_equals_full_batch_gradient(self, tmp_path) -> None:
        seed = 123
        work_dir = str(tmp_path)

        # A fixed global batch of 4 sequences (2 per rank).
        torch.manual_seed(0)
        inputs = torch.randint(0, 16, (4, 4))
        targets = torch.randint(0, 16, (4, 4))
        torch.save({"inputs": inputs, "targets": targets}, os.path.join(work_dir, "batch.pt"))

        # Reference: one process, full batch, mean loss.
        torch.manual_seed(seed)
        ref = KAMUITransformer(_tiny_config())
        ref.train()
        ref(inputs, targets=targets).backward()
        ref_grads = {name: p.grad.detach().clone() for name, p in ref.named_parameters()}

        # Two gloo processes, each on half the batch; DDP averages the gradients.
        spawn_workers(_ddp_grad_check_worker, world_size=2, args=(work_dir, seed))

        ddp_grads = torch.load(os.path.join(work_dir, "ddp_grads.pt"))
        assert set(ddp_grads) == set(ref_grads)
        for name, ref_grad in ref_grads.items():
            assert torch.allclose(ddp_grads[name], ref_grad, atol=1e-5, rtol=1e-4), name
