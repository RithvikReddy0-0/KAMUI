"""Distributed data-parallel (DDP) training for KAMUITransformer.

Data parallelism trains one model across ``world_size`` processes (typically
one per GPU).  Every process holds a full replica of the model and processes a
**different shard** of each global batch; after the backward pass the processes
**average their gradients** with an all-reduce, so every replica takes an
identical optimiser step and the weights stay in sync.

Why gradient averaging is exactly a larger batch:
    Split a global batch of ``2N`` examples across two ranks (``N`` each), each
    computing a *mean* loss over its shard::

        grad_rank0 = (1/N) Σ_{i=1..N}    ∇ℓ_i
        grad_rank1 = (1/N) Σ_{i=N+1..2N} ∇ℓ_i

    DDP averages them::

        (grad_rank0 + grad_rank1) / 2 = (1/2N) Σ_{i=1..2N} ∇ℓ_i

    which is exactly the gradient a single process would compute over the full
    ``2N``-example batch with mean reduction.  This identity (with equal shard
    sizes) is the correctness guarantee of DDP — and it is asserted by a test
    that runs two real ``gloo`` processes on CPU.

This module is a thin, readable layer over ``torch.distributed`` and
``torch.nn.parallel.DistributedDataParallel`` — no magic:

Lifecycle:
    - ``init_process_group`` / ``destroy_process_group`` — join / leave the group.
    - ``spawn_workers``  — launch a worker function across ``world_size`` processes.

Queries (all safe to call when distribution is *not* initialised):
    - ``is_dist_available_and_initialized`` / ``get_rank`` / ``get_world_size``
    - ``is_main_process`` / ``barrier``

Model & reductions:
    - ``wrap_ddp``        — wrap a model in ``DistributedDataParallel``.
    - ``unwrap_model``    — recover the underlying model (for checkpoint saving).
    - ``all_reduce_mean`` — average a scalar / tensor metric across ranks.

Data:
    - ``shard_indices``          — this rank's slice of ``range(n)``.
    - ``DistributedDataLoader``  — a ``DataLoader`` that yields only this rank's shard.

Typical worker (one process per GPU), driven by ``spawn_workers``::

    def worker(rank, world_size, tokens, config):
        init_process_group(rank, world_size, backend="nccl")
        torch.cuda.set_device(rank)
        model = KAMUITransformer(config).cuda(rank)
        model = wrap_ddp(model, device_ids=[rank])
        loader = DistributedDataLoader(TextDataset(tokens, config.context_length),
                                       batch_size=16)
        trainer = Trainer(model, loader, config=TrainingConfig(max_steps=2000))
        trainer.train(2000)
        if is_main_process():
            save_checkpoint("ckpt.pt", unwrap_model(model), trainer.optimizer,
                            trainer.scheduler, trainer.step)
        destroy_process_group()

    spawn_workers(worker, world_size=torch.cuda.device_count(), args=(tokens, config))

Because ``wrap_ddp`` leaves ``model.parameters()`` and the forward signature
unchanged, the existing ``Trainer`` and ``build_optimizer`` work on a wrapped
model with no changes.

Implemented in: v0.3 (multi-GPU training).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable, Iterator
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from kamui.training.data import TextDataset

# ---------------------------------------------------------------------------
# Process-group lifecycle
# ---------------------------------------------------------------------------


def init_process_group(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    init_method: str | None = None,
    timeout_seconds: float = 300.0,
) -> None:
    """Join the distributed process group as ``rank`` of ``world_size``.

    Args:
        rank:            This process's rank in ``[0, world_size)``.
        world_size:      Total number of participating processes (>= 1).
        backend:         ``"gloo"`` (CPU / portable) or ``"nccl"`` (CUDA).
        init_method:     Rendezvous URL.  ``None`` uses ``env://`` with
            ``MASTER_ADDR`` / ``MASTER_PORT`` (defaulting to ``127.0.0.1:29500``);
            a ``file://`` URL uses a shared ``FileStore`` (portable, no ports).
        timeout_seconds: Collective-operation timeout.

    Raises:
        ValueError:   If ``world_size < 1`` or ``rank`` is out of range.
        RuntimeError: If ``torch.distributed`` is unavailable or a group is
            already initialised.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")
    if dist.is_initialized():
        raise RuntimeError("a process group is already initialised")

    if init_method is None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        init_method = "env://"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def destroy_process_group() -> None:
    """Leave the process group if one is initialised (a no-op otherwise)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Queries — all safe to call with no active process group
# ---------------------------------------------------------------------------


def is_dist_available_and_initialized() -> bool:
    """True iff ``torch.distributed`` is available and a group is initialised."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """This process's rank (0 when distribution is not initialised)."""
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    """Total process count (1 when distribution is not initialised)."""
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def is_main_process() -> bool:
    """True on rank 0 (and always true when not initialised)."""
    return get_rank() == 0


def barrier() -> None:
    """Block until all ranks reach this point (a no-op when not initialised)."""
    if is_dist_available_and_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Model wrapping and metric reduction
# ---------------------------------------------------------------------------


def wrap_ddp(
    model: nn.Module,
    device_ids: list[int] | None = None,
    find_unused_parameters: bool = False,
) -> DistributedDataParallel:
    """Wrap ``model`` in ``DistributedDataParallel``.

    On construction, DDP broadcasts rank 0's parameters to every rank, so all
    replicas start identical.  The wrapper preserves ``model.parameters()`` and
    the forward signature, so ``build_optimizer`` and ``Trainer`` work unchanged.

    Args:
        model:                  A model on this rank's device.
        device_ids:             ``[local_gpu]`` for a single-GPU-per-process
            NCCL setup; ``None`` for CPU / ``gloo``.
        find_unused_parameters: Enable only if some parameters do not receive
            gradients every step (it adds overhead).

    Returns:
        The ``DistributedDataParallel``-wrapped model.

    Raises:
        RuntimeError: If no process group is initialised.
    """
    if not is_dist_available_and_initialized():
        raise RuntimeError("call init_process_group before wrap_ddp")
    return DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=find_unused_parameters
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, unwrapping ``DistributedDataParallel``.

    Use before saving a checkpoint so its ``state_dict`` keys are not prefixed
    with ``module.`` and can be loaded into a plain (non-DDP) model.
    """
    return model.module if isinstance(model, DistributedDataParallel) else model


def all_reduce_mean(value: float | Tensor) -> float | Tensor:
    """Average a scalar / tensor across all ranks (identity when not initialised).

    Args:
        value: A Python float or a tensor. A float returns a float; a tensor
            returns a new tensor (the input is not modified).

    Returns:
        The mean of ``value`` over all ranks, in the same type as the input.
    """
    if not is_dist_available_and_initialized():
        return value
    world_size = dist.get_world_size()
    if isinstance(value, Tensor):
        tensor = value.detach().clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / world_size
    tensor = torch.tensor(float(value))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / world_size).item()


# ---------------------------------------------------------------------------
# Data sharding
# ---------------------------------------------------------------------------


def shard_indices(
    n: int,
    rank: int | None = None,
    world_size: int | None = None,
    drop_remainder: bool = True,
) -> list[int]:
    """Return this rank's strided slice of ``range(n)``.

    Rank ``r`` receives indices ``r, r + world_size, r + 2*world_size, …`` so
    the shards are balanced.  With ``drop_remainder=True`` the tail that would
    make shards unequal is dropped, giving every rank exactly ``n // world_size``
    indices — the equal-count condition under which DDP gradient averaging
    equals the full-batch gradient.

    Args:
        n:              Total number of items (>= 0).
        rank:           Rank (defaults to the current rank).
        world_size:     Process count (defaults to the current world size).
        drop_remainder: Drop the uneven tail so all ranks get equal counts.

    Returns:
        The list of indices assigned to ``rank``.

    Raises:
        ValueError: If ``n < 0``, ``world_size < 1``, or ``rank`` is out of range.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rank = get_rank() if rank is None else rank
    world_size = get_world_size() if world_size is None else world_size
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    limit = (n // world_size) * world_size if drop_remainder else n
    return list(range(rank, limit, world_size))


class DistributedDataLoader:
    """A batching loader that yields only this rank's shard of a ``TextDataset``.

    Each epoch, the dataset indices are (optionally) shuffled with an
    epoch-dependent seed shared by all ranks, truncated to a multiple of
    ``world_size``, then split into contiguous per-rank blocks of equal size.
    Every rank therefore sees a disjoint shard and the same number of batches,
    keeping the replicas in lock-step.

    Attributes:
        dataset:    The wrapped dataset.
        batch_size: Sequences per batch.
        rank:       This loader's rank.
        world_size: Total number of ranks.
    """

    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int,
        rank: int | None = None,
        world_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        """Create a distributed loader.

        Args:
            dataset:    A ``TextDataset``.
            batch_size: Sequences per batch (>= 1).
            rank:       Rank (defaults to the current rank).
            world_size: Process count (defaults to the current world size).
            shuffle:    Shuffle the global index order each epoch.
            seed:       Base RNG seed (the epoch index is added to it).
            drop_last:  Drop a trailing partial batch within the shard.

        Raises:
            ValueError: If ``batch_size < 1``, ``world_size < 1``, or ``rank``
                is out of range.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = get_rank() if rank is None else rank
        self.world_size = get_world_size() if world_size is None else world_size
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0
        self._per_rank = len(dataset) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shuffling differs across epochs but agrees across ranks."""
        self._epoch = epoch

    def _shard(self) -> list[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        usable = self._per_rank * self.world_size
        indices = indices[:usable]
        start = self.rank * self._per_rank
        return indices[start : start + self._per_rank]

    def __len__(self) -> int:
        """Number of batches this rank yields per epoch."""
        if self.drop_last:
            return self._per_rank // self.batch_size
        return (self._per_rank + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(inputs, targets)`` batches from this rank's shard, then bump the epoch."""
        shard = self._shard()
        for start in range(0, len(shard), self.batch_size):
            batch_idx = shard[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            pairs = [self.dataset[i] for i in batch_idx]
            inputs = torch.stack([x for x, _ in pairs])
            targets = torch.stack([y for _, y in pairs])
            yield inputs, targets
        self._epoch += 1


# ---------------------------------------------------------------------------
# Multi-process launcher
# ---------------------------------------------------------------------------


def _worker_entry(
    rank: int,
    world_size: int,
    init_method: str,
    backend: str,
    worker_fn: Callable[..., None],
    args: tuple[object, ...],
) -> None:
    """Child-process entry point: join the group, run ``worker_fn``, then leave."""
    init_process_group(rank, world_size, backend=backend, init_method=init_method)
    try:
        worker_fn(rank, world_size, *args)
    finally:
        destroy_process_group()


def spawn_workers(
    worker_fn: Callable[..., None],
    world_size: int,
    args: tuple[object, ...] = (),
    backend: str = "gloo",
    init_method: str | None = None,
) -> None:
    """Launch ``worker_fn`` across ``world_size`` processes and wait for them.

    Each process runs ``worker_fn(rank, world_size, *args)`` with the process
    group already initialised (and torn down on return).  ``worker_fn`` and
    every element of ``args`` must be picklable (defined at module top level),
    because the ``spawn`` start method re-imports them in each child.

    Args:
        worker_fn:   ``worker_fn(rank, world_size, *args) -> None``.
        world_size:  Number of processes to launch (>= 1).
        args:        Extra positional arguments forwarded to ``worker_fn``.
        backend:     ``"gloo"`` (CPU / portable) or ``"nccl"`` (CUDA).
        init_method: Rendezvous URL. ``None`` allocates a private ``FileStore``
            (portable, needs no free TCP port).

    Raises:
        ValueError: If ``world_size < 1``.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if init_method is None:
        store_path = os.path.join(tempfile.gettempdir(), f"kamui_ddp_{uuid.uuid4().hex}")
        init_method = "file:///" + store_path.replace("\\", "/")
    mp.spawn(
        _worker_entry,
        args=(world_size, init_method, backend, worker_fn, args),
        nprocs=world_size,
        join=True,
    )
