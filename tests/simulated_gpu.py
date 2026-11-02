"""A simulated GPU, so that device bugs fail on machines that have no GPU.

Mixing devices only crashes on hardware that has a second device, and CI has
none.  ``SimulatedGPU`` runs everything on the CPU but enforces CUDA's rules:
inside it, ``tensor.to("cuda")`` tags the result as being on the GPU, ``.device``
reports ``cuda:0``, and every torch call checks the tags of its tensors and
raises the error real CUDA would raise.

It catches tensors on the wrong device, not hardware problems, so it complements
a real run of ``kamui-check-device`` rather than replacing one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor
from torch.overrides import TorchFunctionMode
from torch.utils.weak import WeakTensorKeyDictionary

GPU = torch.device("cuda", 0)
CPU = torch.device("cpu")

#: Tensors on the simulated GPU.  This is a side table, not an attribute on each
#: tensor, because attributes survive pickling and a tensor loaded with
#: ``map_location="cpu"`` has to land on the CPU.
_ON_GPU: Any = WeakTensorKeyDictionary()

#: Calls that CUDA allows across devices: a copy, and a comparison of types.
_MAY_MIX = (Tensor.copy_, torch._has_compatible_shallow_copy_type)


def _on_gpu(tensor: Tensor) -> bool:
    return tensor in _ON_GPU


def _set_device(value: Any, on_gpu: bool) -> Any:
    """Tag every tensor in ``value`` and return ``value``."""
    if isinstance(value, Tensor):
        if on_gpu:
            _ON_GPU[value] = True
        else:
            _ON_GPU.pop(value, None)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _set_device(item, on_gpu)
    return value


def _tensors(value: Any) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        yield from _tensors(list(value.values()))


def _is_gpu_spec(spec: Any) -> bool:
    return (isinstance(spec, str) and spec.startswith("cuda")) or (
        isinstance(spec, torch.device) and spec.type == "cuda"
    )


def _mismatch(func_name: str) -> RuntimeError:
    return RuntimeError(
        "Expected all tensors to be on the same device, but found at least two "
        f"devices, cuda:0 and cpu! (when calling {func_name})"
    )


class SimulatedGPU(TorchFunctionMode):
    """Run on the CPU, and fail like CUDA does when devices are mixed."""

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = dict(kwargs or {})

        if func == Tensor.device.__get__:
            return GPU if _on_gpu(args[0]) else func(*args)
        if func in (Tensor.to, Tensor.cuda, Tensor.cpu):
            return self._move(func, args, kwargs)
        if func == Tensor.data.__set__:
            func(*args)
            return _set_device(args[0], _on_gpu(args[1]))
        if func == Tensor.numpy and _on_gpu(args[0]):
            raise TypeError("can't convert cuda:0 device type tensor to numpy.")

        wants_gpu = _is_gpu_spec(kwargs.get("device"))
        if wants_gpu:
            if kwargs.get("generator") is not None:
                raise RuntimeError("Expected a 'cuda' device type for generator but found 'cpu'")
            kwargs["device"] = CPU

        self._check(func, args, kwargs)
        if func == Tensor.__setitem__:
            return func(*args, **kwargs)
        result = func(*args, **kwargs)
        inputs = [t for t in _tensors((args, kwargs)) if _on_gpu(t)]
        if func in (Tensor.copy_, Tensor.__getitem__):
            return _set_device(result, _on_gpu(args[0]))
        if wants_gpu or inputs:
            _set_device(result, True)
        return result

    @staticmethod
    def _check(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        name = getattr(func, "__name__", str(func))
        if func in (Tensor.__getitem__, Tensor.__setitem__):
            # A GPU tensor takes CPU indices, and either side of a copy may be anywhere.
            if not _on_gpu(args[0]) and any(_on_gpu(t) and t.dim() > 0 for t in _tensors(args[1])):
                raise RuntimeError(
                    "indices should be either on cpu or on the same device as the "
                    "indexed tensor (cpu)"
                )
            return
        if func in _MAY_MIX:
            return

        tensors = [t for t in _tensors((args, kwargs)) if t.dim() > 0]  # scalars may mix
        if any(_on_gpu(t) for t in tensors) and not all(_on_gpu(t) for t in tensors):
            raise _mismatch(name)

    @staticmethod
    def _move(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tensor:
        tensor = args[0]
        target: bool | None = None  # None: stay where it is
        dtype = kwargs.get("dtype")
        if func == Tensor.cuda:
            target = True
        elif func == Tensor.cpu:
            target = False
        else:
            for spec in (*args[1:], kwargs.get("device"), kwargs.get("other")):
                if isinstance(spec, torch.dtype):
                    dtype = spec
                elif isinstance(spec, Tensor):
                    target, dtype = _on_gpu(spec), spec.dtype
                elif isinstance(spec, (str, torch.device)):
                    target = _is_gpu_spec(spec)

        here = _on_gpu(tensor)
        moved = tensor if dtype is None else Tensor.to(tensor, dtype=dtype)
        if target is None or target == here:
            return _set_device(moved, here)
        return _set_device(moved.clone(), target)
