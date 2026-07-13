"""Type definitions for the hook system.

Responsibilities:
    Define the type aliases used throughout the hook system so that static
    checkers can verify hook registration and activation access.

Types defined here:
    - ``HookIO``: Literal string union of the valid I/O suffixes that name a
      hook point (e.g. ``"output"``, ``"weights"``).
    - ``ActivationCache``: dict mapping hook-point strings to captured tensors.
    - ``HookFn``: type alias for a PyTorch forward-hook callable.
      Signature: ``(module, input, output) -> Optional[Tensor]``.

Note on HookFn:
    A forward hook may optionally return a tensor to *replace* the module's
    output.  This is the mechanism used by activation patching: the patching
    hook returns the clean activation instead of the corrupted one, causally
    intervening in the forward pass.

Implemented in: Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from torch import Tensor, nn

#: Valid I/O suffixes for a hook point (the part after the module path).
#:   output      — a module's forward output
#:   input       — a module's forward input (via a forward-pre-hook)
#:   weights     — attention probabilities (B, H, S, S)
#:   mid         — FFN hidden activation, post-GELU (B, S, F)
HookIO = Literal["output", "input", "weights", "mid"]

#: A dict of captured activations, keyed by hook-point string.
ActivationCache = dict[str, Tensor]

#: A PyTorch forward hook: ``(module, inputs, output) -> Optional[Tensor]``.
#: Returning a tensor replaces the module's output.
HookFn = Callable[[nn.Module, tuple[Any, ...], Any], Tensor | None]
