"""Type definitions for the hook system.

Responsibilities:
    Define the type aliases and TypedDicts used throughout the hook system
    so that mypy can verify correctness of hook registration and activation
    access.

Types defined here (to be implemented in Phase 3):
    - ``HookPoint``: Literal string union of all valid hook point identifiers
    - ``ActivationCache``: TypedDict mapping hook point strings to tensors
    - ``HookFn``: type alias for a PyTorch forward hook callable
      Signature: (module, input, output) -> Optional[Tensor]

Note on HookFn:
    A forward hook may optionally return a tensor to *replace* the module's
    output.  This is the mechanism used by activation patching:
    the patching hook returns the clean activation instead of the corrupted
    one, causally intervening in the forward pass.

Implemented in: Phase 3, Week 13
"""

# Implementation begins in Phase 3.
