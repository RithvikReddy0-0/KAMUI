# Hook System

## Purpose

The hook system is the bridge between the model and every interpretability
tool in KAMUI. It allows any activation in the forward pass to be captured,
inspected, and replaced — without touching a single line of model code.

---

## The Core Problem

PyTorch's `register_forward_hook` is powerful but dangerous:

```python
# Dangerous pattern — hook handle must be manually removed
handle = model.blocks[3].attn.register_forward_hook(my_fn)
logits = model(ids)
handle.remove()    # ← easy to forget; silent memory leak if missed
```

If `.remove()` is never called:
- Hooks accumulate across forward passes
- The activation cache grows unboundedly
- Old activations contaminate new runs

KAMUI's `HookManager` makes the correct pattern the only possible pattern.

---

## HookManager

`kamui/hooks/manager.py`

A context manager that registers hooks on entry and removes all of them on
exit — including on exceptions.

```python
from kamui.hooks import HookManager

with HookManager(model) as hooks:
    hooks.attach("blocks.3.attn", point="output")
    hooks.attach("blocks.3.attn", point="weights")
    hooks.attach("embed",         point="output")

    logits = model(token_ids)

    attn_out  = hooks.get("blocks.3.attn.output")   # (B, S, D)
    attn_wts  = hooks.get("blocks.3.attn.weights")  # (B, H, S, S)
    embedding = hooks.get("embed.output")           # (B, S, D)

# ← All hooks removed here, even if an exception occurred inside
```

### Key methods

| Method | Description |
|--------|-------------|
| `hooks.attach(module, point)` | Register a hook at the named module and IO point |
| `hooks.get(hook_point)` | Retrieve cached activation tensor |
| `hooks.get_all()` | Return `dict[str, Tensor]` of all cached activations |
| `hooks.clear()` | Clear the cache without removing hooks |

---

## Hook Point Naming

Hook points are strings in the format `"<module_path>.<io>"`.

`module_path` matches PyTorch's `model.named_modules()` keys exactly.
`io` specifies which activation to capture.

### All valid hook points (small config, 6 layers)

```
embed.output                      # combined token + position embeddings  (B, S, D)

blocks.0.attn.pre_softmax         # attention scores before masking       (B, H, S, S)
blocks.0.attn.weights             # attention probabilities (post-softmax)(B, H, S, S)
blocks.0.attn.output              # projected attention output             (B, S, D)
blocks.0.ffn.mid                  # FFN hidden activations (post-GELU)    (B, S, F)
blocks.0.ffn.output               # FFN output                             (B, S, D)

blocks.1.attn.pre_softmax         # (same pattern for layers 1–5)
...
blocks.5.ffn.output

unembed.input                     # residual stream before unembedding    (B, S, D)
```

### Listing all hook points programmatically

```python
from kamui.hooks import HookRegistry
from kamui.model import ModelConfig

config = ModelConfig.from_yaml("configs/small.yaml")
all_points = HookRegistry.all_points(config)
print(all_points)
# ["embed.output", "blocks.0.attn.pre_softmax", ..., "unembed.input"]
```

---

## Activation Patching via Hooks

The hook system supports output replacement: a hook can return a tensor
to *replace* the module's output instead of just observing it. This is
the mechanism behind `ActivationPatcher`.

```python
# Internal mechanism used by ActivationPatcher
def patch_fn(module, input, output):
    return clean_activation   # replaces the corrupted output

handle = model.blocks[3].attn.register_forward_hook(patch_fn)
```

`HookManager` exposes this cleanly:

```python
with HookManager(model) as hooks:
    hooks.attach("blocks.3.attn", point="output", replace_with=clean_tensor)
    patched_logits = model(corrupted_ids)
```

---

## Design Decisions

### Why strings, not module references?

Module references break if the model is reloaded or re-instantiated.
Strings are serialisable, printable, and config-storable.

### Why not a persistent hooks API?

Long-lived hooks are almost always bugs. The context manager enforces the
correct lifecycle: attach → forward pass → detach. Any use case requiring
persistent hooks should be re-examined.

### Why does attaching hooks not change output?

Read-only hooks (observation only) never return a value. PyTorch only
modifies the computation if the hook function returns a non-None tensor.
KAMUI's default hooks return `None` — they only write to the cache.

---

## References

- PyTorch documentation: `torch.nn.Module.register_forward_hook`
- TransformerLens `run_with_cache` — the inspiration for this design
