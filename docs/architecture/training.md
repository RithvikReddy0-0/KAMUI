# Training Loop

## Philosophy

KAMUI's training loop is explicit Python. There is no `model.fit()`,
no callback system, no framework magic. The loop is a `for` loop you can
read, modify, and understand in full.

This is intentional. Understanding *how* a model is trained is as important
as understanding its architecture. The training loop teaches:

- How gradients accumulate and apply
- Why learning rate warmup matters
- What gradient clipping does and why
- How to diagnose training instability from loss curves

---

## The Loop

This is what `Trainer.train(n_steps)` does, in `kamui/training/trainer.py`:

```text
for _ in range(n_steps):
    optimizer.zero_grad()

    # ── 1. Accumulate gradients over micro-batches ───────────────────────
    for _ in range(grad_accum_steps):
        inputs, targets = next(data)                # (B, S) each
        loss = model(inputs, targets=targets)
        (loss / grad_accum_steps).backward()        # scale for accumulation

    # ── 2. Clip, keeping the norm from before clipping ───────────────────
    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)

    # ── 3. Apply the scheduled learning rate and update ──────────────────
    set_lr(optimizer, scheduler.get_lr(step))
    optimizer.step()
    step += 1

    # ── 4. Record, and evaluate if it is time ────────────────────────────
    record = {step, train_loss, lr, grad_norm}
    if eval_interval and val_loader and step % eval_interval == 0:
        record["val_loss"] = evaluate()
```

---

## Where Training Runs

Move the model, and everything else follows it. `KAMUITransformer.forward` moves its
inputs to the model's device, so token IDs can stay on the CPU, which is where they
arrive from a tokenizer.

```python
from kamui.utils import get_device

model.to(get_device())   # CUDA, then Apple's MPS, then the CPU
```

Move the model before you create a `Trainer` or an `EMA`, because an `EMA` keeps its
copy of the weights on the device it started on.

`kamui-train` and `kamui-eval` pick the best device by default. Pass `--device cpu` or
`--device cuda:1` to choose one.

| What | Where it lives |
|------|----------------|
| Token IDs and targets | Anywhere: `forward` moves them to the model |
| Tool results, such as `probs` and `weights` | The model's device, so call `.cpu()` before `.numpy()` |
| `collect_activations` and `LinearProbe` features | The CPU, because they can outgrow a GPU's memory |
| `SparseAutoencoder` | Its own device: activations are moved to it |

!!! warning "Not yet run on a real GPU"
    CI has no GPU, so KAMUI's GPU support has only been tested on a simulated one.
    `tests/simulated_gpu.py` runs every tool on the CPU while enforcing CUDA's rules
    about devices. That catches tensors on the wrong device, but it can't find a
    problem with real hardware.

    Distributed training has run across two CPU processes with `gloo`. It has not
    been tested across several GPUs.

### Checking a Real GPU

`check_device.py` trains a small model on the device, runs every tool on it, compares
its results with the CPU's, and prints PASS or FAIL for each:

```bash
python scripts/check_device.py --device cuda
```

To try it for free on Google Colab, choose **Runtime > Change runtime type > T4 GPU**,
then run this in a cell:

```text
!git clone https://github.com/RithvikReddy0-0/KAMUI
%cd KAMUI
!pip install -q -e ".[all]"
!python scripts/check_device.py
```

---

## What `kamui-train` Reads

`kamui-train` reads these keys from the YAML config:

| Section | Keys |
|---------|------|
| `model` | Every key |
| `training` | `batch_size`, `grad_accum_steps`, `max_steps`, `learning_rate`, `min_lr`, `warmup_steps`, `grad_clip`, `weight_decay`, `eval_interval` |
| `data` | `train_split` |
| `logging` | `output_dir` |

The shipped configs also contain `betas`, `checkpoint_interval`, `mixed_precision`,
`log_interval`, `data.dataset` and `data.seed`, which `kamui-train` doesn't read yet.
It logs every step, saves one checkpoint when training finishes, and always trains
its tokenizer from the corpus you pass with `--corpus`.

---

## Learning Rate Schedule

Linear warmup → cosine decay → floor.

```text
if step < warmup_steps:
    lr = max_lr * (step / warmup_steps)

elif step < max_steps:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

else:
    lr = min_lr
```

**Why warmup?** At step 0, gradients are large and noisy. Starting at
`max_lr` causes loss spikes or divergence. Warmup lets the model reach a
stable region before the full learning rate is applied.

**Why cosine decay?** Smooth transitions, no sudden LR drops that cause loss
spikes. The model gently anneals into a low-loss minimum.

---

## Gradient Accumulation

Gradient accumulation simulates a larger effective batch size without
requiring additional memory.

```text
effective_batch_size = batch_size × grad_accum_steps
```

With `batch_size=32` and `grad_accum_steps=4`, the effective batch is 128 —
equivalent to loading 128 sequences at once but using only 32 sequences
worth of memory per micro-batch.

**Correctness requirement**: the loss must be divided by `grad_accum_steps`
before `.backward()`. Otherwise gradients are scaled by the number of
accumulation steps relative to a true large-batch run.

---

## Gradient Clipping

```python
from torch.nn.utils import clip_grad_norm_

grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Limits the L2 norm of the gradient vector. Prevents occasional large
gradient updates (from difficult training examples) from destabilising
the model.

**Diagnostics**: KAMUI records this pre-clip norm as `grad_norm` at every update
step. If this value is consistently above 5–10, the learning rate is likely
too high or the model is unstable.

---

## AdamW — Weight Decay Separation

AdamW applies L2 regularisation directly to weights. KAMUI uses two
parameter groups:

| Group | Parameters | Weight decay |
|-------|-----------|-------------|
| Decayed | All weight matrices (ndim ≥ 2) | 0.1 |
| Not decayed | Biases, normalisation parameters (ndim < 2) | 0.0 |

Biases and normalisation parameters are left out because they should be
unconstrained. This is the standard separation for transformers.

---

## Diagnosing Training Problems

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease at all | Causal mask broken (future leakage) or zero LR | Check mask uses `-inf`; check warmup |
| Loss decreases then explodes | LR too high | Lower `learning_rate` |
| Loss decreases very slowly | LR too low or gradient vanishing | Raise LR; check `grad_norm` |
| `grad_norm` > 10 consistently | LR too high | Lower LR; tighten `grad_clip` |
| Val loss diverges from train loss | Overfitting | Add dropout; use larger dataset |
| Loss is exactly `log(vocab_size)` after 200 steps | Gradient not flowing | Check `.backward()` is called |
| `need at least context_length + 1 tokens` | Corpus too small for the vocabulary | Use a larger corpus or a smaller `vocab_size` |
