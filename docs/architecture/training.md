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

## The Loop (pseudocode)

```python
for step in range(max_steps):

    # ── 1. Get a batch ───────────────────────────────────────────────────
    input_ids, target_ids = next(dataloader)   # (B, S) each

    # ── 2. Forward pass + loss ───────────────────────────────────────────
    loss = model(input_ids, targets=target_ids)
    loss = loss / grad_accum_steps             # scale for accumulation

    # ── 3. Backward pass ─────────────────────────────────────────────────
    loss.backward()

    # ── 4. Gradient update (every grad_accum_steps micro-batches) ────────
    if (step + 1) % grad_accum_steps == 0:

        grad_norm_pre  = compute_grad_norm(model)
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        grad_norm_post = compute_grad_norm(model)

        lr = scheduler.get_lr(step)
        set_lr(optimiser, lr)

        optimiser.step()
        optimiser.zero_grad()

    # ── 5. Logging ───────────────────────────────────────────────────────
    if step % log_interval == 0:
        log(step, loss, lr, grad_norm_pre, grad_norm_post, tokens_per_sec)

    # ── 6. Evaluation ────────────────────────────────────────────────────
    if step % eval_interval == 0:
        val_loss = evaluate(model, val_dataloader)
        log(step, val_loss=val_loss)

    # ── 7. Checkpoint ────────────────────────────────────────────────────
    if step % checkpoint_interval == 0:
        save_checkpoint(model, optimiser, scheduler, step)
```

---

## Learning Rate Schedule

Linear warmup → cosine decay → floor.

```
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
requiring additional GPU memory.

```
effective_batch_size = batch_size × grad_accum_steps
```

With `batch_size=32` and `grad_accum_steps=4`, the effective batch is 128 —
equivalent to loading 128 sequences at once but using only 32 sequences
worth of GPU memory per step.

**Correctness requirement**: the loss must be divided by `grad_accum_steps`
before `.backward()`. Otherwise gradients are scaled by the number of
accumulation steps relative to a true large-batch run.

---

## Gradient Clipping

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Limits the L2 norm of the gradient vector. Prevents occasional large
gradient updates (from difficult training examples) from destabilising
the model.

**Diagnostics**: KAMUI logs `grad_norm_pre_clip` at every update step.
If this value is consistently above 5–10, the learning rate is likely
too high or the model is unstable.

---

## AdamW — Weight Decay Separation

AdamW applies L2 regularisation directly to weights. KAMUI uses two
parameter groups:

| Group | Parameters | Weight decay |
|-------|-----------|-------------|
| Decayed | All weight matrices (ndim ≥ 2) | 0.1 |
| Not decayed | Biases, LayerNorm γ/β (ndim < 2) | 0.0 |

Applying weight decay to biases and LayerNorm parameters is incorrect —
they should be unconstrained. This separation is subtle but measurably
improves training stability.

---

## Diagnosing Training Problems

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease at all | Causal mask broken (future leakage) or zero LR | Check mask uses `-inf`; check warmup |
| Loss decreases then explodes | LR too high | Lower `learning_rate` |
| Loss decreases very slowly | LR too low or gradient vanishing | Raise LR; check `grad_norm_pre_clip` |
| `grad_norm_pre_clip` > 10 consistently | LR too high | Lower LR; tighten `grad_clip` |
| Val loss diverges from train loss | Overfitting | Add dropout; use larger dataset |
| Loss is exactly `log(vocab_size)` after 200 steps | Gradient not flowing | Check `.backward()` is called |
