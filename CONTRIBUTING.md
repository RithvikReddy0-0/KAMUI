# Contributing to KAMUI

Thank you for your interest in contributing. KAMUI is an open research
project — contributions at every level are welcome, from fixing a typo
to implementing a new interpretability tool.

---

## The one rule

> Every line of code in KAMUI must be understandable by a graduate student
> reading it for the first time.

If you cannot explain a function's logic in plain English in one paragraph,
it is not ready to merge. This is the non-negotiable standard.

---

## Getting started

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/<your-username>/kamui
cd kamui

# 3. Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# 4. Set up pre-commit hooks (runs black + ruff + mypy on every commit)
pre-commit install

# 5. Verify everything passes before you start
pytest tests/unit/ -v
```

---

## What to contribute

### Good first issues (`good-first-issue` label)

- Add a new interpretability tool to `kamui/mechinterp/`
  (the hook system handles activation capture — you only write analysis logic)
- Add a unit test for an untested edge case
- Improve a docstring with a concrete worked example
- Add or improve an educational notebook

### Intermediate

- New positional encoding variant (RoPE, ALiBi) in `kamui/model/embedding.py`
- New sampling strategy in `kamui/evaluate/generation.py`
- Improve the attention visualiser's HTML output

### Advanced / Research (v0.2 targets)

- Sparse autoencoder implementation
  (see `research/future/sae_design.md` for the full design spec)
- Gradient-based attribution (`kamui/mechinterp/attribution.py`)
- Multi-GPU training support

---

## Standards — every PR must pass all of these

### 1. Tests pass

```bash
pytest tests/unit/ -v
```

All existing tests must pass. New functionality must have new tests.

### 2. Linting passes

```bash
ruff check kamui/ tests/
black --check kamui/ tests/
```

Or let pre-commit handle it:

```bash
pre-commit run --all-files
```

### 3. Type checking passes

```bash
mypy kamui/ --ignore-missing-imports
```

Every new public function needs type annotations.

### 4. Docstrings exist

Every public function and class needs a Google-style docstring:

```python
def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (B, H, S, Dh).
        k: Key tensor of shape (B, H, S, Dh).
        v: Value tensor of shape (B, H, S, Dh).
        mask: Boolean causal mask of shape (S, S).
              True entries are masked to -inf before softmax.

    Returns:
        output: Attention-weighted values, shape (B, H, S, Dh).
        weights: Attention probabilities, shape (B, H, S, S).
    """
```

### 5. Coverage does not drop

The coverage baseline must not decrease. Check with:

```bash
pytest tests/unit/ --cov=kamui --cov-report=term-missing
```

---

## Pull request process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the standards above.

3. **Run the full check**:
   ```bash
   pre-commit run --all-files
   pytest tests/unit/ -v
   ```

4. **Update `CHANGELOG.md`** under `[Unreleased]`.

5. **Open a PR** against `main` with:
   - A clear title following the commit message convention
   - A description explaining *what* changed and *why*
   - Reference to the issue being addressed (`Closes #123`)

6. A maintainer will review within 7 days.

---

## Commit message convention

```
<type>(<scope>): <short description>

Types:  feat | fix | docs | test | refactor | perf | chore
Scopes: tokenizer | model | training | hooks | mechinterp | evaluate | docs | ci

Examples:
  feat(mechinterp): add activation patching with layer-level granularity
  fix(attention): correct causal mask — use -inf not 0 before softmax
  test(hooks): add test for hook cleanup on exception path
  docs(mechinterp): add worked example to logit lens documentation
  chore(ci): add Python 3.12 to CI matrix
```

---

## Research contributions

If you are contributing a research finding (new circuit, new interpretability
result), follow the experiment documentation format in
`research/experiments/README.md`. Your PR should include:

- The experiment folder with `config.yaml`, `results.json`, and `notes.md`
- An entry in `research/RESEARCH_LOG.md`
- Any new or updated code that produced the finding

---

## Code of conduct

Be precise, be direct, be kind. Disagreements about implementation are
resolved first by the architecture document, then by the maintainer.
No ad hominem. No dismissiveness. We are all here to understand transformers.

---

## Questions?

Open an issue with the `question` label.
Include what you tried and what confused you.
