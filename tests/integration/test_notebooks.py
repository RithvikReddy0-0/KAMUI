"""Execute every educational notebook against the current API.

Notebooks are documentation that runs; without this test they silently rot as
the library evolves.  Each notebook's code cells are executed in order, in a
fresh namespace, from the ``notebooks/`` directory the notebooks expect.  No
Jupyter dependency is needed because the notebooks use no IPython magics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_notebooks_are_discovered() -> None:
    assert len(NOTEBOOKS) >= 8, f"expected the educational notebooks in {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(NOTEBOOK_DIR)
    monkeypatch.setattr(sys, "path", list(sys.path))  # notebooks prepend ".."
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    namespace: dict[str, object] = {"__name__": "__main__"}
    try:
        code_cells = (c for c in cells if c["cell_type"] == "code")
        for index, cell in enumerate(code_cells):
            source = compile("".join(cell["source"]), f"{path.name}[cell {index}]", "exec")
            exec(source, namespace)
    finally:
        plt.close("all")
