"""Execute only the independent inference cells and persist their visual outputs.

Retains existing outputs of unchanged cells in the full executed notebook.
Does not train models or create registry releases.
"""
from copy import deepcopy
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient


def execute():
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "01_classical_ml_pipeline.ipynb"
    executed = source.with_name("01_classical_ml_pipeline.executed.ipynb")
    notebook = nbformat.read(source, as_version=4)
    full = deepcopy(notebook)
    if executed.exists():
        previous = nbformat.read(executed, as_version=4)
        outputs = {cell.source: cell for cell in previous.cells if cell.cell_type == "code"}
        for cell in full.cells:
            if cell.cell_type == "code" and cell.source in outputs:
                cell.outputs = deepcopy(outputs[cell.source].outputs)
                cell.execution_count = outputs[cell.source].execution_count
    indices = [i for i, cell in enumerate(notebook.cells)
               if "inference-demo" in cell.metadata.get("tags", [])]
    if not indices:
        raise RuntimeError("Rebuild the notebook to add the inference demo cells first")
    demo = nbformat.v4.new_notebook(cells=[deepcopy(notebook.cells[i]) for i in indices],
                                  metadata=deepcopy(notebook.metadata))

    def on_start(cell, cell_index, **kwargs):
        print(f"demo cell {cell_index}: {cell.source.splitlines()[0][:90]}", flush=True)

    NotebookClient(demo, timeout=300, kernel_name="plate-ml-314",
                   resources={"metadata": {"path": str(root)}}, on_cell_start=on_start).execute()
    for index, cell in zip(indices, demo.cells):
        notebook.cells[index] = deepcopy(cell)
        full.cells[index] = deepcopy(cell)
    # Persist demo outputs in the main notebook the user normally opens as well.
    nbformat.write(notebook, source)
    nbformat.write(full, executed)
    return source


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(execute())
