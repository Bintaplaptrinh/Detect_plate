from pathlib import Path
from time import perf_counter
import sys
import nbformat
from nbclient import NotebookClient


def execute(notebook_name="01_classical_ml_pipeline"):
    root = Path(__file__).resolve().parents[1]
    if notebook_name not in {"01_classical_ml_pipeline", "02_vnlp_ocr_ablation", "03_paper_inspired_pipeline", "04_glyph_quality_filter"}:
        raise ValueError("Unknown experiment notebook")
    source = root / "notebooks" / f"{notebook_name}.ipynb"
    output = root / "notebooks" / f"{notebook_name}.executed.ipynb"
    notebook = nbformat.read(source, as_version=4)
    start = perf_counter()

    def on_start(cell, cell_index, **kwargs):
        print(f"cell {cell_index}: {cell.source.splitlines()[0][:90]}", flush=True)

    def on_executed(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)
        print(f"saved cell {cell_index}, elapsed={perf_counter() - start:.1f}s", flush=True)

    client = NotebookClient(notebook, timeout=1800, kernel_name="plate-ml-314",
                            resources={"metadata": {"path": str(root)}},
                            on_cell_start=on_start, on_cell_executed=on_executed)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    return output


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(execute(sys.argv[1] if len(sys.argv) > 1 else "01_classical_ml_pipeline"))
