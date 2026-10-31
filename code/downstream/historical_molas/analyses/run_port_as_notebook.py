from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


NOTEBOOK = Path(__file__).with_name("port_as_analysis.ipynb")
OUTPUT_ROOT = Path(
    "/data1/home/jw1017/AS_Dock/results/MolAS/AL_analysis/port_as_primary_reconciled_20260827_v1"
)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = json.loads(NOTEBOOK.read_text())
    shell = InteractiveShell.instance()
    completed = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        print(f"EXECUTING CELL {index}", flush=True)
        result = shell.run_cell("".join(cell.get("source", [])), store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(f"Notebook cell {index} failed") from error
        completed.append(index)
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": str(NOTEBOOK),
        "notebook_sha256": sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "notebook_cells": len(notebook["cells"]),
        "completed_code_cells": completed,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("ALL NOTEBOOK CODE CELLS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
