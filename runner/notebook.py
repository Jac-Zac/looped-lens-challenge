"""Execute a notebook top to bottom, as the entry file of a submission.

Run as a script (``python notebook.py main.ipynb``) inside the method's own
process, so the notebook's kernel inherits the method's environment variables.
Cell output is printed so it ends up in the run's log, and a failing cell makes
this script exit 1.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def _print_outputs(notebook) -> None:
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                sys.stdout.write(output.text)
            elif output.output_type == "error":
                sys.stdout.write("\n".join(output.traceback) + "\n")
    sys.stdout.flush()


def main(path: str) -> int:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    # Register a kernel that is this exact interpreter. Relying on the name
    # "python3" would pick up whatever kernel of that name the machine has
    # installed, which may belong to a different environment.
    kernels = Path(tempfile.mkdtemp(prefix="leaderboard-kernel-"))
    spec = kernels / "kernels" / "leaderboard"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "leaderboard",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(kernels)

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,  # the runner enforces the time limit on the whole process
        kernel_name="leaderboard",
        resources={"metadata": {"path": str(Path(path).resolve().parent)}},
    )
    try:
        client.execute()
    except CellExecutionError:
        _print_outputs(notebook)
        return 1
    _print_outputs(notebook)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
