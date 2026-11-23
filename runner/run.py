"""Start a submission's entry file once per dataset and report what happened."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

ENTRY_FILES = ("main.py", "main.ipynb")

# Variables with this prefix configure the task around the method (upload URLs
# and the like). They are never passed on to the method's process.
INTERNAL_PREFIX = "LEADERBOARD_"

_LOG_TAIL_LINES = 60


@dataclass(frozen=True)
class Dataset:
    name: str
    inputs: Path
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetRun:
    name: str
    ok: bool
    seconds: float
    # Why the run failed, in words that cannot contain anything the method printed.
    reason: str | None = None
    # The end of the method's log. Filled in only when the caller asks for it.
    log_tail: str | None = None


@dataclass
class RunReport:
    ok: bool
    seconds: float
    datasets: list[DatasetRun]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def load_manifest(path: Path) -> tuple[list[Dataset], dict]:
    """Read a ``manifest.json`` written by ``competition.prepare``. Returns the
    datasets and the runner limits recorded beside them."""
    manifest = json.loads(Path(path).read_text())
    base = Path(path).parent
    datasets = [Dataset(d["name"], base / d["inputs"], d.get("env", {})) for d in manifest["datasets"]]
    return datasets, manifest.get("runner", {})


def find_entry(submission: Path) -> Path:
    """The submission's entry file. Raises ``ValueError`` unless there is exactly one."""
    found = [submission / name for name in ENTRY_FILES if (submission / name).is_file()]
    if len(found) != 1:
        have = ", ".join(p.name for p in found) or "neither"
        raise ValueError(f"a submission needs exactly one of {' or '.join(ENTRY_FILES)} at its top level (found {have})")
    return found[0]


def _command(entry: Path) -> list[str]:
    if entry.suffix == ".ipynb":
        # A notebook is executed by a small script of ours, run by file path so
        # the method's process does not need this package on its import path.
        return [sys.executable, str(Path(__file__).with_name("notebook.py")), entry.name]
    return [sys.executable, entry.name]


def _directory_megabytes(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _run_one(command, cwd: Path, env: dict, log: Path, timeout: float, echo: bool) -> tuple[int | None, float]:
    """Run one process to completion or timeout, copying its output to ``log``
    (and to our stdout when ``echo``). Returns ``(exit code, seconds)``; the
    exit code is ``None`` when the process was killed for running too long."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # its own process group, so a timeout kills its children too
    )

    def copy_output() -> None:
        with log.open("wb") as sink:
            for line in process.stdout:
                sink.write(line)
                sink.flush()
                if echo:
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()

    copier = threading.Thread(target=copy_output, daemon=True)
    copier.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        code = None
    copier.join(timeout=10)
    return code, time.monotonic() - started


def run_submission(
    submission: Path,
    datasets: list[Dataset],
    outputs: Path,
    *,
    env: dict[str, str] | None = None,
    timeout_per_dataset: int = 1800,
    timeout_total: int = 4 * 3600,
    max_output_mb: int = 100,
    reveal_errors: bool = False,
    echo: bool = True,
    on_progress: Callable[[dict], None] | None = None,
) -> RunReport:
    """Run the submission on each dataset in turn, stopping at the first failure.

    Each dataset's results land in ``outputs/<dataset name>/`` and its log in
    ``outputs/<dataset name>.log``. ``env`` carries the submitter's credentials
    (``NDIF_API_KEY``, ``NDIF_HOST``, ``HF_TOKEN``). ``on_progress`` is called
    with ``{"dataset", "index", "total", "state"}`` as each dataset starts and
    ends. Never raises for anything the submission does; that is what the
    report is for.
    """
    started = time.monotonic()
    submission = Path(submission).resolve()
    outputs = Path(outputs).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    runs: list[DatasetRun] = []

    def progress(dataset: Dataset, index: int, state: str) -> None:
        if on_progress:
            on_progress({"dataset": dataset.name, "index": index, "total": len(datasets), "state": state})

    try:
        command = _command(find_entry(submission))
    except ValueError as error:
        return RunReport(False, 0.0, [DatasetRun(d.name, False, 0.0, str(error)) for d in datasets[:1]])

    inherited = {k: v for k, v in os.environ.items() if not k.startswith(INTERNAL_PREFIX)}

    for index, dataset in enumerate(datasets, start=1):
        remaining = timeout_total - (time.monotonic() - started)
        if remaining <= 0:
            runs.append(DatasetRun(dataset.name, False, 0.0, f"the submission used up its total time budget ({timeout_total}s)"))
            break

        output_dir = outputs / dataset.name
        output_dir.mkdir(parents=True, exist_ok=True)
        log = outputs / f"{dataset.name}.log"
        process_env = {
            **inherited,
            "PYTHONUNBUFFERED": "1",
            **(env or {}),
            **dataset.env,
            "DATASET_NAME": dataset.name,
            "DATASET_PATH": str(Path(dataset.inputs).resolve()),
            "OUTPUT_DIR": str(output_dir),
        }

        progress(dataset, index, "running")
        budget = min(timeout_per_dataset, remaining)
        code, seconds = _run_one(command, submission, process_env, log, budget, echo)

        reason = None
        if code is None:
            reason = f"timed out after {int(budget)}s"
        elif code != 0:
            reason = f"exited with code {code}"
        elif not any(output_dir.iterdir()):
            reason = "finished without writing anything to $OUTPUT_DIR"
        elif (size := _directory_megabytes(output_dir)) > max_output_mb:
            reason = f"wrote {size:.0f} MB to $OUTPUT_DIR (the limit is {max_output_mb} MB)"

        run = DatasetRun(dataset.name, reason is None, round(seconds, 2), reason)
        if reason and reveal_errors:
            run.log_tail = "\n".join(log.read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:])
        runs.append(run)
        progress(dataset, index, "done" if run.ok else "failed")
        if not run.ok:
            break

    ok = len(runs) == len(datasets) and all(r.ok for r in runs)
    return RunReport(ok, round(time.monotonic() - started, 2), runs)
