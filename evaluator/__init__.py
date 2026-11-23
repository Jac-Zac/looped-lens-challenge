"""Scores what a run wrote.

``evaluate.py`` in this package is the competition's scoring code: one function
that receives every dataset's output directory together with that dataset's
private data and returns the scores. This file is the small frame around it.
The evaluator task on Fargate and ``submit.py --local`` both go through
``score``. It imports nothing else from this repository.

The results envelope
--------------------
``score`` always returns this shape, and it is the only thing the leaderboard
knows about a submission's quality::

    {
      "ok": true,
      "metrics":   {"accuracy": 0.91, "f1": 0.90},        # headline numbers
      "breakdown": [{"name": "movies",                     # optional rows for
                     "metrics": {"accuracy": 0.93}}],      #   the per-dataset view
      "message":   null,      # shown to the participant
      "details":   {}         # kept for organizers, never shown
    }

When ``ok`` is false, ``message`` says why and the metrics are empty.
"""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    inputs: Path  # what the method was given
    private: Path | None  # what only the evaluator sees, if the dataset has any
    output: Path  # what the method wrote for this dataset


class SubmissionError(Exception):
    """The submission's output cannot be scored. The message is shown to the
    participant, so say what is wrong with the output and nothing about the
    private data."""


def load_manifest(path: Path, outputs: Path) -> list[Dataset]:
    """Read a ``manifest.json`` written by ``competition.prepare`` and pair each
    dataset with its directory under ``outputs``."""
    base = Path(path).parent
    return [
        Dataset(d["name"], base / d["inputs"], base / d["private"] if "private" in d else None, Path(outputs) / d["name"])
        for d in json.loads(Path(path).read_text())["datasets"]
    ]


def _number(value) -> float | None:
    """A metric value as JSON can carry it: a finite float, or null."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metrics(raw) -> dict[str, float | None]:
    if not isinstance(raw, dict):
        raise TypeError("`metrics` must be a dict of metric name to number")
    return {str(name): _number(value) for name, value in raw.items()}


def _failure(message: str, details: dict | None = None) -> dict:
    return {"ok": False, "metrics": {}, "breakdown": [], "message": message, "details": details or {}}


def score(datasets: list[Dataset]) -> dict:
    """Call the competition's ``evaluate`` and return a results envelope.
    Never raises: a problem with the submission or a bug in the scoring code
    both come back as ``ok: false``."""
    from evaluator.evaluate import evaluate

    try:
        raw = evaluate(datasets)
        return {
            "ok": True,
            "metrics": _metrics(raw["metrics"]),
            "breakdown": [{"name": str(row["name"]), "metrics": _metrics(row["metrics"])} for row in raw.get("breakdown", [])],
            "message": raw.get("message"),
            "details": json.loads(json.dumps(raw.get("details", {}), default=str)),
        }
    except SubmissionError as error:
        return _failure(str(error))
    except Exception:
        # A bug in the scoring code, not something the participant can fix.
        return _failure("the evaluator failed while scoring this submission; the organizers have the details", {"traceback": traceback.format_exc()})
