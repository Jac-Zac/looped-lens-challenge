"""The competition's scoring code: hand each family's readouts to the challenge's judge.

A lens writes ``readouts.jsonl`` into each family's output directory, one line per question::

    {"id": "<question id>", "readouts": [{"layer": 12, "tokens": [" first", " second", ...]}, ...]}

with up to 10 layers per question and up to 10 tokens per layer, best first. This file pairs
those lines with the family's full questions, which only the evaluator is given, and scores them
with ``lens_eval`` (final-evals/). A judge model is shown each question's readout and five
options, never the question, and says which options each layer's tokens name; code then gives
each layer one vote, and a question passes when its target has the most.

The judge is the model named in ``lens_eval/config.py``, reached at OpenAI with
``OPENAI_API_KEY``. Without that key, as on a participant's machine, the judge is
``string-match``: no model, and a token names an option only if it equals it. Setting
``LENS_EVAL_JUDGE`` picks either, or another model.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from evaluator import Dataset, SubmissionError

# lens_eval sits beside its prompts in final-evals/, which is not a name Python can import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final-evals"))
import lens_eval  # noqa: E402
from lens_eval import config as judge_config  # noqa: E402
from lens_eval import data as judge_data  # noqa: E402
from lens_eval import judge as judge_models  # noqa: E402

READOUTS = "readouts.jsonl"
QUESTIONS = "questions.jsonl"
JUDGE_VARIABLE = "LENS_EVAL_JUDGE"
MAX_READOUTS_MB = 20  # a family's readouts come to about 0.1 MB
STRING_MATCH = "string-match"
SHOWN_LAYERS = 24  # a lens that read more layers than this is told how many, not which


def _questions(dataset: Dataset) -> tuple[str, dict]:
    """The dataset's family and its full questions, from the directory only the evaluator is
    given. A dataset's name says nothing about what is in it (participants are shown the
    name), so the family is read from the questions."""
    path = dataset.private / QUESTIONS if dataset.private else None
    if path is None or not path.is_file():
        raise RuntimeError(f"{dataset.name}: the evaluator was given no {QUESTIONS} for this dataset")
    questions = judge_data.read_questions(path)
    families = sorted({question["family"] for question in questions.values()})
    if len(families) != 1:
        raise RuntimeError(f"{dataset.name}: its {QUESTIONS} has to hold questions of one family, and holds {families}")
    return families[0], questions


def _readouts(dataset: Dataset, questions: dict) -> tuple[list[dict], int]:
    """What the lens wrote for this dataset: one row per question it answered, and how many
    lines were for ids that are not this dataset's questions (those are left out)."""
    path = dataset.output / READOUTS
    if not path.is_file():
        raise SubmissionError(f"{dataset.name}: no {READOUTS} in $OUTPUT_DIR")
    if (megabytes := path.stat().st_size / 1e6) > MAX_READOUTS_MB:
        raise SubmissionError(f"{dataset.name}: {READOUTS} is {megabytes:.0f} MB; the limit is {MAX_READOUTS_MB} MB")
    try:
        lines = path.read_text(encoding="utf-8").split("\n")  # not splitlines: a token may be a line separator
    except UnicodeDecodeError:
        raise SubmissionError(f"{dataset.name}: {READOUTS} is not UTF-8 text")

    rows, seen, strays = [], set(), 0
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise SubmissionError(f"{dataset.name}: line {number} of {READOUTS} is not JSON")
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise SubmissionError(f'{dataset.name}: line {number} of {READOUTS} must be a JSON object with an "id" and "readouts"')
        if row["id"] not in questions:
            strays += 1
        elif row["id"] in seen:
            raise SubmissionError(f"{dataset.name}: {READOUTS} has two lines for one question")
        else:
            seen.add(row["id"])
            rows.append(row)
    return rows, strays


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _judge() -> str:
    """The judge to score with: the one asked for, else the challenge's judge model if there is
    a key to reach it with, else the judge that needs none."""
    if asked := os.environ.get(JUDGE_VARIABLE):
        return asked
    try:
        judge_models.api_key()
    except RuntimeError:
        return STRING_MATCH
    return judge_config.JUDGE_MODEL


def _message(datasets: list[Dataset], families: dict, card: dict, results: list[dict], rows: list[dict], strays: dict) -> str:
    """What the participant is told beside the scores: how many questions passed, the layers
    that were read, and anything about their file that cost them questions."""
    said = [f"Passed {card['overall']['passed']} of {card['overall']['n']} questions."]
    if card["run"]["judge_model"] == STRING_MATCH:
        said.append("Scored by exact matching, without the leaderboard's judge model.")

    # Only small layer numbers are repeated back. A lens names its layers itself and is run on
    # questions its author may not see, so nothing it could write those questions into (a
    # name, a very large number) goes into this message.
    judged = {result["id"] for result in results if result["status"] == "judged"}
    layers = {readout["layer"] for row in rows if row["id"] in judged for readout in row["readouts"]}
    numbered = sorted(layer for layer in layers if isinstance(layer, int) and 0 <= layer < 1000)
    if len(layers) > SHOWN_LAYERS or len(numbered) < len(layers):
        said.append(f"Layers read: {len(layers)} different ones.")
    elif numbered:
        said.append(f"Layers read: {', '.join(map(str, numbered))}.")

    for dataset in datasets:
        own = [result for result in results if result["family"] == families[dataset.name]]
        notes = []
        if missing := sum(result["status"] == "missing" for result in own):
            notes.append(f"{_count(missing, 'question')} without a readout")
        if invalid := [result for result in own if result["status"] == "invalid"]:
            notes.append(f"{_count(len(invalid), 'question')} with an unusable readout ({invalid[0]['reason']})")
        if strays[dataset.name]:
            notes.append(f"{_count(strays[dataset.name], 'line')} for ids that are not this dataset's questions")
        if notes:
            said.append(f"{dataset.name}: {'; '.join(notes)}.")
    return " ".join(said)


def evaluate(datasets: list[Dataset]) -> dict:
    """Score one submission. Called once, with every family it was run on."""
    started = time.monotonic()

    # Every family's file is read and checked before anything is sent to the judge.
    questions, rows, strays, families = {}, [], {}, {}
    for dataset in datasets:
        family, own = _questions(dataset)
        if family in families.values():
            raise RuntimeError(f"{dataset.name}: another dataset of this submission already holds the family {family}")
        if shared := sorted(own.keys() & questions.keys()):
            raise RuntimeError(f"{dataset.name}: question ids used by another dataset: {shared[:5]}")
        families[dataset.name] = family
        found, strays[dataset.name] = _readouts(dataset, own)
        questions |= own
        rows += found

    model = _judge()
    with tempfile.TemporaryDirectory(prefix="lens-eval-") as work:
        card = lens_eval.evaluate(rows, name="submission", model=model, questions=questions, out_dir=work)
        results = [json.loads(line) for line in (Path(work) / "results.jsonl").read_text().split("\n") if line]

    # A question the judge could not be reached for is not the lens's fault, and a score over
    # fewer questions is not comparable with the others. The evaluation fails instead.
    if failed := [result for result in results if result["status"] == "judge_failed"]:
        raise RuntimeError(f"the judge ({model}) failed on {len(failed)} of {len(results)} questions, so no score is given; first reason: {failed[0]['reason']}")

    seconds = round(time.monotonic() - started, 1)
    # The cost is known only when the judge's service reports it (OpenRouter does, OpenAI does not).
    cost = f", ${card['run']['judge_cost_usd']}" if card["run"]["judge_cost_usd"] else ""
    print(f"[evaluate] {len(rows)} readouts judged by {model} in {seconds}s{cost}", flush=True)
    outcomes = {result["id"]: ("pass" if result["pass"] else "fail") if result["status"] == "judged" else result["status"] for result in results}
    return {
        # The headline is the mean of the families' pass rates.
        "metrics": {"pass_rate": card["overall"]["score"]},
        "breakdown": [{"name": dataset.name, "metrics": {"pass_rate": card["families"][families[dataset.name]]["score"]}} for dataset in datasets],
        "message": _message(datasets, families, card, results, rows, strays),
        "details": {"score_card": card, "outcomes": outcomes, "seconds": seconds},
    }
