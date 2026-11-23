"""The plumbing every lens shares: which questions to read, and how to hand the readouts back.

Each line of ``questions.jsonl`` in $DATASET_PATH is one question: the exact string to run the
model on (``model_input``) and the token of it to read at (``read_position``, with that token's
text in ``read_token`` as a check). This module turns them into token ids and writes readouts in
the format the leaderboard scores into $OUTPUT_DIR.

You should not need to change this file to try a new lens.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

DATASET = os.environ["DATASET_NAME"]
QUESTIONS = Path(os.environ["DATASET_PATH"]) / "questions.jsonl"
OUTPUT = Path(os.environ["OUTPUT_DIR"]) / "readouts.jsonl"

MAX_LAYERS = 10  # a question's readout may name at most this many layers
MAX_TOKENS = 10  # and at most this many tokens per layer, best first


@dataclass(frozen=True)
class Item:
    """One question, ready to run: its model input as token ids and the position to read."""

    id: str
    ids: list[int]
    position: int


def token_strings(tokenizer, token_ids: list[int]) -> list[str]:
    """Each token's text, as the leaderboard's judge will be shown it."""
    return [tokenizer.decode([token_id]) or tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]


def load(tokenizer) -> list[Item]:
    """Every question of this dataset, tokenized the way the questions were built: the model
    input as it is, with no special tokens added."""
    items = []
    for line in QUESTIONS.read_text().splitlines():
        if not line.strip():
            continue
        question = json.loads(line)
        ids = tokenizer(question["model_input"], add_special_tokens=False)["input_ids"]
        position = question["read_position"]
        found = token_strings(tokenizer, [ids[position]])[0]
        if found != question["read_token"]:
            raise ValueError(f"{question['id']}: the token at position {position} is {found!r}, not {question['read_token']!r}; is this the model's tokenizer?")
        items.append(Item(question["id"], ids, position))
    return items


def write(items: list[Item], readouts: dict) -> None:
    """Write the readouts file: one line per question.

    ``readouts[question id]`` is a list of ``{"layer": ..., "tokens": [...]}``, one per layer the
    lens read for that question, each with its top tokens as text, best first. A question left
    out counts as failed, and so does one with more layers than allowed.
    """
    with OUTPUT.open("w") as out:
        for item in items:
            if item.id not in readouts:
                continue
            read = readouts[item.id]
            assert len(read) <= MAX_LAYERS, f"{item.id}: {len(read)} layers; the leaderboard takes at most {MAX_LAYERS}"
            line = {"id": item.id, "readouts": [{"layer": r["layer"], "tokens": list(r["tokens"])[:MAX_TOKENS]} for r in read]}
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
