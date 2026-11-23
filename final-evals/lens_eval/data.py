"""Questions and submissions.

Question (a line of a question file):
    {"id", "family", "prompt", "intermediate", "answer", "options": [5 strings],
     "task_type", "rules", ...}
The correct option is the one equal to `intermediate`. `family` names the group the question is
scored in. `task_type` and `rules` are what the judge is told about the question: the line that
fills {task_type} in its user message, and the rules that fill {rules} in its system prompt.
An optional `groups` ({breakdown name: this question's group}) adds breakdowns to the score card.

Submission (what a participant's code returns for one question):
    {"id": "<question id>", "readouts": [{"layer": 12, "tokens": [" first", " second", ...]}, ...]}
One readout per chosen layer; tokens are that layer's top tokens, best first.
"""
import json
from pathlib import Path

from . import config


def read_questions(path, into=None):
    """Add one question file's questions to `into` ({question id: question}) and return it."""
    out = {} if into is None else into
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        assert q["id"] not in out, f"duplicate question id {q['id']}"
        assert q["intermediate"] in q["options"], q["id"]
        assert isinstance(q.get("task_type"), str) and isinstance(q.get("rules"), str), f"{q['id']}: no task_type or rules"
        out[q["id"]] = q
    return out


def load_questions(families=None):
    """{question id: question} from every question file in config.DATA_DIR, for the given
    families (default: all)."""
    out = {}
    for path in sorted(config.DATA_DIR.glob("*.jsonl")):
        read_questions(path, out)
    return {i: q for i, q in out.items() if not families or q["family"] in families}


def groups_of(q):
    """Breakdowns reported on the score card: {breakdown name: this question's group}."""
    return q.get("groups", {})


def clean_submission(sub, questions):
    """Check one submission dict. Returns (clean dict, None) or (None, reason)."""
    if not isinstance(sub, dict) or not isinstance(sub.get("id"), str) or sub["id"] not in questions:
        return None, f"unknown question id {sub.get('id') if isinstance(sub, dict) else sub!r}"
    rd = sub.get("readouts")
    if not isinstance(rd, list) or not rd:
        return None, "readouts must be a non-empty list"
    if len(rd) > config.MAX_LAYERS:
        return None, f"more than {config.MAX_LAYERS} layers"
    clean, seen = [], set()
    for r in rd:
        if not isinstance(r, dict) or "layer" not in r or not isinstance(r.get("tokens"), list):
            return None, "each readout needs a layer and a list of tokens"
        if isinstance(r["layer"], bool) or not isinstance(r["layer"], (int, str)):
            return None, "layer must be an integer or a string"
        if r["layer"] in seen:
            return None, "a layer appears twice"
        if not all(isinstance(t, str) for t in r["tokens"]):
            return None, "tokens must be strings"
        if any(len(t) > config.MAX_TOKEN_CHARS for t in r["tokens"][:config.N_TOKENS]):
            return None, f"a token is longer than {config.MAX_TOKEN_CHARS} characters"
        seen.add(r["layer"])
        clean.append({"layer": r["layer"], "tokens": r["tokens"][:config.N_TOKENS]})
    return {"id": sub["id"], "readouts": clean}, None
