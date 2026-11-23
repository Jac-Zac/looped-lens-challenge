"""Run an evaluation: submissions in, judge calls saved, score card out.

    from lens_eval import evaluate
    card = evaluate(submissions, name="my-lens")        # submissions: iterable of dicts (data.py)

    python -m lens_eval submissions.jsonl --name my-lens [--model M] [--families a,b]

Writes to runs/<name>/:
    submissions.jsonl   what was received
    judge_calls.jsonl   every judge call: the user message, the raw reply, the parsed mapping
    results.jsonl       one row per question: status, pass, votes per option, vote per layer
    score_card.json     the scores
Re-running the same name reuses judge calls already made for identical prompts.
"""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from . import config, data, prompts, scoring
from .judge import make_judge


# Never shown to a participant's code: the target, and what the judge is told about the question.
HIDDEN = {"family", "intermediate", "answer", "options", "task_type", "rules", "groups"}


def public_questions(families=None):
    """What a participant is given: every question without its target and what the judge is told.
    Each has "id", "prompt", "model_input" (the exact string to run) and "read_position" /
    "read_token" (the token of model_input at which to read)."""
    return [{k: v for k, v in q.items() if k not in HIDDEN} for q in data.load_questions(families).values()]


def collect(lens_fn, families=None):
    """Call a participant's function once per public question and gather the submission dicts."""
    return [lens_fn(q) for q in public_questions(families)]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


async def _run(submissions, name, model, families, judge, questions=None, out_dir=None):
    questions = data.load_questions(families) if questions is None else questions
    out = Path(out_dir) if out_dir else config.RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    submissions = list(submissions)
    _write_jsonl(out / "submissions.jsonl", submissions)

    clean, invalid = {}, {}
    for sub in submissions:
        ok, why = data.clean_submission(sub, questions)
        if ok:
            clean[ok["id"]] = ok
        elif isinstance(sub, dict) and isinstance(sub.get("id"), str) and sub["id"] in questions:
            invalid[sub["id"]] = why
        else:
            print("ignored:", why)

    judge = judge or make_judge(model)
    calls_path = out / "judge_calls.jsonl"
    cache = {}
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            c = json.loads(line)
            if c.get("parsed") is not None:
                cache[c["key"]] = c

    async def ask(sub):
        q = questions[sub["id"]]
        system, user = prompts.build(q, sub["readouts"])
        key = hashlib.sha1(f"{config.PROMPT_VERSION}|{judge.model}|{system}|{user}".encode()).hexdigest()[:16]
        if key in cache:
            return sub["id"], cache[key], 0.0
        reply = await judge.map(system, user)
        call = {"key": key, "id": sub["id"], "model": judge.model, "user": user, **reply}
        with open(calls_path, "a") as fh:
            fh.write(json.dumps(call, ensure_ascii=False) + "\n")
        return sub["id"], call, reply["cost"]

    # A judge model takes seconds per call and a run is hundreds of calls, so say how far it has
    # got, about ten times over the run. The string-match judge is instant and says nothing.
    live = bool(clean) and judge.model != "string-match"
    started, judged, every = time.monotonic(), 0, max(1, -(-len(clean) // 10))
    if live:
        print(f"[judge] {len(clean)} readouts to judge with {judge.model}; this can take a few minutes", flush=True)

    async def judge_one(sub):
        nonlocal judged
        result = await ask(sub)
        judged += 1
        if live and (judged % every == 0 or judged == len(clean)):
            print(f"[judge] {judged} of {len(clean)} judged ({time.monotonic() - started:.0f}s)", flush=True)
        return result

    done = await asyncio.gather(*(judge_one(s) for s in clean.values()))
    calls = {iid: call for iid, call, _ in done}

    results = []
    for iid, q in questions.items():
        row = {"id": iid, "family": q["family"], "status": "judged", "pass": False}
        if iid in invalid:
            row.update(status="invalid", reason=invalid[iid])
        elif iid not in clean:
            row.update(status="missing")
        elif calls[iid]["parsed"] is None:
            row.update(status="judge_failed", **{"pass": None}, reason=calls[iid]["error"])
        else:
            row.update(scoring.score_question(q, clean[iid]["readouts"], calls[iid]["parsed"]))
        results.append(row)
    _write_jsonl(out / "results.jsonl", results)
    failed = [r for r in results if r["status"] == "judge_failed"]
    if failed:
        print(f"WARNING: the judge failed on {len(failed)} questions (left out of the scores); "
              f"first reason: {failed[0]['reason']}")

    card = scoring.score_card(results, questions, {
        "name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "judge_model": judge.model,
        "prompt_version": config.PROMPT_VERSION, "pass_rule": "plurality",
        "max_layers": config.MAX_LAYERS, "tokens_per_layer": config.N_TOKENS,
        "judge_calls_made": sum(1 for _, c, _ in done if c["key"] not in cache),
        "judge_calls_reused": sum(1 for _, c, _ in done if c["key"] in cache),
        "judge_cost_usd": round(sum(cost for _, _, cost in done), 4)})
    (out / "score_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
    return card


def evaluate(submissions, name="run", model=config.JUDGE_MODEL, families=None, judge=None, questions=None, out_dir=None):
    """Judge and score an iterable of submission dicts. Returns the score card (also saved).

    By default the questions are the files named in config.py and the run is saved in runs/<name>/.
    A caller that keeps them elsewhere passes `questions` ({question id: question}, see
    data.read_questions) and `out_dir`."""
    return asyncio.run(_run(submissions, name, model, families, judge, questions, out_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("submissions", help="JSONL file, one submission dict per line")
    ap.add_argument("--name", default="run")
    ap.add_argument("--model", default=config.JUDGE_MODEL, help='judge model, or "string-match" for a free dry run')
    ap.add_argument("--families", help="comma-separated; default all")
    a = ap.parse_args()
    subs = [json.loads(l) for l in open(a.submissions) if l.strip()]
    card = evaluate(subs, a.name, a.model, a.families.split(",") if a.families else None)
    print(json.dumps({"overall": card["overall"],
                      "families": {f: {k: v for k, v in e.items() if k != "by"} for f, e in card["families"].items()}}, indent=1))
    print("saved to", config.RUNS_DIR / a.name)


if __name__ == "__main__":
    main()
