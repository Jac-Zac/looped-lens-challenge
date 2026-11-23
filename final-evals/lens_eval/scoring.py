"""Scoring. All of it is code; the judge only supplied the token-to-option mapping.

1. Verify: a citation counts only if the cited token really is on the cited line.
2. Vote: each layer votes for the named option whose token is ranked best on that line.
   A line that names nothing, or whose best token names two options, casts no vote.
3. Plurality: the question passes if the correct option has strictly more votes than every
   other option. A tie fails.
"""
import re
from collections import Counter, defaultdict

from .data import groups_of
from .prompts import LINE_LABELS, OPTION_LETTERS


def _norm(s):
    return re.sub(r"[\s\.\,\:\;\!\?\"'“”‘’\(\)\[\]\*_]+", "", str(s)).lower()


def verified_mapping(parsed, readouts, n_options):
    """({layer: {option index: best rank}}, number of citations dropped)."""
    lines = {LINE_LABELS[i]: (r["layer"], [_norm(t) for t in r["tokens"]]) for i, r in enumerate(readouts)}
    named, dropped = defaultdict(dict), 0
    for ln in parsed.get("lines") or []:
        label = str(ln.get("line", "")).strip("[] ") if isinstance(ln, dict) else ""
        for e in (ln.get("named") or []) if label in lines else []:
            letter = str(e.get("option", "")).strip().upper()[:1] if isinstance(e, dict) else ""
            token = _norm(e.get("token", "")) if isinstance(e, dict) else ""
            layer, tokens = lines[label]
            idx = OPTION_LETTERS.find(letter) if letter else -1
            if 0 <= idx < n_options and token and token in tokens:
                rank = tokens.index(token) + 1
                named[layer][idx] = min(named[layer].get(idx, rank), rank)
            else:
                dropped += 1
    return dict(named), dropped


def layer_votes(named):
    """{layer: option index it votes for}."""
    votes = {}
    for layer, ranks in named.items():
        best = min(ranks.values())
        top = [o for o, r in ranks.items() if r == best]
        if len(top) == 1:
            votes[layer] = top[0]
    return votes


def plurality(votes, gold):
    """True if `gold` has strictly more layer votes than any other option."""
    count = Counter(votes.values())
    return count[gold] > max((n for o, n in count.items() if o != gold), default=0)


def score_question(q, readouts, parsed):
    gold = q["options"].index(q["intermediate"])
    named, dropped = verified_mapping(parsed, readouts, len(q["options"]))
    votes = layer_votes(named)
    count = Counter(votes.values())
    return {
        "pass": plurality(votes, gold),
        "correct_option": q["intermediate"],
        "votes": {q["options"][o]: n for o, n in count.most_common()},
        "layer_votes": {str(layer): q["options"][o] for layer, o in sorted(votes.items(), key=lambda kv: str(kv[0]))},
        "dropped_citations": dropped,
    }


def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 3), round(c + h, 3)]


def _tally(rows):
    scored = [r for r in rows if r["status"] != "judge_failed"]   # a judge failure is not the lens's fault
    passed = sum(bool(r["pass"]) for r in scored)
    out = {"score": round(passed / len(scored), 4) if scored else None, "passed": passed, "n": len(scored)}
    for status in ("missing", "invalid", "judge_failed"):
        k = sum(r["status"] == status for r in rows)
        if k:
            out[status] = k
    return out


def score_card(results, questions, run_info):
    """results: one row per question (see run.py). Returns the score card dict."""
    fams = defaultdict(list)
    for r in results:
        fams[r["family"]].append(r)
    card = {"run": run_info, "overall": {}, "families": {}}
    for fam, rows in fams.items():
        entry = _tally(rows)
        entry["ci95"] = wilson(entry["passed"], entry["n"])
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for name, group in groups_of(questions[r["id"]]).items():
                by[name][group].append(r)
        if by:
            entry["by"] = {name: {g: _tally(rs) for g, rs in sorted(gs.items())} for name, gs in by.items()}
        card["families"][fam] = entry
    scores = [f["score"] for f in card["families"].values() if f["score"] is not None]
    card["overall"] = {"score": round(sum(scores) / len(scores), 4) if scores else None,   # mean of family scores
                       "passed": sum(f["passed"] for f in card["families"].values()),
                       "n": sum(f["n"] for f in card["families"].values())}
    return card
