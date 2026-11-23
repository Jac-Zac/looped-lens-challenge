"""The judge: maps each readout line onto the options it names. It decides nothing else.

A judge is any object with
    async def map(system, user) -> {"parsed": dict | None, "raw": str, "attempts": int,
                                    "error": str | None, "cost": float}
`parsed` is {"lines": [{"line": "<label>", "named": [{"option": "<letter>", "token": "..."}]}]}.
Two are provided: LLMJudge (a model at OpenAI or on OpenRouter; the real judge) and StringMatchJudge (no model,
exact matches only; free, for dry runs and tests).
"""
import asyncio
import json
import os
import re

from . import config
from .prompts import LINE_LABELS, OPTION_LETTERS


def parse_reply(text):
    """The reply's JSON object, or None if it is not the required shape."""
    if not text:
        return None
    m = re.search(r"\{.*\}", re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()), re.S)
    try:
        j = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return j if isinstance(j, dict) and isinstance(j.get("lines"), list) else None


def at_openai(model):
    """True for a model reached at OpenAI itself: one with no vendor in its name ("gpt-6-luna").
    "vendor/model" names ("openai/gpt-6-luna", "xiaomi/mimo-v2.6-pro") are OpenRouter's."""
    return "/" not in model


def api_key(model=config.JUDGE_MODEL):
    """The key for the service `model` is reached through, from the environment or the .env file."""
    name = config.OPENAI_API_KEY_ENV if at_openai(model) else config.API_KEY_ENV
    key = os.environ.get(name)
    if not key and config.ENV_FILE.exists():
        for line in config.ENV_FILE.read_text().splitlines():
            if line.strip().startswith(name + "="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise RuntimeError(f"set {name} (or put it in {config.ENV_FILE})")
    return key


class LLMJudge:
    def __init__(self, model=config.JUDGE_MODEL, concurrency=config.CONCURRENCY):
        from openai import AsyncOpenAI
        self.model = model
        base = config.OPENAI_API_BASE if at_openai(model) else config.API_BASE
        self.client = AsyncOpenAI(base_url=base, api_key=api_key(model), timeout=300)
        self.sem = asyncio.Semaphore(concurrency)

    def _request(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        reasons = "luna" in self.model or "mimo" in self.model      # reasoning models: keep it short
        if at_openai(self.model):                           # OpenAI's own names for the same settings
            kw = dict(model=self.model, max_completion_tokens=config.MAX_OUTPUT_TOKENS, messages=messages)
            if reasons:
                kw["reasoning_effort"] = "low"
        else:
            extra = {"usage": {"include": True}}            # OpenRouter then reports each call's cost
            kw = dict(model=self.model, max_tokens=config.MAX_OUTPUT_TOKENS, extra_body=extra, messages=messages)
            if reasons:
                extra["reasoning"] = {"effort": "low"}
        if "luna" not in self.model:                        # Luna does not take a temperature
            kw["temperature"] = 0
        return kw

    async def map(self, system, user):
        cost, raw, error = 0.0, "", None
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                async with self.sem:
                    r = await self.client.chat.completions.create(**self._request(system, user))
            except Exception as e:                          # network, rate limit, refusal
                error = f"{type(e).__name__}: {str(e)[:300]}"
                if getattr(e, "status_code", None) in (400, 401, 403, 404):
                    break                                   # will not succeed on a retry
                await asyncio.sleep(3 * attempt)
                continue
            u = r.usage
            cost += (getattr(u, "cost", None) or (u.model_extra or {}).get("cost") or 0.0) if u else 0.0
            raw = (r.choices[0].message.content or "") if r.choices else ""
            parsed = parse_reply(raw)
            if parsed is not None:
                return {"parsed": parsed, "raw": raw, "attempts": attempt, "error": None, "cost": cost}
            error = "reply was not the required JSON"
        return {"parsed": None, "raw": raw, "attempts": attempt, "error": error, "cost": cost}


class StringMatchJudge:
    """No model: a token names an option only if it equals it (ignoring case and spaces)."""
    model = "string-match"

    async def map(self, system, user):
        readout = user.split("\nOPTIONS:")[0].split("best first):\n", 1)[1].strip().splitlines()
        options = {m.group(2).strip().lower(): m.group(1)
                   for m in re.finditer(rf"^([{OPTION_LETTERS}])\. (.*)$", user, re.M)}
        lines = []
        for row in readout:
            label, _, rest = row.partition("] ")
            named = [{"option": options[t.strip().lower()], "token": t}
                     for t in rest.split(" | ") if t.strip().lower() in options]
            if named and label.strip("[") in LINE_LABELS:
                lines.append({"line": label.strip("["), "named": named})
        parsed = {"lines": lines}
        return {"parsed": parsed, "raw": json.dumps(parsed, ensure_ascii=False), "attempts": 1,
                "error": None, "cost": 0.0}


def make_judge(model=config.JUDGE_MODEL):
    return StringMatchJudge() if model == "string-match" else LLMJudge(model)
