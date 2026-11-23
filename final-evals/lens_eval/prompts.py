"""Build the judge's system and user message for one question and its readouts."""
from functools import lru_cache

from . import config

LINE_LABELS = "αβγδεζηθικλμ"      # one per layer; the judge never sees layer numbers
OPTION_LETTERS = "ABCDE"


@lru_cache(maxsize=None)
def _text(name):
    return (config.PROMPT_DIR / name).read_text()


def render_readout(readouts):
    return "\n".join(f"[{LINE_LABELS[i]}] " + " | ".join(t.replace("\n", "\\n") for t in r["tokens"])
                     for i, r in enumerate(readouts))


def build(q, readouts):
    """(system, user). The judge sees the task type, the readout and the options: never the
    question, never which option is correct. The task type and the rules come with the question."""
    system = _text("system_mc.txt").replace("{rules}", q["rules"].strip())
    options = "\n".join(f"{l}. {o}" for l, o in zip(OPTION_LETTERS, q["options"]))
    user = (_text("user_mc.txt").replace("{task_type}", q["task_type"])
            .replace("{n}", str(config.N_TOKENS))
            .replace("{readout}", render_readout(readouts)).replace("{options}", options))
    return system, user
