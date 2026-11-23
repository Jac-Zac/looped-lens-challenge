# final-evals

Scores a lens for the competition. It takes one dict per question (the lens's
readout), asks a judge model which options each readout names, and returns a
score card. It does not run the model or participants' code.

| Path | Contents |
|---|---|
| `datasets/` | Not in this repository. The question files (`*.jsonl`, one question per line) contain the targets and are kept private; put them here to run this folder on its own. |
| `judge_prompts/` | The judge's prompt texts, read at run time (see its README). |
| `lens_eval/` | The scoring code: about 500 lines in six small files. |
| `examples/` | `naive_logit_lens.py`, a participant's lens in full. |
| `runs/<name>/` | Output of each evaluation. Not committed: its files name the questions' options. |

## How a lens is scored

1. For each question the participant's code returns one dict:
   `{"id": "<question id>", "readouts": [{"layer": 12, "tokens": [" first", ...]}, ...]}`
   with up to 10 layers and up to 10 tokens per layer, best first. A token is
   at most 64 characters.
2. A judge model sees the readout and five options (the target plus four
   distractors), never the question or the correct option. For each layer it
   lists which options the tokens name.
3. Code checks every cited token is really on that layer, then each layer
   votes for the named option ranked highest on it.
4. The question passes if the correct option has strictly more votes than any
   other option. A tie fails.

## Running it

```python
from lens_eval import evaluate, collect, public_questions

subs = collect(my_lens_fn)             # calls my_lens_fn(question) once per question
card = evaluate(subs, name="my-lens")  # or pass any iterable of submission dicts
```

or, from this folder:

```bash
python -m lens_eval submissions.jsonl --name try-it --model string-match
```

- **`string-match`** is a free dry run with no judge model: a token names an
  option only if it equals it. It is a quick check that nothing is broken.
- **The leaderboard calls the same function.** `evaluator/evaluate.py` reads
  what a submission wrote and calls `evaluate(..., questions=..., out_dir=...)`
  with the questions the evaluator was given and a temporary directory, instead
  of the files in `datasets/` and `runs/`.
- **The judge is `gpt-6-luna`**, the default in `config.py`, reached at OpenAI.
  It needs the `openai` package and `OPENAI_API_KEY`, set in the environment or
  in a `.env` file in this folder. A full run is 300 judge calls and takes
  about half a minute.
- **Luna was checked against the two judges the suite was first validated
  with,** `xiaomi/mimo-v2.6-pro` and `google/gemini-3.8-flash`, on the
  reference lens's 300 readouts. It gave the same verdict as each of them on
  296; those two agreed with each other on 298.
- **Other models are reached through OpenRouter.** A model named
  `vendor/model` goes there, with `OPENROUTER_API_KEY`:
  `--model xiaomi/mimo-v2.6-pro` selects MiMo, which takes several minutes a
  run and costs about $0.45. Luna is not reached that way because OpenRouter
  limits a new account to 20 requests a minute for it.

`collect` and `public_questions()` give a participant's code each question
without its target and without what the judge is told about it. What is left
includes `model_input` (the exact string to run the model on) and
`read_position` / `read_token` (the token of that string to read at, always the
last one).

A question file's line carries, beside the question and its target, two texts
for the judge: `task_type`, the line that describes the task in the judge's
user message, and `rules`, the rules for that kind of option in its system
prompt. `family` names the group a question is scored in, and an optional
`groups` adds breakdowns to the score card. This code names no family or kind
of question itself.

## What a run writes

`runs/<name>/` gets four files:

| File | Contents |
|---|---|
| `score_card.json` | Overall score (the mean of the three family scores), each family's score with a 95% interval, and breakdowns. The only file safe to show a participant. |
| `results.jsonl` | One row per question: status, pass, votes per option, each layer's vote. Names the correct option. |
| `judge_calls.jsonl` | Every judge call: the message sent, the raw reply, the parsed mapping, cost. |
| `submissions.jsonl` | The submissions as received. |

- A question with no submission, or a malformed one, counts as failed.
- A question where the judge itself failed is left out of the score and
  counted under `judge_failed`. Running the same name again retries those and
  reuses every call that already succeeded.

## Where to change things (`lens_eval/`)

| File | Role |
|---|---|
| `config.py` | Paths, limits (10 layers x 10 tokens), judge model, API settings. Start here. |
| `data.py` | Loads questions; validates a submission dict. |
| `prompts.py` | Builds the judge's messages from `judge_prompts/` and from what each question says the judge is told. |
| `judge.py` | `LLMJudge` (a model at OpenAI or on OpenRouter) and `StringMatchJudge`. Any object with the same `map` method can be passed as `evaluate(..., judge=...)`. |
| `scoring.py` | Citation checks, layer votes, the plurality rule and the score card. |
| `run.py` | `evaluate`, `collect`, `public_questions` and the command line. |

If a file in `judge_prompts/` changes, bump `PROMPT_VERSION` in `config.py`
so that old judge calls are not reused.
