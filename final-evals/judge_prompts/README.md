# Judge prompts

The blind multiple-choice judge, the same for every question. One call per question.
The judge sees a task-type line, the readout and five lettered options; it
never sees the question or which option is correct. It only maps tokens to
options; code does the scoring (each layer votes for its best-ranked named
option; the question passes if the correct option has the most votes).

Prompt version `j3-v3` (2026-11-20): in `system_mc.txt` a word formed
directly from the option's own word names it ("Turkish"
names Turkey, "guitarist" names guitar). A different word for something
related still does not ("Istanbul"). The four-character fragment rule is
unchanged.

## Files

| File | Use |
|---|---|
| `system_mc.txt` | System prompt, shared by all questions. `{rules}` is replaced by the question's `rules`. |
| `user_mc.txt` | User message with `{task_type}`, `{n}`, `{readout}`, `{options}`. |

Two texts are not in this folder, because they say what kind of question is
being judged: each question carries its own `rules` (how an option of its kind
may be named: for number options, only by a token that writes the whole number)
and its own `task_type` (one line describing the task). They are in the question
files, which for the scored questions are private.

## Filling the user message

- `{n}`: tokens per layer (10).
- `{readout}`: one line per layer, labelled with Greek letters, tokens best
  first, separated by ` | ` (as `render_readout` in
  `lens_eval/prompts.py`).
- `{options}`: the question's `options` list, in
  file order, one per line as `A. ...` to `E. ...`. The correct option is
  the one equal to the question's `intermediate`.
