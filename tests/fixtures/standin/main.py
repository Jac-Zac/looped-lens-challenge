"""A lens that needs no model: it says the same three tokens at two layer visits for every
question. It stands in for a real submission in the tests, so they exercise the real question
plumbing and the real readout format without calling NDIF. (questions.py is copied in beside it
by the fixture.)"""

import os

from transformers import AutoTokenizer

import questions

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"])
items = questions.load(tokenizer)
said = [{"layer": visit, "tokens": [" the", " a", " of"]} for visit in (12, 47)]
questions.write(items, {item.id: said for item in items})
print(f"{questions.DATASET}: {len(items)} questions")
