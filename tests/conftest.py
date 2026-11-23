import json
import shutil
from pathlib import Path

import pytest

import competition as comp

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
MODEL = "thinkingmachines/Inkling-2-Mini"
FAMILIES = ["adding", "guessing", "linking"]

# Made-up questions, two per made-up family, in the shape of the competition's own (which are
# on the Hub, the scored ones privately): what to run the model on, the target, five options,
# and what the judge is told about the question.
# The stand-in lens says " the", " a" and " of", so it gets link-902 and nothing else.
MADE_UP = {
    "adding": [
        {"id": "sum-901", "prompt": "Compute (3 + 4) + 1.", "intermediate": "7", "answer": "8", "options": ["7", "3", "5", "1", "9"]},
        {"id": "sum-902", "prompt": "Compute (1 + 3) + 2.", "intermediate": "4", "answer": "6", "options": ["2", "4", "6", "8", "0"]},
    ],
    "guessing": [
        {"id": "player-901", "prompt": "What game does Ada Example play?", "intermediate": "chess", "answer": "chess", "options": ["chess", "darts", "curling", "polo", "rowing"]},
        {"id": "tower-901", "prompt": "Which country is the Example Tower in?", "intermediate": "Peru", "answer": "Peru", "options": ["Chile", "Peru", "Laos", "Fiji", "Oman"]},
    ],
    "linking": [
        {"id": "link-901", "prompt": "Fact: The colour of the example is ", "intermediate": "red", "answer": "r", "options": ["red", "blue", "green", "black", "white"]},
        {"id": "link-902", "prompt": "Fact: The first word of the example is ", "intermediate": "the", "answer": "t", "options": ["an", "the", "this", "that", "one"]},
    ],
}
HIDDEN = ("family", "intermediate", "answer", "options", "task_type", "rules", "groups")  # what a lens is never given
JUDGE_IS_TOLD = {"task_type": "a made-up task", "rules": "- A made-up rule."}


@pytest.fixture(scope="session")
def question_sets(tmp_path_factory) -> Path:
    """MADE_UP laid out the way the competition's datasets are: <family>/inputs/questions.jsonl
    for a lens, <family>/private/questions.jsonl for the evaluator. Read positions come from
    the model's tokenizer, which Hugging Face caches after the first download."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    root = tmp_path_factory.mktemp("questions")
    for family, questions in MADE_UP.items():
        full = []
        for question in questions:
            ids = tokenizer(question["prompt"], add_special_tokens=False)["input_ids"]
            where = {"model_input": question["prompt"], "read_position": len(ids) - 1, "read_token": tokenizer.decode([ids[-1]])}
            full.append({"family": family, **JUDGE_IS_TOLD, **question, **where})
        for part, rows in (("private", full), ("inputs", [{k: v for k, v in q.items() if k not in HIDDEN} for q in full])):
            (root / family / part).mkdir(parents=True)
            (root / family / part / "questions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


@pytest.fixture
def made_up_competition(question_sets) -> comp.Competition:
    """The competition as configured, with its practice and scored sets both replaced by the
    made-up questions, so that nothing is fetched from the Hub."""
    competition = comp.load()
    datasets = [
        comp.Dataset(name=family, inputs=comp.Source(dir=str(question_sets / family / "inputs")), private=comp.Source(dir=str(question_sets / family / "private")), env={"MODEL_ID": MODEL})
        for family in FAMILIES
    ]
    competition.datasets.dev, competition.datasets.eval = datasets, list(datasets)
    return competition


@pytest.fixture
def standin_submission(tmp_path) -> Path:
    """A working submission directory that calls no model: the stand-in lens plus the real
    plumbing module from submission/. It needs the model's tokenizer."""
    submission = Path(shutil.copytree(FIXTURES / "standin", tmp_path / "submission"))
    shutil.copy(ROOT / "submission" / "questions.py", submission / "questions.py")
    return submission


@pytest.fixture
def any_submission(tmp_path) -> Path:
    """A directory that packages as a submission; what it does when run does not matter."""
    submission = tmp_path / "anything"
    submission.mkdir()
    (submission / "main.py").write_text("print('hi')\n")
    return submission
