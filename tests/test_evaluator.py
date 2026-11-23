"""The evaluator on readouts files, scored by the challenge's own code with the judge that needs
no model: a token names an option only if it equals it. The questions here are made up; the
competition's are private."""

import json

import pytest

import evaluator

OPTIONS = ["red", "blue", "green", "black", "white"]
QUESTIONS = {
    "adding": [
        {"id": "sum-901", "intermediate": "7", "options": ["7", "3", "5", "1", "9"], "groups": {"size": "small"}},
        {"id": "sum-902", "intermediate": "4", "options": ["2", "4", "6", "8", "0"]},
    ],
    "guessing": [
        {"id": "player-901", "intermediate": "chess", "options": ["chess", "darts", "curling", "polo", "rowing"]},
        {"id": "tower-901", "intermediate": "Peru", "options": ["Chile", "Peru", "Laos", "Fiji", "Oman"]},
    ],
    "linking": [
        {"id": "link-901", "intermediate": "red", "options": OPTIONS},
        {"id": "link-902", "intermediate": "blue", "options": OPTIONS},
    ],
}
# A lens that gets both arithmetic questions, one guessing question and no linking one.
SAID = {
    "sum-901": [" 7", " the"],
    "sum-902": ["4"],
    "player-901": [" Chess"],
    "tower-901": [" Chile"],
    "link-901": [" the", " of"],
    "link-902": [" red"],
}


def row(question_id: str, layers=(5, 9)) -> dict:
    return {"id": question_id, "readouts": [{"layer": layer, "tokens": SAID[question_id]} for layer in layers]}


def write(dataset: evaluator.Dataset, rows: list) -> None:
    (dataset.output / "readouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(autouse=True)
def no_judge_model(monkeypatch):
    monkeypatch.setenv("LENS_EVAL_JUDGE", "string-match")


@pytest.fixture
def scored(tmp_path) -> list[evaluator.Dataset]:
    """The three families as the evaluator sees them, with the lens's readouts written."""
    datasets = []
    for family, questions in QUESTIONS.items():
        private, output = tmp_path / "data" / family / "private", tmp_path / "out" / family
        private.mkdir(parents=True)
        output.mkdir(parents=True)
        (private / "questions.jsonl").write_text("".join(json.dumps({**q, "family": family, "prompt": "?", "answer": "?", "task_type": "a made-up task", "rules": "- A made-up rule."}) + "\n" for q in questions))
        dataset = evaluator.Dataset(family, tmp_path / "data" / family / "inputs", private, output)
        write(dataset, [row(q["id"]) for q in questions])
        datasets.append(dataset)
    return datasets


def test_envelope_for_complete_readouts(scored):
    results = evaluator.score(scored)
    assert results["ok"], results
    assert results["metrics"] == {"pass_rate": 0.5}  # the mean of the families' rates
    assert results["breakdown"] == [
        {"name": "adding", "metrics": {"pass_rate": 1.0}},
        {"name": "guessing", "metrics": {"pass_rate": 0.5}},
        {"name": "linking", "metrics": {"pass_rate": 0.0}},
    ]
    assert results["message"] == "Passed 3 of 6 questions. Scored by exact matching, without the leaderboard's judge model. Layers read: 5, 9."
    assert results["details"]["score_card"]["run"]["judge_model"] == "string-match"
    # A breakdown comes from the question itself: this code knows no kind of question.
    assert results["details"]["score_card"]["families"]["adding"]["by"] == {"size": {"small": {"n": 1, "passed": 1, "score": 1.0}}}
    assert results["details"]["outcomes"]["player-901"] == "pass" and results["details"]["outcomes"]["link-902"] == "fail"
    json.dumps(results)


def test_a_question_passes_only_when_its_target_has_the_most_votes(scored):
    # One layer names the target and another names a different option: a tie, which fails.
    write(scored[2], [{"id": "link-901", "readouts": [{"layer": 5, "tokens": ["red"]}, {"layer": 9, "tokens": ["blue"]}]}, row("link-902")])
    assert evaluator.score(scored)["breakdown"][2]["metrics"] == {"pass_rate": 0.0}
    write(scored[2], [{"id": "link-901", "readouts": [{"layer": 5, "tokens": ["red"]}, {"layer": 9, "tokens": ["red", "blue"]}]}, row("link-902")])
    assert evaluator.score(scored)["breakdown"][2]["metrics"] == {"pass_rate": 0.5}


def test_missing_readouts_file(scored):
    (scored[1].output / "readouts.jsonl").unlink()
    results = evaluator.score(scored)
    assert not results["ok"] and results["metrics"] == {}
    assert results["message"] == "guessing: no readouts.jsonl in $OUTPUT_DIR"


def test_a_line_that_is_not_json(scored):
    with (scored[0].output / "readouts.jsonl").open("a") as file:
        file.write("{not json\n")
    assert evaluator.score(scored)["message"] == "adding: line 3 of readouts.jsonl is not JSON"


def test_a_line_that_is_not_a_readout(scored):
    write(scored[0], [["sum-901"]])
    assert evaluator.score(scored)["message"] == 'adding: line 1 of readouts.jsonl must be a JSON object with an "id" and "readouts"'


def test_two_lines_for_one_question(scored):
    write(scored[2], [row("link-901"), row("link-902"), row("link-901")])
    assert evaluator.score(scored)["message"] == "linking: readouts.jsonl has two lines for one question"


def test_missing_and_unusable_questions_count_as_failed(scored):
    write(scored[0], [row("sum-901", layers=range(11))])  # too many layers, and the other question is absent
    results = evaluator.score(scored)
    assert results["ok"], results
    assert results["breakdown"][0]["metrics"] == {"pass_rate": 0.0}
    assert results["message"] == (
        "Passed 1 of 6 questions. Scored by exact matching, without the leaderboard's judge model. Layers read: 5, 9. "
        "adding: 1 question without a readout; 1 question with an unusable readout (more than 10 layers)."
    )
    assert results["details"]["outcomes"]["sum-901"] == "invalid" and results["details"]["outcomes"]["sum-902"] == "missing"


def test_text_is_not_a_token(scored):
    write(scored[2], [{"id": "link-901", "readouts": [{"layer": 5, "tokens": ["red " * 20]}]}, row("link-902")])
    results = evaluator.score(scored)
    assert results["ok"] and "linking: 1 question with an unusable readout (a token is longer than 64 characters)." in results["message"]


def test_lines_for_another_familys_questions_are_left_out(scored):
    write(scored[0], [row("sum-901"), row("sum-902"), {"id": "link-902", "readouts": [{"layer": 5, "tokens": ["blue"]}]}, {"id": "nope", "readouts": []}])
    results = evaluator.score(scored)
    assert results["ok"], results
    assert results["breakdown"][2]["metrics"] == {"pass_rate": 0.0}  # link-902 is scored on what linking's own file says
    assert results["message"].endswith("adding: 2 lines for ids that are not this dataset's questions.")


def test_a_datasets_name_need_not_be_its_family(scored):
    # Participants are shown names such as dataset_a, so the family comes from the questions.
    masked = [evaluator.Dataset(f"dataset_{letter}", d.inputs, d.private, d.output) for letter, d in zip("abc", scored)]
    write(masked[2], [row("link-901")])
    results = evaluator.score(masked)
    assert results["ok"], results
    assert results["breakdown"] == [
        {"name": "dataset_a", "metrics": {"pass_rate": 1.0}},
        {"name": "dataset_b", "metrics": {"pass_rate": 0.5}},
        {"name": "dataset_c", "metrics": {"pass_rate": 0.0}},
    ]
    assert results["message"].endswith("dataset_c: 1 question without a readout.")
    assert "linking" not in results["message"]


def test_what_the_participant_is_told_names_no_target(scored):
    results = evaluator.score(scored)
    shown = json.dumps({key: results[key] for key in ("metrics", "breakdown", "message")})
    assert not any(option in shown for option in ("chess", "Peru", "blue", "darts"))


def test_nothing_a_lens_wrote_is_repeated_back_to_it(scored):
    """A lens runs on questions its author may not see, and names its own layers. A layer name,
    or a number big enough to spell something, could carry those questions out."""
    secret = "Compute (3 + 4) + 1."
    write(scored[2], [{"id": "link-901", "readouts": [{"layer": secret, "tokens": ["red"]}, {"layer": 10**40, "tokens": ["red"]}]}, row("link-902")])
    write(scored[0], [{"id": "sum-901", "readouts": [{"layer": secret, "tokens": ["7"]}, {"layer": secret, "tokens": ["7"]}]}, row("sum-902")])
    results = evaluator.score(scored)
    assert results["ok"], results
    shown = json.dumps({key: results[key] for key in ("metrics", "breakdown", "message")})
    assert "Compute" not in shown and "0000000000" not in shown
    assert "Layers read: 4 different ones." in results["message"]
    assert "1 question with an unusable readout (a layer appears twice)" in results["message"]


def test_the_judge_is_exact_matching_only_when_there_is_no_key_for_the_model(scored, monkeypatch, tmp_path):
    from evaluator import evaluate as scoring

    monkeypatch.delenv("LENS_EVAL_JUDGE")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(scoring.judge_config, "ENV_FILE", tmp_path / "no-such-file")
    assert scoring._judge() == "string-match"
    monkeypatch.setenv("OPENAI_API_KEY", "a key that is never used")
    assert scoring._judge() == scoring.judge_config.JUDGE_MODEL
    monkeypatch.setenv("LENS_EVAL_JUDGE", "another/model")
    assert scoring._judge() == "another/model"


def test_no_score_when_the_judge_cannot_be_reached(scored, monkeypatch):
    import lens_eval.run

    class Unreachable:
        model = "some/model"

        async def map(self, system, user):
            return {"parsed": None, "raw": "", "attempts": 5, "error": "APIConnectionError: no route", "cost": 0.0}

    monkeypatch.setattr(lens_eval.run, "make_judge", lambda model: Unreachable())
    results = evaluator.score(scored)
    assert not results["ok"] and results["metrics"] == {}
    assert results["message"] == "the evaluator failed while scoring this submission; the organizers have the details"
    assert "failed on 6 of 6 questions" in results["details"]["traceback"] and "APIConnectionError" in results["details"]["traceback"]


def test_an_evaluator_without_the_questions_says_so_to_the_organizers(scored):
    (scored[0].private / "questions.jsonl").unlink()
    results = evaluator.score(scored)
    assert not results["ok"] and "organizers" in results["message"]
    assert "was given no questions.jsonl" in results["details"]["traceback"]


def test_a_judge_model_without_a_vendor_is_reached_at_openai(monkeypatch, tmp_path):
    from lens_eval import judge

    monkeypatch.setattr(judge.config, "ENV_FILE", tmp_path / "no-such-file")
    monkeypatch.setenv("OPENROUTER_API_KEY", "an openrouter key")
    monkeypatch.setenv("OPENAI_API_KEY", "an openai key")

    direct = judge.LLMJudge("gpt-6-luna")
    assert str(direct.client.base_url).startswith("https://api.openai.com/") and direct.client.api_key == "an openai key"
    request = direct._request("system", "user")
    assert request["reasoning_effort"] == "low" and request["max_completion_tokens"] and "extra_body" not in request and "temperature" not in request

    routed = judge.LLMJudge("openai/gpt-6-luna")
    assert str(routed.client.base_url).startswith("https://openrouter.ai/") and routed.client.api_key == "an openrouter key"
    assert routed._request("system", "user")["extra_body"] == {"usage": {"include": True}, "reasoning": {"effort": "low"}}
    assert judge.LLMJudge("xiaomi/mimo-v2.6-pro")._request("system", "user")["temperature"] == 0

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="set OPENAI_API_KEY"):
        judge.LLMJudge("gpt-6-luna")
