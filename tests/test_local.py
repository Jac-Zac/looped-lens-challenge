"""`python submit.py --local`, end to end, with the stand-in lens: the real question plumbing,
the real readout format and the challenge's scoring code. No call to NDIF and no judge model:
without a key for one, a local run scores by exact matching."""

import sys

import pytest

import submit as client


@pytest.fixture
def local(monkeypatch, capfd, tmp_path, made_up_competition):
    """Run `python submit.py --local <args>` in this process on the made-up questions.
    Returns its exit code and everything it printed."""
    monkeypatch.setattr(client.comp, "load", lambda: made_up_competition)
    monkeypatch.setattr(client, "WORK", tmp_path / "work")
    monkeypatch.setattr(client, "hf_token", lambda given: None)
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LENS_EVAL_JUDGE"):
        monkeypatch.delenv(name, raising=False)
    import evaluator.evaluate as scoring

    monkeypatch.setattr(scoring.judge_config, "ENV_FILE", tmp_path / "no-such-file")  # nor a key saved beside the judge

    def local(*args: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["submit.py", "--local", *args])
        try:
            code, said = client.main(), ""
        except SystemExit as stop:
            code, said = (1, stop.code) if isinstance(stop.code, str) else (stop.code, "")
        printed = capfd.readouterr()
        return code, printed.out + printed.err + said

    return local


def test_local_run_reads_and_scores_the_practice_questions(local, standin_submission):
    code, said = local("--submission", str(standin_submission))
    assert code == 0, said
    assert "── linking  (3/3)" in said
    assert "linking: 2 questions" in said  # the lens's own output, live
    assert "pass_rate" in said and "overall" in said and "0.1667" in said  # one of linking's two, and no other
    assert "Passed 1 of 6 questions. Scored by exact matching, without the leaderboard's judge model. Layers read: 12, 47." in said


def test_local_run_on_one_family(local, standin_submission):
    code, said = local("--dataset", "linking", "--submission", str(standin_submission))
    assert code == 0, said
    assert "── linking  (1/1)" in said and "Passed 1 of 2 questions." in said


def test_local_run_shows_the_real_error(local, tmp_path):
    (tmp_path / "main.py").write_text("raise RuntimeError('my lens is broken')\n")
    code, said = local("--dataset", "linking", "--submission", str(tmp_path))
    assert code == 1
    assert "RuntimeError: my lens is broken" in said  # on your own machine the log is yours to see
    assert "linking: exited with code 1" in said


def test_local_run_explains_unusable_output(local, tmp_path):
    (tmp_path / "main.py").write_text("import os; open(os.environ['OUTPUT_DIR'] + '/notes.txt', 'w').write('hi')\n")
    code, said = local("--dataset", "linking", "--submission", str(tmp_path))
    assert code == 1
    assert "not scored: linking: no readouts.jsonl in $OUTPUT_DIR" in said


def test_unknown_dev_dataset_is_refused(local, tmp_path):
    (tmp_path / "main.py").write_text("pass\n")
    code, said = local("--dataset", "nope", "--submission", str(tmp_path))
    assert code != 0 and "no dev dataset named" in said
