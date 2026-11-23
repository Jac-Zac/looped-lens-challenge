"""`submit.py` against a running Space: the client and the API agree."""

import socket
import sys
import threading
import time

import pytest
import uvicorn

import competition as comp
import submit as client
from space import app as space_app
from test_space import fake_whoami, score, scored_datasets, state  # noqa: F401  (fixtures)


@pytest.fixture
def competition() -> comp.Competition:
    """The example as it ships: no categories to declare."""
    competition = comp.load()
    competition.submission.rate_limit.max, competition.submission.rate_limit.window_hours = 3, 24
    competition.ndif.required_tags = ["example-competition"]  # the tag the made-up accounts carry
    competition.datasets.eval = scored_datasets()
    return competition


@pytest.fixture
def space_url(competition, state):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(space_app.create_app(competition, state, whoami=fake_whoami), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def run(monkeypatch, capsys, competition, space_url):
    """Run `python submit.py <args>` in this process against the test Space.
    Returns its exit code and everything it printed."""
    monkeypatch.setattr(client.comp, "load", lambda: competition)
    monkeypatch.setattr(client, "hf_token", lambda given: None)  # never send this machine's own token

    def run(*args: str, key: str = "key-ada") -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["submit.py", "--space-url", space_url, "--ndif-api-key", key, *args])
        try:
            code, said = client.main(), ""
        except SystemExit as stop:  # sys.exit(message), or argparse refusing the arguments
            code, said = (1, stop.code) if isinstance(stop.code, str) else (stop.code, "")
        printed = capsys.readouterr()
        return code, printed.out + printed.err + said

    return run


def test_submit_then_follow_to_the_scores(run, state, any_submission):
    code, said = run("--team", "Ada", "--submission", str(any_submission))
    assert code == 0, said
    assert "packaged 1 files" in said and "submitted as Ada: " in said
    (stored,) = state.submissions()
    assert stored.fields == {} and stored.id in said

    code, said = run("--status")
    assert "team Ada" in said and "2 of 3 left in this 24h window" in said
    assert stored.id in said and "running" in said

    score(state, stored.id, 0.875)
    code, said = run("--status", stored.id)
    assert code == 0, said
    assert "scored" in said and "0.8750" in said and "Dataset A" in said and "dataset_a" not in said


def test_declared_categories_become_required_flags(run, competition, state, any_submission):
    competition.submission.fields = [comp.SubmissionField(name="track", choices=["open", "closed"])]

    code, said = run("--team", "Ada", "--submission", str(any_submission))
    assert code != 0 and "--track" in said and state.submissions() == []

    code, said = run("--team", "Ada", "--track", "open", "--submission", str(any_submission))
    assert code == 0, said
    assert state.submissions()[0].fields == {"track": "open"}


def test_a_flag_for_an_undeclared_category_is_refused(run, any_submission):
    code, said = run("--team", "Ada", "--track", "open", "--submission", str(any_submission))
    assert code == 2 and "unrecognized arguments: --track open" in said


def test_refusals_reach_the_participant_in_words(run, any_submission):
    code, said = run("--team", "Dee", "--submission", str(any_submission), key="key-nobody")
    assert code != 0 and "NDIF does not recognise this API key" in said


def test_a_submission_without_an_entry_file_is_not_uploaded(run, state, tmp_path):
    (tmp_path / "notes.txt").write_text("no method here")
    code, said = run("--team", "Ada", "--submission", str(tmp_path))
    assert code != 0 and "exactly one of main.py or main.ipynb" in said
    assert state.submissions() == []
