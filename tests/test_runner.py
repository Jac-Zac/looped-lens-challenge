"""The runner on toy datasets: these tests are about how a method is started and what is
reported, not about the benchmark."""

import json
import os
import sys
from pathlib import Path

import nbformat
import pytest

import runner


def dataset(tmp_path: Path, name: str) -> runner.Dataset:
    inputs = tmp_path / "inputs" / name
    inputs.mkdir(parents=True)
    (inputs / "plan.jsonl").write_text("{}\n")
    return runner.Dataset(name, inputs, {"MODEL_ID": "some/model"})


@pytest.fixture
def one(tmp_path) -> list[runner.Dataset]:
    return [dataset(tmp_path, "first")]


@pytest.fixture
def two(tmp_path) -> list[runner.Dataset]:
    return [dataset(tmp_path, "first"), dataset(tmp_path, "second")]


def write(submission: Path, source: str) -> Path:
    submission.mkdir(exist_ok=True)
    (submission / "main.py").write_text(source)
    return submission


def test_runs_once_per_dataset_and_collects_outputs(tmp_path, two):
    source = "import os\nopen(os.environ['OUTPUT_DIR'] + '/out.txt', 'w').write(os.environ['DATASET_NAME'])\nprint('ran', os.environ['DATASET_NAME'])\n"
    events = []
    report = runner.run_submission(write(tmp_path / "s", source), two, tmp_path / "out", echo=False, on_progress=events.append)

    assert report.ok and [r.name for r in report.datasets] == ["first", "second"]
    assert (tmp_path / "out/second/out.txt").read_text() == "second"
    assert "ran first" in (tmp_path / "out/first.log").read_text()
    assert [(e["dataset"], e["state"]) for e in events] == [("first", "running"), ("first", "done"), ("second", "running"), ("second", "done")]


def test_method_sees_the_contract_variables_and_not_internal_ones(tmp_path, one, monkeypatch):
    monkeypatch.setenv("LEADERBOARD_LINKS_URL", "https://example.invalid/secret")
    source = "import json, os\nopen(os.path.join(os.environ['OUTPUT_DIR'], 'env.json'), 'w').write(json.dumps(dict(os.environ)))\n"
    report = runner.run_submission(write(tmp_path / "s", source), one, tmp_path / "out", env={"NDIF_API_KEY": "key", "NDIF_HOST": "https://ndif.test"}, echo=False)
    seen = json.loads((tmp_path / "out/first/env.json").read_text())

    assert report.ok
    assert seen["DATASET_NAME"] == "first" and seen["MODEL_ID"] == "some/model"
    assert Path(seen["DATASET_PATH"], "plan.jsonl").is_file()
    assert seen["NDIF_API_KEY"] == "key" and seen["NDIF_HOST"] == "https://ndif.test"
    assert not [name for name in seen if name.startswith("LEADERBOARD_")]


@pytest.mark.parametrize(
    "source, reason",
    [
        ("raise SystemExit(3)", "exited with code 3"),
        ("pass", "without writing anything"),
        ("import time; time.sleep(60)", "timed out after 1s"),
        ("import os; open(os.environ['OUTPUT_DIR'] + '/big', 'wb').write(b'0' * 2_000_000)", "the limit is 1 MB"),
    ],
)
def test_failures_are_reported_with_a_reason(tmp_path, one, source, reason):
    report = runner.run_submission(write(tmp_path / "s", source), one, tmp_path / "out", echo=False, timeout_per_dataset=1, max_output_mb=1)
    assert not report.ok and reason in report.datasets[0].reason
    assert report.datasets[0].log_tail is None


def test_stops_at_the_first_failure(tmp_path, two):
    report = runner.run_submission(write(tmp_path / "s", "raise ValueError('what the method printed')"), two, tmp_path / "out", echo=False)
    assert not report.ok and [r.name for r in report.datasets] == ["first"]
    assert "what the method printed" not in report.to_json()


def test_log_tail_is_returned_only_when_revealing_errors(tmp_path, one):
    report = runner.run_submission(write(tmp_path / "s", "raise ValueError('what went wrong')"), one, tmp_path / "out", echo=False, reveal_errors=True)
    assert "what went wrong" in report.datasets[0].log_tail


def test_a_submission_needs_exactly_one_entry_file(tmp_path, one):
    report = runner.run_submission(tmp_path / "inputs", one, tmp_path / "out", echo=False)
    assert not report.ok and "exactly one of main.py or main.ipynb" in report.datasets[0].reason

    both = write(tmp_path / "s", "pass")
    (both / "main.ipynb").write_text("{}")
    with pytest.raises(ValueError, match="found main.py, main.ipynb"):
        runner.find_entry(both)


def test_notebook_entry_runs_in_this_interpreter(tmp_path, one):
    cell = (
        "import json, os, sys\n"
        "print('hello from the notebook')\n"
        "json.dump({'python': sys.executable, 'dataset': os.environ['DATASET_NAME']}, open(os.environ['OUTPUT_DIR'] + '/out.json', 'w'))\n"
    )
    submission = tmp_path / "s"
    submission.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell)]), submission / "main.ipynb")

    report = runner.run_submission(submission, one, tmp_path / "out", echo=False)
    assert report.ok, (tmp_path / "out/first.log").read_text()
    written = json.loads((tmp_path / "out/first/out.json").read_text())
    assert written["dataset"] == "first"
    assert os.path.realpath(written["python"]) == os.path.realpath(sys.executable)
    assert "hello from the notebook" in (tmp_path / "out/first.log").read_text()


def test_failing_notebook_cell_fails_the_run(tmp_path, one):
    submission = tmp_path / "s"
    submission.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise RuntimeError('boom')")]), submission / "main.ipynb")
    report = runner.run_submission(submission, one, tmp_path / "out", echo=False)
    assert not report.ok and report.datasets[0].reason == "exited with code 1"
    assert "RuntimeError" in (tmp_path / "out/first.log").read_text()
