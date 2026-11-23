"""The two Fargate entry points, run as they are in the task: as processes that
are handed nothing but URLs. A local HTTP server plays the part of S3."""

import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import competition as comp

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ["adding", "guessing", "linking"]


@pytest.fixture
def bucket():
    """An object store over HTTP: PUT stores a body at a path, GET returns it."""
    objects: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = objects.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.end_headers()
            self.wfile.write(body or b"")

        def do_PUT(self):
            objects[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    objects["url"] = f"http://127.0.0.1:{server.server_port}".encode()
    yield objects
    server.shutdown()


def zipped(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in directory.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    return buffer.getvalue()


def task(module: str, env: dict) -> subprocess.CompletedProcess:
    # The evaluator task scores by exact matching here: no judge model is called, whatever key
    # this machine has in its environment or saved beside the judge.
    inherited = {name: value for name, value in os.environ.items() if name not in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")}
    return subprocess.run([sys.executable, "-m", module], cwd=ROOT, env={**inherited, "LENS_EVAL_JUDGE": "string-match", **env}, capture_output=True, text=True)


def run_both_tasks(bucket: dict, submission: Path, tmp_path: Path, competition: comp.Competition) -> tuple[dict, dict]:
    """Run the runner task, then the evaluator task on what it uploaded. Each
    gets its own data directory, built the way the two images are."""
    url = bucket["url"].decode()
    comp.prepare(competition, "eval", tmp_path / "runner-data", private=False)
    comp.prepare(competition, "eval", tmp_path / "evaluator-data", private=True)
    bucket["/submission.zip"] = zipped(submission)
    bucket["/credentials.json"] = json.dumps({"NDIF_API_KEY": "key", "NDIF_HOST": "https://ndif.test", "HF_TOKEN": None}).encode()

    def links(name: str, **targets: str) -> str:
        bucket[f"/{name}-links.json"] = json.dumps({key: url + path for key, path in targets.items()}).encode()
        return url + f"/{name}-links.json"

    ran = task(
        "runner.task",
        {
            "LEADERBOARD_DATA": str(tmp_path / "runner-data"),
            "LEADERBOARD_LINKS_URL": links(
                "runner",
                submission="/submission.zip",
                credentials="/credentials.json",
                output="/outputs.tar.gz",
                report="/report.json",
                progress="/progress.json",
            ),
        },
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    report = json.loads(bucket["/report.json"])

    scored = task(
        "evaluator.task",
        {"LEADERBOARD_DATA": str(tmp_path / "evaluator-data"), "LEADERBOARD_LINKS_URL": links("evaluator", output="/outputs.tar.gz", results="/results.json")},
    )
    assert scored.returncode == 0, scored.stdout + scored.stderr
    return report, json.loads(bucket["/results.json"])


def test_runner_task_hands_its_outputs_to_the_evaluator_task(bucket, standin_submission, tmp_path, made_up_competition):
    report, results = run_both_tasks(bucket, standin_submission, tmp_path, made_up_competition)

    assert report["ok"] and [d["name"] for d in report["datasets"]] == FAMILIES
    assert json.loads(bucket["/progress.json"]) == {"dataset": FAMILIES[-1], "index": 3, "total": 3, "state": "done"}
    assert results["ok"], results
    assert [row["name"] for row in results["breakdown"]] == FAMILIES
    assert results["breakdown"][2]["metrics"] == {"pass_rate": 0.5}  # the stand-in gets one of linking's two
    assert results["message"].startswith("Passed 1 of 6 questions.") and results["message"].endswith("Layers read: 12, 47.")


def test_a_failed_run_reports_no_log_and_scores_as_unusable(bucket, tmp_path, made_up_competition):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "main.py").write_text("import os; print(open(os.environ['DATASET_PATH'] + '/questions.jsonl').read()); raise RuntimeError('the lens crashed')\n")
    report, results = run_both_tasks(bucket, submission, tmp_path, made_up_competition)

    assert not report["ok"] and report["datasets"][0]["reason"] == "exited with code 1"
    # The scored questions are private, and this lens printed them: none of its log comes back.
    assert report["datasets"][0]["log_tail"] is None
    assert "Compute" not in json.dumps(report)
    assert not results["ok"] and "no readouts.jsonl" in results["message"]


def test_evaluator_refuses_an_archive_that_escapes_its_directory(bucket, tmp_path, made_up_competition):
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("../../escaped.txt")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"hi"))
    bucket["/outputs.tar.gz"] = buffer.getvalue()
    comp.prepare(made_up_competition, "eval", tmp_path / "data", private=True)

    url = bucket["url"].decode()
    bucket["/links.json"] = json.dumps({"output": url + "/outputs.tar.gz", "results": url + "/results.json"}).encode()
    done = task("evaluator.task", {"LEADERBOARD_DATA": str(tmp_path / "data"), "LEADERBOARD_LINKS_URL": url + "/links.json"})
    assert done.returncode != 0 and "/results.json" not in bucket
    assert not (tmp_path.parent / "escaped.txt").exists()
