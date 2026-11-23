"""Entry point of the runner task on Fargate.

Fetches one submission, runs it on the datasets baked into this image, and
uploads what it wrote. Everything in this container is assumed to be under the
submission's control once the method starts, so the task is given no AWS
identity. It is handed links instead, each good for one HTTP method on one
object.

    LEADERBOARD_LINKS_URL   GET  a JSON object of the links this task may use:
        submission    GET  the submission zip
        credentials   GET  JSON of variables for the method
                           (NDIF_API_KEY, NDIF_HOST, HF_TOKEN)
        output        PUT  outputs.tar.gz, one directory per dataset
        report        PUT  report.json, a runner.RunReport plus the image's build info
        progress      PUT  progress.json, the latest progress event
    LEADERBOARD_DATA        directory holding manifest.json (default /data)

Exits 0 once a report has been uploaded, whatever the report says. A non-zero
exit means the task itself broke.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from runner.run import load_manifest, run_submission

MAX_UNPACKED_MB = 2000


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _put(url: str, body: bytes) -> None:
    request = urllib.request.Request(url, data=body, method="PUT", headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=600):
        pass


def _unpack(archive: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        if sum(member.file_size for member in package.infolist()) > MAX_UNPACKED_MB * 1e6:
            raise ValueError(f"the submission unpacks to more than {MAX_UNPACKED_MB} MB")
        package.extractall(dest)  # zipfile drops absolute paths and ".." components


def _pack(outputs: Path, names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in names:
            if (outputs / name).is_dir():
                archive.add(outputs / name, arcname=name)
    return buffer.getvalue()


def main() -> int:
    env = os.environ
    manifest = Path(env.get("LEADERBOARD_DATA", "/data")) / "manifest.json"
    datasets, limits = load_manifest(manifest)
    links = json.loads(_get(env["LEADERBOARD_LINKS_URL"]))
    credentials = json.loads(_get(links["credentials"]))

    def report_progress(event: dict) -> None:
        try:
            _put(links["progress"], json.dumps(event).encode())
        except Exception as error:  # progress is a courtesy; never fail a run over it
            print(f"[runner] could not report progress: {error}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="leaderboard-") as work:
        submission, outputs = Path(work) / "submission", Path(work) / "outputs"
        _unpack(_get(links["submission"]), submission)
        report = run_submission(
            submission,
            datasets,
            outputs,
            env={name: value for name, value in credentials.items() if value},
            on_progress=report_progress,
            **limits,
        )
        _put(links["output"], _pack(outputs, [d.name for d in datasets]))
        # The report also says which image produced it: its code and data versions.
        stamped = {**json.loads(report.to_json()), "build": json.loads(manifest.read_text()).get("build", {})}
        _put(links["report"], json.dumps(stamped).encode())

    print(f"[runner] ok={report.ok} seconds={report.seconds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
