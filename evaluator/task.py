"""Entry point of the evaluator task on Fargate.

Fetches what a run wrote, scores it against the private data baked into this
image, and uploads the results envelope. Like the runner task it is handed
single-object URLs rather than an AWS identity; unlike the runner it has open
network access and whatever secrets the competition configured.

    LEADERBOARD_LINKS_URL   GET  a JSON object of the links this task may use:
        output    GET  outputs.tar.gz written by the runner task
        results   PUT  results.json, the envelope
    LEADERBOARD_DATA        directory holding manifest.json (default /data)
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from evaluator import load_manifest, score

MAX_UNPACKED_MB = 2000


def _unpack(archive: bytes, dest: Path) -> None:
    """Extract the runner's archive. It was produced inside the untrusted task,
    so it is treated as hostile: the "data" filter refuses links and paths that
    leave ``dest``, and the total size is capped."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as outputs:
        members = outputs.getmembers()
        if sum(member.size for member in members) > MAX_UNPACKED_MB * 1e6:
            raise ValueError(f"the run's output unpacks to more than {MAX_UNPACKED_MB} MB")
        outputs.extractall(dest, members=members, filter="data")


def main() -> int:
    env = os.environ
    with urllib.request.urlopen(env["LEADERBOARD_LINKS_URL"], timeout=300) as response:
        links = json.loads(response.read())
    with tempfile.TemporaryDirectory(prefix="leaderboard-") as outputs:
        with urllib.request.urlopen(links["output"], timeout=600) as response:
            _unpack(response.read(), Path(outputs))
        results = score(load_manifest(Path(env.get("LEADERBOARD_DATA", "/data")) / "manifest.json", Path(outputs)))

    request = urllib.request.Request(
        links["results"],
        data=json.dumps(results).encode(),
        method="PUT",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300):
        pass
    print(f"[evaluator] ok={results['ok']} metrics={results['metrics']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
