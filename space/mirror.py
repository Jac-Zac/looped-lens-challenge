"""A copy of every submission in a private Hugging Face dataset.

DynamoDB is the system of record; this is a readable backup that outlives the
AWS stack. It is one file, ``submissions.jsonl``, rewritten whole whenever
anything has changed. It holds everything, emails and organizer-only details
included, so the dataset must stay private.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from space.state import State


class Mirror:
    def __init__(self, repo: str, token: str):
        from huggingface_hub import HfApi

        self.repo = repo
        self.api = HfApi(token=token)
        self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        self._last = None

    def sync(self, state: State) -> bool:
        """Upload the current submissions if they differ from the last upload."""
        body = "".join(json.dumps(asdict(s), sort_keys=True) + "\n" for s in state.submissions()).encode()
        digest = hashlib.sha256(body).hexdigest()
        if digest == self._last:
            return False
        self.api.upload_file(
            path_or_fileobj=body,
            path_in_repo="submissions.jsonl",
            repo_id=self.repo,
            repo_type="dataset",
            commit_message="Update submissions",
        )
        self._last = digest
        return True
