"""The competition definition: the schema of ``competition.yaml`` and how its
data is fetched.

Everything else in the repository reads the competition through this module:

- ``load()`` parses and validates a configuration file: ``competition.yaml``, which
  is what a participant's tools read, or ``competition.prod.yaml``, the organizers'
  full configuration, which ``manage.py`` reads and which is never committed.
- ``prepare()`` downloads a list of datasets into a directory and writes the
  ``manifest.json`` that the runner and the evaluator read. ``submit.py --local``
  calls it for the ``dev`` datasets; ``manage.py build`` calls it for the
  ``eval`` datasets when it bakes the two images.

The framework never opens a dataset's files. A dataset is two directories of
bytes (``inputs`` for the method, ``private`` for the evaluator) and a few
environment variables; what the files mean is between the method and the
evaluator.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "competition.yaml"  # what a participant's tools read; in the repository
PROD_FILE = ROOT / "competition.prod.yaml"  # the organizers' full configuration; never committed
MANIFEST = "manifest.json"

# A name becomes a directory, an environment variable value and part of AWS
# resource names, so it is restricted to characters that are safe in all three.
_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class _Section(BaseModel):
    # A misspelt key in competition.yaml is an error, not a silently ignored setting.
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------- data --


class Source(_Section):
    """Where one directory of files comes from: a Hugging Face dataset repo
    (optionally one subfolder of it, at one revision) or a directory in this
    repository."""

    repo: str | None = None
    path: str = ""
    revision: str = "main"
    dir: str | None = None

    @model_validator(mode="after")
    def _exactly_one_origin(self) -> "Source":
        if (self.repo is None) == (self.dir is None):
            raise ValueError("a source needs exactly one of `repo` or `dir`")
        return self


class Dataset(_Section):
    """One unit a submission is run on. The method is started once per dataset."""

    name: str = Field(pattern=_NAME)
    # What participants are shown instead of `name`, when the name itself would
    # give something away.
    codename: str | None = None
    inputs: Source  # baked into the runner image; the method reads it
    # Baked into the evaluator image only. None when scoring needs nothing beyond the inputs.
    private: Source | None = None
    env: dict[str, str] = {}  # extra variables for the method, e.g. MODEL_ID

    @property
    def label(self) -> str:
        return self.codename or self.name


class Datasets(_Section):
    dev: list[Dataset] = []  # public, private part included; used by local runs
    eval: list[Dataset] = []  # hidden; every submission is run on all of them

    @model_validator(mode="after")
    def _unique_names(self) -> "Datasets":
        # Unique within each list. The same dataset may be in both: a competition
        # whose items are public scores on the data people develop against.
        for names in ([d.name for d in self.dev], [d.name for d in self.eval]):
            if duplicates := sorted({n for n in names if names.count(n) > 1}):
                raise ValueError(f"dataset names must be unique: {duplicates}")
        return self


# --------------------------------------------------------------- components --


class Ndif(_Section):
    host: str = "https://api.ndif.us"
    # A key must carry at least one of these tags to submit. Empty accepts any
    # key NDIF recognises.
    required_tags: list[str] = []


class Egress(_Section):
    """The only hosts the runner task can reach. Matching is by suffix."""

    allow: list[str] = ["api.ndif.us"]  # tunnelled untouched
    read_only: list[str] = ["huggingface.co", "hf.co"]  # GET and HEAD only


class Runner(_Section):
    """The untrusted task that executes a submission."""

    timeout_per_dataset: int = 1800  # seconds
    timeout_total: int = 4 * 3600
    max_output_mb: int = 100  # per dataset
    # Return the tail of a failed run's log to the participant. Leave off when
    # the eval inputs are secret: a traceback can quote them.
    reveal_errors: bool = False
    cpu: int = 1024  # Fargate CPU units
    memory: int = 4096  # MiB
    egress: Egress = Egress()


class Evaluator(_Section):
    """The trusted task that scores what a run wrote."""

    timeout: int = 3600
    cpu: int = 1024
    memory: int = 4096
    # AWS Secrets Manager secret names, each injected as an environment variable
    # of the same name (API keys for an LLM judge, for example).
    secrets: list[str] = []


class SubmissionField(_Section):
    """An optional category every submission must declare (a track, a class of
    method), shown as a label and a filter. A competition has none by default."""

    name: str = Field(pattern=_NAME)
    choices: list[str] = Field(min_length=1)
    help: str = ""


class RateLimit(_Section):
    max: int = 0  # submissions per window per team; 0 is unlimited
    window_hours: float = 24
    exempt: list[str] = []  # team names


class Submission(_Section):
    max_size_mb: int = 200
    fields: list[SubmissionField] = []
    rate_limit: RateLimit = RateLimit()
    # How many submissions run at once across all teams. The rest queue in order.
    max_concurrent: int = 4


class Leaderboard(_Section):
    # Metric names from the evaluator's results. The first ranks; the rest break ties.
    rank_by: list[str] = Field(min_length=1)
    lower_is_better: list[str] = []
    columns: list[str] = []  # shown in this order; defaults to rank_by
    # Teams shown as reference rows and left out of the ranking.
    baselines: list[str] = []
    intro: str = ""  # the line under the title
    accent: str = "#b771cf"  # the colour a link's underline takes under the pointer
    links: dict[str, str] = {}  # shown in the page's header, label to address

    @model_validator(mode="after")
    def _default_columns(self) -> "Leaderboard":
        if not self.columns:
            self.columns = list(self.rank_by)
        return self


class Space(_Section):
    repo: str  # the Hugging Face Space, "org/name"
    url: str | None = None  # defaults to the Space's hf.space address
    # A private Space is reachable only by members of its organization: the
    # way to try a competition out before opening it.
    private: bool = False
    # What the Space runs on, by Hugging Face's name for it. Free hardware
    # ("cpu-basic") sleeps when idle and is not offered to every organization.
    hardware: str = "cpu-basic"
    # A private Hugging Face dataset that receives a copy of every submission.
    mirror: str | None = None

    @model_validator(mode="after")
    def _default_url(self) -> "Space":
        if self.url is None:
            self.url = "https://" + self.repo.lower().replace("/", "-").replace("_", "-") + ".hf.space"
        return self


class Aws(_Section):
    region: str = "us-east-1"


class Competition(_Section):
    name: str = Field(pattern=_NAME)  # prefixes every AWS resource
    title: str
    ndif: Ndif = Ndif()
    datasets: Datasets
    runner: Runner = Runner()
    evaluator: Evaluator = Evaluator()
    submission: Submission = Submission()
    leaderboard: Leaderboard
    space: Space
    aws: Aws = Aws()


def load(path: Path = CONFIG_FILE) -> Competition:
    return parse(Path(path).read_text())


def parse(text: str) -> Competition:
    """A configuration from its YAML text. The Space is handed the organizers' one this way."""
    return Competition.model_validate(yaml.safe_load(text))


# ----------------------------------------------------------------- fetching --


def fetch(source: Source, dest: Path, token: str | None = None) -> str:
    """Put the source's files in ``dest`` (replacing what was there) and return
    the version that was fetched: the commit hash for a Hugging Face repo,
    ``"local"`` for a directory."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source.dir is not None:
        origin = (ROOT / source.dir).resolve()
        if not origin.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {origin}")
        shutil.copytree(origin, dest)
        return "local"

    from huggingface_hub import snapshot_download

    subfolder = source.path.strip("/")
    snapshot = Path(
        snapshot_download(
            source.repo,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=[f"{subfolder}/**"] if subfolder else None,
            token=token,
        )
    )
    origin = snapshot / subfolder
    if not origin.is_dir():
        raise FileNotFoundError(f"{source.repo}@{source.revision} has no directory {subfolder!r}")
    # The snapshot is a tree of symlinks into the Hub cache; copy the real files
    # so the result can be used as a Docker build context.
    shutil.copytree(origin, dest, symlinks=False)
    return snapshot.name  # snapshots are stored under their commit hash


def prepare(
    competition: Competition,
    which: Literal["dev", "eval"],
    dest: Path,
    *,
    private: bool,
    token: str | None = None,
    build: dict | None = None,
) -> Path:
    """Fetch the ``dev`` or ``eval`` datasets into ``dest`` and write the
    manifest describing them. Returns the manifest's path.

    ``private=False`` leaves the private directories out entirely; that is how
    the runner image is built, so labels are never in the same image as a
    submission.

    The manifest's ``build`` block records the exact version of every directory
    that was fetched, plus whatever the caller passes in ``build`` (the git
    commit and lockfile, for an image). The runner copies it into each report,
    so every submission says which data and code it ran against.

    Layout::

        dest/manifest.json
        dest/<dataset>/inputs/...
        dest/<dataset>/private/...     (only with private=True)
    """
    dest.mkdir(parents=True, exist_ok=True)
    entries, revisions = [], {}
    for dataset in getattr(competition.datasets, which):
        entry = {"name": dataset.name, "inputs": f"{dataset.name}/inputs", "env": dataset.env}
        revisions[dataset.name] = {"inputs": fetch(dataset.inputs, dest / dataset.name / "inputs", token)}
        if private and dataset.private is not None:
            entry["private"] = f"{dataset.name}/private"
            revisions[dataset.name]["private"] = fetch(dataset.private, dest / dataset.name / "private", token)
        entries.append(entry)

    manifest = {
        "datasets": entries,
        "runner": competition.runner.model_dump(include={"timeout_per_dataset", "timeout_total", "max_output_mb", "reveal_errors"}),
        "build": {**(build or {}), "revisions": revisions},
    }
    path = dest / MANIFEST
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
