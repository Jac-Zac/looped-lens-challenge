"""The organizers' commands that change what is stored, against moto's DynamoDB
and S3. Each is run the way `python manage.py <command>` runs it, with what the
organizer would type at its question."""

from argparse import Namespace

import pytest

import manage
from test_space import competition, state  # noqa: F401  (fixtures)


@pytest.fixture
def operate(monkeypatch, capsys, competition, state):
    """Run one command on the test state. Returns its exit code and what it printed."""
    monkeypatch.setattr(manage, "deployed_state", lambda competition: (state, {}))

    def operate(command, typed: str = "", **args) -> tuple[int, str]:
        monkeypatch.setattr("builtins.input", lambda question: typed)
        try:
            code, said = command(competition, Namespace(**args)), ""
        except SystemExit as stop:
            code, said = 1, str(stop.code)
        return code, capsys.readouterr().out + said

    return operate


def test_new_phase_asks_first_and_then_starts_one(operate, state):
    code, said = operate(manage.new_phase, typed="yes")
    assert code == 1 and "Nothing was changed" in said and state.phase_starts() == []

    code, said = operate(manage.new_phase, typed="2")
    assert code == 0 and "Phase 2 has begun" in said and len(state.phase_starts()) == 1
