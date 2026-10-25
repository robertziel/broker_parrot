"""``queue-orchestrator`` parses its real command line: ``--help`` answers before
any database is touched (it once parsed ``argv or []`` and so ignored sys.argv)."""
from __future__ import annotations

import sys

import pytest

from queue_workflows import orchestrator


def test_help_exits_before_touching_the_db(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["queue-orchestrator", "--help"])
    with pytest.raises(SystemExit) as exc:
        orchestrator.main()
    assert exc.value.code == 0
    assert "queue-orchestrator" in capsys.readouterr().out
