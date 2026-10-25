"""The package root's public API (``__all__``) must list every ``set_*`` /
``register_*`` hook a host wires — a hook that's defined but missing from
``__all__`` is a silent export bug (``from queue_workflows import *`` skips it,
and it reads as private).
"""

from __future__ import annotations

import queue_workflows


def test_all_names_are_resolvable() -> None:
    for name in queue_workflows.__all__:
        assert hasattr(queue_workflows, name), f"{name} in __all__ but not defined"
