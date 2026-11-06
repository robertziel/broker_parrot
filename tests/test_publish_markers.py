"""The private ``local`` branch stays strippable: every local-only block is balanced, and no
private term lives outside one (or outside a ``.publish-ignore`` path).

``scripts/release/strip_local.py`` turns the ``local`` tree into the public tree; this test is
what makes that strip well-defined on every commit, not just on publish day. Pure filesystem —
no database, so it runs on both backends.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TERMS = ROOT / "scripts" / "release" / "private_terms.txt"

_spec = importlib.util.spec_from_file_location("strip_local", ROOT / "scripts" / "release" / "strip_local.py")
strip_local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strip_local)  # type: ignore[union-attr]

START, END = strip_local.START, strip_local.END


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


def _terms() -> list[str]:
    if not _TERMS.exists():
        pytest.skip("no private_terms.txt (public tree)")
    return [t.strip().lower() for t in _TERMS.read_text().splitlines() if t.strip() and not t.startswith("#")]


def test_strip_removes_block_and_keeps_the_rest():
    text = f"a\n{START}\nprivate\n{END}\nb\n"
    assert strip_local.strip_text(text) == "a\nb\n"
    # a block set off by blank lines leaves exactly one blank line behind
    assert strip_local.strip_text(f"a\n\n{START}\nx\n{END}\n\nb\n") == "a\n\nb\n"


@pytest.mark.parametrize("text", [
    f"{START}\nx\n", f"x\n{END}\n", f"{START}\n{START}\n{END}\n{END}\n",
    f"keep {START}\nx\n{END}\n",  # a marker must sit alone on its line
])
def test_strip_refuses_unbalanced_or_nested(text):
    with pytest.raises(strip_local.MarkerError):
        strip_local.strip_text(text)


def test_strip_keeps_crlf_and_handles_eof():
    assert strip_local.strip_text(f"a\r\n{START}\r\nx\r\n{END}\r\nb\r\n") == "a\r\nb\r\n"
    assert strip_local.strip_text(f"a\n{START}\nx\n{END}") == "a\n"


def test_strip_tree_never_writes_through_a_symlink(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text(f"{START}\nx\n{END}\n")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "link.md").symlink_to(outside)
    assert strip_local.strip_tree(tmp_path / "root", files=["link.md"]) == []
    assert START in outside.read_text()


def test_strip_tree_drops_ignored_paths(tmp_path):
    (tmp_path / ".publish-ignore").write_text("notes/*.local.md\n")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.local.md").write_text("secret")
    (tmp_path / "doc.md").write_text(f"pub\n{START}\nsecret\n{END}\n")
    changed = strip_local.strip_tree(tmp_path)
    assert not (tmp_path / "notes" / "a.local.md").exists()
    assert (tmp_path / "doc.md").read_text() == "pub\n"
    assert sorted(changed) == ["removed notes/a.local.md", "stripped doc.md"]


def test_every_tracked_file_has_balanced_markers():
    for rel in _tracked():
        text = (ROOT / rel).read_bytes().decode("utf-8", errors="replace")
        strip_local.check_markers(text, path=rel)


def test_no_private_term_outside_a_local_only_block():
    terms = _terms()
    patterns = strip_local.read_ignore(ROOT)
    hits = []
    for rel in _tracked():
        if strip_local.ignored(rel, patterns):
            continue
        # bytes decoded with replacement: a non-UTF-8 file is scanned too, never skipped
        raw = (ROOT / rel).read_bytes().decode("utf-8", errors="replace")
        text = strip_local.strip_text(raw, path=rel).lower()
        hits += [f"{rel}: {t}" for t in terms if t in text]
    assert hits == [], "private terms outside a local-only block:\n" + "\n".join(hits)
