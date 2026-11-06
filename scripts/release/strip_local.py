#!/usr/bin/env python3
"""Strip the private parts of a working tree before it becomes a public commit.

The private ``local`` branch may carry text that must never reach the public ``main``:
blocks fenced by the local-only markers (an HTML comment pair, so they render as nothing
in Markdown) and whole paths listed in ``.publish-ignore``.  This removes both, in place,
from the tree rooted at ``root``.  It runs on the
public release branch before the commit is made;
``tests/test_publish_markers.py`` keeps ``local`` well-formed so the strip is always
well-defined (balanced markers, each on its own line, no private term outside a block).

Pure stdlib, no engine import: it is a release tool, not part of the package.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

# Built by concatenation so this file itself never contains a literal marker
# (a public tree must not contain one; the test scans for them).
START = "<!-- " + "local-only:start" + " -->"
END = "<!-- " + "local-only:end" + " -->"
# A block is whole lines: the start marker alone on its line ... the end marker alone on its line,
# plus one following blank line (so a block set off by blank lines leaves exactly one behind).
_EOL = r"(?:\r?\n|\Z)"
_BLOCK = re.compile(
    r"^[ \t]*" + re.escape(START) + r"[ \t]*\r?\n.*?^[ \t]*" + re.escape(END) + r"[ \t]*" + _EOL + r"(?:[ \t]*\r?\n)?",
    re.S | re.M,
)
_ALONE = re.compile(r"^[ \t]*(?:" + re.escape(START) + "|" + re.escape(END) + r")[ \t]*\r?$", re.M)


class MarkerError(ValueError):
    """A file whose local-only markers do not pair up — stripping it would be a guess."""


def strip_text(text: str, *, path: str = "<text>") -> str:
    """Remove every local-only block; raise MarkerError on unbalanced or nested markers."""
    check_markers(text, path=path)
    return _BLOCK.sub("", text)


def check_markers(text: str, *, path: str = "<text>") -> None:
    depth = 0
    for m in re.finditer(re.escape(START) + "|" + re.escape(END), text):
        depth += 1 if m.group(0) == START else -1
        if depth not in (0, 1):
            line = text.count("\n", 0, m.start()) + 1
            raise MarkerError(f"{path}:{line}: unbalanced or nested local-only marker")
    if depth:
        raise MarkerError(f"{path}: local-only block opened and never closed")
    for m in re.finditer(re.escape(START) + "|" + re.escape(END), text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if not _ALONE.fullmatch(text[line_start: len(text) if line_end < 0 else line_end]):
            line = text.count("\n", 0, m.start()) + 1
            raise MarkerError(f"{path}:{line}: a local-only marker must sit alone on its line")


def read_ignore(root: Path) -> list[str]:
    f = root / ".publish-ignore"
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def ignored(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def tree_files(root: Path) -> list[str]:
    """Every file of the tree: ``git ls-files`` in a checkout, else a walk that skips ``.git``."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
        return [p for p in out.decode().split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [p.relative_to(root).as_posix() for p in root.rglob("*") if ".git" not in p.parts]


def strip_tree(root: Path, *, files: list[str] | None = None) -> list[str]:
    """Strip ``files`` (repo-relative; default = :func:`tree_files`). Returns what changed.

    Symlinks are never followed or written through; line endings are kept as they are.
    """
    patterns = read_ignore(root)
    rels = files if files is not None else tree_files(root)
    changed: list[str] = []
    for rel in sorted(rels):
        p = root / rel
        if ignored(rel, patterns):
            if p.exists() or p.is_symlink():
                p.unlink()
                changed.append(f"removed {rel}")
            continue
        if p.is_symlink() or not p.is_file():
            continue
        data = p.read_bytes()
        if START.encode() not in data and END.encode() not in data:
            continue
        text = data.decode("utf-8")  # a marked file must be UTF-8; anything else fails loudly
        p.write_bytes(strip_text(text, path=rel).encode("utf-8"))
        changed.append(f"stripped {rel}")
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("files", nargs="*", help="repo-relative files to strip (default: the whole tree)")
    a = ap.parse_args(argv)
    try:
        for line in strip_tree(Path(a.root), files=a.files or None):
            print(line)
    except MarkerError as e:
        print(f"strip_local: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
