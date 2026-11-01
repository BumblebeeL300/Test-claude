"""Workspace snapshots: take, diff (difflib) and restore.

A :class:`Snapshot` records the content of every non-ignored file in the workspace. It is used for
three things: the unified diff in the final report, "files changed" statistics, and
``--rollback-on-fail``. ``.forge/``, ``.git/``, caches and virtualenvs are never part of a snapshot,
so reports survive a rollback.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .sandbox import (
    IGNORED_DIR_NAMES,
    IGNORED_FILE_NAMES,
    IGNORED_FILE_SUFFIXES,
    looks_binary,
)

__all__ = ["FileChange", "FileState", "RestoreResult", "Snapshot", "WorkspaceDiff"]

ChangeStatus = Literal["added", "modified", "deleted"]


class FileChange(BaseModel):
    """Per-file diff statistics."""

    path: str
    status: ChangeStatus
    additions: int = 0
    deletions: int = 0
    binary: bool = False


class WorkspaceDiff(BaseModel):
    """Everything that differs between a snapshot and the current workspace."""

    changes: list[FileChange] = []
    unified: str = ""

    @property
    def empty(self) -> bool:
        return not self.changes

    @property
    def additions(self) -> int:
        return sum(c.additions for c in self.changes)

    @property
    def deletions(self) -> int:
        return sum(c.deletions for c in self.changes)


@dataclass
class FileState:
    """Recorded state of one file (``data`` is None when it was too large to keep)."""

    kind: Literal["file", "symlink"]
    sha256: str
    size: int
    mode: int = 0o644
    data: bytes | None = None
    link: str | None = None

    @property
    def restorable(self) -> bool:
        return self.kind == "symlink" or self.data is not None


@dataclass
class RestoreResult:
    """What :meth:`Snapshot.restore` did."""

    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unrestorable: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.restored) + len(self.removed)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scan(
    root: Path, *, max_file_bytes: int, max_total_bytes: int
) -> tuple[dict[str, FileState], set[str]]:
    files: dict[str, FileState] = {}
    dirs: set[str] = set()
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        kept: list[str] = []
        for d in sorted(dirnames):
            if d in IGNORED_DIR_NAMES:
                continue
            full = base / d
            if full.is_symlink():
                # Directory symlinks are tracked as links, never followed.
                rel = full.relative_to(root).as_posix()
                files[rel] = FileState(
                    "symlink", _sha(os.readlink(full).encode()), 0, link=os.readlink(full)
                )
                continue
            kept.append(d)
            dirs.add(full.relative_to(root).as_posix())
        dirnames[:] = kept
        for name in sorted(filenames):
            if name in IGNORED_FILE_NAMES or name.endswith(IGNORED_FILE_SUFFIXES):
                continue
            full = base / name
            rel = full.relative_to(root).as_posix()
            try:
                if full.is_symlink():
                    target = os.readlink(full)
                    files[rel] = FileState("symlink", _sha(target.encode()), 0, link=target)
                    continue
                st = full.stat()
                if not stat.S_ISREG(st.st_mode):
                    continue
                data: bytes | None = None
                if st.st_size <= max_file_bytes and total + st.st_size <= max_total_bytes:
                    data = full.read_bytes()
                    total += len(data)
                    digest = _sha(data)
                else:
                    h = hashlib.sha256()
                    with full.open("rb") as fh:
                        for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                            h.update(chunk)
                    digest = h.hexdigest()
                files[rel] = FileState("file", digest, st.st_size, stat.S_IMODE(st.st_mode), data)
            except OSError:
                continue
    return files, dirs


def _text_lines(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n\\ No newline at end of file\n"
    return lines


class Snapshot:
    """In-memory copy of a workspace at one point in time."""

    def __init__(
        self,
        root: Path,
        files: dict[str, FileState],
        dirs: set[str],
        limits: tuple[int, int],
    ) -> None:
        self.root = root
        self.files = files
        self.dirs = dirs
        self.created_at = time.time()
        self._limits = limits

    @classmethod
    def take(
        cls,
        root: str | os.PathLike[str],
        *,
        max_file_bytes: int = 2_000_000,
        max_total_bytes: int = 64 * 1024 * 1024,
    ) -> Snapshot:
        resolved = Path(root).resolve()
        files, dirs = _scan(
            resolved, max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes
        )
        return cls(resolved, files, dirs, (max_file_bytes, max_total_bytes))

    def _current(self) -> dict[str, FileState]:
        return _scan(self.root, max_file_bytes=self._limits[0], max_total_bytes=self._limits[1])[0]

    def diff(self, max_diff_chars: int = 400_000) -> WorkspaceDiff:
        """Compare the workspace against this snapshot (text files get a unified diff)."""
        now = self._current()
        changes: list[FileChange] = []
        chunks: list[str] = []
        for path in sorted(set(self.files) | set(now)):
            old, new = self.files.get(path), now.get(path)
            if old and new and old.sha256 == new.sha256 and old.kind == new.kind:
                continue
            status: ChangeStatus = (
                "added" if old is None else "deleted" if new is None else "modified"
            )
            old_bytes = old.data if old and old.kind == "file" else b""
            new_bytes = new.data if new and new.kind == "file" else b""
            binary = (
                old is not None
                and old.kind == "file"
                and (old.data is None or looks_binary(old.data))
            ) or (
                new is not None
                and new.kind == "file"
                and (new.data is None or looks_binary(new.data))
            )
            if binary or old_bytes is None or new_bytes is None:
                changes.append(FileChange(path=path, status=status, binary=True))
                chunks.append(f"Binary or oversized file {status}: {path}\n")
                continue
            a_lines = _text_lines(old_bytes) if old and old.kind == "file" else []
            b_lines = _text_lines(new_bytes) if new and new.kind == "file" else []
            if old and old.kind == "symlink":
                a_lines = [f"-> {old.link}\n"]
            if new and new.kind == "symlink":
                b_lines = [f"-> {new.link}\n"]
            fromfile = "/dev/null" if status == "added" else f"a/{path}"
            tofile = "/dev/null" if status == "deleted" else f"b/{path}"
            body = list(difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile))
            added = sum(1 for ln in body if ln.startswith("+") and not ln.startswith("+++"))
            removed = sum(1 for ln in body if ln.startswith("-") and not ln.startswith("---"))
            changes.append(FileChange(path=path, status=status, additions=added, deletions=removed))
            if body:
                chunks.append(f"diff --git a/{path} b/{path}\n" + "".join(body))
        unified = "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
        if len(unified) > max_diff_chars:
            unified = unified[:max_diff_chars] + "\n... [diff truncated]\n"
        return WorkspaceDiff(changes=changes, unified=unified)

    def restore(self) -> RestoreResult:
        """Put the workspace back exactly as it was when the snapshot was taken."""
        result = RestoreResult()
        now = self._current()
        for path in sorted(set(now) - set(self.files)):
            target = self.root / path
            try:
                target.unlink()
                result.removed.append(path)
            except OSError:
                result.unrestorable.append(path)
        for path, old in sorted(self.files.items()):
            cur = now.get(path)
            if cur and cur.sha256 == old.sha256 and cur.kind == old.kind:
                continue
            target = self.root / path
            if not old.restorable:
                result.unrestorable.append(path)
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if cur is not None or target.is_symlink():
                    target.unlink()
                if old.kind == "symlink" and old.link is not None:
                    os.symlink(old.link, target)
                else:
                    assert old.data is not None
                    target.write_bytes(old.data)
                    os.chmod(target, old.mode)
                result.restored.append(path)
            except OSError:
                result.unrestorable.append(path)
        # Remove directories that did not exist before and are now empty (deepest first).
        for dirpath, _dirnames, _filenames in os.walk(self.root, topdown=False):
            rel = Path(dirpath).relative_to(self.root).as_posix()
            parts = Path(rel).parts
            if rel == "." or any(p in IGNORED_DIR_NAMES for p in parts) or rel in self.dirs:
                continue
            try:
                os.rmdir(dirpath)
            except OSError:
                continue
        return result
