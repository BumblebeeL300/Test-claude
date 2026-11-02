"""Snapshots: unified diffs against the initial state and exact rollback."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nimlab.forge import Snapshot


def put(root: Path, name: str, text: str | bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def tree(root: Path) -> dict[str, bytes]:
    """All files (not .forge/.git) with content, for exact before/after comparison."""
    out: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".forge", ".git"}]
        for name in filenames:
            full = Path(dirpath, name)
            out[full.relative_to(root).as_posix()] = full.read_bytes()
    return out


@pytest.fixture
def populated(ws: Path) -> Path:
    put(ws, "README.md", "# Project\n")
    put(ws, "app.py", "def main():\n    return 1\n")
    put(ws, "pkg/__init__.py", "")
    put(ws, "pkg/util.py", "def helper():\n    return 'h'\n")
    (ws / "empty_dir").mkdir()
    return ws


class TestDiff:
    def test_unchanged_workspace_has_empty_diff(self, populated: Path) -> None:
        snap = Snapshot.take(populated)
        diff = snap.diff()
        assert diff.empty and diff.unified == ""

    def test_added_modified_deleted(self, populated: Path) -> None:
        snap = Snapshot.take(populated)
        put(populated, "new.py", "x = 1\n")
        put(populated, "app.py", "def main():\n    return 2\n\n\ndef other():\n    pass\n")
        (populated / "README.md").unlink()
        diff = snap.diff()
        status = {c.path: c.status for c in diff.changes}
        assert status == {"new.py": "added", "app.py": "modified", "README.md": "deleted"}
        stats = {c.path: (c.additions, c.deletions) for c in diff.changes}
        assert stats["new.py"] == (1, 0)
        assert stats["README.md"] == (0, 1)
        assert stats["app.py"][0] >= 4 and stats["app.py"][1] == 1
        assert diff.additions == sum(a for a, _ in stats.values())

    def test_unified_format_headers(self, populated: Path) -> None:
        snap = Snapshot.take(populated)
        put(populated, "new.py", "x = 1\n")
        put(populated, "app.py", "def main():\n    return 2\n")
        (populated / "README.md").unlink()
        text = snap.diff().unified
        assert "diff --git a/app.py b/app.py" in text
        assert "--- a/app.py\n+++ b/app.py" in text
        assert "--- /dev/null\n+++ b/new.py" in text
        assert "--- a/README.md\n+++ /dev/null" in text
        assert "-    return 1\n+    return 2" in text
        assert "@@" in text

    def test_ignored_paths_do_not_show_up(self, populated: Path) -> None:
        snap = Snapshot.take(populated)
        put(populated, "__pycache__/x.cpython-313.pyc", b"\x00bytes")
        put(populated, ".forge/forge_report.json", "{}")
        put(populated, ".pytest_cache/v/cache", "x")
        put(populated, ".git/HEAD", "ref")
        put(populated, "mod.pyc", b"\x00")
        assert snap.diff().empty

    def test_missing_final_newline_is_marked(self, ws: Path) -> None:
        put(ws, "a.txt", "one\ntwo")
        snap = Snapshot.take(ws)
        put(ws, "a.txt", "one\ntwo\nthree")
        assert "\\ No newline at end of file" in snap.diff().unified

    def test_binary_changes_are_listed_without_content(self, ws: Path) -> None:
        put(ws, "img.bin", b"\x00\x01\x02")
        snap = Snapshot.take(ws)
        put(ws, "img.bin", b"\x00\x01\x03\x04")
        put(ws, "blob2.bin", b"\x00abc")
        diff = snap.diff()
        assert {c.path for c in diff.changes if c.binary} == {"img.bin", "blob2.bin"}
        assert "\x01" not in diff.unified

    def test_diff_is_truncated_when_huge(self, ws: Path) -> None:
        snap = Snapshot.take(ws)
        put(ws, "big.txt", "line\n" * 5000)
        diff = snap.diff(max_diff_chars=1000)
        assert len(diff.unified) < 1100 and "[diff truncated]" in diff.unified
        assert diff.changes[0].additions == 5000  # stats stay exact

    @pytest.mark.skipif(shutil.which("patch") is None, reason="patch(1) not installed")
    def test_diff_applies_with_patch(self, populated: Path, tmp_path: Path) -> None:
        original = tmp_path / "original"
        shutil.copytree(populated, original)
        snap = Snapshot.take(populated)
        put(populated, "app.py", "def main():\n    return 42\n")
        put(populated, "pkg/new.py", "y = 2\n")
        (populated / "pkg" / "util.py").unlink()
        patch_file = tmp_path / "changes.patch"
        patch_file.write_text(snap.diff().unified)
        result = subprocess.run(
            ["patch", "-p1", "-i", str(patch_file)], cwd=original, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert tree(original) == tree(populated)


class TestRestore:
    def test_restores_modified_added_and_deleted(self, populated: Path) -> None:
        before = tree(populated)
        snap = Snapshot.take(populated)
        put(populated, "app.py", "BROKEN")
        put(populated, "brand/new/file.py", "x")
        (populated / "pkg" / "util.py").unlink()
        shutil.rmtree(populated / "pkg" / "__pycache__", ignore_errors=True)
        result = snap.restore()
        assert tree(populated) == before
        assert sorted(result.restored) == ["app.py", "pkg/util.py"]
        assert result.removed == ["brand/new/file.py"]
        assert result.unrestorable == []
        assert snap.diff().empty

    def test_directories_created_by_the_run_are_removed(self, populated: Path) -> None:
        snap = Snapshot.take(populated)
        put(populated, "a/b/c/deep.py", "x")
        snap.restore()
        assert not (populated / "a").exists()
        assert (populated / "empty_dir").is_dir()  # pre-existing empty dir stays

    def test_forge_dir_survives_restore(self, populated: Path) -> None:
        snap = Snapshot.take(populated)
        put(populated, ".forge/forge_report.json", "{}")
        put(populated, "x.py", "1")
        snap.restore()
        assert (populated / ".forge" / "forge_report.json").exists()
        assert not (populated / "x.py").exists()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
    def test_file_mode_is_restored(self, populated: Path) -> None:
        script = put(populated, "run.sh", "#!/bin/sh\n")
        script.chmod(0o755)
        snap = Snapshot.take(populated)
        script.unlink()
        put(populated, "run.sh", "changed")
        snap.restore()
        assert (populated / "run.sh").read_text() == "#!/bin/sh\n"
        assert os.stat(populated / "run.sh").st_mode & 0o777 == 0o755

    def test_binary_files_roundtrip(self, ws: Path) -> None:
        payload = bytes(range(256))
        put(ws, "data.bin", payload)
        snap = Snapshot.take(ws)
        put(ws, "data.bin", b"\x00corrupted")
        snap.restore()
        assert (ws / "data.bin").read_bytes() == payload

    def test_oversized_files_cannot_be_restored_and_are_reported(self, ws: Path) -> None:
        put(ws, "huge.dat", "z" * 500)
        put(ws, "small.txt", "keep")
        snap = Snapshot.take(ws, max_file_bytes=100)
        put(ws, "huge.dat", "changed")
        put(ws, "small.txt", "changed")
        result = snap.restore()
        assert result.unrestorable == ["huge.dat"]
        assert (ws / "small.txt").read_text() == "keep"

    @pytest.mark.skipif(os.name != "posix", reason="symlinks")
    def test_symlinks_are_tracked_not_followed(self, ws: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")
        put(ws, "real.txt", "real")
        (ws / "link.txt").symlink_to("real.txt")
        snap = Snapshot.take(ws)
        (ws / "escape.txt").symlink_to(outside)  # added by the run
        (ws / "link.txt").unlink()
        put(ws, "link.txt", "now a file")
        snap.restore()
        assert (ws / "link.txt").is_symlink() and os.readlink(ws / "link.txt") == "real.txt"
        assert not (ws / "escape.txt").exists() and not (ws / "escape.txt").is_symlink()
        assert outside.read_text() == "outside"  # the target was never touched

    def test_restore_is_idempotent(self, populated: Path) -> None:
        before = tree(populated)
        snap = Snapshot.take(populated)
        put(populated, "app.py", "x")
        snap.restore()
        again = snap.restore()
        assert again.total == 0
        assert tree(populated) == before
