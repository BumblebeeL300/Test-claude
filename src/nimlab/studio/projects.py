"""Forge projects on disk: ``<projects_dir>/<slug>/``. Listing, tree, safe file reads, diff."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_FILE_BYTES",
    "ProjectError",
    "ProjectStore",
    "slugify",
]

MAX_FILE_BYTES = 512 * 1024
MAX_TREE_ENTRIES = 2000
SKIP_DIRS = frozenset(
    {
        ".git",
        ".forge",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
    }
)
SKIP_FILES = frozenset({".DS_Store"})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_LANGUAGES = {
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".txt": "text",
    ".cfg": "ini",
    ".ini": "ini",
    ".rs": "rust",
    ".go": "go",
}


class ProjectError(Exception):
    """A project/file request that must be refused; ``status`` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def slugify(text: str, fallback: str = "project", max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len].strip("-")
    return slug or fallback


def _read_json(path: Path) -> dict[str, Any]:
    """A JSON object from ``path``, or ``{}`` when it is missing or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ resolving
    def path_for(self, project_id: str) -> Path:
        """Directory of an existing project; ``ProjectError`` if the id is bad or unknown."""
        if not isinstance(project_id, str) or not _ID_RE.match(project_id):
            raise ProjectError("invalid project id", 400)
        base = self.root.resolve()
        path = (base / project_id).resolve()
        if path.parent != base or not path.is_dir():
            raise ProjectError(f"unknown project {project_id!r}", 404)
        return path

    def choose_id(self, prompt: str, requested: str | None) -> str:
        """Id for a new build: the requested name (reused if it exists) or a fresh unique one."""
        if requested and requested.strip():
            return slugify(requested)
        base = slugify(prompt)
        candidate, n = base, 2
        while (self.root / candidate).exists():
            candidate = f"{base[:36]}-{n}"
            n += 1
        return candidate

    def prepare(self, project_id: str, name: str, prompt: str) -> Path:
        """Create the directory (if needed) and remember what it was asked to build."""
        path = self.root / project_id
        path.mkdir(parents=True, exist_ok=True)
        meta_dir = path / ".forge"
        meta_dir.mkdir(exist_ok=True)
        meta_file = meta_dir / "studio.json"
        meta = _read_json(meta_file)
        meta.setdefault("created_at", _iso(datetime.now(timezone.utc).timestamp()))
        meta["name"] = meta.get("name") or name
        meta["last_prompt"] = prompt[:2000]
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ listing
    def _walk(self, base: Path) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
            )
            for name in sorted(filenames):
                if name in SKIP_FILES or name.endswith((".pyc", ".pyo")):
                    continue
                out.append(Path(dirpath) / name)
                if len(out) >= MAX_TREE_ENTRIES:
                    return out
        return out

    def report(self, path: Path) -> dict[str, Any] | None:
        data = _read_json(path / ".forge" / "forge_report.json")
        return data or None

    def summary(self, path: Path) -> dict[str, Any]:
        files = self._walk(path)
        mtimes = []
        for f in files:
            try:
                mtimes.append(f.stat().st_mtime)
            except OSError:
                continue
        with contextlib.suppress(OSError):
            mtimes.append(path.stat().st_mtime)
        report = self.report(path)
        meta = _read_json(path / ".forge" / "studio.json")
        return {
            "id": path.name,
            "name": str(meta.get("name") or path.name),
            "path": str(path),
            "files": len(files),
            "updated_at": _iso(max(mtimes)) if mtimes else None,
            "last_status": report.get("status") if report else None,
        }

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        rows = [
            self.summary(p)
            for p in self.root.iterdir()
            if p.is_dir() and _ID_RE.match(p.name) and not p.is_symlink()
        ]
        rows.sort(key=lambda r: r["updated_at"] or "", reverse=True)
        return rows

    # ------------------------------------------------------------------ tree / files
    def tree(self, project_id: str) -> dict[str, Any]:
        base = self.path_for(project_id)
        flat: list[dict[str, Any]] = []
        root: list[dict[str, Any]] = []
        index: dict[str, dict[str, Any]] = {}
        for f in self._walk(base):
            rel = f.relative_to(base)
            parent_children = root
            for i, part in enumerate(rel.parts[:-1]):
                key = "/".join(rel.parts[: i + 1])
                node = index.get(key)
                if node is None:
                    node = {"name": part, "path": key, "type": "dir", "children": []}
                    index[key] = node
                    parent_children.append(node)
                parent_children = node["children"]
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            entry = {"name": rel.name, "path": rel.as_posix(), "type": "file", "size": size}
            parent_children.append(entry)
            flat.append({"path": entry["path"], "size": size})
        return {"id": project_id, "path": str(base), "tree": root, "files": flat}

    def read_file(self, project_id: str, rel: str) -> dict[str, Any]:
        base = self.path_for(project_id)
        if not rel or "\x00" in rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel):
            raise ProjectError("invalid path", 400)
        target = (base / rel).resolve()
        if target != base and base not in target.parents:
            raise ProjectError("path escapes the project", 400)
        parts = target.relative_to(base).parts
        if any(p in SKIP_DIRS for p in parts[:-1]) or (parts and parts[0] == ".forge"):
            raise ProjectError("that folder is not browsable", 403)
        if not target.is_file():
            raise ProjectError("no such file", 404)
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ProjectError(
                f"file is too large to preview ({size} bytes; limit {MAX_FILE_BYTES})", 413
            )
        raw = target.read_bytes()
        if b"\x00" in raw:
            raise ProjectError("binary files cannot be previewed", 415)
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ProjectError("only UTF-8 text files can be previewed", 415) from None
        return {
            "path": target.relative_to(base).as_posix(),
            "size": size,
            "lines": content.count("\n") + (0 if content.endswith("\n") or not content else 1),
            "language": _LANGUAGES.get(target.suffix.lower(), "text"),
            "content": content,
        }

    # ------------------------------------------------------------------ diff / reveal
    def diff(self, project_id: str) -> dict[str, Any]:
        base = self.path_for(project_id)
        report = self.report(base)
        if report is None:
            return {
                "project": project_id,
                "has_report": False,
                "status": None,
                "diff": "",
                "files": [],
            }
        return {
            "project": project_id,
            "has_report": True,
            "status": report.get("status"),
            "summary": report.get("summary", ""),
            "diff": str(report.get("diff") or ""),
            "files": report.get("files_changed") or [],
            "started_at": report.get("started_at"),
        }

    def reveal(self, project_id: str) -> dict[str, Any]:
        """Show the folder in Finder (macOS ``open -R``); a no-op returning the path elsewhere."""
        path = self.path_for(project_id)
        revealed = False
        if sys.platform == "darwin":
            try:
                subprocess.run(["open", "-R", str(path)], check=False, timeout=5)
                revealed = True
            except (OSError, subprocess.SubprocessError):
                revealed = False
        return {"ok": True, "path": str(path), "revealed": revealed}
