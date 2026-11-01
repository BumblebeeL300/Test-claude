"""Explore phase helper: a compact, deterministic summary of a workspace."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .sandbox import Sandbox, SandboxError, looks_binary

__all__ = ["WorkspaceSummary", "summarize_workspace"]

_KEY_FILE_PRIORITY = (
    "readme.md",
    "readme.rst",
    "readme.txt",
    "readme",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "requirements.txt",
    "pytest.ini",
    "tox.ini",
    "conftest.py",
    "makefile",
    "main.py",
    "app.py",
    "__main__.py",
)


@dataclass
class WorkspaceSummary:
    """What Forge knows about the workspace before it plans."""

    root: str
    files: list[str] = field(default_factory=list)
    tree: str = ""
    key_files: dict[str, str] = field(default_factory=dict)
    symbols: dict[str, list[str]] = field(default_factory=dict)
    test_files: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.files

    @property
    def has_tests(self) -> bool:
        return bool(self.test_files)

    def render(self, max_chars: int = 7000) -> str:
        """Markdown-ish text block for prompts."""
        if self.empty:
            return "The workspace is empty. Create everything from scratch."
        parts = [f"Workspace: {len(self.files)} file(s)", "", "File tree:", self.tree]
        if self.test_files:
            parts += ["", "Existing tests: " + ", ".join(self.test_files[:20])]
        else:
            parts += ["", "There are no tests yet."]
        if self.symbols:
            parts += ["", "Python outline:"]
            for path, syms in self.symbols.items():
                parts.append(f"  {path}: " + "; ".join(syms))
        for path, head in self.key_files.items():
            parts += ["", f"--- {path} (head) ---", head]
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[: max_chars - 30].rstrip() + "\n... [summary truncated]"
        return text


def _render_tree(paths: list[str], max_entries: int) -> str:
    lines: list[str] = []
    seen_dirs: set[str] = set()
    for shown, path in enumerate(paths):
        if shown >= max_entries:
            lines.append(f"... ({len(paths) - shown} more files)")
            break
        parts = PurePosixPath(path).parts
        for depth in range(len(parts) - 1):
            d = "/".join(parts[: depth + 1])
            if d not in seen_dirs:
                seen_dirs.add(d)
                lines.append("  " * depth + parts[depth] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
    return "\n".join(lines)


def _outline(source: str, limit: int = 14) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ["(syntax error)"]
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            out.append(f"def {node.name}({args})")
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            out.append(f"class {node.name}" + (f" [{', '.join(methods[:8])}]" if methods else ""))
        if len(out) >= limit:
            out.append("...")
            break
    return out


def _is_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def summarize_workspace(
    sandbox: Sandbox,
    *,
    max_tree_entries: int = 120,
    max_key_files: int = 5,
    max_key_chars: int = 1200,
    max_outline_files: int = 30,
) -> WorkspaceSummary:
    """Walk the workspace and collect a tree, key file heads and a Python outline."""
    paths = sorted(sandbox.relpath(p) for p in sandbox.iter_files())
    summary = WorkspaceSummary(root=str(sandbox.root), files=paths)
    if not paths:
        return summary
    summary.tree = _render_tree(paths, max_tree_entries)
    summary.test_files = [p for p in paths if _is_test_file(p)]

    ranked: list[tuple[int, int, str]] = []
    for path in paths:
        name = PurePosixPath(path).name.lower()
        if name in _KEY_FILE_PRIORITY:
            ranked.append((len(PurePosixPath(path).parts), _KEY_FILE_PRIORITY.index(name), path))
    for _, _, path in sorted(ranked)[:max_key_files]:
        try:
            text = sandbox.read_text(path, max_bytes=60_000)
        except SandboxError:
            continue
        head = "\n".join(text.splitlines()[:40])
        summary.key_files[path] = head[:max_key_chars]

    for path in paths:
        if len(summary.symbols) >= max_outline_files:
            break
        if not path.endswith(".py") or _is_test_file(path):
            continue
        try:
            raw = sandbox.resolve(path).read_bytes()[:200_000]
        except OSError:
            continue
        if looks_binary(raw):
            continue
        outline = _outline(raw.decode("utf-8", errors="replace"))
        if outline:
            summary.symbols[path] = outline
    return summary
