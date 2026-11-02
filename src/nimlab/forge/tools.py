"""The coder's tool surface: ``list_dir``, ``read_file``, ``write_file``, ``apply_patch``,
``run_tests``, ``run_python`` and ``search_code``.

:class:`WorkspaceTools` is deliberately forgiving about *how* a model calls it (wrong types, missing
arguments, unknown tool names, JSON passed as a string) and strict about *what* it may do (every
path goes through :class:`~nimlab.forge.sandbox.Sandbox`). ``call`` never raises: failures come
back as ``ToolOutcome(ok=False, output="Error: ...")`` so the model can recover.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .events import EventBus
from .patching import PatchError, apply_patch_text
from .pytest_parser import PytestReport, parse_pytest_output
from .sandbox import (
    IGNORED_DIR_NAMES,
    CommandNotAllowedError,
    Sandbox,
    SandboxError,
    looks_binary,
)

__all__ = ["TOOL_NAMES", "ToolOutcome", "ToolSpec", "WorkspaceTools", "clip"]


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of one tool (also renders its OpenAI schema)."""

    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                },
            },
        }

    def usage(self) -> str:
        parts = []
        for name, prop in self.properties.items():
            opt = "" if name in self.required else "?"
            parts.append(f"{name}{opt}: {prop.get('type', 'string')}")
        return f"{self.name}({', '.join(parts)})"


# list_dir is first on purpose: the offline mock backend calls the *first* tool, and listing a
# directory is the most harmless thing it can do.
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_dir",
        "List files and directories in a workspace directory (relative path).",
        {
            "path": {"type": "string", "description": "Directory, default '.'"},
            "recursive": {"type": "boolean", "description": "Include subdirectories (max depth 3)"},
        },
    ),
    ToolSpec(
        "read_file",
        "Read a text file from the workspace, optionally a line range (1-based, inclusive).",
        {
            "path": {"type": "string", "description": "File path relative to the workspace"},
            "start_line": {"type": "integer", "description": "First line to return"},
            "end_line": {"type": "integer", "description": "Last line to return"},
        },
        ("path",),
    ),
    ToolSpec(
        "write_file",
        "Create or fully overwrite a file with the given content (parent folders are created). "
        "Use apply_patch for small edits to existing files.",
        {
            "path": {"type": "string", "description": "File path relative to the workspace"},
            "content": {"type": "string", "description": "Complete file content"},
        },
        ("path", "content"),
    ),
    ToolSpec(
        "apply_patch",
        "Edit an existing file with SEARCH/REPLACE blocks. Each block is\n"
        "<<<<<<< SEARCH\n<exact lines to find>\n=======\n<replacement lines>\n>>>>>>> REPLACE\n"
        "SEARCH must match exactly one place in the file. Multiple blocks are applied in order; "
        "if any block fails nothing is changed.",
        {
            "path": {"type": "string", "description": "File path relative to the workspace"},
            "patch": {"type": "string", "description": "One or more SEARCH/REPLACE blocks"},
        },
        ("path", "patch"),
    ),
    ToolSpec(
        "run_tests",
        "Run pytest in the workspace and return a condensed report of failures.",
        {
            "path": {"type": "string", "description": "Optional test file or directory"},
            "keyword": {"type": "string", "description": "Optional pytest -k expression"},
            "maxfail": {
                "type": "integer",
                "description": "Stop after this many failures (0 = all)",
            },
        },
    ),
    ToolSpec(
        "run_python",
        "Run a Python script that exists in the workspace and return its output. "
        "Only script files are allowed (no -c, no shell).",
        {
            "script": {"type": "string", "description": "Script path relative to the workspace"},
            "args": {"type": "array", "items": {"type": "string"}, "description": "CLI arguments"},
        },
        ("script",),
    ),
    ToolSpec(
        "search_code",
        "Search files for a regular expression (grep). Returns path:line: text.",
        {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": "string", "description": "Directory or file to search, default '.'"},
            "glob": {"type": "string", "description": "Filename glob, e.g. '*.py'"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
        },
        ("pattern",),
    ),
)
_SPEC_BY_NAME = {s.name: s for s in TOOL_SPECS}
TOOL_NAMES = tuple(_SPEC_BY_NAME)


@dataclass
class ToolOutcome:
    """Result of one tool call."""

    tool: str
    ok: bool
    output: str
    changed: list[str] = field(default_factory=list)


class ToolArgError(ValueError):
    """The model passed an argument we could not interpret."""


def clip(text: str, limit: int) -> str:
    """Shorten ``text`` to ~``limit`` chars keeping the head and the tail."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} chars omitted] ...\n{text[-tail:]}"


def _coerce(name: str, prop: dict[str, Any], value: Any) -> Any:
    kind = prop.get("type", "string")
    if value is None:
        return None
    if kind == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        return str(value)
    if kind == "integer":
        if isinstance(value, bool):
            raise ToolArgError(f"argument '{name}' must be an integer")
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
            return int(value)
        raise ToolArgError(f"argument '{name}' must be an integer, got {value!r}")
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
            return True
        if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", ""}:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ToolArgError(f"argument '{name}' must be a boolean, got {value!r}")
    if kind == "array":
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    loaded = json.loads(text)
                    if isinstance(loaded, list):
                        return [str(v) for v in loaded]
                except json.JSONDecodeError:
                    pass
            return text.split() if text else []
        raise ToolArgError(f"argument '{name}' must be a list of strings")
    return value


def _syntax_note(path: str, content: str) -> str:
    if not path.endswith(".py"):
        return ""
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return f"\nWARNING: {path} has a SyntaxError at line {exc.lineno}: {exc.msg}. Fix it."
    return ""


class WorkspaceTools:
    """Executes the coder's tool calls against a :class:`Sandbox`."""

    def __init__(
        self,
        sandbox: Sandbox,
        bus: EventBus | None = None,
        *,
        max_output_chars: int = 8000,
        phase: str = "tools",
    ) -> None:
        self.sandbox = sandbox
        self.bus = bus
        self.max_output_chars = max_output_chars
        self.phase = phase
        self.last_pytest: PytestReport | None = None
        self.changed_files: set[str] = set()

    # ------------------------------------------------------------------ public API

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        """OpenAI-format tool definitions (``list_dir`` first, see module comment)."""
        return [s.schema() for s in TOOL_SPECS]

    async def call(self, name: str, arguments: Any) -> ToolOutcome:
        """Execute a tool call. Never raises; errors are returned as failed outcomes."""
        spec = _SPEC_BY_NAME.get(name)
        if spec is None:
            hint = ", ".join(TOOL_NAMES)
            return self._finish(
                name, ToolOutcome(name, False, f"Error: unknown tool {name!r}. Available: {hint}")
            )
        self._emit("tool_call", f"{name} {_preview(arguments)}", tool=name)
        try:
            kwargs = self._prepare(spec, arguments)
            handler: Callable[..., Any] = getattr(self, f"_t_{name}")
            outcome = await handler(**kwargs)
        except ToolArgError as exc:
            outcome = ToolOutcome(name, False, f"Error: {exc}. Usage: {spec.usage()}")
        except PatchError as exc:
            outcome = ToolOutcome(name, False, f"Error: patch not applied. {exc}")
        except CommandNotAllowedError as exc:
            outcome = ToolOutcome(name, False, f"Error: not allowed: {exc}")
        except SandboxError as exc:
            outcome = ToolOutcome(name, False, f"Error: {exc}")
        except OSError as exc:
            outcome = ToolOutcome(name, False, f"Error: {type(exc).__name__}: {exc}")
        except Exception as exc:  # tool bugs must never kill the run
            outcome = ToolOutcome(
                name, False, f"Error: internal tool failure: {type(exc).__name__}: {exc}"
            )
        return self._finish(name, outcome)

    async def run_tests(
        self, targets: tuple[str, ...] = (), keyword: str = "", maxfail: int = 0
    ) -> PytestReport:
        """Run pytest and return the parsed report (also remembered as ``last_pytest``)."""
        proc = await self.sandbox.run_pytest(targets, keyword=keyword, maxfail=maxfail)
        report = parse_pytest_output(
            proc.stdout,
            exit_code=proc.returncode,
            stderr=proc.stderr,
            timed_out=proc.timed_out,
            root=str(self.sandbox.root),
        )
        if proc.truncated:
            report.output_tail = "[output truncated]\n" + report.output_tail
        self.last_pytest = report
        return report

    # ------------------------------------------------------------------ plumbing

    def _prepare(self, spec: ToolSpec, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolArgError(f"arguments are not valid JSON ({exc.msg})") from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolArgError("arguments must be an object")
        kwargs: dict[str, Any] = {}
        for pname, prop in spec.properties.items():
            if pname in arguments and arguments[pname] is not None:
                kwargs[pname] = _coerce(pname, prop, arguments[pname])
        missing = [p for p in spec.required if p not in kwargs]
        if missing:
            raise ToolArgError(f"missing required argument(s): {', '.join(missing)}")
        return kwargs

    def _finish(self, name: str, outcome: ToolOutcome) -> ToolOutcome:
        outcome.output = (
            clip(outcome.output, self.max_output_chars) if outcome.output else "(no output)"
        )
        for path in outcome.changed:
            self.changed_files.add(path)
        self._emit(
            "tool_result",
            f"{name}: {'ok' if outcome.ok else 'failed'}",
            tool=name,
            ok=outcome.ok,
            chars=len(outcome.output),
        )
        return outcome

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        if self.bus is not None:
            self.bus.emit(kind, self.phase, message, **data)

    # ------------------------------------------------------------------ handlers

    async def _t_list_dir(self, path: str = ".", recursive: bool = False) -> ToolOutcome:
        base = self.sandbox.resolve(path)
        if not base.exists():
            return ToolOutcome("list_dir", False, f"Error: no such directory: {path!r}")
        if not base.is_dir():
            return ToolOutcome("list_dir", False, f"Error: not a directory: {path!r}")
        rel_base = self.sandbox.relpath(base)
        lines: list[str] = []
        max_entries = 200

        def walk(directory: Any, depth: int) -> None:
            try:
                entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            except OSError:
                return
            for entry in entries:
                if len(lines) >= max_entries:
                    return
                if entry.is_dir() and not entry.is_symlink():
                    if entry.name in IGNORED_DIR_NAMES:
                        continue
                    lines.append("  " * depth + entry.name + "/")
                    if recursive and depth < 2:
                        walk(entry, depth + 1)
                elif entry.is_file():
                    if entry.name.endswith((".pyc", ".pyo")):
                        continue
                    lines.append("  " * depth + f"{entry.name} ({entry.stat().st_size} B)")
                else:
                    lines.append("  " * depth + entry.name + " (link)")

        walk(base, 0)
        header = f"{rel_base}/ : {len(lines)} entr{'y' if len(lines) == 1 else 'ies'}"
        if len(lines) >= max_entries:
            header += " (truncated)"
        return ToolOutcome(
            "list_dir", True, header + ("\n" + "\n".join(lines) if lines else "\n(empty)")
        )

    async def _t_read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> ToolOutcome:
        target = self.sandbox.resolve(path)
        if not target.exists():
            return ToolOutcome("read_file", False, f"Error: file not found: {path!r}")
        if start_line is None and end_line is None:
            text = self.sandbox.read_text(path)
            n = len(text.splitlines())
            return ToolOutcome(
                "read_file", True, f"=== {self.sandbox.relpath(target)} ({n} lines) ===\n{text}"
            )
        text = self.sandbox.read_text(
            path, max_bytes=max(self.sandbox.limits.max_read_bytes, 2_000_000)
        )
        lines = text.splitlines()
        lo = max(1, start_line or 1)
        hi = min(len(lines), end_line if end_line is not None else len(lines))
        if lo > len(lines):
            return ToolOutcome("read_file", False, f"Error: file has only {len(lines)} lines")
        body = "\n".join(lines[lo - 1 : hi])
        return ToolOutcome(
            "read_file",
            True,
            f"=== {self.sandbox.relpath(target)} (lines {lo}-{hi} of {len(lines)}) ===\n{body}",
        )

    async def _t_write_file(self, path: str, content: str) -> ToolOutcome:
        target = self.sandbox.resolve(path, write=True)
        existed = target.is_file()
        old = target.read_text(encoding="utf-8", errors="replace") if existed else None
        if old == content:
            return ToolOutcome(
                "write_file", True, f"{self.sandbox.relpath(target)} unchanged (identical content)"
            )
        self.sandbox.write_text(path, content)
        rel = self.sandbox.relpath(target)
        n = len(content.splitlines())
        verb = "Overwrote" if existed else "Created"
        note = _syntax_note(rel, content)
        return ToolOutcome("write_file", True, f"{verb} {rel} ({n} lines){note}", changed=[rel])

    async def _t_apply_patch(self, path: str, patch: str) -> ToolOutcome:
        target = self.sandbox.resolve(path, write=True)
        if target.exists() and not target.is_file():
            return ToolOutcome("apply_patch", False, f"Error: not a file: {path!r}")
        original = self.sandbox.read_text(path) if target.is_file() else ""
        result = apply_patch_text(original, patch)
        if result.text == original:
            return ToolOutcome(
                "apply_patch", True, "Patch applied but the file content is unchanged."
            )
        self.sandbox.write_text(path, result.text)
        rel = self.sandbox.relpath(target)
        note = _syntax_note(rel, result.text)
        return ToolOutcome(
            "apply_patch",
            True,
            f"Applied {result.applied} edit block(s) to {rel} "
            f"(+{result.added}/-{result.removed} lines){note}",
            changed=[rel],
        )

    async def _t_run_tests(
        self, path: str = "", keyword: str = "", maxfail: int = 0
    ) -> ToolOutcome:
        targets = (path,) if path.strip() else ()
        report = await self.run_tests(targets, keyword.strip(), maxfail)
        return ToolOutcome("run_tests", report.ok, report.condensed())

    async def _t_run_python(self, script: str, args: list[str] | None = None) -> ToolOutcome:
        proc = await self.sandbox.run_python(script, args or [])
        status = (
            f"timed out after {proc.timeout_s:.0f}s"
            if proc.timed_out
            else f"exit code {proc.returncode}"
        )
        body = proc.output or "(no output)"
        if proc.truncated:
            body += "\n[output truncated]"
        return ToolOutcome("run_python", proc.ok, f"{status}\n{body}")

    async def _t_search_code(
        self, pattern: str, path: str = ".", glob: str = "", ignore_case: bool = False
    ) -> ToolOutcome:
        flags = re.IGNORECASE if ignore_case else 0
        note = ""
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)
            note = "(pattern was not a valid regex; searched literally)\n"
        base = self.sandbox.resolve(path)
        if not base.exists():
            return ToolOutcome("search_code", False, f"Error: no such path: {path!r}")
        hits: list[str] = []
        limit = 60
        files_scanned = 0
        for file in self.sandbox.iter_files(path):
            rel = self.sandbox.relpath(file)
            if glob and not (
                fnmatch.fnmatch(PurePosixPath(rel).name, glob) or fnmatch.fnmatch(rel, glob)
            ):
                continue
            try:
                if file.stat().st_size > 1_000_000:
                    continue
                raw = file.read_bytes()
            except OSError:
                continue
            if looks_binary(raw):
                continue
            files_scanned += 1
            for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        if not hits:
            return ToolOutcome("search_code", True, f"{note}No matches in {files_scanned} file(s).")
        suffix = f"\n(stopped at {limit} matches)" if len(hits) >= limit else ""
        return ToolOutcome("search_code", True, note + "\n".join(hits) + suffix)


def _preview(arguments: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(arguments, default=str) if not isinstance(arguments, str) else arguments
    except (TypeError, ValueError):
        text = str(arguments)
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."
