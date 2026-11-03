"""The Foundry builder: run Forge for a prompt and translate its events for the UI.

Studio event types: ``phase.start``, ``phase.end``, ``plan``, ``tool.call``, ``tool.result``,
``file.write``, ``test.run``, ``test.result``, ``review``, ``log``, ``usage``.
"""

from __future__ import annotations

import ast
import asyncio
import os
import time
from pathlib import Path
from typing import Any

from nimlab.core import NIMError
from nimlab.forge import Forge, ForgeConfig, ForgeEvent

from .projects import ProjectError
from .runs import Run
from .simulate import SimForgeClient
from .state import StudioState

__all__ = ["build_project", "start_build"]

PHASE_LABELS = {
    "init": "Snapshot",
    "explore": "Explore",
    "plan": "Plan",
    "implement": "Implement",
    "verify": "Verify",
    "debug": "Debug",
    "escalate": "Root cause",
    "review": "Review",
    "fix": "Fix review findings",
    "report": "Report",
}
_WRITE_TOOLS = ("write_file", "apply_patch")
_SKIP = {".forge", "__pycache__", ".pytest_cache", ".git"}


def _scan(root: Path) -> dict[str, tuple[int, int]]:
    """``{relative path: (mtime_ns, size)}`` of the project's files (Forge's own state excluded)."""
    found: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            found[path.relative_to(root).as_posix()] = (st.st_mtime_ns, st.st_size)
    return found


def _test_names(root: Path) -> list[str]:
    """Best-effort ``test_file.py::test_name`` list from the project's test files."""
    names: list[str] = []
    for rel in sorted(_scan(root)):
        base = rel.rsplit("/", 1)[-1]
        if not (base.startswith("test_") and base.endswith(".py")):
            continue
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                names.append(f"{rel}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                names.extend(
                    f"{rel}::{node.name}::{f.name}"
                    for f in node.body
                    if isinstance(f, ast.FunctionDef) and f.name.startswith("test")
                )
    return names


class _Bridge:
    """Subscribes to Forge's bus and re-emits Studio events on the run."""

    def __init__(self, run: Run, root: Path) -> None:
        self.run = run
        self.root = root
        self.files = _scan(root)
        self.phase_started: dict[str, float] = {}
        self.totals = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
        self.test_names = _test_names(root)

    def __call__(self, ev: ForgeEvent) -> None:
        try:
            self._handle(ev)
        except Exception:  # a UI mapping bug must never abort the build
            self.run.emit(
                "log", {"level": "warning", "message": f"event mapping failed: {ev.kind}"}
            )

    def _handle(self, ev: ForgeEvent) -> None:
        run, d, phase = self.run, ev.data, ev.phase or None
        kind = ev.kind
        if kind == "phase_start":
            self.phase_started[ev.phase] = time.monotonic()
            run.emit(
                "phase.start",
                {"phase": ev.phase, "label": PHASE_LABELS.get(ev.phase, ev.phase)},
                phase,
            )
            if ev.phase == "verify":
                self.test_names = _test_names(self.root)
                run.emit("test.run", {"iteration": None, "total": len(self.test_names)}, phase)
        elif kind == "phase_end":
            began = self.phase_started.pop(ev.phase, None)
            secs = round(time.monotonic() - began, 3) if began is not None else 0.0
            run.emit("phase.end", {"phase": ev.phase, "seconds": secs}, phase)
        elif kind == "plan":
            goal = ev.message.split(": ", 1)[1] if ": " in ev.message else ev.message
            steps = [
                {"id": i, "description": str(s)}
                for i, s in enumerate(d.get("steps") or [], start=1)
            ]
            run.emit("plan", {"goal": goal, "steps": steps}, phase)
        elif kind == "tool_call":
            run.emit("tool.call", {"tool": d.get("tool"), "summary": ev.message}, phase)
        elif kind == "tool_result":
            run.emit(
                "tool.result",
                {"tool": d.get("tool"), "ok": bool(d.get("ok")), "chars": d.get("chars", 0)},
                phase,
            )
            if d.get("tool") in _WRITE_TOOLS and d.get("ok"):
                self._emit_file_writes(phase)
        elif kind == "tests":
            failing = list(d.get("failing") or [])
            names = self.test_names
            results = [
                {
                    "name": n,
                    "status": "failed"
                    if any(f == n or f.startswith(n + "[") for f in failing)
                    else "passed",
                }
                for n in names
            ]
            run.emit(
                "test.result",
                {
                    "passed": d.get("passed", 0),
                    "failed": d.get("failed", 0),
                    "errors": d.get("errors", 0),
                    "ok": bool(d.get("ok")),
                    "iteration": d.get("iteration"),
                    "summary": ev.message,
                    "names": names,
                    "failing": failing,
                    "results": results,
                },
                phase,
            )
        elif kind == "review":
            run.emit(
                "review",
                {
                    "verdict": d.get("verdict"),
                    "issues": d.get("issues", 0),
                    "blocking": d.get("blocking", 0),
                    "message": ev.message,
                },
                phase,
            )
        elif kind == "llm":
            self.totals["calls"] += 1
            self.totals["tokens_in"] += int(d.get("tokens_in") or 0)
            self.totals["tokens_out"] += int(d.get("tokens_out") or 0)
            run.emit(
                "usage",
                {
                    "model": d.get("model"),
                    "tokens_in": d.get("tokens_in", 0),
                    "tokens_out": d.get("tokens_out", 0),
                    "latency_s": d.get("latency_s", 0.0),
                    "totals": dict(self.totals),
                },
                phase,
            )
        elif kind in (
            "info",
            "warning",
            "error",
            "stall",
            "escalate",
            "rollback",
            "run_start",
            "run_end",
        ):
            level = {"warning": "warning", "error": "error", "stall": "warning"}.get(kind, "info")
            run.emit("log", {"level": level, "message": ev.message or kind, "kind": kind}, phase)
        # "state" transitions are noise for the UI; phase.start/end already show progress

    def _emit_file_writes(self, phase: str | None) -> None:
        now = _scan(self.root)
        for rel in sorted(now):
            before = self.files.get(rel)
            if before == now[rel]:
                continue
            size = now[rel][1]
            preview = ""
            lines = 0
            try:
                text = (self.root / rel).read_text(encoding="utf-8")
                lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
                preview = "\n".join(text.splitlines()[:10])[:600]
            except (OSError, UnicodeDecodeError):
                pass
            self.run.emit(
                "file.write",
                {
                    "path": rel,
                    "action": "created" if before is None else "modified",
                    "bytes": size,
                    "lines": lines,
                    "preview": preview,
                },
                phase,
            )
        self.files = now


def start_build(
    state: StudioState, prompt: str, project: str | None, max_iters: int | None
) -> tuple[Run, str]:
    """Register a Forge run; returns ``(run, project_id)``."""
    project_id = state.projects.choose_id(prompt, project)
    name = (project or "").strip() or project_id
    simulated = state.mock
    if any(r.meta.get("project") == project_id for r in state.runs.active("forge")):
        raise ProjectError(f"project {project_id!r} is already being built", 409)
    # Create the folder now so two quick builds with the same prompt get distinct ids.
    state.projects.prepare(project_id, name, prompt)

    async def work(run: Run) -> dict[str, Any]:
        return await build_project(state, run, prompt, project_id, name, max_iters, simulated)

    title = prompt.strip().splitlines()[0][:80]
    run = state.runs.start("forge", title, work, simulated=simulated, meta={"project": project_id})
    return run, project_id


async def build_project(
    state: StudioState,
    run: Run,
    prompt: str,
    project_id: str,
    name: str,
    max_iters: int | None,
    simulated: bool,
) -> dict[str, Any]:
    path = state.projects.prepare(project_id, name, prompt)  # no-op refresh of the metadata
    run.emit(
        "log",
        {
            "level": "info",
            "message": f"Project folder: {path}",
            "project": project_id,
            "simulated": simulated,
        },
    )
    client: Any
    if simulated:
        client = SimForgeClient(prompt, usage=state.usage, delay=state.settings.sim_delay)
    else:
        client = state.client()
    cfg = ForgeConfig(max_iters=max_iters) if max_iters else ForgeConfig()
    bridge = _Bridge(run, path)
    forge = Forge(client, cfg, on_event=bridge)
    try:
        report = await forge.run(prompt, path)
    except NIMError as exc:  # pragma: no cover - Forge converts these into a report
        raise RuntimeError(str(exc)) from exc
    except asyncio.CancelledError:
        run.emit("log", {"level": "warning", "message": "Build cancelled"})
        raise
    data = report.model_dump(mode="json", exclude={"events"})
    diff = data.get("diff") or ""
    if len(diff) > 200_000:
        data["diff"] = diff[:200_000] + "\n... [diff truncated]"
    result = {
        "project": {
            "id": project_id,
            "name": name,
            "path": str(path),
            "files": state.projects.summary(path)["files"],
        },
        "status": report.status,
        "ok": report.status == "success",
        "summary": report.summary,
        "ended_reason": report.ended_reason,
        "iterations": report.iterations,
        "duration_s": round(report.duration_s, 2),
        "tests": data.get("tests"),
        "test_rounds": data.get("test_rounds"),
        "files_changed": data.get("files_changed"),
        "plan": data.get("plan"),
        "review": data.get("review"),
        "models": data.get("models"),
        "usage": data.get("usage"),
        "timings": data.get("timings"),
        "warnings": data.get("warnings"),
        "errors": data.get("errors"),
        "diff": data.get("diff"),
        "simulated": simulated,
    }
    if report.status in ("error",):
        # surface fatal problems (e.g. a rejected key) as a failed run with the reason
        raise RuntimeError(report.summary)
    return result
