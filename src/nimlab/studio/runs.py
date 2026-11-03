"""The run manager: every long-running job is a :class:`Run` with a replayable event log.

A run is an asyncio task that calls ``run.emit(type, data)``. Events are kept in order, so an SSE
subscriber that connects late first receives the past events and then the live ones. The terminal
event has type ``run.done`` or ``run.failed`` (a cancelled run ends with ``run.failed`` and
``data.cancelled = true``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["APP_IDS", "TERMINAL_TYPES", "Run", "RunManager", "iso_now", "sse_format"]

_log = logging.getLogger("nimlab.studio")

APP_IDS = ("council", "atlas", "forge", "arena", "prism", "sentinel", "loom", "distill")
TERMINAL_TYPES = ("run.done", "run.failed")
ACTIVE = ("queued", "running")
MAX_EVENTS_PER_RUN = 20000
MAX_PERSISTED_RESULT_BYTES = 120_000
PERSISTED_RESULTS = 20


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so events never hold objects that cannot be serialised."""
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def sse_format(event: dict[str, Any]) -> str:
    """One SSE message: ``id`` = seq, default ``message`` event, JSON in ``data``."""
    return f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


class Run:
    """One job: metadata, an ordered event log and live subscribers."""

    def __init__(
        self,
        app: str,
        title: str,
        *,
        simulated: bool = False,
        run_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.id = run_id or uuid.uuid4().hex[:12]
        self.app = app
        self.title = title
        self.status = "queued"
        self.started_at = iso_now()
        self.finished_at: str | None = None
        self.simulated = simulated
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.meta: dict[str, Any] = meta or {}
        self.events: list[dict[str, Any]] = []
        self.task: asyncio.Task[Any] | None = None
        self._subs: list[asyncio.Queue[dict[str, Any]]] = []
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ events
    @property
    def finished(self) -> bool:
        return self.status not in ACTIVE

    @property
    def elapsed_s(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def emit(
        self, type_: str, data: dict[str, Any] | None = None, phase: str | None = None
    ) -> dict[str, Any]:
        """Append an event and push it to every live subscriber."""
        event: dict[str, Any] = {
            "seq": len(self.events) + 1,
            "ts": iso_now(),
            "run_id": self.id,
            "type": type_,
            "data": _jsonable(data or {}),
        }
        if phase:
            event["phase"] = phase
        if len(self.events) >= MAX_EVENTS_PER_RUN and type_ not in TERMINAL_TYPES:
            return event  # runaway producer: keep the log bounded, never drop the terminal event
        self.events.append(event)
        for queue in list(self._subs):
            queue.put_nowait(event)
        return event

    def subscribe(
        self, after: int = 0
    ) -> tuple[list[dict[str, Any]], asyncio.Queue[dict[str, Any]]]:
        """``(past events after seq, live queue)``. No ``await`` between snapshot and register."""
        past = [e for e in self.events if e["seq"] > after]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subs.append(queue)
        return past, queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with contextlib.suppress(ValueError):
            self._subs.remove(queue)

    async def stream(self, after: int = 0, heartbeat_s: float = 15.0) -> AsyncIterator[str]:
        """SSE text: past events, then live ones, heartbeat comments while idle."""
        past, queue = self.subscribe(after)
        try:
            yield "retry: 3000\n\n"
            last = after
            for event in past:
                last = event["seq"]
                yield sse_format(event)
                if event["type"] in TERMINAL_TYPES:
                    return
            if self.finished and not self._subs_pending(queue):
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=max(heartbeat_s, 0.01))
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if event["seq"] <= last:
                    continue
                last = event["seq"]
                yield sse_format(event)
                if event["type"] in TERMINAL_TYPES:
                    return
        finally:
            self.unsubscribe(queue)

    @staticmethod
    def _subs_pending(queue: asyncio.Queue[dict[str, Any]]) -> bool:
        return not queue.empty()

    # ------------------------------------------------------------------ views
    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "app": self.app,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "simulated": self.simulated,
        }

    def detail(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "result": self.result,
            "error": self.error,
            "event_count": len(self.events),
            "elapsed_s": self.elapsed_s,
            **({"project": self.meta["project"]} if "project" in self.meta else {}),
        }


Work = Callable[[Run], Awaitable[dict[str, Any] | None]]


class RunManager:
    """Creates runs, tracks them and keeps the small JSON history."""

    def __init__(
        self,
        history_path: Path | None = None,
        *,
        max_history: int = 100,
        max_memory_runs: int = 200,
    ) -> None:
        self.history_path = history_path
        self.max_history = max_history
        self.max_memory_runs = max_memory_runs
        self.runs: dict[str, Run] = {}
        self.counts: dict[str, int] = dict.fromkeys(APP_IDS, 0)
        self.last_run_at: dict[str, str | None] = dict.fromkeys(APP_IDS)
        self._restored: list[dict[str, Any]] = []
        self._load_history()

    # ------------------------------------------------------------------ history
    def _load_history(self) -> None:
        if self.history_path is None:
            return
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        counts = data.get("counts")
        if isinstance(counts, dict):
            for app, n in counts.items():
                if isinstance(n, int) and n >= 0:
                    self.counts[app] = n
        last = data.get("last_run_at")
        if isinstance(last, dict):
            for app, ts in last.items():
                if isinstance(ts, str):
                    self.last_run_at[app] = ts
        for item in data.get("runs", []):
            if not isinstance(item, dict) or "id" not in item or "app" not in item:
                continue
            run = Run(
                str(item["app"]),
                str(item.get("title", "")),
                simulated=bool(item.get("simulated")),
                run_id=str(item["id"]),
            )
            status = str(item.get("status", "failed"))
            run.status = "failed" if status in ACTIVE else status
            run.started_at = str(item.get("started_at") or run.started_at)
            run.finished_at = item.get("finished_at") or run.started_at
            run.result = item.get("result") if isinstance(item.get("result"), dict) else None
            run.error = item.get("error") or (
                "interrupted: Studio was closed while this run was active"
                if status in ACTIVE
                else None
            )
            if "project" in item:
                run.meta["project"] = item["project"]
            terminal = "run.done" if run.status == "done" else "run.failed"
            payload: dict[str, Any] = (
                {"result": run.result} if run.result is not None else {"error": run.error}
            )
            payload["restored"] = True
            if run.status == "cancelled":
                payload["cancelled"] = True
            run.emit(terminal, payload)
            self.runs[run.id] = run

    def save_history(self) -> None:
        if self.history_path is None:
            return
        ordered = sorted(self.runs.values(), key=lambda r: r.started_at, reverse=True)[
            : self.max_history
        ]
        rows: list[dict[str, Any]] = []
        for i, run in enumerate(ordered):
            row: dict[str, Any] = {**run.summary(), "error": run.error}
            if "project" in run.meta:
                row["project"] = run.meta["project"]
            if i < PERSISTED_RESULTS and run.result is not None:
                blob = json.dumps(run.result, default=str)
                if len(blob) <= MAX_PERSISTED_RESULT_BYTES:
                    row["result"] = run.result
            rows.append(row)
        payload = {
            "version": 1,
            "counts": self.counts,
            "last_run_at": self.last_run_at,
            "runs": rows,
        }
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".history-", dir=self.history_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, default=str)
                os.replace(tmp, self.history_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except OSError as exc:  # history is a convenience; never fail a run over it
            _log.warning("could not write studio history: %s", exc)

    # ------------------------------------------------------------------ lifecycle
    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def recent(self, limit: int = 50) -> list[Run]:
        ordered = sorted(self.runs.values(), key=lambda r: r.started_at, reverse=True)
        return ordered[:limit]

    def active(self, app: str | None = None) -> list[Run]:
        return [r for r in self.runs.values() if not r.finished and (app is None or r.app == app)]

    def start(
        self,
        app: str,
        title: str,
        work: Work,
        *,
        simulated: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> Run:
        """Register a run and schedule ``work(run)`` on the running event loop."""
        run = Run(app, title, simulated=simulated, meta=meta)
        self.runs[run.id] = run
        self.counts[app] = self.counts.get(app, 0) + 1
        self.last_run_at[app] = run.started_at
        self._trim()
        run.task = asyncio.get_running_loop().create_task(self._execute(run, work))
        run.task.add_done_callback(lambda _task: self._settle(run))
        return run

    def _settle(self, run: Run) -> None:
        """A task cancelled before it ever started never runs ``_execute``: close it out here."""
        if run.finished:
            return
        run.status = "cancelled"
        run.error = "cancelled"
        run.finished_at = iso_now()
        run.emit("run.failed", {"error": "cancelled", "cancelled": True})
        self.save_history()

    def _trim(self) -> None:
        if len(self.runs) <= self.max_memory_runs:
            return
        finished = sorted((r for r in self.runs.values() if r.finished), key=lambda r: r.started_at)
        for run in finished[: len(self.runs) - self.max_memory_runs]:
            del self.runs[run.id]

    async def _execute(self, run: Run, work: Work) -> None:
        run.status = "running"
        run.emit("run.start", {"app": run.app, "title": run.title, "simulated": run.simulated})
        try:
            result = await work(run)
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.error = "cancelled"
            run.finished_at = iso_now()
            run.emit("run.failed", {"error": "cancelled", "cancelled": True})
        except Exception as exc:
            _log.debug("run %s failed", run.id, exc_info=True)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}" if not str(exc) else str(exc)
            run.finished_at = iso_now()
            run.emit("run.failed", {"error": run.error})
        else:
            run.status = "done"
            run.result = _jsonable(result) if result is not None else {}
            run.finished_at = iso_now()
            run.emit("run.done", {"result": run.result})
        finally:
            self.save_history()

    def cancel(self, run_id: str) -> tuple[Run | None, bool]:
        """``(run, requested)``: ``requested`` is False when the run is unknown or already over."""
        run = self.runs.get(run_id)
        if run is None or run.finished or run.task is None:
            return run, False
        run.task.cancel()
        return run, True

    async def shutdown(self) -> None:
        """Cancel everything still running (app exit) and wait for the tasks to settle."""
        tasks = [r.task for r in self.runs.values() if r.task is not None and not r.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.save_history()
