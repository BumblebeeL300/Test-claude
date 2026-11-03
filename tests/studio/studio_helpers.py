"""Helpers shared by the Studio tests (importable because conftest puts this folder on sys.path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from nimlab.studio.settings import StudioSettings

HOST = "http://127.0.0.1:8765"


def make_settings(tmp_path: Path, **kw: Any) -> StudioSettings:
    values: dict[str, Any] = {"home": tmp_path / "home", "token": None, "sim_delay": 0.0}
    values.update(kw)
    return StudioSettings(**values)


def parse_sse(lines: Any) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        if line.startswith("data:"):
            events.append(json.loads(line[5:]))
    return events


def events_of(
    c: TestClient, run_id: str, headers: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """All events of a run, read through the SSE endpoint (blocks until the terminal event)."""
    with c.stream("GET", f"/api/runs/{run_id}/events", headers=headers or {}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return parse_sse(resp.iter_lines())


def finish(c: TestClient, run_id: str) -> dict[str, Any]:
    """Wait for a run to end and return its detail."""
    events_of(c, run_id)
    detail: dict[str, Any] = c.get(f"/api/runs/{run_id}").json()
    assert detail["status"] not in ("queued", "running")
    return detail
