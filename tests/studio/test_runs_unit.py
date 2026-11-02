"""Run manager and SSE stream behaviour without the HTTP layer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nimlab.studio.runs import Run, RunManager


async def collect(run: Run, **kw: Any) -> list[str]:
    return [chunk async for chunk in run.stream(**kw)]


def data_events(chunks: list[str]) -> list[dict[str, Any]]:
    return [json.loads(c.split("data: ", 1)[1]) for c in chunks if "data: " in c]


async def test_heartbeat_and_live_events() -> None:
    mgr = RunManager(None)
    release = asyncio.Event()

    async def work(run: Run) -> dict[str, Any]:
        run.emit("step", {"n": 1})
        await release.wait()
        run.emit("step", {"n": 2}, phase="late")
        return {"ok": True}

    run = mgr.start("council", "t", work)
    task = asyncio.create_task(collect(run, heartbeat_s=0.02))
    await asyncio.sleep(0.12)  # idle long enough for several heartbeats
    release.set()
    chunks = await asyncio.wait_for(task, 2)
    assert chunks[0] == "retry: 3000\n\n"
    assert chunks.count(": heartbeat\n\n") >= 2
    events = data_events(chunks)
    assert [e["type"] for e in events] == ["run.start", "step", "step", "run.done"]
    assert events[2]["phase"] == "late" and "phase" not in events[1]
    assert chunks[1].startswith("id: 1\n")


async def test_late_subscriber_gets_replay_then_live() -> None:
    mgr = RunManager(None)
    gate = asyncio.Event()

    async def work(run: Run) -> dict[str, Any]:
        for i in range(3):
            run.emit("tick", {"i": i})
        await gate.wait()
        run.emit("tick", {"i": 3})
        return {}

    run = mgr.start("atlas", "t", work)
    await asyncio.sleep(0.02)
    task = asyncio.create_task(collect(run))
    await asyncio.sleep(0.02)
    gate.set()
    events = data_events(await asyncio.wait_for(task, 2))
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5, 6]
    assert [e["data"].get("i") for e in events if e["type"] == "tick"] == [0, 1, 2, 3]


async def test_resume_after_seq() -> None:
    mgr = RunManager(None)

    async def work(run: Run) -> dict[str, Any]:
        run.emit("tick", {})
        return {}

    run = mgr.start("forge", "t", work)
    await run.task  # type: ignore[misc]
    events = data_events(await collect(run, after=2))
    assert [e["type"] for e in events] == ["run.done"]


async def test_cancel_before_the_task_starts() -> None:
    mgr = RunManager(None)

    async def work(run: Run) -> dict[str, Any]:  # never runs
        raise AssertionError

    run = mgr.start("arena", "t", work)
    got, requested = mgr.cancel(run.id)
    assert requested and got is run
    await asyncio.gather(run.task, return_exceptions=True)  # type: ignore[arg-type]
    assert run.status == "cancelled" and run.events[-1]["type"] == "run.failed"
    assert run.events[-1]["data"]["cancelled"] is True
    assert mgr.cancel(run.id) == (run, False)
    assert mgr.cancel("missing") == (None, False)


async def test_failure_is_captured_and_non_json_data_is_stringified() -> None:
    mgr = RunManager(None)

    async def work(run: Run) -> dict[str, Any]:
        run.emit("odd", {"path": Path("/tmp/x"), "s": {1, 2}.__class__})
        raise ValueError("boom")

    run = mgr.start("council", "t", work)
    await asyncio.gather(run.task, return_exceptions=True)  # type: ignore[arg-type]
    assert run.status == "failed" and run.error == "boom"
    assert run.events[1]["data"]["path"] == "/tmp/x"
    assert run.events[-1] == {**run.events[-1], "type": "run.failed", "data": {"error": "boom"}}


async def test_history_roundtrip_and_pruning(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    mgr = RunManager(path, max_history=3)

    async def work(run: Run) -> dict[str, Any]:
        return {"answer": run.title}

    runs = []
    for i in range(5):
        r = mgr.start("council", f"q{i}", work)
        await r.task  # type: ignore[misc]
        runs.append(r)
        await asyncio.sleep(0.002)
    saved = json.loads(path.read_text())
    assert saved["counts"]["council"] == 5 and len(saved["runs"]) == 3
    assert [r["title"] for r in saved["runs"]] == ["q4", "q3", "q2"]
    reloaded = RunManager(path)
    assert reloaded.counts["council"] == 5
    assert [r.title for r in reloaded.recent()] == ["q4", "q3", "q2"]
    assert reloaded.get(runs[4].id).result == {"answer": "q4"}  # type: ignore[union-attr]
    assert not list(tmp_path.glob(".history-*"))  # atomic write leaves no temp files


async def test_memory_is_bounded() -> None:
    mgr = RunManager(None, max_memory_runs=3)

    async def work(run: Run) -> dict[str, Any]:
        return {}

    for i in range(6):
        r = mgr.start("forge", str(i), work)
        await r.task  # type: ignore[misc]
        await asyncio.sleep(0.002)
    assert len(mgr.runs) <= 4
    assert mgr.counts["forge"] == 6


async def test_shutdown_cancels_active_runs(tmp_path: Path) -> None:
    mgr = RunManager(tmp_path / "h.json")

    async def work(run: Run) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {}

    run = mgr.start("forge", "slow", work)
    await asyncio.sleep(0.01)
    await mgr.shutdown()
    assert run.status == "cancelled"
