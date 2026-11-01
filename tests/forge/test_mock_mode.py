"""Forge on the offline mock backend: the mock's tool calls are nonsense, the run must still end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nimlab.core import NIMClient, NIMConfig
from nimlab.forge import Forge, ForgeReport

# ruff: noqa: ASYNC240


async def test_full_run_in_mock_mode_completes_and_reports(
    mock_client: NIMClient, ws: Path
) -> None:
    events = []
    report = await Forge(mock_client).run(
        "Write a function slugify(text) that lowercases and joins words with dashes",
        ws,
        on_event=events.append,
    )
    assert mock_client.config.mock
    assert report.mock is True
    assert report.status in {"failed", "success"}
    assert report.errors == []  # no phase crashed
    assert report.plan is not None and report.plan.steps  # mock JSON was schema-valid
    assert report.models is not None
    assert "Mock mode" in report.summary

    # Placeholder tool calls hit the first tool (list_dir) with junk arguments: nothing is written.
    assert sorted(p.name for p in ws.iterdir()) == [".forge"]
    tool_calls = [e.data["tool"] for e in events if e.kind == "tool_call"]
    assert tool_calls and set(tool_calls) == {"list_dir"}

    # The report exists on disk and is readable.
    on_disk = ForgeReport.load(ws)
    assert on_disk.status == report.status and on_disk.mock
    data = json.loads((ws / ".forge" / "forge_report.json").read_text())
    assert data["usage"]["total"]["calls"] > 0
    assert (ws / ".forge" / "forge_report.md").read_text().startswith("# Forge report:")
    assert {t.phase for t in report.timings} >= {"explore", "plan", "implement", "verify", "report"}


async def test_mock_run_terminates_via_no_progress_detection(
    mock_client: NIMClient, ws: Path
) -> None:
    report = await Forge(mock_client).run("Build something", ws, max_iters=6)
    assert report.status == "failed"
    assert report.ended_reason.startswith("no progress")
    assert report.escalations == 2 and report.iterations < 6
    assert len(report.root_causes) == 2  # the mock reasoning model answered the RCA prompts
    assert report.tests is not None and report.tests.no_tests


async def test_mock_run_with_nothing_changed_skips_review(mock_client: NIMClient, ws: Path) -> None:
    (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (ws / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    events = []
    report = await Forge(mock_client).run("Add a subtract function", ws, on_event=events.append)
    assert report.status == "success", report.summary  # tests were green from the start
    assert report.errors == [] and report.test_rounds[0].ok
    assert report.review is None and report.files_changed == []
    assert any(e.message == "no changes to review" for e in events)
    assert "Mock mode" in report.summary
    assert (ws / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


async def test_mock_review_and_fix_round_on_real_mock_client(
    mock_client: NIMClient, ws: Path
) -> None:
    """Make the diff non-empty (as a real coder would) so the mock reviewer is exercised."""
    (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (ws / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )

    def simulated_coder(event) -> None:
        if event.kind == "phase_start" and event.phase == "implement":
            (ws / "subtract.py").write_text("def subtract(a, b):\n    return a - b\n")

    report = await Forge(mock_client).run("Add subtract", ws, on_event=simulated_coder)
    assert report.status == "success", report.summary
    assert report.errors == []
    assert report.review is not None and report.review.verdict in {"approve", "request_changes"}
    assert [c.path for c in report.files_changed] == ["subtract.py"]
    assert report.review_rounds in {0, 1}
    if report.review.verdict == "request_changes" and report.review.blocking_issues:
        assert report.review_rounds == 1 and report.iterations == 1  # one fix round, no re-review
    reviewed = [m for m in report.usage["models"] if m == report.models.reviewer]
    assert reviewed  # the reviewer model was called


async def test_mock_json_matches_lenient_models(mock_client: NIMClient) -> None:
    from nimlab.forge import Plan, Review

    plan = await mock_client.chat_json("m", [{"role": "user", "content": "plan a parser"}], Plan)
    review = await mock_client.chat_json("m", [{"role": "user", "content": "review it"}], Review)
    assert isinstance(plan, Plan) and plan.goal and plan.steps
    assert review.verdict in {"approve", "request_changes"}
    assert all(i.severity in {"blocker", "major", "minor", "nit"} for i in review.issues)


async def test_mock_dry_run(mock_client: NIMClient, ws: Path) -> None:
    report = await Forge(mock_client).run("Anything", ws, dry_run=True)
    assert report.status == "dry_run" and report.plan is not None
    assert list(ws.iterdir()) == []


async def test_mock_rollback_leaves_workspace_pristine(mock_client: NIMClient, ws: Path) -> None:
    (ws / "keep.txt").write_text("keep\n")
    report = await Forge(mock_client).run("Build something", ws, rollback_on_fail=True)
    assert report.status == "failed"
    assert (ws / "keep.txt").read_text() == "keep\n"
    assert sorted(p.name for p in ws.iterdir()) == [".forge", "keep.txt"]


async def test_mock_usage_is_tracked_per_model(mock_client: NIMClient, ws: Path) -> None:
    report = await Forge(mock_client).run(
        "Build something", ws, planner="p/plan", coder="c/code", reviewer="r/rev"
    )
    models = report.usage["models"]
    assert set(models) == {"p/plan", "c/code"}  # no tests passed, so no reviewer call
    assert all(m["calls"] > 0 and m["tokens_in"] > 0 for m in models.values())
    assert report.usage["total"]["calls"] == sum(m["calls"] for m in models.values())


@pytest.mark.parametrize("seed_tests", [False, True])
async def test_mock_run_is_deterministic(ws: Path, tmp_path: Path, seed_tests: bool) -> None:
    async def once(target: Path) -> ForgeReport:
        target.mkdir()
        if seed_tests:
            (target / "m.py").write_text("X = 1\n")
            (target / "test_m.py").write_text(
                "from m import X\n\n\ndef test_x():\n    assert X == 1\n"
            )
        async with NIMClient(NIMConfig(mock=True)) as client:
            return await Forge(client).run("Do the thing", target)

    a = await once(tmp_path / "a")
    b = await once(tmp_path / "b")
    assert (a.status, a.iterations, a.escalations) == (b.status, b.iterations, b.escalations)
    assert a.plan == b.plan and a.review == b.review
