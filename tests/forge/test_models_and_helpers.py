"""Data models (lenient parsing, report rendering), progress helpers, workspace summary, display."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from forge_fakes import FakeClient
from rich.console import Console

from nimlab.core import UsageTracker
from nimlab.core.results import Usage
from nimlab.forge import ForgeConfig, ForgeEvent, ForgeReport, Plan, Review, Sandbox
from nimlab.forge.display import ForgeDisplay, print_report
from nimlab.forge.events import EventBus
from nimlab.forge.models import Issue, ModelAssignment, PhaseTiming, Step, VerifyRound
from nimlab.forge.progress import PhaseTimer, StallDetector, UsageProbe
from nimlab.forge.pytest_parser import PytestReport
from nimlab.forge.roles import model_family, resolve_models
from nimlab.forge.snapshot import FileChange
from nimlab.forge.workspace import local_imports, summarize_workspace

# --------------------------------------------------------------------------- lenient models


class TestPlanParsing:
    def test_minimal_and_full(self) -> None:
        plan = Plan.model_validate({"goal": "g"})
        assert plan.steps == [] and plan.risks == [] and plan.test_strategy == ""
        full = Plan.model_validate(
            {"goal": "g", "steps": [{"id": 2, "description": "d", "files": ["a.py"]}],
             "test_strategy": "pytest", "risks": ["r"]}
        )  # fmt: skip
        assert full.steps[0] == Step(id=2, description="d", files=["a.py"])

    def test_models_wobble_types(self) -> None:
        plan = Plan.model_validate(
            {"goal": "g", "steps": [{"id": "Step 3", "description": "d", "files": "a.py"}],
             "risks": "just one risk", "test_strategy": ["unit", "edge cases"], "extra": "ignored"}
        )  # fmt: skip
        assert plan.steps[0].id == 3 and plan.steps[0].files == ["a.py"]
        assert plan.risks == ["just one risk"]
        assert plan.test_strategy == "unit; edge cases"

    def test_json_schema_has_descriptions_for_the_llm(self) -> None:
        schema = Plan.model_json_schema()
        assert set(schema["properties"]) == {"goal", "steps", "test_strategy", "risks"}
        assert schema["required"] == ["goal"]


class TestReviewParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("approve", "approve"), ("APPROVED", "approve"), ("LGTM", "approve"),
         ("request_changes", "request_changes"), ("Request Changes", "request_changes"),
         ("changes-requested", "request_changes"), ("reject", "request_changes")],
    )  # fmt: skip
    def test_verdict_aliases(self, raw: str, expected: str) -> None:
        assert Review.model_validate({"verdict": raw}).verdict == expected

    def test_bad_verdict_rejected(self) -> None:
        with pytest.raises(ValueError):
            Review.model_validate({"verdict": "maybe"})

    @pytest.mark.parametrize(
        ("raw", "expected", "blocking"),
        [("blocker", "blocker", True), ("Critical", "blocker", True), ("high", "major", True),
         ("medium", "major", True), ("low", "minor", False), ("warning", "minor", False),
         ("nit", "nit", False), ("style", "nit", False)],
    )  # fmt: skip
    def test_severity_aliases_and_blocking(self, raw: str, expected: str, blocking: bool) -> None:
        issue = Issue.model_validate({"severity": raw, "description": "x"})
        assert issue.severity == expected and issue.blocking is blocking

    def test_line_number_coercion_and_blocking_issues(self) -> None:
        review = Review.model_validate(
            {"verdict": "request_changes",
             "issues": [{"severity": "major", "line": "12", "description": "a"},
                        {"severity": "nit", "line": "n/a", "description": "b"}]}
        )  # fmt: skip
        assert [i.line for i in review.issues] == [12, None]
        assert [i.description for i in review.blocking_issues] == ["a"]


class TestConfig:
    def test_defaults_and_validation(self) -> None:
        cfg = ForgeConfig()
        assert (cfg.max_iters, cfg.dry_run, cfg.rollback_on_fail) == (6, False, False)
        with pytest.raises(ValueError):
            ForgeConfig(max_iters=-1)
        with pytest.raises(ValueError):
            ForgeConfig(unknown_option=True)  # type: ignore[call-arg]
        with pytest.raises(ValueError):
            ForgeConfig(stall_threshold=1)


# --------------------------------------------------------------------------- report


def sample_report() -> ForgeReport:
    return ForgeReport(
        task="Build | a thing\nwith a pipe",
        workspace="/w",
        status="failed",
        ended_reason="ran out of fix rounds (--max-iters)",
        summary="Not finished.",
        models=ModelAssignment(planner="p", coder="c", reviewer="r"),
        plan=Plan(
            goal="g",
            steps=[Step(id=1, description="do it", files=["a.py"])],
            test_strategy="pytest",
            risks=["edge"],
        ),
        iterations=2,
        escalations=1,
        rolled_back=True,
        test_rounds=[
            VerifyRound(index=1, summary="1 failed", failed=1),
            VerifyRound(index=2, summary="ok", passed=3, ok=True),
        ],
        review=Review(
            verdict="request_changes",
            summary="bad",
            issues=[Issue(severity="major", file="a.py", line=3, description="broken")],
        ),
        root_causes=["Root cause: x"],
        files_changed=[FileChange(path="a.py", status="added", additions=1)],
        tests=PytestReport(exit_code=1, passed=3, failed=1),
        diff="diff --git a/a.py b/a.py\n--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+x = 1\n",
        timings=[
            PhaseTiming(phase="plan", seconds=1.5, runs=1, llm_calls=1, tokens_in=10, tokens_out=5)
        ],
        usage={
            "total": {"calls": 2, "tokens_in": 10, "tokens_out": 5},
            "models": {"p": {"calls": 2, "tokens_in": 10, "tokens_out": 5}},
        },
        warnings=["w1"],
        errors=["e1"],
    )


class TestReport:
    def test_markdown_contains_all_sections(self) -> None:
        md = sample_report().to_markdown()
        assert md.startswith("# Forge report: FAILED")
        sections = (
            "## Summary", "## Models", "## Plan", "## Tests", "## Review", "## Files changed",
            "## Diff", "## Timings", "## Token usage", "## Warnings", "## Errors",
            "## Root-cause analyses",
        )  # fmt: skip
        for section in sections:
            assert section in md, section
        for needle in ("workspace rolled back", "- [major] a.py:3: broken", "1. do it (`a.py`)"):
            assert needle in md
        assert "Build | a thing" in md  # task text is not escaped away

    def test_json_roundtrip_and_load(self, tmp_path: Path) -> None:
        report = sample_report()
        paths = report.write(tmp_path / ".forge")
        assert set(paths) == {"json", "markdown", "patch"}
        loaded = ForgeReport.load(tmp_path)
        assert loaded == report
        assert ForgeReport.load(tmp_path / ".forge") == report

    def test_load_errors(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ForgeReport.load(tmp_path)
        (tmp_path / ".forge").mkdir()
        (tmp_path / ".forge" / "forge_report.json").write_text("{}")
        with pytest.raises(ValueError):
            ForgeReport.load(tmp_path)

    def test_ok_property(self) -> None:
        assert ForgeReport(task="t", workspace="w", status="success").ok
        assert ForgeReport(task="t", workspace="w", status="dry_run").ok
        assert not ForgeReport(task="t", workspace="w", status="failed").ok
        assert not ForgeReport(task="t", workspace="w", status="error").ok

    def test_rich_rendering(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, width=120, color_system=None)
        print_report(console, sample_report(), show_diff=True)
        out = buf.getvalue()
        for needle in (
            "FAILED",
            "Test rounds",
            "Plan",
            "Phase timings",
            "Review: request_changes",
            "x = 1",
        ):
            assert needle in out


# --------------------------------------------------------------------------- progress helpers


class TestStallDetector:
    def test_basic(self) -> None:
        det = StallDetector()
        assert det.repeats == 0 and not det.stalled
        det.record("A")
        assert det.repeats == 1 and not det.stalled
        det.record("A")
        assert det.stalled and det.repeats == 2

    def test_window_forgets_old_rounds(self) -> None:
        det = StallDetector(threshold=2, window=3)
        for sig in ("A", "B", "C", "D", "A"):
            det.record(sig)
        assert not det.stalled  # the first A fell out of the window

    def test_threshold_three(self) -> None:
        det = StallDetector(threshold=3, window=5)
        for _ in range(2):
            det.record("A")
        assert not det.stalled
        det.record("A")
        assert det.stalled

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            StallDetector(threshold=1)


class TestUsageAndTimer:
    def test_probe_without_tracker(self) -> None:
        class Bare:
            pass

        probe = UsageProbe(Bare())
        assert probe.totals() == {"calls": 0, "tokens_in": 0, "tokens_out": 0}
        assert probe.delta()["total"]["calls"] == 0

    def test_probe_delta_ignores_earlier_usage(self) -> None:
        client = FakeClient()
        client.usage.record("m1", Usage(prompt_tokens=100, completion_tokens=10), 0.5)
        probe = UsageProbe(client)
        client.usage.record("m1", Usage(prompt_tokens=7, completion_tokens=3), 0.25)
        client.usage.record("m2", Usage(prompt_tokens=1, completion_tokens=1), 0.1, error=True)
        delta = probe.delta()
        assert delta["models"]["m1"]["calls"] == 1 and delta["models"]["m1"]["tokens_in"] == 7
        assert delta["models"]["m2"]["errors"] == 1
        assert delta["total"]["calls"] == 2 and delta["total"]["tokens_out"] == 4

    def test_broken_tracker_is_tolerated(self) -> None:
        class Broken:
            class usage:
                @staticmethod
                def summary() -> dict:
                    raise RuntimeError("nope")

        assert UsageProbe(Broken()).delta()["total"]["calls"] == 0

    def test_phase_timer_accumulates_repeats_and_usage(self) -> None:
        tracker = UsageTracker()

        class C:
            usage = tracker

        timer = PhaseTimer(UsageProbe(C()))
        timer.start("verify")
        time.sleep(0.02)
        timer.start("debug")  # implicitly stops 'verify'
        tracker.record("m", Usage(prompt_tokens=10, completion_tokens=2), 0.1)
        timer.start("verify")
        time.sleep(0.02)
        mid = {t.phase: t for t in timer.snapshot()}  # includes the running phase
        assert mid["verify"].runs == 2 and mid["verify"].seconds >= 0.04
        timer.stop()
        timer.stop()  # idempotent
        final = {t.phase: t for t in timer.snapshot()}
        assert list(final) == ["verify", "debug"]
        assert final["debug"].llm_calls == 1 and final["debug"].tokens_in == 10
        assert final["verify"].llm_calls == 0


# --------------------------------------------------------------------------- roles


class TestRoles:
    def test_defaults_use_registry_roles(self) -> None:
        models = resolve_models()
        assert "r1" in models.planner or "reason" in models.planner or "nemotron" in models.planner
        assert "coder" in models.coder.lower()
        assert model_family(models.reviewer) != model_family(models.coder)

    def test_env_override_flows_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NIMLAB_MODEL_CODER", "acme/super-coder")
        assert resolve_models().coder == "acme/super-coder"

    def test_explicit_arguments_win(self) -> None:
        models = resolve_models("a/p", "b/c", "c/r")
        assert (models.planner, models.coder, models.reviewer) == ("a/p", "b/c", "c/r")

    def test_model_family(self) -> None:
        assert model_family("qwen/qwen2.5-coder-32b") == "qwen"
        assert model_family("NVIDIA/x") == "nvidia"
        assert model_family("bare-model") == "bare-model"


# --------------------------------------------------------------------------- workspace summary


class TestWorkspaceSummary:
    def test_empty(self, sandbox: Sandbox) -> None:
        summary = summarize_workspace(sandbox)
        assert summary.empty and "empty" in summary.render()

    def test_tree_key_files_outline_and_tests(self, sandbox: Sandbox, ws: Path) -> None:
        (ws / "README.md").write_text("# Hello\nThis project parses things.\n")
        (ws / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
        (ws / "src" / "demo").mkdir(parents=True)
        (ws / "src" / "demo" / "core.py").write_text(
            "class Parser:\n    def parse(self, s): ...\n    def reset(self): ...\n\n"
            "def helper(a, b):\n    return a + b\n\nasync def fetch(url): ...\n"
        )
        (ws / "tests").mkdir()
        (ws / "tests" / "test_core.py").write_text("def test_it(): ...\n")
        (ws / "broken.py").write_text("def (:\n")
        (ws / "__pycache__").mkdir()
        (ws / "__pycache__" / "x.pyc").write_bytes(b"\x00")
        summary = summarize_workspace(sandbox)
        text = summary.render()
        assert summary.has_tests and summary.test_files == ["tests/test_core.py"]
        assert "src/\n  demo/\n    core.py" in summary.tree
        assert "__pycache__" not in text
        assert "class Parser [parse, reset]" in text and "def helper(a, b)" in text
        assert "async" not in text or "def fetch(url)" in text
        assert "broken.py: (syntax error)" in text
        assert "--- README.md (head) ---" in text and "This project parses things." in text
        assert "tests/test_core.py" not in summary.symbols  # test files are not outlined

    def test_render_is_capped(self, sandbox: Sandbox, ws: Path) -> None:
        for i in range(300):
            (ws / f"module_{i}.py").write_text(f"def func_{i}(a, b, c):\n    return {i}\n")
        summary = summarize_workspace(sandbox)
        text = summary.render(max_chars=2000)
        assert len(text) <= 2000 and "truncated" in text
        assert "more files" in summary.tree

    def test_local_imports(self, sandbox: Sandbox, ws: Path) -> None:
        (ws / "mathlib.py").write_text("")
        (ws / "pkg").mkdir()
        (ws / "pkg" / "__init__.py").write_text("")
        (ws / "src").mkdir()
        (ws / "src" / "inner.py").write_text("")
        source = (
            "import os, mathlib\nfrom pkg import x\nfrom inner import y\n"
            "from . import z\nimport missing\n"
        )
        assert local_imports(sandbox, source) == ["mathlib.py", "pkg/__init__.py", "src/inner.py"]
        assert local_imports(sandbox, "def (:") == []


# --------------------------------------------------------------------------- display / events


class TestEventsAndDisplay:
    def test_bus_history_cap_and_failing_subscriber(self) -> None:
        bus = EventBus(max_history=5)
        seen: list[ForgeEvent] = []

        def bad(_: ForgeEvent) -> None:
            raise RuntimeError("ui crashed")

        bus.subscribe(bad)
        bus.subscribe(seen.append)
        for i in range(8):
            bus.emit("info", "x", f"m{i}", n=i)
        assert len(seen) == 8 and len(bus.history) == 5
        assert bus.history[0].message == "m3"
        d = bus.history[-1].to_dict()
        assert d["kind"] == "info" and d["data"] == {"n": 7} and json.dumps(d)

    def test_plain_display_prints_notable_events_only(self) -> None:
        buf = io.StringIO()
        display = ForgeDisplay(Console(file=buf, width=100, color_system=None), live=False)
        bus = EventBus()
        bus.subscribe(display)
        bus.emit("phase_start", "plan", "plan")
        bus.emit("llm", "plan", "noise")
        bus.emit("tool_call", "implement", "write_file {...}", tool="write_file")
        bus.emit("tests", "verify", "round 1: 2 failed", ok=False)
        bus.emit("warning", "verify", "careful [with brackets]")
        bus.emit("stall", "verify", "same failures 2x")
        out = buf.getvalue()
        assert "> plan" in out and "write_file" in out and "round 1: 2 failed" in out
        assert "warning: careful [with brackets]" in out  # markup in messages is escaped
        assert "noise" not in out

    def test_quiet_display_prints_nothing(self) -> None:
        buf = io.StringIO()
        display = ForgeDisplay(Console(file=buf), quiet=True)
        display(ForgeEvent(kind="warning", message="x"))
        assert buf.getvalue() == ""

    def test_live_display_context_manager(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=True, width=80)
        display = ForgeDisplay(console, live=True)
        with display:
            display(ForgeEvent(kind="phase_start", phase="plan", message="plan"))
            display(
                ForgeEvent(kind="tests", phase="verify", message="round 1: ok", data={"ok": True})
            )
            display(ForgeEvent(kind="llm", phase="plan", data={"tokens_in": 5, "tokens_out": 6}))
        assert display._counters["tokens"] == 11
