"""End-to-end Forge runs against a scripted FakeClient (real pytest, temp workspaces, offline)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from forge_fakes import (
    APPROVE_JSON,
    FIZZBUZZ_BUGGY,
    FIZZBUZZ_PATCH,
    FIZZBUZZ_TESTS,
    PLAN_JSON,
    FakeCall,
    FakeClient,
    calls,
    say,
    tc,
)

from nimlab.core import NIMAuthError, NIMJSONError, NIMResponseError
from nimlab.forge import Forge, ForgeConfig, ForgeEvent, ForgeReport, apply_patch_text
from nimlab.forge.context import TRANSITIONS, State
from nimlab.forge.models import REPORT_JSON, REPORT_MD
from nimlab.forge.roles import model_family

PLANNER = "test/planner-r1"
CODER = "qwen/coder-test"
REVIEWER = "meta/reviewer-test"
MODELS = {"planner": PLANNER, "coder": CODER, "reviewer": REVIEWER}


def fizzbuzz_script() -> dict[str, list[Any]]:
    """plan -> write buggy code+tests -> run_tests (fails) -> [verify fails] -> patch -> run_tests
    (passes) -> final text -> [verify passes] -> review approve."""
    return {
        "chat": [
            calls(
                tc("write_file", path="fizzbuzz.py", content=FIZZBUZZ_BUGGY),
                tc("write_file", path="test_fizzbuzz.py", content=FIZZBUZZ_TESTS),
            ),
            calls(tc("run_tests")),
            say("Implemented fizzbuzz and its tests."),
            # --- debug round 1
            calls(tc("read_file", path="fizzbuzz.py")),
            calls(tc("apply_patch", path="fizzbuzz.py", patch=FIZZBUZZ_PATCH)),
            calls(tc("run_tests")),
            say("Checked multiples of 15 first; all tests pass."),
        ],
        "json": [PLAN_JSON, APPROVE_JSON],
    }


async def run_forge(
    client: FakeClient, ws: Path, **kw: Any
) -> tuple[ForgeReport, list[ForgeEvent]]:
    events: list[ForgeEvent] = []
    report = await Forge(client).run(
        "Write fizzbuzz(n) with pytest tests", ws, on_event=events.append, **{**MODELS, **kw}
    )
    return report, events


@dataclass
class Happy:
    """One shared run of the headline scenario (running real pytest four times is not free)."""

    client: FakeClient
    report: ForgeReport
    events: list[ForgeEvent]
    ws: Path


@pytest.fixture(scope="module")
def happy(tmp_path_factory: pytest.TempPathFactory) -> Happy:
    ws = tmp_path_factory.mktemp("happy") / "workspace"
    ws.mkdir()
    client = FakeClient(**fizzbuzz_script())
    report, events = asyncio.run(run_forge(client, ws))
    return Happy(client, report, events, ws)


class TestHappyPath:
    """The headline scenario: a bug is introduced, caught by tests, debugged and reviewed."""

    def test_drives_workspace_to_green(self, happy: Happy) -> None:
        report, ws = happy.report, happy.ws
        assert happy.client.unexpected == []
        assert report.status == "success", report.summary
        assert report.ended_reason == "all tests pass"
        assert report.tests is not None and report.tests.ok
        assert (report.tests.passed, report.tests.failed) == (7, 0)
        source = (ws / "fizzbuzz.py").read_text()
        assert source.index("n % 15") < source.index("n % 3")

    def test_debug_loop_triggered_exactly_once(self, happy: Happy) -> None:
        report, client = happy.report, happy.client
        assert report.iterations == 1 and report.escalations == 0 and report.review_rounds == 0
        assert [r.ok for r in report.test_rounds] == [False, True]
        failing = report.test_rounds[0]
        assert failing.failed == 4 and "test_fizzbuzz.py::test_fizzbuzz_15" in failing.failing
        assert any("test_fizzbuzz.py::test_multiples_of_15[30]" in f for f in failing.failing)
        assert [e.kind for e in happy.events if e.kind == "tests"] == ["tests", "tests"]

        # The coder received the condensed failure report, not raw pytest output.
        debug_call = next(c for c in client.of_kind("chat") if "fix round 1" in c.last_user)
        assert "FAIL test_fizzbuzz.py::test_fizzbuzz_15" in debug_call.last_user
        assert "AssertionError: assert 'Fizz' == 'FizzBuzz'" in debug_call.last_user
        assert "=====" not in debug_call.last_user and str(happy.ws) not in debug_call.last_user
        assert "Do not weaken or delete tests" in debug_call.last_user

    def test_diff_and_files_changed(self, happy: Happy) -> None:
        report, ws = happy.report, happy.ws
        assert {(c.path, c.status) for c in report.files_changed} == {
            ("fizzbuzz.py", "added"),
            ("test_fizzbuzz.py", "added"),
        }
        assert report.diff.startswith("diff --git a/fizzbuzz.py b/fizzbuzz.py")
        assert "--- /dev/null\n+++ b/fizzbuzz.py" in report.diff
        # The diff is of the final state: the fixed ordering, never the buggy one.
        assert (
            '+    if n % 15 == 0:\n+        return "FizzBuzz"\n+    if n % 3 == 0:' in report.diff
        )
        assert '+    if n % 5 == 0:\n+        return "Buzz"\n+    if n % 15 == 0' not in report.diff
        assert "+def test_fizzbuzz_15():" in report.diff
        added = sum(c.additions for c in report.files_changed)
        expected = len((ws / "fizzbuzz.py").read_text().splitlines()) + len(
            FIZZBUZZ_TESTS.splitlines()
        )
        assert added == expected
        assert "2 file(s) changed" in report.summary

    def test_report_files_written_and_roundtrip(self, happy: Happy) -> None:
        report, ws = happy.report, happy.ws
        forge_dir = ws / ".forge"
        data = json.loads((forge_dir / REPORT_JSON).read_text())
        assert data["status"] == "success" and data["iterations"] == 1
        assert data["plan"]["goal"] == PLAN_JSON["goal"]
        assert data["review"]["verdict"] == "approve"
        assert data["models"] == MODELS
        assert (forge_dir / "changes.patch").read_text() == report.diff

        md = (forge_dir / REPORT_MD).read_text()
        assert md.startswith("# Forge report: SUCCESS")
        for needle in (
            "## Plan",
            "## Tests",
            "## Review",
            "## Files changed",
            "## Diff",
            "## Timings",
        ):
            assert needle in md
        assert "fizzbuzz.py" in md and "```diff" in md

        loaded = ForgeReport.load(ws)
        assert loaded.status == report.status and loaded.diff == report.diff
        assert loaded.tests is not None and loaded.tests.passed == 7
        assert ForgeReport.load(forge_dir).task == report.task

    def test_workspace_has_only_expected_files(self, happy: Happy) -> None:
        top = sorted(p.name for p in happy.ws.iterdir())
        assert top == [".forge", "fizzbuzz.py", "test_fizzbuzz.py"]  # no __pycache__/.pytest_cache

    def test_roles_use_the_right_models(self, happy: Happy) -> None:
        client = happy.client
        plan_call, review_call = client.of_kind("chat_json")
        assert plan_call.model == PLANNER and plan_call.schema.__name__ == "Plan"
        assert review_call.model == REVIEWER and review_call.schema.__name__ == "Review"
        assert {c.model for c in client.of_kind("chat")} == {CODER}
        assert all(c.tools for c in client.of_kind("chat"))
        assert "diff --git a/fizzbuzz.py" in review_call.last_user  # reviewer sees the diff
        assert "pytest: 7 passed" in review_call.last_user
        assert "Write fizzbuzz(n)" in plan_call.last_user

    def test_timings_and_usage(self, happy: Happy) -> None:
        report, client = happy.report, happy.client
        by_phase = {t.phase: t for t in report.timings}
        assert list(by_phase) == [
            "init", "explore", "plan", "implement", "verify", "debug", "review", "report",
        ]  # fmt: skip
        assert by_phase["verify"].runs == 2 and by_phase["debug"].runs == 1
        assert all(t.seconds >= 0 for t in by_phase.values())
        assert by_phase["plan"].llm_calls == 1 and by_phase["implement"].llm_calls == 3
        assert by_phase["debug"].llm_calls == 4 and by_phase["review"].llm_calls == 1
        assert by_phase["verify"].llm_calls == 0 and by_phase["verify"].seconds > 0
        total = report.usage["total"]
        assert total["calls"] == 9 == len(client.calls)
        assert total["tokens_in"] == sum(t.tokens_in for t in report.timings) > 0
        assert set(report.usage["models"]) == {PLANNER, CODER, REVIEWER}
        assert report.usage["models"][CODER]["calls"] == 7
        assert report.duration_s > 0 and report.started_at > 0

    def test_events_cover_the_pipeline(self, happy: Happy) -> None:
        events = happy.events
        kinds = {e.kind for e in events}
        assert {
            "run_start", "phase_start", "phase_end", "state", "plan", "llm", "tool_call",
            "tool_result", "tests", "review", "run_end",
        } <= kinds  # fmt: skip
        transitions = [
            (e.data["from"], e.data["to"]) for e in events if e.kind == "state" and e.data["from"]
        ]
        assert transitions[0] == ("init", "explore")
        assert transitions[-1] == ("report", "done")
        for src, dst in transitions:
            assert State(dst) in TRANSITIONS[State(src)]
        visited = [t[1] for t in transitions]
        assert visited == [
            "explore", "plan", "implement", "verify", "debug", "verify", "review", "report", "done",
        ]  # fmt: skip
        assert events[0].kind == "run_start" and events[-1].kind == "run_end"
        assert events[-1].data["status"] == "success"
        tool_names = [e.data["tool"] for e in events if e.kind == "tool_call"]
        assert tool_names == [
            "write_file", "write_file", "run_tests", "read_file", "apply_patch", "run_tests",
        ]  # fmt: skip

    def test_plan_is_normalised_and_in_prompts(self, happy: Happy) -> None:
        assert happy.report.plan is not None
        assert [s.id for s in happy.report.plan.steps] == [1, 2]
        implement_prompt = happy.client.of_kind("chat")[0].messages[1]["content"]
        assert "Implement fizzbuzz" in implement_prompt
        assert "multiples of 15 must be checked" in implement_prompt
        assert "The workspace is empty" in implement_prompt

    async def test_existing_workspace_is_explored_and_preserved(self, ws: Path) -> None:
        (ws / "README.md").write_text("# Demo project\n")
        (ws / "util.py").write_text("def helper(x):\n    return x\n")
        client = FakeClient(**fizzbuzz_script())
        report, _ = await run_forge(client, ws)
        assert (ws / "README.md").read_text() == "# Demo project\n"
        assert {c.path for c in report.files_changed} == {"fizzbuzz.py", "test_fizzbuzz.py"}
        implement_prompt = client.of_kind("chat")[0].messages[1]["content"]
        assert "README.md" in implement_prompt and "def helper(x)" in implement_prompt


class TestNoChanges:
    GREEN = {
        "fizzbuzz.py": apply_patch_text(FIZZBUZZ_BUGGY, FIZZBUZZ_PATCH).text,
        "test_fizzbuzz.py": FIZZBUZZ_TESTS,
    }

    def seed(self, ws: Path) -> None:
        for name, text in self.GREEN.items():
            (ws / name).write_text(text)

    async def test_green_but_untouched_workspace_is_nudged_then_succeeds(self, ws: Path) -> None:
        self.seed(ws)
        extra_test = (
            FIZZBUZZ_TESTS + "\n\ndef test_big():\n    assert fizzbuzz(1500) == 'FizzBuzz'\n"
        )

        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return say("The code already works, nothing to do.")  # lazy coder
            if call.messages[-1]["role"] == "tool":
                return say("Added a test.")
            return calls(tc("write_file", path="test_fizzbuzz.py", content=extra_test))

        client = FakeClient(chat=chat, json=[PLAN_JSON, APPROVE_JSON])
        report, events = await run_forge(client, ws)
        assert report.status == "success" and report.iterations == 1
        assert [r.ok for r in report.test_rounds] == [True, True]
        nudge = client.of_kind("chat")[1].last_user
        assert "no file in the workspace has been created or modified" in nudge
        assert [c.path for c in report.files_changed] == ["test_fizzbuzz.py"]
        assert any("nothing was changed" in e.message for e in events)

    async def test_coder_that_never_acts_ends_as_not_done(self, ws: Path) -> None:
        self.seed(ws)
        client = FakeClient(chat=lambda _call: say("Looks good to me."), json=[PLAN_JSON])
        report, _ = await run_forge(client, ws, rollback_on_fail=True)
        assert report.status == "failed" and report.ended_reason.startswith("the tests pass but no")
        assert report.iterations == 2 and report.rolled_back is False  # nothing to roll back
        assert report.review is None and report.files_changed == []
        assert "not done" in report.summary or "no files were changed" in report.summary

    async def test_nudging_respects_max_iters(self, ws: Path) -> None:
        self.seed(ws)
        client = FakeClient(chat=lambda _call: say("done"), json=[PLAN_JSON])
        report, _ = await run_forge(client, ws, max_iters=0)
        assert report.status == "failed" and report.iterations == 0 and len(report.test_rounds) == 1


class TestReviewRound:
    async def test_blocking_review_triggers_one_fix_round(self, ws: Path) -> None:
        extra_patch = (
            "<<<<<<< SEARCH\n    if n % 15 == 0:\n=======\n"
            "    if n < 1:\n        raise ValueError('n must be positive')\n    if n % 15 == 0:\n"
            ">>>>>>> REPLACE\n"
        )
        review = {
            "verdict": "request_changes",
            "summary": "No input validation.",
            "issues": [
                {"severity": "major", "file": "fizzbuzz.py", "line": 3,
                 "description": "n <= 0 is accepted",
                 "suggestion": "raise ValueError"},
                {"severity": "nit", "description": "docstring could be longer"},
            ],
        }  # fmt: skip
        script = fizzbuzz_script()
        script["chat"] += [
            calls(tc("apply_patch", path="fizzbuzz.py", patch=extra_patch)),
            calls(tc("run_tests")),
            say("Added validation."),
        ]
        script["json"] = [PLAN_JSON, review]  # exactly one review: a second would be 'unexpected'
        client = FakeClient(**script)
        report, events = await run_forge(client, ws)

        assert client.unexpected == []
        assert report.status == "success"
        assert report.review is not None and report.review.verdict == "request_changes"
        assert report.review_rounds == 1 and report.iterations == 2
        assert "raise ValueError" in (ws / "fizzbuzz.py").read_text()
        fix_prompt = client.of_kind("chat")[-3].last_user
        assert "independent reviewer requested changes" in fix_prompt
        assert (
            "[major] (fizzbuzz.py:3) n <= 0 is accepted Suggested fix: raise ValueError"
            in fix_prompt
        )
        visited = [e.data["to"] for e in events if e.kind == "state"]
        assert visited[-5:] == ["review", "fix", "verify", "report", "done"]
        assert report.test_rounds[-1].ok and len(report.test_rounds) == 3

    async def test_nit_only_request_changes_does_not_trigger_fix(self, ws: Path) -> None:
        review = {
            "verdict": "request_changes",
            "summary": "meh",
            "issues": [{"severity": "nit", "description": "x"}],
        }
        script = fizzbuzz_script()
        script["json"] = [PLAN_JSON, review]
        report, _ = await run_forge(FakeClient(**script), ws)
        assert report.review_rounds == 0 and report.status == "success"

    async def test_review_rounds_can_be_disabled(self, ws: Path) -> None:
        review = {
            "verdict": "request_changes",
            "summary": "bad",
            "issues": [{"severity": "blocker", "description": "x"}],
        }
        script = fizzbuzz_script()
        script["json"] = [PLAN_JSON, review]
        client = FakeClient(**script)
        report = await Forge(client, ForgeConfig(max_review_rounds=0, **MODELS)).run("t", ws)
        assert report.review_rounds == 0 and report.status == "success"
        assert any("no fix budget" in w for w in report.warnings)

    async def test_review_failure_is_non_fatal(self, ws: Path) -> None:
        script = fizzbuzz_script()
        script["json"] = [PLAN_JSON, NIMJSONError("not json", raw="???")]
        report, _ = await run_forge(FakeClient(**script), ws)
        assert report.status == "success" and report.review is None
        assert any("review unavailable" in w for w in report.warnings)

    async def test_reviewer_defaults_to_a_different_family(self, ws: Path) -> None:
        from nimlab.forge.roles import resolve_models

        models = resolve_models()
        assert model_family(models.reviewer) != model_family(models.coder)
        assert resolve_models(coder="nvidia/nemotron-3-super-120b-a12b").reviewer.split("/")[0] != "meta"
        assert resolve_models(reviewer="my/reviewer").reviewer == "my/reviewer"


class TestStallAndEscalation:
    @staticmethod
    def stuck_client(
        rca_text: str = "Root cause: the checks are ordered wrongly.\nFix: test 15 first.",
    ) -> FakeClient:
        def chat(call: FakeCall) -> Any:
            if call.model == PLANNER:  # root-cause analysis
                return say(rca_text)
            if call.index == 0:
                return calls(
                    tc("write_file", path="fizzbuzz.py", content=FIZZBUZZ_BUGGY),
                    tc("write_file", path="test_fizzbuzz.py", content=FIZZBUZZ_TESTS),
                )
            return say("I reviewed the code and believe it is correct.")  # never changes anything

        return FakeClient(chat=chat, json=[PLAN_JSON])

    async def test_no_progress_detected_and_escalated_then_abandoned(self, ws: Path) -> None:
        client = self.stuck_client()
        report, events = await run_forge(client, ws, max_iters=6)

        assert report.status == "failed"
        assert report.ended_reason.startswith("no progress")
        assert report.escalations == 2 and report.iterations == 3  # < max_iters: gave up early
        signatures = {r.signature for r in report.test_rounds}
        assert len(report.test_rounds) == 4 and len(signatures) == 1
        assert len(report.root_causes) == 2 and "ordered wrongly" in report.root_causes[0]
        assert [e.kind for e in events].count("stall") == 3
        assert [e.kind for e in events].count("escalate") == 2

        rca_calls = client.with_model(PLANNER)
        rca_calls = [c for c in rca_calls if c.kind == "chat"]
        assert len(rca_calls) == 2 and not rca_calls[0].tools
        prompt = rca_calls[0].messages[1]["content"]
        assert "## Repeating failures" in prompt and "test_fizzbuzz_15" in prompt
        assert "```python" in prompt and "def fizzbuzz" in prompt  # source files are attached
        assert "believe it is correct" in prompt  # the coder's own claim is included

    async def test_rca_is_fed_to_the_next_coder_round(self, ws: Path) -> None:
        client = self.stuck_client("Root cause: ORDER-BUG-MARKER")
        await run_forge(client, ws)
        coder_prompts = [c.last_user for c in client.of_kind("chat") if c.model == CODER]
        with_rca = [p for p in coder_prompts if "ORDER-BUG-MARKER" in p]
        without = [p for p in coder_prompts if "fix round" in p and "ORDER-BUG-MARKER" not in p]
        assert len(with_rca) == 2 and len(without) == 1
        assert "A senior engineer analysed the repeated failure" in with_rca[0]

    async def test_escalation_budget_is_configurable(self, ws: Path) -> None:
        client = self.stuck_client()
        report = await Forge(client, ForgeConfig(max_escalations=1, **MODELS)).run("t", ws)
        assert report.escalations == 1 and report.iterations == 2
        assert report.ended_reason.startswith("no progress")

    async def test_escalation_can_unblock_the_coder(self, ws: Path) -> None:
        def chat(call: FakeCall) -> Any:
            if call.model == PLANNER:
                return say("Root cause: wrong order. Fix: check 15 first.")
            if call.index == 0:
                return calls(
                    tc("write_file", path="fizzbuzz.py", content=FIZZBUZZ_BUGGY),
                    tc("write_file", path="test_fizzbuzz.py", content=FIZZBUZZ_TESTS),
                )
            if "wrong order" in call.last_user and not any(
                m["role"] == "tool" for m in call.messages[-1:]
            ):
                return calls(tc("apply_patch", path="fizzbuzz.py", patch=FIZZBUZZ_PATCH))
            return say("done" if call.messages[-1]["role"] == "tool" else "no change")

        client = FakeClient(chat=chat, json=[PLAN_JSON, APPROVE_JSON])
        report, _ = await run_forge(client, ws)
        assert report.status == "success"
        assert report.escalations == 1 and report.iterations == 2
        assert [r.ok for r in report.test_rounds] == [False, False, True]

    async def test_oscillating_failures_count_as_stall(self, ws: Path) -> None:
        """A -> B -> A is also 'no progress' (the window looks beyond consecutive rounds)."""
        from nimlab.forge.progress import StallDetector

        det = StallDetector(threshold=2, window=4)
        for sig in ("A", "B"):
            det.record(sig)
            assert not det.stalled
        det.record("A")
        assert det.stalled and det.repeats == 2
        det.record("C")
        assert not det.stalled

    async def test_max_iters_exhaustion_without_stall(self, ws: Path) -> None:
        """Each fix round changes the failure, so it is not a stall; the budget ends the run."""
        counter = iter(range(100))

        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return calls(
                    tc("write_file", path="m.py", content="VALUE = 0\n"),
                    tc(
                        "write_file",
                        path="test_m.py",
                        content="from m import VALUE\n\ndef test_v():\n    assert VALUE == 99\n",
                    ),
                )
            if call.messages[-1]["role"] == "tool":
                return say("tweaked")
            return calls(tc("write_file", path="m.py", content=f"VALUE = {next(counter) + 1}\n"))

        client = FakeClient(chat=chat, json=[PLAN_JSON])
        report, _ = await run_forge(client, ws, max_iters=2)
        assert report.status == "failed" and report.ended_reason.startswith("ran out of fix rounds")
        assert report.iterations == 2 and report.escalations == 0
        assert len({r.signature for r in report.test_rounds}) == 3
        assert report.review is None  # failing code is not reviewed

    async def test_max_iters_zero_means_verify_once(self, ws: Path) -> None:
        client = self.stuck_client()
        report, _ = await run_forge(client, ws, max_iters=0)
        assert report.status == "failed" and report.iterations == 0 and len(report.test_rounds) == 1
        assert report.ended_reason.startswith("ran out of fix rounds")


class TestRollback:
    @staticmethod
    def seed(ws: Path) -> dict[str, str]:
        files = {
            "README.md": "# Keep me\n",
            "existing.py": "ORIGINAL = True\n",
            "pkg/mod.py": "x = 1\n",
        }
        for name, text in files.items():
            (ws / name).parent.mkdir(parents=True, exist_ok=True)
            (ws / name).write_text(text)
        return files

    @staticmethod
    def destructive_client() -> FakeClient:
        def chat(call: FakeCall) -> Any:
            if call.model == PLANNER:
                return say("Root cause: unknown.")
            if call.index == 0:
                return calls(
                    tc("write_file", path="fizzbuzz.py", content=FIZZBUZZ_BUGGY),
                    tc("write_file", path="test_fizzbuzz.py", content=FIZZBUZZ_TESTS),
                    tc("write_file", path="existing.py", content="ORIGINAL = False\n"),
                    tc("write_file", path="brand/new/dir/file.py", content="y = 2\n"),
                    tc("write_file", path="pkg/mod.py", content="x = 'clobbered'\n"),
                )
            return say("shrug")

        return FakeClient(chat=chat, json=[PLAN_JSON])

    async def test_rollback_restores_workspace_when_final_tests_fail(self, ws: Path) -> None:
        original = self.seed(ws)
        report, events = await run_forge(self.destructive_client(), ws, rollback_on_fail=True)

        assert report.status == "failed" and report.rolled_back is True
        assert {p.name for p in ws.iterdir()} == {".forge", "README.md", "existing.py", "pkg"}
        for name, text in original.items():
            assert (ws / name).read_text() == text
        assert not (ws / "brand").exists() and not (ws / "fizzbuzz.py").exists()
        # The report records what was attempted, and survived the rollback.
        assert {c.path for c in report.files_changed} == {
            "fizzbuzz.py", "test_fizzbuzz.py", "existing.py", "brand/new/dir/file.py", "pkg/mod.py",
        }  # fmt: skip
        assert "-ORIGINAL = True\n+ORIGINAL = False" in report.diff
        assert "rolled back" in report.summary
        assert ForgeReport.load(ws).rolled_back is True
        assert any(e.kind == "rollback" and e.data["removed"] == 3 for e in events)

    async def test_no_rollback_by_default(self, ws: Path) -> None:
        self.seed(ws)
        report, _ = await run_forge(self.destructive_client(), ws)
        assert report.status == "failed" and report.rolled_back is False
        assert (ws / "existing.py").read_text() == "ORIGINAL = False\n"
        assert (ws / "fizzbuzz.py").exists()

    async def test_rollback_not_applied_on_success(self, ws: Path) -> None:
        report, _ = await run_forge(FakeClient(**fizzbuzz_script()), ws, rollback_on_fail=True)
        assert report.status == "success" and not report.rolled_back
        assert (ws / "fizzbuzz.py").exists()

    async def test_rollback_after_llm_outage(self, ws: Path) -> None:
        self.seed(ws)

        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return calls(tc("write_file", path="partial.py", content="x = 1\n"))
            return NIMResponseError("HTTP 503")

        client = FakeClient(chat=chat, json=[PLAN_JSON])
        report, _ = await run_forge(client, ws, rollback_on_fail=True)
        assert report.status == "failed" and report.ended_reason.startswith(
            "the coder model was unavailable"
        )
        assert report.rolled_back and not (ws / "partial.py").exists()


class TestDryRun:
    async def test_plans_without_writing_anything(self, ws: Path) -> None:
        (ws / "keep.py").write_text("x = 1\n")
        client = FakeClient(
            json=[PLAN_JSON]
        )  # no chat script: any coder call would be 'unexpected'
        report, events = await run_forge(client, ws, dry_run=True)

        assert report.status == "dry_run" and report.dry_run and report.ok
        assert report.plan is not None and len(report.plan.steps) == 2
        assert client.unexpected == [] and not client.of_kind("chat")
        assert sorted(p.name for p in ws.iterdir()) == ["keep.py"]  # not even .forge/
        assert report.files_changed == [] and report.diff == "" and report.tests is None
        assert report.summary.startswith("Dry run: planned 2 step(s). No files were written.")
        visited = [e.data["to"] for e in events if e.kind == "state" and e.data["from"]]
        assert visited == ["explore", "plan", "report", "done"]

    async def test_dry_run_on_missing_workspace_creates_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "not" / "there"
        report, _ = await run_forge(FakeClient(json=[PLAN_JSON]), target, dry_run=True)
        assert report.status == "dry_run" and not target.exists() and not target.parent.exists()
        assert report.workspace == str(target)

    async def test_real_run_creates_missing_workspace(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "project"
        report, _ = await run_forge(FakeClient(**fizzbuzz_script()), target)
        assert report.status == "success" and (target / "fizzbuzz.py").exists()
        assert (target / ".forge" / REPORT_JSON).exists()

    async def test_dry_run_sandbox_is_read_only(self, ws: Path) -> None:
        # Even if something tried to use the tools during a dry run, writes are refused.
        from nimlab.forge import Sandbox
        from nimlab.forge.tools import WorkspaceTools

        out = await WorkspaceTools(Sandbox(ws, read_only=True)).call(
            "write_file", {"path": "a.py", "content": "x"}
        )
        assert not out.ok and not (ws / "a.py").exists()


class TestFailureModes:
    async def test_planner_failure_falls_back_to_single_step_plan(self, ws: Path) -> None:
        client = FakeClient(json=[NIMJSONError("garbage", raw="???")])
        report, _ = await run_forge(client, ws, dry_run=True)
        assert report.status == "dry_run"
        assert report.plan is not None and len(report.plan.steps) == 1
        assert "Write fizzbuzz" in report.plan.steps[0].description
        assert any("planner failed" in w for w in report.warnings)

    async def test_empty_plan_from_model_is_repaired(self, ws: Path) -> None:
        client = FakeClient(
            json=[{"goal": "", "steps": [{"id": "7", "description": "  "}], "risks": "careful"}]
        )
        report, _ = await run_forge(client, ws, dry_run=True)
        assert report.plan is not None
        assert report.plan.goal and [s.id for s in report.plan.steps] == [1]
        assert report.plan.risks == ["careful"]

    async def test_auth_error_is_reported_not_raised(self, ws: Path) -> None:
        (ws / "keep.py").write_text("x = 1\n")
        client = FakeClient(json=[NIMAuthError("invalid key", status_code=401)])
        report, _ = await run_forge(client, ws)
        assert report.status == "error" and not report.ok
        assert "authentication failed" in report.summary and report.errors
        assert (ws / ".forge" / REPORT_JSON).exists()
        assert (ws / "keep.py").read_text() == "x = 1\n"

    async def test_coder_auth_error_with_rollback(self, ws: Path) -> None:
        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return calls(tc("write_file", path="a.py", content="x = 1\n"))
            return NIMAuthError("revoked", status_code=403)

        report, _ = await run_forge(
            FakeClient(chat=chat, json=[PLAN_JSON]), ws, rollback_on_fail=True
        )
        assert report.status == "error" and report.rolled_back and not (ws / "a.py").exists()

    async def test_llm_outage_during_debugging_gives_up_cleanly(self, ws: Path) -> None:
        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return calls(
                    tc("write_file", path="fizzbuzz.py", content=FIZZBUZZ_BUGGY),
                    tc("write_file", path="test_fizzbuzz.py", content=FIZZBUZZ_TESTS),
                )
            return NIMResponseError("HTTP 502")

        client = FakeClient(chat=chat, json=[PLAN_JSON])
        report, _ = await run_forge(client, ws)
        assert report.status == "failed" and "unavailable" in report.ended_reason
        assert any("coder LLM call failed" in w for w in report.warnings)

    async def test_phase_crash_still_produces_a_report(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nimlab.forge import phases

        async def explode(self: Any, ctx: Any) -> State:
            raise RuntimeError("bug in verify")

        monkeypatch.setattr(phases.VerifyPhase, "run", explode)
        report, _ = await run_forge(FakeClient(chat=[say("hi")], json=[PLAN_JSON]), ws)
        assert report.status == "error"
        assert any("verify phase crashed: RuntimeError: bug in verify" in e for e in report.errors)
        assert (ws / ".forge" / REPORT_JSON).exists()

    async def test_optional_phase_crash_is_survivable(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nimlab.forge import phases

        async def explode(self: Any, ctx: Any) -> State:
            raise RuntimeError("reviewer bug")

        monkeypatch.setattr(phases.ReviewPhase, "run", explode)
        report, _ = await run_forge(FakeClient(**fizzbuzz_script()), ws)
        assert report.status == "success" and any(
            "review phase crashed" in e for e in report.errors
        )

    async def test_illegal_transition_is_a_loud_bug(
        self, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nimlab.forge import phases
        from nimlab.forge.context import IllegalTransition

        async def jump(self: Any, ctx: Any) -> State:
            return State.REVIEW

        monkeypatch.setattr(phases.PlanPhase, "run", jump)
        with pytest.raises(IllegalTransition):
            await run_forge(FakeClient(json=[PLAN_JSON]), ws)

    async def test_empty_task_rejected(self, ws: Path) -> None:
        with pytest.raises(ValueError, match="task"):
            await Forge(FakeClient()).run("   ", ws)

    async def test_sandbox_blocks_malicious_coder_actions_during_a_run(
        self, ws: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.txt"

        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return calls(
                    tc("write_file", path="../outside.txt", content="pwned"),
                    tc("write_file", path=str(outside), content="pwned"),
                    tc("run_python", script="../outside.py"),
                    tc("write_file", path=".forge/forge_report.json", content="{}"),
                    tc("write_file", path="ok.py", content="x = 1\n"),
                )
            return say("done")

        client = FakeClient(chat=chat, json=[PLAN_JSON])
        report, _ = await run_forge(client, ws, max_iters=0)
        assert not outside.exists()
        assert (ws / "ok.py").exists()
        assert [c.path for c in report.files_changed] == ["ok.py"]
        json.loads((ws / ".forge" / REPORT_JSON).read_text())  # the report was not clobbered

    async def test_tests_removed_by_coder_raise_a_warning(self, ws: Path) -> None:
        def chat(call: FakeCall) -> Any:
            if call.index == 0:
                return calls(
                    tc("write_file", path="m.py", content="VALUE = 1\n"),
                    tc("write_file", path="test_m.py",
                       content="from m import VALUE\n\ndef test_a():\n    assert VALUE == 1\n\n"
                               "def test_b():\n    assert VALUE == 2\n"),
                )  # fmt: skip
            if call.messages[-1]["role"] == "tool":
                return say("fixed")
            return calls(
                tc(
                    "write_file",
                    path="test_m.py",
                    content="from m import VALUE\n\ndef test_a():\n    assert VALUE == 1\n",
                )
            )  # "fixes" the failure by deleting the failing test

        script = {"chat": chat, "json": [PLAN_JSON, APPROVE_JSON]}
        report, _ = await run_forge(FakeClient(**script), ws)
        assert report.status == "success"
        assert any("test count dropped from 2 to 1" in w for w in report.warnings)


async def test_workspace_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        await Forge(FakeClient(json=[PLAN_JSON])).run("task", target, dry_run=True)


async def test_concurrent_runs_do_not_interfere(tmp_path: Path) -> None:
    dirs = [tmp_path / "one", tmp_path / "two"]
    for d in dirs:
        d.mkdir()
    clients = [FakeClient(**fizzbuzz_script()) for _ in dirs]
    reports = await asyncio.gather(*(run_forge(c, d) for c, d in zip(clients, dirs, strict=True)))
    for (report, _), d, client in zip(reports, dirs, clients, strict=True):
        assert report.status == "success" and report.workspace == str(d.resolve())
        assert client.unexpected == []
        assert report.usage["total"]["calls"] == len(client.calls) == 9
        assert sorted(p.name for p in d.iterdir()) == [".forge", "fizzbuzz.py", "test_fizzbuzz.py"]
