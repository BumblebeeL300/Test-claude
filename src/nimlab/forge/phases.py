"""The pipeline phases. Each phase is a class with ``async run(ctx) -> next State``.

``Init -> Explore -> Plan -> Implement -> Verify <-> Debug (<- Escalate) -> Review
(-> Fix -> Verify) -> Report``. The :class:`~nimlab.forge.orchestrator.Orchestrator` owns the
order; phases only decide which state comes next.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import ClassVar

from nimlab.core import NIMAuthError, NIMError, strip_think

from .agent import run_tool_loop
from .context import RunContext, State
from .models import ForgeReport, Plan, Review, RunStatus, Step, VerifyRound
from .prompts import (
    CODER_SYSTEM,
    PLANNER_SYSTEM,
    RCA_SYSTEM,
    REVIEWER_SYSTEM,
    fix_prompt,
    implement_prompt,
    plan_prompt,
    rca_prompt,
    review_fix_prompt,
    review_prompt,
)
from .sandbox import SandboxError
from .snapshot import Snapshot
from .tools import clip
from .workspace import summarize_workspace

__all__ = [
    "DebugPhase",
    "EscalatePhase",
    "ExplorePhase",
    "FixPhase",
    "ImplementPhase",
    "InitPhase",
    "Phase",
    "PlanPhase",
    "ReportPhase",
    "ReviewPhase",
    "VerifyPhase",
    "default_phases",
]


class Phase(ABC):
    """One step of the pipeline."""

    state: ClassVar[State]

    @property
    def name(self) -> str:
        return self.state.value

    @abstractmethod
    async def run(self, ctx: RunContext) -> State:
        """Do the work and return the next state."""


class InitPhase(Phase):
    """Take the pre-run snapshot (skipped for dry runs, which never write)."""

    state = State.INIT

    async def run(self, ctx: RunContext) -> State:
        if not ctx.config.dry_run:
            ctx.snapshot = await asyncio.to_thread(Snapshot.take, ctx.sandbox.root)
            ctx.bus.emit("info", self.name, f"snapshot taken ({len(ctx.snapshot.files)} files)")
        return State.EXPLORE


class ExplorePhase(Phase):
    """Summarise the workspace (tree, key files, Python outline) for the planner and the coder."""

    state = State.EXPLORE

    async def run(self, ctx: RunContext) -> State:
        ctx.summary = await asyncio.to_thread(summarize_workspace, ctx.sandbox)
        n = len(ctx.summary.files)
        ctx.bus.emit(
            "info",
            self.name,
            "empty workspace"
            if ctx.summary.empty
            else f"{n} file(s), {len(ctx.summary.test_files)} test file(s)",
            files=n,
        )
        return State.PLAN


def _normalise_plan(plan: Plan, task: str) -> Plan:
    goal = plan.goal.strip() or task.strip().splitlines()[0][:200]
    steps = [s for s in plan.steps if s.description.strip()][:12]
    if not steps:
        steps = [Step(id=1, description=task.strip()[:500])]
    for i, step in enumerate(steps, start=1):
        step.id = i
    return plan.model_copy(update={"goal": goal, "steps": steps})


class PlanPhase(Phase):
    """Ask the REASONING model for a structured :class:`Plan`."""

    state = State.PLAN

    async def run(self, ctx: RunContext) -> State:
        summary = ctx.summary.render() if ctx.summary else "(unknown)"
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": plan_prompt(ctx.task, summary)},
        ]
        plan: Plan
        try:
            plan = await ctx.client.chat_json(
                ctx.models.planner,
                messages,
                Plan,
                temperature=0.2,
                max_tokens=ctx.config.planner_max_tokens,
            )
        except NIMAuthError:
            raise
        except (NIMError, ValueError) as exc:
            ctx.warn(
                f"planner failed ({type(exc).__name__}: {exc}); using a single-step plan", self.name
            )
            plan = Plan(goal="", steps=[])
        ctx.plan = _normalise_plan(plan, ctx.task)
        ctx.bus.emit(
            "plan",
            self.name,
            f"{len(ctx.plan.steps)} step(s): {ctx.plan.goal}",
            steps=[s.description for s in ctx.plan.steps],
        )
        if ctx.config.dry_run:
            ctx.ended_reason = "dry_run"
            return State.REPORT
        return State.IMPLEMENT


class ImplementPhase(Phase):
    """The CODER model writes tests and code with the workspace tools."""

    state = State.IMPLEMENT

    async def run(self, ctx: RunContext) -> State:
        summary = ctx.summary.render() if ctx.summary else "(unknown)"
        ctx.coder_messages = [
            {"role": "system", "content": CODER_SYSTEM},
            {"role": "user", "content": implement_prompt(ctx.task, ctx.plan, summary)},
        ]
        loop = await run_tool_loop(
            client=ctx.client,
            model=ctx.models.coder,
            messages=ctx.coder_messages,
            tools=ctx.tools,
            bus=ctx.bus,
            phase=self.name,
            max_steps=ctx.config.max_tool_steps,
            max_tokens=ctx.config.coder_max_tokens,
        )
        ctx.note_loop(loop, self.name)
        return State.VERIFY


class VerifyPhase(Phase):
    """Run the whole test suite and decide: done, debug, escalate or give up."""

    state = State.VERIFY

    async def run(self, ctx: RunContext) -> State:
        cfg = ctx.config
        report = await ctx.tools.run_tests()
        ctx.last_tests = report
        ctx.detector.record(report.signature)
        index = len(ctx.test_rounds) + 1
        ctx.test_rounds.append(
            VerifyRound(
                index=index,
                summary=report.summary_line,
                passed=report.passed,
                failed=report.failed,
                errors=report.errors,
                ok=report.ok,
                signature=report.signature,
                failing=[f.nodeid for f in report.failures][:20],
            )
        )
        ctx.bus.emit(
            "tests",
            self.name,
            f"round {index}: {report.summary_line}",
            ok=report.ok,
            passed=report.passed,
            failed=report.failed,
            errors=report.errors,
            iteration=ctx.iterations,
        )
        if report.ok:
            if ctx.max_tests_seen and report.total < ctx.max_tests_seen:
                ctx.warn(
                    f"test count dropped from {ctx.max_tests_seen} to {report.total}; "
                    "tests may have been removed",
                    self.name,
                )
            ctx.max_tests_seen = max(ctx.max_tests_seen, report.total)
            ctx.ended_reason = "tests_pass"
            return State.REPORT if ctx.post_review else State.REVIEW
        ctx.max_tests_seen = max(ctx.max_tests_seen, report.total)

        if ctx.llm_error_streak >= 2:
            ctx.ended_reason = "llm_unavailable"
            return State.REPORT
        if ctx.iterations >= cfg.max_iters:
            ctx.ended_reason = "max_iterations"
            return State.REPORT
        if ctx.detector.stalled:
            ctx.bus.emit(
                "stall",
                self.name,
                f"same failures {ctx.detector.repeats}x ({report.signature})",
                repeats=ctx.detector.repeats,
                signature=report.signature,
            )
            if ctx.escalations < cfg.max_escalations:
                return State.ESCALATE
            ctx.ended_reason = "no_progress"
            return State.REPORT
        return State.DEBUG


class DebugPhase(Phase):
    """Feed the condensed failure report (plus any root-cause analysis) back to the coder."""

    state = State.DEBUG

    async def run(self, ctx: RunContext) -> State:
        assert ctx.last_tests is not None
        ctx.iterations += 1
        prompt = fix_prompt(
            ctx.last_tests.condensed(), ctx.iterations, ctx.config.max_iters, ctx.pending_rca
        )
        ctx.pending_rca = None
        ctx.bus.emit("info", self.name, f"fix round {ctx.iterations}/{ctx.config.max_iters}")
        ctx.coder_messages.append({"role": "user", "content": prompt})
        loop = await run_tool_loop(
            client=ctx.client,
            model=ctx.models.coder,
            messages=ctx.coder_messages,
            tools=ctx.tools,
            bus=ctx.bus,
            phase=self.name,
            max_steps=ctx.config.max_tool_steps,
            max_tokens=ctx.config.coder_max_tokens,
        )
        ctx.note_loop(loop, self.name)
        return State.VERIFY


class EscalatePhase(Phase):
    """Ask the REASONING model for a root-cause analysis when the coder is going in circles."""

    state = State.ESCALATE

    def _sources(self, ctx: RunContext) -> dict[str, str]:
        paths: list[str] = []
        if ctx.last_tests:
            for failure in ctx.last_tests.failures:
                for fr in failure.frames:
                    paths.append(fr.file)
                paths.append(failure.nodeid.split("::", 1)[0])
        out: dict[str, str] = {}
        for path in dict.fromkeys(p for p in paths if p.endswith(".py")):
            if len(out) >= 5:
                break
            try:
                out[path] = clip(ctx.sandbox.read_text(path), 3500)
            except SandboxError:
                continue
        return out

    async def run(self, ctx: RunContext) -> State:
        assert ctx.last_tests is not None
        ctx.escalations += 1
        model = ctx.models.planner
        ctx.bus.emit(
            "escalate", self.name, f"root-cause analysis #{ctx.escalations} by {model}", model=model
        )
        history = [f"round {r.index}: {r.summary}" for r in ctx.test_rounds[-5:]]
        messages = [
            {"role": "system", "content": RCA_SYSTEM},
            {
                "role": "user",
                "content": rca_prompt(
                    ctx.task,
                    ctx.plan,
                    ctx.last_tests.condensed(),
                    history,
                    self._sources(ctx),
                    ctx.coder_final,
                ),
            },
        ]
        try:
            res = await ctx.client.chat(
                model, messages, temperature=0.1, max_tokens=ctx.config.planner_max_tokens
            )
        except NIMAuthError:
            raise
        except NIMError as exc:
            ctx.warn(f"root-cause analysis failed: {exc}", self.name)
            return State.DEBUG
        text, _ = strip_think(res.content or "")
        text = text.strip()
        if text:
            ctx.pending_rca = text
            ctx.root_causes.append(text)
        return State.DEBUG


class ReviewPhase(Phase):
    """An independent model reviews the unified diff."""

    state = State.REVIEW

    async def run(self, ctx: RunContext) -> State:
        cfg = ctx.config
        if ctx.snapshot is None:
            return State.REPORT
        diff = await asyncio.to_thread(ctx.snapshot.diff)
        if diff.empty:
            ctx.bus.emit("info", self.name, "no changes to review")
            return State.REPORT
        tests = ctx.last_tests.condensed() if ctx.last_tests else "(no test run)"
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM},
            {
                "role": "user",
                "content": review_prompt(
                    ctx.task, ctx.plan, tests, clip(diff.unified, cfg.max_diff_chars_for_review)
                ),
            },
        ]
        try:
            review: Review = await ctx.client.chat_json(
                ctx.models.reviewer,
                messages,
                Review,
                temperature=0.1,
                max_tokens=cfg.reviewer_max_tokens,
            )
        except NIMAuthError:
            raise
        except (NIMError, ValueError) as exc:
            ctx.warn(f"review unavailable ({type(exc).__name__}: {exc})", self.name)
            return State.REPORT
        ctx.review = review
        blocking = review.blocking_issues
        ctx.bus.emit(
            "review",
            self.name,
            f"{review.verdict}: {len(review.issues)} issue(s), {len(blocking)} blocking",
            verdict=review.verdict,
            issues=len(review.issues),
            blocking=len(blocking),
        )
        if review.verdict == "request_changes" and blocking:
            if ctx.review_rounds < cfg.max_review_rounds and ctx.iterations < cfg.max_iters:
                return State.FIX
            ctx.warn("reviewer requested changes but no fix budget is left", self.name)
        return State.REPORT


class FixPhase(Phase):
    """One more coder round to address blocking review findings."""

    state = State.FIX

    async def run(self, ctx: RunContext) -> State:
        assert ctx.review is not None
        ctx.review_rounds += 1
        ctx.iterations += 1
        ctx.post_review = True
        ctx.bus.emit(
            "info",
            self.name,
            f"addressing {len(ctx.review.blocking_issues)} blocking review issue(s)",
        )
        ctx.coder_messages.append({"role": "user", "content": review_fix_prompt(ctx.review)})
        loop = await run_tool_loop(
            client=ctx.client,
            model=ctx.models.coder,
            messages=ctx.coder_messages,
            tools=ctx.tools,
            bus=ctx.bus,
            phase=self.name,
            max_steps=ctx.config.max_tool_steps,
            max_tokens=ctx.config.coder_max_tokens,
        )
        ctx.note_loop(loop, self.name)
        return State.VERIFY


def _status(ctx: RunContext) -> RunStatus:
    if ctx.fatal_error:
        return "error"
    if ctx.config.dry_run:
        return "dry_run"
    if ctx.last_tests is not None and ctx.last_tests.ok:
        return "success"
    return "failed"


_REASONS = {
    "tests_pass": "all tests pass",
    "max_iterations": "ran out of fix rounds (--max-iters)",
    "no_progress": "no progress: the same failures kept repeating after root-cause analysis",
    "llm_unavailable": "the coder model was unavailable",
    "dry_run": "dry run (nothing was written)",
}


class ReportPhase(Phase):
    """Compute the diff, roll back if asked, and write ``.forge/forge_report.{md,json}``."""

    state = State.REPORT

    def _summary(self, ctx: RunContext, status: RunStatus, report: ForgeReport) -> str:
        n = len(report.files_changed)
        adds = sum(c.additions for c in report.files_changed)
        dels = sum(c.deletions for c in report.files_changed)
        tests = ctx.last_tests.summary_line if ctx.last_tests else "not run"
        if status == "dry_run":
            steps = len(ctx.plan.steps) if ctx.plan else 0
            text = f"Dry run: planned {steps} step(s). No files were written."
        elif status == "success":
            text = (
                f"Done. Tests: {tests} after {ctx.iterations} fix round(s). "
                f"{n} file(s) changed (+{adds}/-{dels})."
            )
        elif status == "error":
            text = f"Run aborted: {ctx.fatal_error}"
        else:
            reason = _REASONS.get(ctx.ended_reason, ctx.ended_reason or "tests did not pass")
            text = f"Not finished: {reason}. Tests: {tests}. {n} file(s) changed (+{adds}/-{dels})."
        if ctx.rolled_back:
            text += " The workspace was rolled back to its initial state."
        if ctx.review is not None:
            text += f" Review: {ctx.review.verdict}."
        if ctx.mock:
            text += " (Mock mode: model output is placeholder text, so no real work was done.)"
        if ctx.coder_final and status != "dry_run":
            text += "\n\nCoder notes: " + clip(ctx.coder_final, 800)
        return text

    async def run(self, ctx: RunContext) -> State:
        status = _status(ctx)
        diff = None
        if ctx.snapshot is not None:
            diff = await asyncio.to_thread(ctx.snapshot.diff)
            ctx.diff = diff
            if ctx.config.rollback_on_fail and status in ("failed", "error") and not diff.empty:
                result = await asyncio.to_thread(ctx.snapshot.restore)
                ctx.rolled_back = True
                ctx.bus.emit(
                    "rollback",
                    self.name,
                    f"restored {len(result.restored)} file(s), removed {len(result.removed)}",
                    restored=len(result.restored),
                    removed=len(result.removed),
                )
                if result.unrestorable:
                    ctx.warn(f"could not restore: {', '.join(result.unrestorable[:5])}", self.name)
        report = ForgeReport(
            task=ctx.task,
            workspace=str(ctx.workspace),
            status=status,
            ended_reason=_REASONS.get(ctx.ended_reason, ctx.ended_reason),
            mock=ctx.mock,
            dry_run=ctx.config.dry_run,
            models=ctx.models,
            plan=ctx.plan,
            iterations=ctx.iterations,
            escalations=ctx.escalations,
            review_rounds=ctx.review_rounds,
            rolled_back=ctx.rolled_back,
            tests=ctx.last_tests,
            test_rounds=ctx.test_rounds,
            review=ctx.review,
            files_changed=diff.changes if diff else [],
            diff=diff.unified if diff else "",
            root_causes=ctx.root_causes,
            warnings=list(dict.fromkeys(ctx.warnings)),
            errors=ctx.errors,
            started_at=ctx.started_at,
        )
        report.summary = self._summary(ctx, status, report)
        report.timings = ctx.timer.snapshot()
        report.usage = ctx.probe.delta()
        report.duration_s = round(time.monotonic() - ctx.started_monotonic, 3)
        report.events = [e.to_dict() for e in ctx.bus.history[-400:]]
        ctx.report = report
        if ctx.config.write_report and not ctx.config.dry_run:
            try:
                paths = await asyncio.to_thread(report.write, ctx.sandbox.root / ".forge")
                ctx.bus.emit("info", self.name, f"report written to {paths['markdown'].parent}")
            except OSError as exc:
                ctx.warn(f"could not write report: {exc}", self.name)
                report.warnings.append(f"could not write report: {exc}")
        return State.DONE


def default_phases() -> dict[State, Phase]:
    """The standard phase set, keyed by the state each one handles."""
    phases: list[Phase] = [
        InitPhase(),
        ExplorePhase(),
        PlanPhase(),
        ImplementPhase(),
        VerifyPhase(),
        DebugPhase(),
        EscalatePhase(),
        ReviewPhase(),
        FixPhase(),
        ReportPhase(),
    ]
    return {p.state: p for p in phases}
