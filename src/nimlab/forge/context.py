"""Run state shared by the phases: the :class:`State` machine definition and :class:`RunContext`."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .agent import LoopResult
from .events import EventBus
from .models import (
    ForgeConfig,
    ForgeReport,
    ModelAssignment,
    Plan,
    Review,
    VerifyRound,
)
from .progress import PhaseTimer, StallDetector, UsageProbe
from .pytest_parser import PytestReport
from .sandbox import Sandbox
from .snapshot import Snapshot, WorkspaceDiff
from .tools import WorkspaceTools
from .workspace import WorkspaceSummary

__all__ = ["TRANSITIONS", "IllegalTransition", "RunContext", "State"]


class State(str, Enum):
    """States of the orchestrator's state machine (also the phase names in events/timings)."""

    INIT = "init"
    EXPLORE = "explore"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    DEBUG = "debug"
    ESCALATE = "escalate"
    REVIEW = "review"
    FIX = "fix"
    REPORT = "report"
    DONE = "done"


#: Allowed transitions. The orchestrator refuses anything else (a bug in a phase, not user error).
TRANSITIONS: dict[State, frozenset[State]] = {
    State.INIT: frozenset({State.EXPLORE, State.REPORT}),
    State.EXPLORE: frozenset({State.PLAN, State.REPORT}),
    State.PLAN: frozenset({State.IMPLEMENT, State.REPORT}),  # REPORT directly on --dry-run
    State.IMPLEMENT: frozenset({State.VERIFY, State.REPORT}),
    State.VERIFY: frozenset({State.REVIEW, State.DEBUG, State.ESCALATE, State.REPORT}),
    State.DEBUG: frozenset({State.VERIFY, State.REPORT}),
    State.ESCALATE: frozenset({State.DEBUG, State.REPORT}),
    State.REVIEW: frozenset({State.FIX, State.REPORT}),
    State.FIX: frozenset({State.VERIFY, State.REPORT}),
    State.REPORT: frozenset({State.DONE}),
    State.DONE: frozenset(),
}


class IllegalTransition(RuntimeError):
    """A phase returned a next state the transition table does not allow."""


@dataclass
class RunContext:
    """Mutable state of one Forge run, passed to every phase."""

    task: str
    workspace: Path  # as requested by the caller (used in the report)
    config: ForgeConfig
    models: ModelAssignment
    client: Any
    sandbox: Sandbox
    tools: WorkspaceTools
    bus: EventBus
    timer: PhaseTimer
    probe: UsageProbe
    detector: StallDetector
    mock: bool = False
    started_at: float = 0.0
    started_monotonic: float = 0.0

    snapshot: Snapshot | None = None
    summary: WorkspaceSummary | None = None
    plan: Plan | None = None
    coder_messages: list[dict[str, Any]] = field(default_factory=list)
    coder_final: str = ""
    last_loop: LoopResult | None = None
    llm_error_streak: int = 0

    last_tests: PytestReport | None = None
    test_rounds: list[VerifyRound] = field(default_factory=list)
    max_tests_seen: int = 0
    iterations: int = 0  # debug + review-fix rounds used (shared budget: config.max_iters)
    escalations: int = 0
    pending_rca: str | None = None
    root_causes: list[str] = field(default_factory=list)

    review: Review | None = None
    review_rounds: int = 0
    post_review: bool = False  # set once a review fix round ran: no second review

    diff: WorkspaceDiff | None = None
    rolled_back: bool = False
    ended_reason: str = ""
    fatal_error: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report: ForgeReport | None = None

    def warn(self, message: str, phase: str = "") -> None:
        self.warnings.append(message)
        self.bus.emit("warning", phase, message)

    def error(self, message: str, phase: str = "") -> None:
        self.errors.append(message)
        self.bus.emit("error", phase, message)

    def note_loop(self, loop: LoopResult, phase: str) -> None:
        """Record the outcome of a coder turn."""
        self.last_loop = loop
        if loop.final_text:
            self.coder_final = loop.final_text
        if loop.ended_reason == "llm_error":
            self.llm_error_streak += 1
            self.warnings.append(f"{phase}: coder LLM call failed: {loop.error}")
        else:
            self.llm_error_streak = 0
        if loop.tool_calls and loop.failed_tool_calls == loop.tool_calls:
            self.warnings.append(f"{phase}: every tool call of the coder failed")
