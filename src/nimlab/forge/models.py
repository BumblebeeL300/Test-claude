"""Typed data model for Forge: plan, review, configuration and the final report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .pytest_parser import PytestReport
from .snapshot import FileChange

__all__ = [
    "REPORT_JSON",
    "REPORT_MD",
    "ForgeConfig",
    "ForgeReport",
    "Issue",
    "ModelAssignment",
    "PhaseTiming",
    "Plan",
    "Review",
    "RunStatus",
    "Step",
    "VerifyRound",
]

REPORT_JSON = "forge_report.json"
REPORT_MD = "forge_report.md"

RunStatus = Literal["success", "failed", "dry_run", "error"]


def _as_list(value: Any) -> Any:
    """Models sometimes return a bare string where a list is expected."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return value


class Step(BaseModel):
    """One implementation step of a plan."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(default=1, description="1-based step number")
    description: str = Field(default="", description="What to do in this step")
    files: list[str] = Field(default_factory=list, description="Files to create or modify")

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> Any:
        if isinstance(v, str):
            digits = "".join(ch for ch in v if ch.isdigit())
            return int(digits) if digits else 1
        return v

    @field_validator("files", mode="before")
    @classmethod
    def _files_list(cls, v: Any) -> Any:
        return _as_list(v)


class Plan(BaseModel):
    """The planner's output: what will be built and how it will be tested."""

    model_config = ConfigDict(extra="ignore")

    goal: str = Field(description="One-sentence restatement of the task")
    steps: list[Step] = Field(default_factory=list, description="Ordered implementation steps")
    test_strategy: str = Field(default="", description="How the work will be verified with pytest")
    risks: list[str] = Field(default_factory=list, description="Risks and edge cases to watch")

    @field_validator("risks", mode="before")
    @classmethod
    def _risks_list(cls, v: Any) -> Any:
        return _as_list(v)

    @field_validator("test_strategy", mode="before")
    @classmethod
    def _strategy_text(cls, v: Any) -> Any:
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return v if v is not None else ""


Severity = Literal["blocker", "major", "minor", "nit"]
_SEVERITY_ALIASES = {
    "critical": "blocker",
    "blocking": "blocker",
    "error": "blocker",
    "high": "major",
    "medium": "major",
    "moderate": "major",
    "warning": "minor",
    "low": "minor",
    "info": "nit",
    "style": "nit",
    "suggestion": "nit",
}


class Issue(BaseModel):
    """One code-review finding."""

    model_config = ConfigDict(extra="ignore")

    severity: Severity = Field(default="minor", description="blocker, major, minor or nit")
    file: str = Field(default="", description="File the issue is in")
    line: int | None = Field(default=None, description="Line number, if known")
    description: str = Field(default="", description="What is wrong")
    suggestion: str = Field(default="", description="How to fix it")

    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, v: Any) -> Any:
        if isinstance(v, str):
            low = v.strip().lower()
            return _SEVERITY_ALIASES.get(low, low)
        return v

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, v: Any) -> Any:
        if isinstance(v, str):
            return int(v) if v.strip().isdigit() else None
        return v

    @property
    def blocking(self) -> bool:
        return self.severity in ("blocker", "major")


class Review(BaseModel):
    """Verdict of the independent code review."""

    model_config = ConfigDict(extra="ignore")

    verdict: Literal["approve", "request_changes"] = Field(description="approve or request_changes")
    summary: str = Field(default="", description="Two or three sentence assessment")
    issues: list[Issue] = Field(default_factory=list, description="Findings, most severe first")

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v: Any) -> Any:
        if isinstance(v, str):
            low = v.strip().lower().replace("-", "_").replace(" ", "_")
            if low in {"approved", "lgtm", "accept", "accepted", "pass"}:
                return "approve"
            if low in {"changes_requested", "request_change", "reject", "rejected", "needs_work"}:
                return "request_changes"
            return low
        return v

    @property
    def blocking_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.blocking]


class ModelAssignment(BaseModel):
    """Which NIM model plays which role in a run."""

    planner: str
    coder: str
    reviewer: str


class ForgeConfig(BaseModel):
    """Tunable behaviour of a Forge run."""

    model_config = ConfigDict(extra="forbid")

    max_iters: int = Field(
        default=6, ge=0, description="Max debug/fix rounds after the first build"
    )
    dry_run: bool = False
    rollback_on_fail: bool = False
    planner: str | None = None
    coder: str | None = None
    reviewer: str | None = None

    command_timeout_s: float = Field(default=90.0, gt=0)
    max_output_bytes: int = Field(default=48_000, gt=0)
    max_tool_steps: int = Field(default=14, ge=1, description="Tool-call rounds per coder turn")
    coder_max_tokens: int = Field(default=4096, ge=256)
    planner_max_tokens: int = Field(default=4096, ge=256)
    reviewer_max_tokens: int = Field(default=2048, ge=256)

    stall_threshold: int = Field(
        default=2, ge=2, description="Identical failures that count as a stall"
    )
    max_escalations: int = Field(default=2, ge=0, description="Root-cause analyses allowed per run")
    max_review_rounds: int = Field(default=1, ge=0, description="Fix rounds after a review")
    max_diff_chars_for_review: int = Field(default=24_000, ge=1000)
    write_report: bool = True


class PhaseTiming(BaseModel):
    """Wall-clock time and LLM usage attributed to one phase (summed over repeats)."""

    phase: str
    seconds: float = 0.0
    runs: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class VerifyRound(BaseModel):
    """Result of one verification (pytest) pass."""

    index: int
    summary: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    ok: bool = False
    signature: str = ""
    failing: list[str] = Field(default_factory=list)


class ForgeReport(BaseModel):
    """Everything a run produced; serialised to ``.forge/forge_report.json``."""

    task: str
    workspace: str
    status: RunStatus
    ended_reason: str = ""
    summary: str = ""
    mock: bool = False
    dry_run: bool = False
    models: ModelAssignment | None = None
    plan: Plan | None = None
    iterations: int = 0
    escalations: int = 0
    review_rounds: int = 0
    rolled_back: bool = False
    tests: PytestReport | None = None
    test_rounds: list[VerifyRound] = Field(default_factory=list)
    review: Review | None = None
    files_changed: list[FileChange] = Field(default_factory=list)
    diff: str = ""
    root_causes: list[str] = Field(default_factory=list)
    timings: list[PhaseTiming] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: float = 0.0
    duration_s: float = 0.0
    events: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("success", "dry_run")

    # ------------------------------------------------------------------ persistence

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write forge_report.md, forge_report.json and changes.patch into ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out / REPORT_JSON,
            "markdown": out / REPORT_MD,
        }
        paths["json"].write_text(self.to_json() + "\n", encoding="utf-8")
        paths["markdown"].write_text(self.to_markdown(), encoding="utf-8")
        if self.diff:
            paths["patch"] = out / "changes.patch"
            paths["patch"].write_text(self.diff, encoding="utf-8")
        return paths

    @classmethod
    def load(cls, workspace_or_forge_dir: str | Path) -> ForgeReport:
        """Load the last report from ``<workspace>/.forge/`` (or from a ``.forge`` dir directly)."""
        base = Path(workspace_or_forge_dir)
        candidates = [base / REPORT_JSON, base / ".forge" / REPORT_JSON]
        for path in candidates:
            if path.is_file():
                return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        raise FileNotFoundError(
            f"no Forge report found under {base} (looked for .forge/{REPORT_JSON})"
        )

    # ------------------------------------------------------------------ rendering

    def to_markdown(self) -> str:
        return render_markdown(self)


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(r: ForgeReport) -> str:
    """Human-readable report."""
    badge = {"success": "SUCCESS", "failed": "FAILED", "dry_run": "DRY RUN", "error": "ERROR"}[
        r.status
    ]
    lines = [f"# Forge report: {badge}", "", f"**Task:** {r.task.strip()}", ""]
    meta = [
        f"workspace `{r.workspace}`",
        f"{r.duration_s:.1f}s",
        f"{r.iterations} debug round(s)",
        f"{r.escalations} escalation(s)",
    ]
    if r.mock:
        meta.append("MOCK MODE (placeholder model output)")
    if r.rolled_back:
        meta.append("workspace rolled back")
    lines += [" | ".join(meta), ""]
    if r.ended_reason:
        lines += [f"**Ended:** {r.ended_reason}", ""]
    lines += ["## Summary", "", r.summary.strip() or "(no summary)", ""]

    if r.models:
        lines += [
            "## Models",
            "",
            f"- planner: `{r.models.planner}`",
            f"- coder: `{r.models.coder}`",
            f"- reviewer: `{r.models.reviewer}`",
            "",
        ]
    if r.plan:
        lines += ["## Plan", "", f"**Goal:** {r.plan.goal}", ""]
        for step in r.plan.steps:
            files = f" (`{'`, `'.join(step.files)}`)" if step.files else ""
            lines.append(f"{step.id}. {step.description}{files}")
        if r.plan.test_strategy:
            lines += ["", f"**Test strategy:** {r.plan.test_strategy}"]
        if r.plan.risks:
            lines += ["", "**Risks:**"] + [f"- {x}" for x in r.plan.risks]
        lines.append("")

    if r.tests:
        lines += [
            "## Tests",
            "",
            f"Final run: **{r.tests.summary_line}** (exit code {r.tests.exit_code})",
            "",
        ]
        if not r.tests.ok and r.tests.failures:
            lines += ["```text", r.tests.condensed(), "```", ""]
    if r.test_rounds:
        lines += ["| round | result | passed | failed | errors |", "|---|---|---|---|---|"]
        for t in r.test_rounds:
            result = "pass" if t.ok else "FAIL"
            lines.append(f"| {t.index} | {result} | {t.passed} | {t.failed} | {t.errors} |")
        lines.append("")
    if r.root_causes:
        lines += ["## Root-cause analyses", ""]
        for i, rc in enumerate(r.root_causes, start=1):
            lines += [f"**RCA {i}**", "", rc.strip(), ""]

    if r.review:
        lines += [
            "## Review",
            "",
            f"Verdict: **{r.review.verdict}**",
            "",
            r.review.summary.strip(),
            "",
        ]
        for issue in r.review.issues:
            loc = f"{issue.file}:{issue.line}" if issue.file and issue.line else issue.file
            lines.append(f"- [{issue.severity}] {loc + ': ' if loc else ''}{issue.description}")
        lines.append("")

    if r.files_changed:
        lines += ["## Files changed", "", "| file | status | + | - |", "|---|---|---|---|"]
        for c in r.files_changed:
            lines.append(f"| `{_cell(c.path)}` | {c.status} | {c.additions} | {c.deletions} |")
        lines.append("")
    if r.diff:
        lines += ["## Diff", "", "```diff", r.diff.rstrip("\n"), "```", ""]

    if r.timings:
        lines += [
            "## Timings",
            "",
            "| phase | seconds | runs | LLM calls | tokens in | tokens out |",
            "|---|---|---|---|---|---|",
        ]
        for timing in r.timings:
            lines.append(
                f"| {timing.phase} | {timing.seconds:.2f} | {timing.runs} | {timing.llm_calls} "
                f"| {timing.tokens_in} | {timing.tokens_out} |"
            )
        lines.append("")
    total = (r.usage or {}).get("total")
    if total:
        lines += [
            "## Token usage",
            "",
            f"{total.get('calls', 0)} calls, {total.get('tokens_in', 0)} tokens in, "
            f"{total.get('tokens_out', 0)} tokens out.",
            "",
        ]
        models = (r.usage or {}).get("models") or {}
        if models:
            lines += ["| model | calls | tokens in | tokens out |", "|---|---|---|---|"]
            for name, s in models.items():
                lines.append(
                    f"| `{name}` | {s.get('calls', 0)} | {s.get('tokens_in', 0)} "
                    f"| {s.get('tokens_out', 0)} |"
                )
            lines.append("")
    if r.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in r.warnings] + [""]
    if r.errors:
        lines += ["## Errors", ""] + [f"- {e}" for e in r.errors] + [""]
    return "\n".join(lines).rstrip() + "\n"
