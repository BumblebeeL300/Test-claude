"""Rich rendering: a live phase/event view while Forge runs, and the final report."""

from __future__ import annotations

from collections import deque
from types import TracebackType

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from nimlab.core import get_console

from .events import ForgeEvent
from .models import ForgeReport

__all__ = ["ForgeDisplay", "print_report", "render_report"]

_PIPELINE = ("explore", "plan", "implement", "verify", "debug", "review", "report")
_STATUS_STYLE = {"success": "green", "failed": "red", "dry_run": "cyan", "error": "bold red"}
_QUIET_KINDS = {"llm", "tool_result", "phase_end", "state", "run_start", "run_end"}


class ForgeDisplay:
    """Event subscriber that shows progress.

    On a terminal it renders a live table (phases, counters, recent events); otherwise (pipes,
    CI, tests) it prints one plain line per notable event. ``quiet=True`` prints nothing.
    """

    def __init__(
        self, console: Console | None = None, *, live: bool | None = None, quiet: bool = False
    ) -> None:
        self.console = console or get_console()
        self.quiet = quiet
        self.live_mode = (self.console.is_terminal if live is None else live) and not quiet
        self._live: Live | None = None
        self._status: dict[str, str] = {}
        self._current = ""
        self._recent: deque[str] = deque(maxlen=8)
        self._tests = ""
        self._counters = {"tools": 0, "llm": 0, "tokens": 0}

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> ForgeDisplay:
        if self.live_mode:
            self._live = Live(
                self._render(), console=self.console, refresh_per_second=8, transient=True
            )
            self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    # ------------------------------------------------------------------ events

    def __call__(self, event: ForgeEvent) -> None:
        self._track(event)
        line = self._format(event)
        if line is None or self.quiet:
            return
        if self.live_mode and self._live is not None:
            self._recent.append(line)
            self._live.update(self._render())
        elif not self.live_mode:
            self.console.print(line, highlight=False)

    def _track(self, event: ForgeEvent) -> None:
        if event.kind == "phase_start":
            self._current = event.phase
            self._status[event.phase] = "running"
        elif event.kind == "phase_end" and self._status.get(event.phase) == "running":
            self._status[event.phase] = "done"
        elif event.kind == "tests":
            self._tests = event.message
        elif event.kind == "tool_call":
            self._counters["tools"] += 1
        elif event.kind == "llm":
            self._counters["llm"] += 1
            self._counters["tokens"] += int(event.data.get("tokens_in", 0)) + int(
                event.data.get("tokens_out", 0)
            )

    def _format(self, event: ForgeEvent) -> str | None:
        if event.kind in _QUIET_KINDS:
            return None
        msg = escape(event.message)
        phase = f"[dim]{event.phase:<9}[/]" if event.phase else " " * 9
        if event.kind == "phase_start":
            return f"[bold cyan]> {escape(event.phase)}[/]"
        if event.kind == "tool_call":
            return f"  {phase} [dim]{msg}[/]"
        if event.kind == "warning":
            return f"  {phase} [yellow]warning:[/] {msg}"
        if event.kind == "error":
            return f"  {phase} [red]error:[/] {msg}"
        if event.kind == "tests":
            color = "green" if event.data.get("ok") else "red"
            return f"  {phase} [{color}]{msg}[/]"
        if event.kind in ("stall", "escalate"):
            return f"  {phase} [magenta]{event.kind}:[/] {msg}"
        if event.kind in ("review", "plan", "rollback"):
            return f"  {phase} [bold]{event.kind}:[/] {msg}"
        return f"  {phase} {msg}"

    # ------------------------------------------------------------------ live view

    def _render(self) -> RenderableType:
        table = Table.grid(padding=(0, 2))
        for name in _PIPELINE:
            state = self._status.get(name)
            if name == self._current and state == "running":
                mark = "[bold yellow]>[/]"
            elif state == "done":
                mark = "[green]ok[/]"
            else:
                mark = "[dim]-[/]"
            table.add_row(mark, name)
        stats = (
            f"tools {self._counters['tools']}  llm calls {self._counters['llm']}  "
            f"tokens {self._counters['tokens']:,}"
        )
        parts: list[RenderableType] = [table, Text(stats, style="dim")]
        if self._tests:
            parts.append(Text(self._tests))
        parts.extend(Text.from_markup(line) for line in self._recent)
        return Panel(Group(*parts), title="Forge", border_style="cyan", expand=False)


def render_report(report: ForgeReport, *, show_diff: bool = False) -> list[RenderableType]:
    """Rich renderables summarising a :class:`ForgeReport`."""
    color = _STATUS_STYLE.get(report.status, "white")
    head = Text()
    head.append(f"{report.status.upper()}", style=f"bold {color}")
    head.append(f"  {report.duration_s:.1f}s  {report.iterations} fix round(s)")
    if report.escalations:
        head.append(f"  {report.escalations} escalation(s)")
    if report.rolled_back:
        head.append("  rolled back", style="yellow")
    body: list[RenderableType] = [head, Text(report.summary.strip())]
    if report.models:
        body.append(
            Text(
                f"planner {report.models.planner} | coder {report.models.coder} | "
                f"reviewer {report.models.reviewer}",
                style="dim",
            )
        )
    out: list[RenderableType] = [Panel(Group(*body), title="Forge report", border_style=color)]

    if report.plan and report.plan.steps:
        plan = Table(title="Plan", title_justify="left", show_header=False)
        plan.add_column(justify="right", style="dim")
        plan.add_column(overflow="fold")
        for step in report.plan.steps:
            files = f"  [dim]{escape(', '.join(step.files))}[/]" if step.files else ""
            plan.add_row(str(step.id), escape(step.description) + files)
        out.append(plan)

    if report.test_rounds:
        rounds = Table(title="Test rounds", title_justify="left")
        for col in ("round", "result", "passed", "failed", "errors"):
            rounds.add_column(col, justify="right" if col != "result" else "left")
        for r in report.test_rounds:
            rounds.add_row(
                str(r.index),
                "[green]pass[/]" if r.ok else "[red]FAIL[/]",
                str(r.passed),
                str(r.failed),
                str(r.errors),
            )
        out.append(rounds)

    if report.review:
        issues = "\n".join(
            f"  [{i.severity}] {i.file + ': ' if i.file else ''}{i.description}"
            for i in report.review.issues[:8]
        )
        out.append(
            Text(f"Review: {report.review.verdict}. {report.review.summary}\n{issues}".rstrip())
        )

    if report.files_changed:
        files = Table(title="Files changed", title_justify="left")
        files.add_column("file")
        files.add_column("status")
        files.add_column("+", justify="right", style="green")
        files.add_column("-", justify="right", style="red")
        for c in report.files_changed:
            files.add_row(escape(c.path), c.status, str(c.additions), str(c.deletions))
        out.append(files)

    if report.timings:
        timings = Table(title="Phase timings", title_justify="left")
        for col in ("phase", "seconds", "runs", "llm calls", "tokens in", "tokens out"):
            timings.add_column(col, justify="left" if col == "phase" else "right")
        for t in report.timings:
            timings.add_row(
                t.phase,
                f"{t.seconds:.2f}",
                str(t.runs),
                str(t.llm_calls),
                f"{t.tokens_in:,}",
                f"{t.tokens_out:,}",
            )
        out.append(timings)

    for item in report.warnings[:6]:
        out.append(Text(f"warning: {item}", style="yellow"))
    for item in report.errors[:6]:
        out.append(Text(f"error: {item}", style="red"))
    if show_diff and report.diff:
        out.append(Syntax(report.diff, "diff", theme="ansi_dark", word_wrap=True))
    return out


def print_report(console: Console, report: ForgeReport, *, show_diff: bool = False) -> None:
    for renderable in render_report(report, show_diff=show_diff):
        console.print(renderable)
