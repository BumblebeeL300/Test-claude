"""Live terminal view: a rich tree of models working in parallel, grouped by phase."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from nimlab.core.logging import get_console

from .events import (
    CallFinished,
    CallStarted,
    ConsensusComputed,
    CouncilEvent,
    PhaseStarted,
    RunFinished,
    RunStarted,
)
from .records import StepType

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
MODEL_STYLES = (
    "cyan",
    "magenta",
    "green",
    "yellow",
    "blue",
    "bright_red",
    "bright_cyan",
    "orange3",
)


@dataclass
class _Call:
    index: int
    model: str
    label: str
    type: StepType
    started: float
    status: str = "running"  # running | ok | error
    duration: float = 0.0
    tokens: int | None = None
    error: str | None = None


@dataclass
class _Phase:
    name: str
    calls: list[_Call] = field(default_factory=list)
    note: str = ""


def short_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


class CouncilLive:
    """Observer + context manager that renders the run as it happens.

    Use standalone (``with CouncilLive() as view: await council.run(..., observers=[view])``) or via
    ``Council.run(..., live=True)``. Rendering is a pure function of the events seen so far
    (:meth:`render`), so it can also be driven manually and snapshot-tested.
    """

    def __init__(self, console: Console | None = None, *, refresh_per_second: float = 10.0) -> None:
        self.console = console or get_console()
        self.refresh_per_second = refresh_per_second
        self.strategy = ""
        self.question = ""
        self.models: list[str] = []
        self.phases: list[_Phase] = []
        self.finished = False
        self.error: str | None = None
        self._calls: dict[int, _Call] = {}
        self._styles: dict[str, str] = {}
        self._t0 = time.perf_counter()
        self._elapsed_final: float | None = None
        self._live: Live | None = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> CouncilLive:
        self._live = Live(
            self,
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            transient=False,
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
            self._live.refresh()
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    # ------------------------------------------------------------------ events
    def __call__(self, event: CouncilEvent) -> None:
        if isinstance(event, RunStarted):
            self.strategy, self.question = event.strategy, event.question
            self.models = list(event.models)
            self._t0 = time.perf_counter()
            for model in self.models:
                self._style(model)
        elif isinstance(event, PhaseStarted):
            self._open_phase(event.phase)
        elif isinstance(event, CallStarted):
            phase = self._open_phase(event.phase)
            call = _Call(
                event.step_index, event.model, event.label, event.step_type, time.perf_counter()
            )
            self._calls[event.step_index] = call
            phase.calls.append(call)
            self._style(event.model)
        elif isinstance(event, CallFinished):
            call = self._calls.get(event.step_index)
            if call is not None:
                call.status = "ok" if event.ok else "error"
                call.duration = event.duration_s
                call.tokens = event.total_tokens
                call.error = event.error
        elif isinstance(event, ConsensusComputed):
            phase = self._open_phase(event.phase)
            outliers = f", outliers: {', '.join(event.outliers)}" if event.outliers else ""
            phase.note = f"agreement {event.agreement:.2f}{outliers}"
        elif isinstance(event, RunFinished):
            self.finished = True
            self.error = event.error
            self._elapsed_final = time.perf_counter() - self._t0

    def _open_phase(self, name: str) -> _Phase:
        name = name or "Run"
        for phase in self.phases:
            if phase.name == name:
                return phase
        phase = _Phase(name)
        self.phases.append(phase)
        return phase

    def _style(self, model: str) -> str:
        if model not in self._styles:
            self._styles[model] = MODEL_STYLES[len(self._styles) % len(MODEL_STYLES)]
        return self._styles[model]

    # ------------------------------------------------------------------ rendering
    @property
    def elapsed(self) -> float:
        return (
            self._elapsed_final
            if self._elapsed_final is not None
            else time.perf_counter() - self._t0
        )

    def __rich__(self) -> RenderableType:
        return self.render()

    def render(self, now: float | None = None) -> RenderableType:
        """Current state as a rich renderable (``now`` is injectable for deterministic tests)."""
        now = time.perf_counter() if now is None else now
        title = Text.assemble(
            ("Council", "bold"),
            " · ",
            (self.strategy or "starting", "bold cyan"),
        )
        if self.question:
            title.append("  ")
            title.append(_clip(self.question, 70), style="italic dim")
        root = Tree(title, guide_style="dim")
        for phase in self.phases:
            running = sum(c.status == "running" for c in phase.calls)
            done = sum(c.status != "running" for c in phase.calls)
            head = Text(phase.name, style="bold")
            if phase.calls:
                head.append(f"  {done}/{len(phase.calls)}", style="dim")
            if phase.note:
                head.append(f"  · {phase.note}", style="green" if not running else "dim")
            branch = root.add(head)
            for call in phase.calls:
                branch.add(self._call_line(call, now))
        if not self.phases:
            root.add(Text("waiting for the first model call…", style="dim"))
        return Group(root, self._footer())

    def _call_line(self, call: _Call, now: float) -> Text:
        line = Text()
        if call.status == "running":
            frame = SPINNER[int((now - call.started) * self.refresh_per_second) % len(SPINNER)]
            line.append(f"{frame} ", style="yellow")
        elif call.status == "ok":
            line.append("✔ ", style="green")
        else:
            line.append("✘ ", style="red")
        line.append(f"{call.label or call.type.value:<14}", style="dim")
        line.append(short_model(call.model), style=self._style(call.model))
        line.append(f"  {call.type.value}", style="dim")
        if call.status == "running":
            line.append(f"  {now - call.started:.1f}s", style="yellow")
        else:
            line.append(f"  {call.duration:.2f}s", style="dim")
            if call.tokens:
                line.append(f" · {call.tokens:,} tok", style="dim")
        if call.error:
            line.append(f"  {_clip(call.error, 60)}", style="red")
        return line

    def _footer(self) -> RenderableType:
        calls = list(self._calls.values())
        ok = sum(c.status == "ok" for c in calls)
        err = sum(c.status == "error" for c in calls)
        run = sum(c.status == "running" for c in calls)
        tokens = sum(c.tokens or 0 for c in calls)
        text = Text()
        text.append(f"{self.elapsed:.1f}s", style="bold")
        text.append(f"  ·  {run} running  ·  {ok} done", style="dim")
        if err:
            text.append(f"  ·  {err} failed", style="red")
        text.append(f"  ·  {tokens:,} tokens", style="dim")
        if self.finished:
            text.append("  ·  ", style="dim")
            text.append(
                "failed" if self.error else "finished", style="red" if self.error else "green"
            )
        return Panel(text, border_style="dim", padding=(0, 1))


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
