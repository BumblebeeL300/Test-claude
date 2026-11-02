"""Structured run events and a tiny synchronous event bus."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EventBus", "EventCallback", "ForgeEvent"]

_log = logging.getLogger("nimlab.forge")

#: Event kinds emitted by Forge. ``data`` carries kind-specific details.
EVENT_KINDS = (
    "run_start",  # task, workspace, models
    "state",  # from, to
    "phase_start",
    "phase_end",  # seconds
    "info",
    "warning",
    "error",
    "plan",  # steps
    "llm",  # model, role, tokens, latency_s, tool_calls
    "tool_call",  # tool, args
    "tool_result",  # tool, ok, chars
    "tests",  # passed, failed, errors, ok, iteration
    "stall",  # repeats, signature
    "escalate",  # model
    "review",  # verdict, issues
    "rollback",  # restored, removed
    "run_end",  # status
)


@dataclass
class ForgeEvent:
    """One thing that happened during a run."""

    kind: str
    phase: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "phase": self.phase,
            "message": self.message,
            "data": self.data,
            "ts": round(self.ts, 3),
            "elapsed_s": round(self.elapsed_s, 3),
        }


EventCallback = Callable[[ForgeEvent], None]


class EventBus:
    """Collects events and fans them out to subscribers.

    Subscribers are plain callables; an exception inside one is logged and swallowed so a broken
    UI can never abort a run.
    """

    def __init__(self, max_history: int = 2000) -> None:
        self._subscribers: list[EventCallback] = []
        self.history: list[ForgeEvent] = []
        self._max_history = max_history
        self._t0 = time.monotonic()

    def subscribe(self, callback: EventCallback) -> None:
        self._subscribers.append(callback)

    def emit(self, kind: str, phase: str = "", message: str = "", **data: Any) -> ForgeEvent:
        event = ForgeEvent(
            kind=kind,
            phase=phase,
            message=message,
            data=data,
            elapsed_s=time.monotonic() - self._t0,
        )
        self.history.append(event)
        if len(self.history) > self._max_history:
            del self.history[: len(self.history) - self._max_history]
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:  # UI errors must not break the run
                _log.exception("forge event subscriber failed")
        return event
