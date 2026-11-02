"""Lightweight run events that observers (the live terminal UI, tests, loggers) can subscribe to."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .records import StepType, Trace


@dataclass(frozen=True)
class CouncilEvent:
    """Base class; observers dispatch with ``isinstance``."""


@dataclass(frozen=True)
class RunStarted(CouncilEvent):
    strategy: str
    question: str
    models: tuple[str, ...]


@dataclass(frozen=True)
class PhaseStarted(CouncilEvent):
    phase: str


@dataclass(frozen=True)
class CallStarted(CouncilEvent):
    step_index: int
    model: str
    label: str
    phase: str
    step_type: StepType


@dataclass(frozen=True)
class CallFinished(CouncilEvent):
    step_index: int
    ok: bool
    duration_s: float = 0.0
    total_tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ConsensusComputed(CouncilEvent):
    phase: str
    agreement: float
    outliers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RunFinished(CouncilEvent):
    trace: Trace
    error: str | None = None


Observer = Callable[[CouncilEvent], None]
