"""The :class:`Strategy` protocol, its outcome type, and the pluggable registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from ..errors import NotEnoughModelsError, UnknownStrategyError
from ..records import Candidate, ConsensusReport

if TYPE_CHECKING:
    from ..context import Answer, RunContext


@dataclass
class StrategyOutcome:
    """What a strategy hands back to :class:`~nimlab.council.council.Council`."""

    answer: str
    rationale: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    consensus: ConsensusReport | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """A deliberation algorithm. Implement this (and :func:`register_strategy`) to add your own."""

    name: ClassVar[str]
    description: ClassVar[str]

    async def run(self, ctx: RunContext) -> StrategyOutcome:
        """Run the strategy to completion, calling models only through ``ctx``."""
        ...


class BaseStrategy:
    """Convenience base: metadata for the CLI plus shared helpers."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    aliases: ClassVar[tuple[str, ...]] = ()
    options: ClassVar[tuple[str, ...]] = ()  # CLI/config knobs that matter for this strategy
    min_models: ClassVar[int] = 1

    def check_models(self, ctx: RunContext) -> None:
        if len(ctx.models) < self.min_models:
            raise NotEnoughModelsError(
                f"strategy {self.name!r} needs at least {self.min_models} models, "
                f"got {len(ctx.models)}: {', '.join(ctx.models) or '(none)'}",
                trace=ctx.trace,
            )

    async def run(self, ctx: RunContext) -> StrategyOutcome:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def candidates_from(
        answers: list[Answer], report: ConsensusReport | None = None
    ) -> list[Candidate]:
        """Candidates for ``answers``; outlier flags come from ``report`` when it matches."""
        flagged: set[str] = set()
        if report is not None:
            flagged = {report.labels[i] for i in report.outliers}
        return [
            Candidate(
                label=a.label,
                model=a.model,
                text=a.text,
                step_index=a.step_index,
                outlier=a.label in flagged,
            )
            for a in answers
        ]


def normalise_name(name: str) -> str:
    return re.sub(r"[\s_]+", "-", name.strip().lower())


_REGISTRY: dict[str, Strategy] = {}
_ALIASES: dict[str, str] = {}


def register_strategy(strategy: Strategy | type[Strategy]) -> Strategy | type[Strategy]:
    """Register a strategy instance (or class, which is instantiated); usable as a decorator."""
    instance: Strategy = strategy() if isinstance(strategy, type) else strategy
    key = normalise_name(instance.name)
    if not key:
        raise ValueError("a strategy needs a non-empty name")
    _REGISTRY[key] = instance
    for alias in getattr(instance, "aliases", ()):
        _ALIASES[normalise_name(alias)] = key
    return strategy


def unregister_strategy(name: str) -> None:
    key = _ALIASES.get(normalise_name(name), normalise_name(name))
    _REGISTRY.pop(key, None)
    for alias, target in list(_ALIASES.items()):
        if target == key:
            del _ALIASES[alias]


def get_strategy(name: str) -> Strategy:
    key = normalise_name(name)
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise UnknownStrategyError(f"unknown strategy {name!r}; available: {known}") from None


def list_strategies() -> list[Strategy]:
    return list(_REGISTRY.values())
