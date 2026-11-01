"""Grader protocol, result type and the pluggable registry."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, Field

__all__ = [
    "GradeContext",
    "GradeResult",
    "Grader",
    "GraderError",
    "available_graders",
    "get_grader",
    "grader_class",
    "register_grader",
]


class GraderError(RuntimeError):
    """Grading itself failed (e.g. the judge model was unreachable).

    Distinct from a wrong answer: the runner records such items as *errored* and excludes them
    from the model's score instead of counting them as a failure.
    """


class GradeResult(BaseModel):
    """Outcome of grading one model output. ``score`` is normalised to ``[0, 1]``."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    detail: str = ""
    extracted: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass
class GradeContext:
    """Everything a grader may look at."""

    output: str
    params: dict[str, Any]
    prompt: str = ""
    reference: str | None = None
    item_id: str = ""
    suite: str = ""
    client: Any = None  # an LLM client (needed by judge graders only)
    judge_model: str | None = None


class Grader(abc.ABC):
    """Base class. Subclasses declare a pydantic ``Params`` model and implement :meth:`run`."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    needs_llm: ClassVar[bool] = False
    Params: ClassVar[type[BaseModel]]

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> BaseModel:
        """Parse ``params`` into the grader's typed model (raises ``ValueError`` if invalid)."""
        return cls.Params.model_validate(params)

    async def grade(self, ctx: GradeContext) -> GradeResult:
        params = self.validate_params(ctx.params)
        return await self.run(params, ctx)

    @abc.abstractmethod
    async def run(self, params: Any, ctx: GradeContext) -> GradeResult: ...


_REGISTRY: dict[str, type[Grader]] = {}
_ALIASES = {
    "numeric_tolerance": "numeric",
    "contains-all": "contains_all",
    "programmatic": "checks",
    "programmatic_checks": "checks",
    "code": "code_exec",
    "judge": "llm_judge",
}

G = TypeVar("G", bound=type[Grader])


def _canon(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return _ALIASES.get(key, key)


def register_grader(cls: G) -> G:
    """Class decorator adding a grader to the registry under ``cls.name``."""
    _REGISTRY[_canon(cls.name)] = cls
    return cls


def grader_class(name: str) -> type[Grader]:
    try:
        return _REGISTRY[_canon(name)]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown grader type {name!r}; available: {known}") from None


def get_grader(name: str) -> Grader:
    """Instantiate the grader registered as ``name`` (hyphens and common aliases accepted)."""
    return grader_class(name)()


def available_graders() -> list[str]:
    return sorted(_REGISTRY)
