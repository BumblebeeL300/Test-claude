"""Structured-output schemas the council asks models to produce via ``chat_json``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import Severity

SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3}

_SEVERITY_ALIASES = {
    "critical": "high",
    "severe": "high",
    "major": "high",
    "blocker": "high",
    "moderate": "medium",
    "med": "medium",
    "minor": "low",
    "trivial": "low",
    "nit": "low",
    "info": "low",
}


class Issue(BaseModel):
    """One problem a critic found in a draft."""

    severity: Severity = Field(
        default="low", description="'high' = wrong/misleading/missing something essential"
    )
    category: str = Field(
        default="other",
        description="accuracy | completeness | reasoning | clarity | safety | other",
    )
    description: str = Field(default="", description="What is wrong, concretely")
    suggestion: str = Field(default="", description="How to fix it")

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _SEVERITY_ALIASES.get(key, key)
        return v

    @property
    def rank(self) -> int:
        return SEVERITY_RANK[self.severity]


class Critique(BaseModel):
    """A critic's structured review of a draft answer."""

    issues: list[Issue] = Field(
        default_factory=list, description="Empty list when the answer needs no changes"
    )
    summary: str = Field(default="", description="One or two sentence overall assessment")

    def blocking(self, threshold: Severity = "high") -> list[Issue]:
        """Issues at or above ``threshold``."""
        floor = SEVERITY_RANK[threshold]
        return [i for i in self.issues if i.rank >= floor]


class JudgeVerdict(BaseModel):
    """A judge's decision between (or synthesis of) the candidate answers."""

    selected: int | None = Field(
        default=None,
        description="1-based number of the best answer, or null if you wrote a synthesis",
    )
    final_answer: str = Field(
        default="",
        description="The final answer for the user (chosen answer, improved, or a synthesis)",
    )
    rationale: str = Field(default="", description="Why this is the best answer")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return min(1.0, max(0.0, float(v)))
        return v


class RubricScore(BaseModel):
    """LLM-as-judge rubric score for one candidate answer."""

    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall quality from 0 (useless/wrong) to 10 (excellent, correct, complete)",
    )
    rationale: str = Field(default="", description="One or two sentences justifying the score")

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return min(10.0, max(0.0, float(v)))
        return v
