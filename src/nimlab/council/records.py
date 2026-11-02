"""Pydantic data model shared by every part of Council: trace, candidates, consensus, results."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class StepType(str, Enum):
    """What a recorded step *is* (drives colouring and grouping in the reports)."""

    PROPOSE = "propose"  # an independent first answer
    AGGREGATE = "aggregate"  # a synthesis over several earlier answers (MoA layers / final)
    DRAFT = "draft"  # first draft in the critic loop
    CRITIQUE = "critique"  # a review of someone's answer
    REVISE = "revise"  # an answer revised after seeing peers / critiques
    JUDGE = "judge"  # a judge picking or synthesising the final answer
    SCORE = "score"  # a reward-model or LLM-as-judge score for a candidate
    CONSENSUS = "consensus"  # an embedding pass measuring agreement


class TraceStep(BaseModel):
    """One model call (or embedding pass) made during a council run."""

    index: int
    type: StepType
    model: str
    label: str = ""
    phase: str = ""
    status: Literal["running", "ok", "error"] = "running"
    prompt_summary: str = ""
    prompt: str | None = None
    output: str = ""
    reasoning: str | None = None
    temperature: float | None = None
    started_s: float = 0.0  # seconds since the run started
    duration_s: float = 0.0  # wall-clock duration of the call (includes queueing)
    latency_s: float = 0.0  # latency reported by the client
    prompt_tokens: int | None = None  # None = unknown
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def ended_s(self) -> float:
        return self.started_s + self.duration_s


class ConsensusReport(BaseModel):
    """Pairwise agreement between a set of answers."""

    phase: str
    method: Literal["embedding", "lexical"] = "embedding"
    embed_model: str | None = None
    labels: list[str]
    models: list[str]
    matrix: list[list[float]]
    agreement: float  # mean off-diagonal cosine, clipped to [0, 1]
    min_pairwise: float
    centrality: list[float]  # mean similarity of each answer to all the others
    outliers: list[int] = Field(default_factory=list)
    medoid: int | None = None  # index of the most central answer

    @property
    def outlier_labels(self) -> list[str]:
        return [self.labels[i] for i in self.outliers]

    @property
    def size(self) -> int:
        return len(self.labels)


class Candidate(BaseModel):
    """An answer that was in the running when the strategy finished."""

    label: str
    model: str
    text: str
    score: float | None = None
    score_source: Literal["reward", "judge"] | None = None
    selected: bool = False
    outlier: bool = False
    step_index: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    """Per-model usage totals for a run."""

    model: str
    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0


class Trace(BaseModel):
    """A complete, serialisable record of one council run."""

    schema_version: int = SCHEMA_VERSION
    question: str
    strategy: str
    models: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    mock: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    wall_time_s: float = 0.0
    steps: list[TraceStep] = Field(default_factory=list)
    consensus: list[ConsensusReport] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    final_answer: str = ""
    rationale: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    usage: list[ModelUsage] = Field(default_factory=list)
    error: str | None = None

    # ------------------------------------------------------------------ queries
    @property
    def all_models(self) -> list[str]:
        """Participating models, configured ones first, then any others seen in steps."""
        seen: dict[str, None] = dict.fromkeys(self.models)
        for step in self.steps:
            seen.setdefault(step.model, None)
        return list(seen)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.usage) or sum(
            s.total_tokens or 0 for s in self.steps
        )

    @property
    def final_consensus(self) -> ConsensusReport | None:
        return self.consensus[-1] if self.consensus else None

    def steps_for(self, model: str) -> list[TraceStep]:
        return [s for s in self.steps if s.model == model]

    def steps_of(self, type_: StepType) -> list[TraceStep]:
        return [s for s in self.steps if s.type == type_]

    def usage_from_steps(self) -> list[ModelUsage]:
        """Per-model usage derived purely from the recorded steps."""
        rows: dict[str, ModelUsage] = defaultdict(lambda: ModelUsage(model=""))
        for step in self.steps:
            row = rows[step.model]
            row.model = step.model
            row.calls += 1
            row.errors += int(step.status == "error")
            row.prompt_tokens += step.prompt_tokens or 0
            row.completion_tokens += step.completion_tokens or 0
            row.total_tokens += step.total_tokens or 0
            row.latency_s += step.duration_s
        return sorted(rows.values(), key=lambda r: r.model)

    # ------------------------------------------------------------ serialisation
    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Trace:
        return cls.model_validate_json(text)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class CouncilResult(BaseModel):
    """What :meth:`Council.run` returns."""

    question: str
    strategy: str
    answer: str
    rationale: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    consensus: ConsensusReport | None = None
    trace: Trace

    @property
    def agreement(self) -> float | None:
        return self.consensus.agreement if self.consensus else None

    @property
    def models(self) -> list[str]:
        return self.trace.models

    @property
    def usage(self) -> list[ModelUsage]:
        return self.trace.usage

    @property
    def selected(self) -> Candidate | None:
        return next((c for c in self.candidates if c.selected), None)

    def save_json(self, path: str | Path) -> Path:
        """Write the full trace (which includes the answer) as JSON."""
        return self.trace.save(path)

    def html(self) -> str:
        from .report import render_html

        return render_html(self.trace)

    def save_html(self, path: str | Path) -> Path:
        from .report import save_html

        return save_html(self.trace, path)
