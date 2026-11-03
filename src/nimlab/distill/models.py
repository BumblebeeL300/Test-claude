"""Data structures: seeds, samples, run options and the persisted run description."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .recipes import Recipe

__all__ = [
    "DROP_REASONS",
    "SAMPLE_STAGES",
    "STAGES",
    "ExportFormat",
    "RunMeta",
    "RunOptions",
    "Sample",
    "Seed",
    "StageName",
    "utc_now",
]

StageName = Literal["seed", "generate", "score", "filter", "dedup", "decontaminate", "finalize"]
ExportFormat = Literal["openai-chat", "alpaca", "sharegpt", "dpo"]

#: Pipeline order. ``seed`` and ``finalize`` never drop samples.
STAGES: tuple[str, ...] = (
    "seed",
    "generate",
    "score",
    "filter",
    "dedup",
    "decontaminate",
    "finalize",
)
#: Stages at which a sample can be dropped, in order.
SAMPLE_STAGES: tuple[str, ...] = ("generate", "score", "filter", "dedup", "decontaminate")

DROP_REASONS: dict[str, str] = {
    "error": "the model call failed",
    "unscored": "neither reward model nor judge produced a score",
    "low_score": "score below the threshold",
    "too_short": "response shorter than the minimum length",
    "too_long": "response longer than the maximum length",
    "short_prompt": "prompt shorter than the minimum length",
    "refusal": "the response is a refusal or non-answer",
    "language": "text does not look English",
    "repetition": "the response repeats itself",
    "low_margin": "chosen and rejected scores are too close",
    "ungrounded": "the answer is not supported by the source excerpt",
    "too_easy": "difficulty estimate below the minimum",
    "duplicate_exact": "exact duplicate of a better sample",
    "duplicate_near": "near-duplicate of a better sample",
    "contaminated": "overlaps with the decontamination eval set",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Seed(BaseModel):
    """One unit of work produced by the seed stage."""

    idx: int
    topic: str = ""
    subtopic: str = ""
    #: A ready-made prompt (instruction or question); empty for grounded QA, where the model
    #: writes the question from ``context``.
    prompt: str = ""
    context: str = ""
    source: str = ""
    #: Which of several questions about the same excerpt this seed asks for.
    variant: int = 0


class Sample(BaseModel):
    """One dataset row and its journey through the pipeline."""

    id: str
    seed_idx: int
    prompt: str
    response: str
    rejected: str | None = None
    model: str
    rejected_model: str | None = None
    temperature: float = 0.0
    topic: str = ""
    subtopic: str = ""
    source: str = ""
    context: str = ""
    score: float | None = None
    rejected_score: float | None = None
    score_source: Literal["reward", "judge"] | None = None
    difficulty: float | None = None
    cluster: int | None = None
    dropped_at: str | None = None
    reason: str | None = None
    detail: str = ""
    tokens: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.dropped_at is None

    def drop(self, stage: str, reason: str, detail: str = "") -> None:
        self.dropped_at = stage
        self.reason = reason
        self.detail = detail

    def alive_after(self, stage: str) -> bool:
        """Did this sample survive ``stage`` (and every stage before it)?"""
        if self.dropped_at is None:
            return True
        return SAMPLE_STAGES.index(self.dropped_at) > SAMPLE_STAGES.index(stage)


class RunOptions(BaseModel):
    """Per-run knobs (CLI flags and the Studio ``options`` object)."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    seed_model: str | None = None
    reward_model: str | None = None
    judge_model: str | None = None
    embed_model: str | None = None
    #: Eval sets: the weaker model whose answer sets the difficulty estimate.
    weak_model: str | None = None
    domains: list[str] | None = None
    concurrency: int | None = Field(default=None, ge=1, le=64)
    checkpoint_every: int = Field(default=25, ge=1)
    max_tokens: int | None = Field(default=None, ge=16, le=8192)
    dedup_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    #: Partial override of the recipe's ``filters`` object.
    filters: dict[str, Any] = Field(default_factory=dict)
    include_meta: bool | None = None
    #: Seconds of artificial pause per simulated call (Studio animation). Ignored with real models.
    sim_delay: float = Field(default=0.0, ge=0.0, le=5.0)
    #: Documents folder for ``qa_from_docs`` and eval-set file for decontamination.
    docs: str | None = None
    eval_set: str | None = None


class RunMeta(BaseModel):
    """Everything needed to resume a run, stored in ``run.json``."""

    id: str
    recipe: Recipe
    n: int
    models: list[str]
    seed_model: str
    reward_model: str
    judge_model: str
    embed_model: str
    weak_model: str | None = None
    docs: str | None = None
    eval_set: str | None = None
    format: ExportFormat = "openai-chat"
    out: str | None = None
    options: RunOptions = Field(default_factory=RunOptions)
    simulated: bool = False
    status: Literal["created", "running", "interrupted", "done", "failed"] = "created"
    stages_done: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    error: str | None = None
    usage: dict[str, dict[str, int]] = Field(default_factory=dict)
    cost_usd: float = 0.0
    files: list[str] = Field(default_factory=list)
    version: str = ""
