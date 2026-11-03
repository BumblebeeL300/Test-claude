"""Quality metrics: the pipeline funnel, score histograms, diversity and top/bottom samples."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .costs import summarize
from .dedup import mean_pairwise_distance
from .models import SAMPLE_STAGES, RunMeta, Sample
from .textutil import distinct_n

__all__ = [
    "Diversity",
    "FunnelRow",
    "Histogram",
    "ModelRow",
    "QualityReport",
    "SampleView",
    "build_report",
    "compute_funnel",
    "histogram",
]

STAGE_LABELS = {
    "generate": "Generated",
    "score": "Scored",
    "filter": "Passed filters",
    "dedup": "After dedup",
    "decontaminate": "After decontamination",
}


class FunnelRow(BaseModel):
    stage: str
    label: str
    in_: int = Field(alias="in")
    out: int
    dropped: int
    reasons: dict[str, int] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class Histogram(BaseModel):
    source: str
    edges: list[float]
    all: list[int]
    kept: list[int]


class Diversity(BaseModel):
    n: int = 0
    mean_pairwise_distance: float | None = None
    distinct_1: float = 0.0
    distinct_2: float = 0.0
    distinct_3: float = 0.0
    response_distinct_2: float = 0.0


class ModelRow(BaseModel):
    model: str
    generated: int
    kept: int
    kept_rate: float
    mean_score: float | None = None
    score_source: str | None = None


class SampleView(BaseModel):
    id: str
    prompt: str
    response: str
    rejected: str | None = None
    model: str
    score: float | None = None
    score_source: str | None = None
    topic: str = ""
    difficulty: float | None = None
    reason: str | None = None


class QualityReport(BaseModel):
    run_id: str
    recipe: str
    kind: str
    simulated: bool
    n_requested: int
    n_final: int
    funnel: list[FunnelRow]
    drop_reasons: dict[str, int]
    histograms: dict[str, Histogram]
    models: list[ModelRow]
    diversity: Diversity
    lengths: dict[str, float]
    top: list[SampleView]
    bottom: list[SampleView]
    usage: dict[str, Any]
    dominant_source: str | None = None

    @property
    def retained(self) -> float:
        return self.n_final / self.n_requested if self.n_requested else 0.0


def compute_funnel(samples: list[Sample], n_seeds: int) -> list[FunnelRow]:
    """Counts after each sample-level stage (``in`` of a stage is the previous stage's ``out``)."""
    rows: list[FunnelRow] = []
    previous = n_seeds or len(samples)
    for stage in SAMPLE_STAGES:
        out = sum(1 for s in samples if s.alive_after(stage))
        reasons = Counter(s.reason or "unknown" for s in samples if s.dropped_at == stage)
        rows.append(
            FunnelRow(
                stage=stage,
                label=STAGE_LABELS[stage],
                **{"in": previous},
                out=out,
                dropped=previous - out,
                reasons=dict(reasons.most_common()),
            )
        )
        previous = out
    return rows


def histogram(
    values: list[float], kept: list[float], source: str, bins: int = 12
) -> Histogram | None:
    if not values:
        return None
    if source == "judge":
        lo, hi = 0.0, 10.0
    else:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    all_counts, _ = np.histogram(values, bins=edges)
    kept_counts, _ = np.histogram(kept, bins=edges) if kept else (np.zeros(bins, dtype=int), None)
    return Histogram(
        source=source,
        edges=[round(float(e), 4) for e in edges],
        all=[int(c) for c in all_counts],
        kept=[int(c) for c in kept_counts],
    )


def _view(s: Sample) -> SampleView:
    return SampleView(
        id=s.id,
        prompt=s.prompt,
        response=s.response,
        rejected=s.rejected,
        model=s.model,
        score=None if s.score is None else round(s.score, 3),
        score_source=s.score_source,
        topic=s.topic,
        difficulty=s.difficulty,
        reason=s.reason,
    )


def build_report(
    meta: RunMeta,
    samples: list[Sample],
    n_seeds: int,
    vectors: dict[str, list[float]] | None = None,
    *,
    top_k: int = 5,
) -> QualityReport:
    """Assemble the quality report from the sample table (and embeddings, if available)."""
    kept = [s for s in samples if s.alive]
    scored = [s for s in samples if s.score is not None and s.score_source]
    sources = Counter(s.score_source for s in scored)
    dominant = sources.most_common(1)[0][0] if sources else None

    histograms: dict[str, Histogram] = {}
    for source in ("reward", "judge"):
        values = [s.score for s in scored if s.score_source == source and s.score is not None]
        keep_values = [s.score for s in kept if s.score_source == source and s.score is not None]
        h = histogram(values, keep_values, source)
        if h is not None:
            histograms[source] = h

    rows: list[ModelRow] = []
    for model in dict.fromkeys(s.model for s in samples):
        mine = [s for s in samples if s.model == model]
        sc = [s.score for s in mine if s.score is not None and s.score_source == dominant]
        rows.append(
            ModelRow(
                model=model,
                generated=len(mine),
                kept=sum(1 for s in mine if s.alive),
                kept_rate=round(sum(1 for s in mine if s.alive) / len(mine), 4),
                mean_score=round(statistics.fmean(sc), 3) if sc else None,
                score_source=dominant if sc else None,
            )
        )

    prompts = [s.prompt for s in kept]
    responses = [s.response for s in kept]
    diversity = Diversity(
        n=len(kept),
        distinct_1=round(distinct_n(prompts, 1), 4),
        distinct_2=round(distinct_n(prompts, 2), 4),
        distinct_3=round(distinct_n(prompts, 3), 4),
        response_distinct_2=round(distinct_n(responses, 2), 4),
    )
    if vectors:
        mat = [vectors[s.id] for s in kept if s.id in vectors]
        if len(mat) >= 2:
            diversity.mean_pairwise_distance = round(mean_pairwise_distance(np.array(mat)), 4)

    ranked = sorted(
        (s for s in kept if s.score is not None and s.score_source == dominant),
        key=lambda s: (-(s.score or 0.0), s.id),
    )
    lengths = [len(s.response) for s in kept]
    length_stats = (
        {
            "mean": round(statistics.fmean(lengths), 1),
            "median": float(statistics.median(lengths)),
            "min": float(min(lengths)),
            "max": float(max(lengths)),
        }
        if lengths
        else {}
    )
    funnel = compute_funnel(samples, n_seeds)
    reasons: Counter[str] = Counter()
    for row in funnel:
        reasons.update(row.reasons)
    return QualityReport(
        run_id=meta.id,
        recipe=meta.recipe.name,
        kind=meta.recipe.kind,
        simulated=meta.simulated,
        n_requested=meta.n,
        n_final=len(kept),
        funnel=funnel,
        drop_reasons=dict(reasons.most_common()),
        histograms=histograms,
        models=rows,
        diversity=diversity,
        lengths=length_stats,
        top=[_view(s) for s in ranked[:top_k]],
        bottom=[_view(s) for s in ranked[::-1][:top_k]],
        usage=summarize(meta.usage),
        dominant_source=dominant,
    )
