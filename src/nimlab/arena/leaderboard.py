"""Aggregate a :class:`RunResult` into rankings, per-suite scores, speed stats and Elo."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from nimlab.core import default_registry
from nimlab.core.registry import ModelRegistry

from .battle import position_bias
from .elo import Outcome, bootstrap_elo, outcomes_from_scores
from .results import ItemResult, RunResult

__all__ = ["Leaderboard", "ModelStanding", "pareto_frontier"]

EloSource = Literal["battles", "scores", "none"]


class ModelStanding(BaseModel):
    """One row of the leaderboard."""

    model: str
    rank: int = 0
    overall: float | None = None
    suite_scores: dict[str, float | None] = Field(default_factory=dict)
    suite_latency_s: dict[str, float | None] = Field(default_factory=dict)
    pass_rate: float | None = None
    n_scored: int = 0
    n_errors: int = 0
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    ttft_p50_s: float | None = None
    tokens_per_s: float | None = None
    tokens_out: int = 0
    cost_tier: int | None = None
    elo: float | None = None
    elo_lo: float | None = None
    elo_hi: float | None = None
    elo_games: int = 0
    pareto: bool = False


class Leaderboard(BaseModel):
    """Ranked models plus the metadata needed to render a report or route requests."""

    run_id: str
    created_at: str = ""
    mock: bool = False
    judge_model: str | None = None
    suites: list[str] = Field(default_factory=list)
    suite_categories: dict[str, str] = Field(default_factory=dict)
    standings: list[ModelStanding] = Field(default_factory=list)
    elo_source: EloSource = "none"
    elo_games: int = 0
    position_bias: float | None = None
    n_boot: int = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_run(
        cls,
        run: RunResult,
        *,
        registry: ModelRegistry | None = None,
        n_boot: int = 200,
        seed: int = 0,
        elo_source: Literal["auto", "battles", "scores"] = "auto",
    ) -> Leaderboard:
        reg = registry or default_registry()
        suites = sorted({r.suite for r in run.items} | set(run.suites))
        by_model: dict[str, list[ItemResult]] = defaultdict(list)
        for r in run.items:
            by_model[r.model].append(r)
        for m in run.models:
            by_model.setdefault(m, [])

        standings = [_standing(m, rs, suites, reg) for m, rs in by_model.items()]

        source: EloSource = "none"
        outcomes: list[Outcome] = []
        if elo_source in ("auto", "battles") and run.battles:
            source = "battles"
            outcomes = [Outcome(b.model_a, b.model_b, b.score_a) for b in run.battles]
        elif elo_source in ("auto", "scores"):
            per_item: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
            for r in run.items:
                if r.scored:
                    per_item[(r.suite, r.item_id)][r.model] = r.score
            outcomes = outcomes_from_scores(per_item)
            source = "scores" if outcomes else "none"
        if outcomes:
            rated = bootstrap_elo(outcomes, n_boot=n_boot, seed=seed)
            for s in standings:
                rating = rated.get(s.model)
                if rating:
                    s.elo, s.elo_lo, s.elo_hi, s.elo_games = (
                        round(rating.elo, 1),
                        round(rating.lo, 1),
                        round(rating.hi, 1),
                        rating.games,
                    )

        standings.sort(
            key=lambda s: (
                s.overall is None,
                -(s.overall or 0.0),
                -(s.elo or 0.0),
                s.latency_p50_s if s.latency_p50_s is not None else float("inf"),
                s.model,
            )
        )
        for i, s in enumerate(standings, 1):
            s.rank = i
        for model in pareto_frontier([(s.model, s.latency_p50_s, s.overall) for s in standings]):
            next(s for s in standings if s.model == model).pareto = True

        categories = {
            name: (run.suites[name].category if name in run.suites else "general_chat")
            for name in suites
        }
        return cls(
            run_id=run.run_id,
            created_at=run.created_at,
            mock=run.mock,
            judge_model=run.judge_model,
            suites=suites,
            suite_categories=categories,
            standings=standings,
            elo_source=source,
            elo_games=len(outcomes),
            position_bias=position_bias(run.battles),
            n_boot=n_boot if outcomes else 0,
        )

    # ------------------------------------------------------------------ queries
    def get(self, model: str) -> ModelStanding | None:
        return next((s for s in self.standings if s.model == model), None)

    def models(self) -> list[str]:
        return [s.model for s in self.standings]

    def best_for_suite(self, suite: str) -> str | None:
        scored = [(s.suite_scores.get(suite), s.model) for s in self.standings]
        scored = [(v, m) for v, m in scored if v is not None]
        return max(scored)[1] if scored else None

    def suites_in(self, category: str) -> list[str]:
        return [s for s in self.suites if self.suite_categories.get(s) == category]

    def category_quality(self, model: str, category: str) -> float | None:
        """Mean score over the suites of ``category`` (all suites for ``general_chat``)."""
        standing = self.get(model)
        if standing is None:
            return None
        if category == "general_chat":
            return standing.overall
        scores = [standing.suite_scores.get(s) for s in self.suites_in(category)]
        present = [v for v in scores if v is not None]
        return float(np.mean(present)) if present else None

    def category_latency(self, model: str, category: str) -> float | None:
        standing = self.get(model)
        if standing is None:
            return None
        names = self.suites if category == "general_chat" else self.suites_in(category)
        latencies = [standing.suite_latency_s.get(s) for s in names]
        known = [v for v in latencies if v is not None]
        if known:
            return float(np.mean(known))
        return standing.latency_p50_s

    def covered_categories(self) -> list[str]:
        return sorted({c for c in self.suite_categories.values()})


def _median(vals: list[float]) -> float | None:
    return float(np.median(vals)) if vals else None


def _standing(
    model: str, results: list[ItemResult], suites: list[str], registry: ModelRegistry
) -> ModelStanding:
    info = registry.find(model)
    scored = [r for r in results if r.scored]
    suite_scores: dict[str, float | None] = {}
    suite_latency: dict[str, float | None] = {}
    for suite in suites:
        rs = [r for r in scored if r.suite == suite]
        wsum = sum(r.weight for r in rs)
        suite_scores[suite] = sum(r.score * r.weight for r in rs) / wsum if wsum else None
        suite_latency[suite] = _median([r.latency_s for r in rs])
    present = [v for v in suite_scores.values() if v is not None]
    latencies = [r.latency_s for r in scored]
    tps = [r.tokens_per_s for r in scored if r.tokens_per_s]
    ttft = [r.ttft_s for r in scored if r.ttft_s is not None]
    return ModelStanding(
        model=model,
        overall=float(np.mean(present)) if present else None,
        suite_scores=suite_scores,
        suite_latency_s=suite_latency,
        pass_rate=(sum(1 for r in scored if r.passed) / len(scored)) if scored else None,
        n_scored=len(scored),
        n_errors=len(results) - len(scored),
        latency_p50_s=_median(latencies),
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
        ttft_p50_s=_median(ttft),
        tokens_per_s=_median(tps),
        tokens_out=sum(r.tokens_out for r in scored),
        cost_tier=info.cost_tier if info else None,
    )


def pareto_frontier(points: list[tuple[str, float | None, float | None]]) -> list[str]:
    """Names on the quality-vs-latency Pareto frontier.

    ``points`` are ``(name, latency, quality)``; a point is dominated when another is at least as
    fast **and** at least as good, and strictly better in one of them.
    """
    pts = [(n, lat, q) for n, lat, q in points if lat is not None and q is not None]
    front: list[str] = []
    for n, lat, q in pts:
        dominated = any(
            (l2 <= lat and q2 >= q) and (l2 < lat or q2 > q) for n2, l2, q2 in pts if n2 != n
        )
        if not dominated:
            front.append(n)
    return front
