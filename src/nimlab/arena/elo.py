"""Bradley-Terry rating fit (MM algorithm, numpy) on an Elo scale, with bootstrap intervals.

Model: ``P(i beats j) = p_i / (p_i + p_j)`` with positive strengths ``p``. Given the matrix ``W``
of points ``i`` scored against ``j`` (win = 1, tie = 1/2, fractional scores allowed) and
``N = W + W^T`` the games played, the maximum-likelihood strengths are the fixed point of the
minorise-maximise update (Hunter, 2004)::

    p_i  <-  W_i / sum_j N_ij / (p_i + p_j)           (W_i = sum_j W_ij)

followed by a rescale to unit geometric mean. A *virtual opponent* of average strength (1) that
ties ``prior`` games with everybody keeps the estimate finite for a model that never lost (a weak
ridge-like prior). Strengths map to Elo points with ``elo_i = base + 400 * log10(p_i)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BTFit",
    "EloRating",
    "Outcome",
    "bootstrap_elo",
    "elo_scale",
    "expected_score",
    "fit_bradley_terry",
    "outcomes_from_scores",
]

ELO_BASE = 1000.0
ELO_SPREAD = 400.0


@dataclass(frozen=True)
class Outcome:
    """One game: ``score_a`` is 1 if ``a`` won, 0.5 for a tie, 0 if ``b`` won."""

    a: str
    b: str
    score_a: float


@dataclass
class BTFit:
    models: list[str]
    strengths: np.ndarray
    games: np.ndarray  # games played per model
    iterations: int
    converged: bool
    elo: dict[str, float] = field(default_factory=dict)

    def rating(self, model: str) -> float:
        return self.elo[model]


@dataclass(frozen=True)
class EloRating:
    model: str
    elo: float
    lo: float
    hi: float
    games: int

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2


def elo_scale(strengths: np.ndarray, base: float = ELO_BASE) -> np.ndarray:
    return base + ELO_SPREAD * np.log10(strengths)


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that ``a`` beats ``b`` according to the Elo difference."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / ELO_SPREAD))


def _win_matrix(n: int, ia: np.ndarray, ib: np.ndarray, s: np.ndarray) -> np.ndarray:
    w = np.zeros((n, n))
    np.add.at(w, (ia, ib), s)
    np.add.at(w, (ib, ia), 1.0 - s)
    return w


def _mm(
    w: np.ndarray,
    prior: float,
    max_iter: int,
    tol: float,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool]:
    """MM iterations with the strengths renormalised to unit geometric mean at every step."""
    n = w.shape[0]
    games = w + w.T
    wins = w.sum(axis=1) + 0.5 * prior
    p = np.ones(n) if start is None else start.copy()
    played = games.sum(axis=1) > 0
    converged, iters = False, 0
    for step in range(1, max_iter + 1):
        iters = step
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1) + prior / (p + 1.0)
        new = np.where(denom > 0, wins / np.where(denom > 0, denom, 1.0), 1.0)
        if played.any():  # fix the gauge: unit geometric mean (the virtual opponent is "average")
            new = new / math.exp(np.log(new[played]).mean())
        converged = bool(np.max(np.abs(np.log(new) - np.log(p))) < tol)
        p = new
        if converged:
            break
    return p, iters, converged


def _encode(
    outcomes: Sequence[Outcome], models: Sequence[str] | None
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = (
        list(models) if models is not None else sorted({m for o in outcomes for m in (o.a, o.b)})
    )
    idx = {m: i for i, m in enumerate(names)}
    keep = [o for o in outcomes if o.a in idx and o.b in idx and o.a != o.b]
    ia = np.array([idx[o.a] for o in keep], dtype=int)
    ib = np.array([idx[o.b] for o in keep], dtype=int)
    s = np.array([o.score_a for o in keep], dtype=float)
    if np.any((s < 0) | (s > 1)):
        raise ValueError("outcome scores must lie in [0, 1]")
    return names, ia, ib, s


def fit_bradley_terry(
    outcomes: Iterable[Outcome],
    models: Sequence[str] | None = None,
    *,
    prior: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-10,
    base: float = ELO_BASE,
) -> BTFit:
    """Maximum-likelihood (lightly regularised) Bradley-Terry strengths and Elo ratings."""
    outs = list(outcomes)
    names, ia, ib, s = _encode(outs, models)
    n = len(names)
    if n == 0:
        raise ValueError("no models to rate")
    w = _win_matrix(n, ia, ib, s)
    p, iters, ok = _mm(w, prior, max_iter, tol)
    elo = elo_scale(p, base)
    games = (w + w.T).sum(axis=1)
    return BTFit(
        models=names,
        strengths=p,
        games=games,
        iterations=iters,
        converged=ok,
        elo={m: float(elo[i]) for i, m in enumerate(names)},
    )


def bootstrap_elo(
    outcomes: Iterable[Outcome],
    models: Sequence[str] | None = None,
    *,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 0,
    prior: float = 1.0,
    base: float = ELO_BASE,
) -> dict[str, EloRating]:
    """Elo ratings with percentile bootstrap confidence intervals.

    The games are resampled with replacement ``n_boot`` times, the model is refit each time
    (warm-started from the full fit) and the ``ci`` central percentile interval is reported.
    Models with very few games get wide intervals, as they should.
    """
    outs = list(outcomes)
    names, ia, ib, s = _encode(outs, models)
    n = len(names)
    full = fit_bradley_terry(outs, names, prior=prior, base=base)
    if not len(s):
        return {m: EloRating(m, base, base, base, 0) for m in names}
    if n_boot <= 0:  # point estimates only
        return {
            m: EloRating(m, full.elo[m], full.elo[m], full.elo[m], int(full.games[i]))
            for i, m in enumerate(names)
        }
    rng = np.random.default_rng(seed)
    m = len(s)
    samples = np.empty((n_boot, n))
    for b in range(n_boot):
        pick = rng.integers(0, m, size=m)
        w = _win_matrix(n, ia[pick], ib[pick], s[pick])
        p, _, _ = _mm(w, prior, 500, 1e-8, start=full.strengths)
        samples[b] = elo_scale(p, base)
    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(samples, alpha, axis=0)
    hi = np.quantile(samples, 1.0 - alpha, axis=0)
    out: dict[str, EloRating] = {}
    for i, name in enumerate(names):
        elo = full.elo[name]
        out[name] = EloRating(
            name,
            elo,
            float(min(lo[i], elo)),
            float(max(hi[i], elo)),
            int(full.games[i]),
        )
    return out


def outcomes_from_scores(
    scores: dict[tuple[str, str], dict[str, float]], tie_margin: float = 0.05
) -> list[Outcome]:
    """Pairwise outcomes from per-item scores.

    ``scores`` maps ``(suite, item_id)`` to ``{model: score}``. Every pair of models that both
    answered an item plays one game; scores within ``tie_margin`` are ties.
    """
    out: list[Outcome] = []
    for per_model in scores.values():
        models = sorted(per_model)
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                diff = per_model[a] - per_model[b]
                score = 0.5 if abs(diff) <= tie_margin else (1.0 if diff > 0 else 0.0)
                out.append(Outcome(a, b, score))
    return out
