"""Rank fusion (RRF), diversification (MMR) and context packing. Pure functions, easy to test."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

import numpy as np

H = TypeVar("H", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[H]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[H, float]]:
    """Fuse several best-first rankings: ``score(d) = sum_i w_i / (k + rank_i(d))`` (rank from 1).

    Items missing from a ranking simply contribute nothing for it. A repeated item inside one
    ranking is counted once, at its best rank. Ties keep first-seen order. Returns ``(item, score)``
    pairs, best first.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[H, float] = {}
    order: dict[H, int] = {}
    for i, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[i])
        seen: set[H] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            order.setdefault(item, len(order))
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def minmax(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant vector maps to all ones (everything equally relevant)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def mmr_select(
    relevance: Sequence[float] | np.ndarray,
    vectors: np.ndarray,
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance selection.

    Greedily picks ``argmax  lambda * rel(d) - (1 - lambda) * max_{s in selected} cos(d, s)``.
    ``relevance`` should already be on a comparable scale (see :func:`minmax`); ``vectors`` are the
    candidates' embeddings (rows are L2-normalised here). Returns candidate indices in selection
    order; ties resolve to the lowest index, so with ``lambda_ = 1`` the input order is preserved.
    """
    rel = np.asarray(relevance, dtype=np.float64)
    n = rel.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    mat = np.asarray(vectors, dtype=np.float64)
    if mat.shape[0] != n:
        raise ValueError("vectors and relevance must have the same length")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1.0, norms)
    sim = mat @ mat.T
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    redundancy = np.zeros(n, dtype=np.float64)
    for step in range(k):
        score = lambda_ * rel - (1.0 - lambda_) * redundancy
        score = np.where(available, score, -np.inf)
        pick = int(np.argmax(score))
        selected.append(pick)
        available[pick] = False
        redundancy = sim[pick].copy() if step == 0 else np.maximum(redundancy, sim[pick])
    return selected


def truncate_at_boundary(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` chars, preferring a sentence or word boundary."""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[: max(limit, 0)]
    room = limit - 4
    cut = text[:room]
    floor = int(room * 0.6)
    best = max(cut.rfind(". "), cut.rfind("\n"), cut.rfind("! "), cut.rfind("? "))
    if best >= floor:
        return cut[: best + 1].rstrip() + " ..."
    space = cut.rfind(" ")
    if space >= floor:
        cut = cut[:space]
    return cut.rstrip() + " ..."


def pack_context(
    texts: Sequence[str], budget_chars: int, max_items: int
) -> list[tuple[int, str]]:
    """Choose which passages fit a character budget.

    Walks the (best-first) passages; a passage that no longer fits is skipped so that smaller,
    lower-ranked ones can still be used. If even the very first passage exceeds the budget it is
    truncated at a sentence boundary instead of dropping all context. Returns ``(index, text)``.
    """
    packed: list[tuple[int, str]] = []
    used = 0
    for i, text in enumerate(texts):
        if len(packed) >= max_items:
            break
        if used + len(text) <= budget_chars:
            packed.append((i, text))
            used += len(text)
        elif not packed:
            packed.append((i, truncate_at_boundary(text, budget_chars)))
            used = budget_chars
    return packed
