"""Exact and near-duplicate detection (numpy only).

* **Exact**: samples whose normalised text (case, punctuation and whitespace removed) hashes alike.
* **Near**: greedy clustering on cosine similarity. Rows are visited best-first (highest priority),
  and a row joins the first existing cluster whose representative is at least ``threshold`` similar,
  otherwise it founds a new cluster. The representative of each cluster is therefore its best row.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "cluster_members",
    "exact_duplicates",
    "greedy_clusters",
    "mean_pairwise_distance",
    "normalize_rows",
]


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows (zero rows stay zero)."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2-D array")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _order(n: int, priority: Sequence[float] | None) -> list[int]:
    if priority is None:
        return list(range(n))
    if len(priority) != n:
        raise ValueError("priority must have one entry per row")
    return sorted(range(n), key=lambda i: (-priority[i], i))


def exact_duplicates(
    keys: Sequence[str], priority: Sequence[float] | None = None
) -> dict[int, int]:
    """``{duplicate index: kept index}``; the highest-priority member of each group is kept."""
    kept: dict[str, int] = {}
    dups: dict[int, int] = {}
    for i in _order(len(keys), priority):
        if keys[i] in kept:
            dups[i] = kept[keys[i]]
        else:
            kept[keys[i]] = i
    return dups


def greedy_clusters(
    vectors: np.ndarray, threshold: float, priority: Sequence[float] | None = None
) -> list[int]:
    """Representative row index for every row (a representative maps to itself)."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    unit = normalize_rows(vectors)
    n = unit.shape[0]
    rep_of = [-1] * n
    reps = np.empty((n, unit.shape[1]), dtype=np.float64)
    rep_rows: list[int] = []
    for i in _order(n, priority):
        if rep_rows:
            sims = reps[: len(rep_rows)] @ unit[i]
            best = int(np.argmax(sims))
            if sims[best] >= threshold - 1e-12:
                rep_of[i] = rep_rows[best]
                continue
        reps[len(rep_rows)] = unit[i]
        rep_rows.append(i)
        rep_of[i] = i
    return rep_of


def cluster_members(rep_of: Sequence[int]) -> dict[int, list[int]]:
    """``{representative: [members including itself]}`` in ascending member order."""
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(rep_of):
        groups.setdefault(r, []).append(i)
    return groups


def mean_pairwise_distance(vectors: np.ndarray, max_items: int = 400, seed: int = 0) -> float:
    """Average cosine distance (1 - cosine) over all pairs; subsampled above ``max_items`` rows."""
    arr = np.asarray(vectors, dtype=np.float64)
    n = arr.shape[0]
    if n < 2:
        return 0.0
    if n > max_items:
        idx = np.random.default_rng(seed).choice(n, size=max_items, replace=False)
        arr = arr[np.sort(idx)]
        n = max_items
    unit = normalize_rows(arr)
    sims = unit @ unit.T
    off_diagonal = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off_diagonal)
