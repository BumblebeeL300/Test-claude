"""Consensus metrics: pairwise cosine similarity, agreement score and outlier detection.

The maths is pure numpy and independent of where the vectors come from, so it is trivially
testable. :func:`lexical_vectors` is an offline fallback used when no embedding model is reachable.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .records import ConsensusReport

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    re.findall(
        r"\w+",
        "a an the of and or to in on for with is are was were be been being it its this that "
        "these those as at by from but not no do does did can could should would will may might "
        "have has had i you he she we they them his her our your their my me what which who how "
        "why when where if then than so such into about over under also",
    )
)


def cosine_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Pairwise cosine similarity (n x n). Zero vectors have similarity 0 to everything else."""
    if len(vectors) == 0:
        return np.zeros((0, 0))
    mat = np.asarray(vectors, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {mat.shape}")
    norms = np.linalg.norm(mat, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = mat / safe[:, None]
    sims = unit @ unit.T
    sims = np.clip(sims, -1.0, 1.0)
    zero = norms == 0.0
    sims[zero, :] = 0.0
    sims[:, zero] = 0.0
    np.fill_diagonal(sims, 1.0)
    return sims


def lexical_vectors(texts: Sequence[str], dim: int = 512) -> np.ndarray:
    """Deterministic hashed bag-of-words vectors (sub-linear tf, L2-normalised)."""
    out = np.zeros((len(texts), dim), dtype=float)
    for row, text in enumerate(texts):
        counts = Counter(t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1)
        for tok, c in counts.items():
            out[row, zlib.crc32(tok.encode("utf-8")) % dim] += 1.0 + math.log(c)
        norm = np.linalg.norm(out[row])
        if norm:
            out[row] /= norm
    return out


def agreement_score(matrix: np.ndarray) -> float:
    """Mean off-diagonal similarity clipped to ``[0, 1]``; a single answer agrees with itself."""
    n = matrix.shape[0]
    if n < 2:
        return 1.0
    off = matrix[~np.eye(n, dtype=bool)]
    return float(min(1.0, max(0.0, off.mean())))


def centrality(matrix: np.ndarray) -> np.ndarray:
    """Mean similarity of each item to all the others."""
    n = matrix.shape[0]
    if n < 2:
        return np.ones(n)
    return (matrix.sum(axis=1) - np.diag(matrix)) / (n - 1)


def find_outliers(
    cent: np.ndarray, *, z: float = 1.0, margin: float = 0.1, min_items: int = 3
) -> list[int]:
    """Indices whose centrality is lower than ``mean - max(margin, z * std)``.

    Needs at least ``min_items`` answers (with two, "the odd one out" is undefined). The absolute
    ``margin`` stops tiny, meaningless differences from being flagged when everyone agrees.
    """
    if cent.shape[0] < min_items:
        return []
    mean = float(cent.mean())
    std = float(cent.std())
    cutoff = mean - max(margin, z * std)
    return [int(i) for i in np.flatnonzero(cent < cutoff)]


def build_report(
    vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    labels: Sequence[str],
    models: Sequence[str],
    phase: str,
    method: str = "embedding",
    embed_model: str | None = None,
    outlier_z: float = 1.0,
    outlier_margin: float = 0.1,
) -> ConsensusReport:
    """Assemble a :class:`ConsensusReport` from one vector per answer."""
    if len(labels) != len(vectors) or len(models) != len(vectors):
        raise ValueError("labels, models and vectors must have the same length")
    matrix = cosine_matrix(vectors)
    n = matrix.shape[0]
    cent = centrality(matrix)
    off = matrix[~np.eye(n, dtype=bool)] if n > 1 else np.array([1.0])
    return ConsensusReport(
        phase=phase,
        method="lexical" if method == "lexical" else "embedding",
        embed_model=embed_model,
        labels=list(labels),
        models=list(models),
        matrix=[[round(float(x), 6) for x in row] for row in matrix],
        agreement=round(agreement_score(matrix), 6),
        min_pairwise=round(float(off.min()), 6),
        centrality=[round(float(x), 6) for x in cent],
        outliers=find_outliers(cent, z=outlier_z, margin=outlier_margin),
        medoid=int(np.argmax(cent)) if n else None,
    )
