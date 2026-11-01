"""In-house Okapi BM25 over an inverted index (numpy-vectorised scoring)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


class BM25Index:
    """Okapi BM25 with the non-negative Lucene IDF.

    ``score(d, q) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))``

    with ``idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))``. Each distinct query term counts once.
    Documents are addressed by their position in the list passed to :meth:`build`.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_len = np.zeros(0, dtype=np.float32)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}

    @classmethod
    def build(
        cls, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        """Index pre-tokenised documents."""
        index = cls(k1=k1, b=b)
        index.n_docs = len(docs)
        index.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        index.avgdl = float(index.doc_len.mean()) if docs else 0.0
        staging: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                ids, tfs = staging.setdefault(term, ([], []))
                ids.append(doc_id)
                tfs.append(tf)
        n = index.n_docs
        for term, (ids, tfs) in staging.items():
            index._postings[term] = (
                np.asarray(ids, dtype=np.int32),
                np.asarray(tfs, dtype=np.float32),
            )
            df = len(ids)
            index._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return index

    def __len__(self) -> int:
        return self.n_docs

    def idf(self, term: str) -> float:
        """IDF of ``term`` (0.0 when unseen)."""
        return self._idf.get(term, 0.0)

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """BM25 score of every document for the query (float64 vector of length ``n_docs``)."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs or self.avgdl <= 0:
            return out
        for term in set(query_tokens):
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[ids] / self.avgdl)
            out[ids] += self._idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm)
        return out

    def search(self, query_tokens: Sequence[str], k: int = 10) -> list[tuple[int, float]]:
        """Top-``k`` ``(doc_index, score)`` pairs with a positive score, best first.

        Ties are broken by document position so results are deterministic.
        """
        if k <= 0:
            return []
        scores = self.scores(query_tokens)
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        order = positive[np.lexsort((positive, -scores[positive]))]
        return [(int(i), float(scores[i])) for i in order[:k]]
