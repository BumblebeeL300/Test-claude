"""Hybrid retrieval: dense + BM25 -> RRF -> rerank -> MMR -> context packing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nimlab.core import NIMAuthError, NIMError

from .config import AtlasConfig, ModelSet
from .fusion import minmax, mmr_select, pack_context, reciprocal_rank_fusion
from .models import SourceRef, StoredChunk
from .store import AtlasStore, normalize_rows


@dataclass
class Hit:
    """One retrieved chunk with the score of every stage that touched it."""

    chunk: StoredChunk
    score: float = 0.0  # final ordering score (rerank score, else fused score)
    dense: float | None = None
    bm25: float | None = None
    fused: float | None = None
    rerank: float | None = None
    text: str | None = None  # packed (possibly truncated) text; None means chunk.text

    @property
    def id(self) -> int:
        return self.chunk.id

    @property
    def context_text(self) -> str:
        return self.text if self.text is not None else self.chunk.text

    def to_source(self, n: int) -> SourceRef:
        c = self.chunk
        return SourceRef(
            n=n,
            chunk_id=c.id,
            source=c.doc_name,
            path=c.doc_path,
            title=c.doc_title,
            heading_path=list(c.heading_path),
            page=c.page,
            text=self.context_text,
            score=round(float(self.score), 6),
            dense_score=None if self.dense is None else round(float(self.dense), 6),
            bm25_score=None if self.bm25 is None else round(float(self.bm25), 6),
            rerank_score=None if self.rerank is None else round(float(self.rerank), 6),
        )


@dataclass
class RetrievalResult:
    hits: list[Hit]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def sources_from_hits(hits: Sequence[Hit]) -> list[SourceRef]:
    """Number hits ``1..n`` in order."""
    return [h.to_source(i) for i, h in enumerate(hits, start=1)]


def merge_hits(
    lists: Sequence[Sequence[Hit]], *, rrf_k: int = 60, limit: int | None = None
) -> list[Hit]:
    """Merge several retrieval results (e.g. one per sub-question) with RRF over chunk ids.

    The merged hit keeps the best-scoring stage details seen for a chunk; ``score`` becomes the RRF
    score, so the ordering reflects agreement across searches.
    """
    fused = reciprocal_rank_fusion([[h.id for h in lst] for lst in lists], k=rrf_k)
    best: dict[int, Hit] = {}
    for lst in lists:
        for h in lst:
            cur = best.get(h.id)
            if cur is None or h.score > cur.score:
                best[h.id] = h
    merged = [
        Hit(
            chunk=best[cid].chunk,
            score=score,
            dense=best[cid].dense,
            bm25=best[cid].bm25,
            fused=score,
            rerank=best[cid].rerank,
            text=best[cid].text,
        )
        for cid, score in fused
    ]
    return merged[:limit] if limit is not None else merged


class Retriever:
    """Runs the retrieval pipeline against one :class:`AtlasStore`."""

    def __init__(self, store: AtlasStore, client: Any, config: AtlasConfig, models: ModelSet):
        self.store = store
        self.client = client
        self.config = config
        self.models = models

    # ------------------------------------------------------------------ query embedding
    async def query_vector(self, query: str, hyde_text: str | None = None) -> np.ndarray:
        """Unit-length query vector; with ``hyde_text`` it is blended with the HyDE embedding."""
        coros = [self.client.embed(self.models.embed, [query], input_type="query")]
        if hyde_text:
            coros.append(self.client.embed(self.models.embed, [hyde_text], input_type="passage"))
        results = await asyncio.gather(*coros)
        q = normalize_rows(np.asarray(results[0][0]))[0]
        if hyde_text:
            h = normalize_rows(np.asarray(results[1][0]))[0]
            w = self.config.hyde_weight
            q = normalize_rows((1.0 - w) * q + w * h)[0]
        return q

    # ------------------------------------------------------------------ pipeline
    async def retrieve(
        self,
        query: str,
        collection: str,
        *,
        k: int | None = None,
        hyde_text: str | None = None,
        use_rerank: bool | None = None,
        use_mmr: bool | None = None,
        context_chars: int | None = None,
    ) -> RetrievalResult:
        cfg = self.config
        k = k or cfg.k
        use_rerank = cfg.use_rerank if use_rerank is None else use_rerank
        use_mmr = cfg.use_mmr if use_mmr is None else use_mmr
        budget = context_chars or cfg.context_chars
        t0 = time.perf_counter()
        warnings: list[str] = []
        index = self.store.index(collection)
        stats: dict[str, Any] = {"query": query, "collection": collection, "indexed": len(index)}
        if not len(index):
            return RetrievalResult([], warnings, stats)

        # 1) dense + sparse candidates
        dense: list[tuple[int, float]] = []
        try:
            dense = index.dense_top(await self.query_vector(query, hyde_text), cfg.candidates)
        except NIMAuthError:
            raise
        except NIMError as exc:
            warnings.append(f"dense retrieval unavailable ({exc}); falling back to BM25 only")
        sparse = index.bm25_top(query, cfg.candidates)
        stats["dense_candidates"] = len(dense)
        stats["bm25_candidates"] = len(sparse)

        # 2) reciprocal rank fusion
        rankings = [[p for p, _ in dense], [p for p, _ in sparse]]
        weights = [cfg.dense_weight, cfg.bm25_weight]
        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k, weights=weights)
        dense_score = dict(dense)
        sparse_score = dict(sparse)
        pool_size = max(cfg.rerank_candidates, k)
        pool = [
            Hit(
                chunk=index.chunks[pos],
                score=score,
                dense=dense_score.get(pos),
                bm25=sparse_score.get(pos),
                fused=score,
            )
            for pos, score in fused[:pool_size]
        ]
        stats["fused_candidates"] = len(fused)
        positions = [index.id_to_pos[h.id] for h in pool]

        # 3) cross-encoder rerank
        reranked = False
        if use_rerank and len(pool) > 1:
            try:
                results = await self.client.rerank(
                    self.models.rerank, query, [h.chunk.indexed_text for h in pool]
                )
            except NIMAuthError:
                raise
            except NIMError as exc:
                warnings.append(f"reranker unavailable ({exc}); keeping fused order")
            else:
                scores = {r.index: float(r.score) for r in results if 0 <= r.index < len(pool)}
                floor = min(scores.values()) - 1.0 if scores else 0.0
                for i, hit in enumerate(pool):
                    hit.rerank = scores.get(i)
                    hit.score = hit.rerank if hit.rerank is not None else floor
                order = sorted(range(len(pool)), key=lambda i: (-pool[i].score, i))
                pool = [pool[i] for i in order]
                positions = [positions[i] for i in order]
                reranked = True
        stats["reranked"] = reranked

        # 4) MMR diversification over the (re)ranked pool
        selected = list(range(min(len(pool), k)))
        diversified = False
        if use_mmr and len(pool) > 1 and cfg.mmr_lambda < 1.0:
            rel = minmax([h.score for h in pool])
            selected = mmr_select(rel, index.matrix[positions], k, cfg.mmr_lambda)
            diversified = True
        stats["mmr"] = diversified
        chosen = [pool[i] for i in selected]

        # 5) pack into the character budget
        packed = pack_context([h.chunk.text for h in chosen], budget, k)
        hits: list[Hit] = []
        for i, text in packed:
            hit = chosen[i]
            if text != hit.chunk.text:
                hit.text = text
            hits.append(hit)
        stats["returned"] = len(hits)
        stats["elapsed_s"] = round(time.perf_counter() - t0, 4)
        return RetrievalResult(hits, warnings, stats)
