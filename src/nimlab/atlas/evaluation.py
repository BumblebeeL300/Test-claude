"""Evaluation harness: retrieval recall@k, MRR and answer groundedness over a QA set."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from nimlab.core import NIMAuthError

from .config import DEFAULT_COLLECTION
from .engine import Atlas
from .models import AtlasError, Mode, SourceRef


class QAItem(BaseModel):
    """One evaluation question. ``sources`` lists the files that contain the answer."""

    question: str
    sources: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    id: str | None = None


class ItemResult(BaseModel):
    id: str
    question: str
    expected: list[str]
    retrieved: list[str] = Field(description="Source of each retrieved passage, best first")
    first_rank: int | None = Field(description="1-based rank of the first relevant passage")
    recall: dict[int, float] = Field(default_factory=dict)
    keyword_coverage: float | None = None
    groundedness: float | None = None
    unsupported_claims: int | None = None
    latency_s: float = 0.0
    answer: str | None = None
    error: str | None = None

    @property
    def hit(self) -> bool:
        return self.first_rank is not None


class EvalReport(BaseModel):
    name: str = ""
    collection: str
    mode: str
    k: int
    n_questions: int
    ks: list[int]
    recall_at_k: dict[int, float]
    mrr: float
    hit_rate: float
    keyword_coverage: float | None = None
    mean_groundedness: float | None = None
    answered: bool = False
    mock: bool = False
    elapsed_s: float = 0.0
    items: list[ItemResult] = Field(default_factory=list)


def load_qa(path: str | Path) -> tuple[str, list[QAItem]]:
    """Load a QA file: ``{"name":..., "items":[...]}`` or a bare list of items."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtlasError(f"cannot read QA file {path}: {exc}") from exc
    name = ""
    if isinstance(data, dict):
        name = str(data.get("name", ""))
        data = data.get("items") or data.get("questions") or []
    if not isinstance(data, list) or not data:
        raise AtlasError("QA file must contain a non-empty list of questions")
    items: list[QAItem] = []
    for i, raw in enumerate(data, start=1):
        if isinstance(raw, str):
            raw = {"question": raw}
        if not isinstance(raw, dict):
            raise AtlasError(f"QA item {i} must be an object")
        if "sources" not in raw and "source" in raw:
            raw = {**raw, "sources": [raw["source"]]}
        raw = {k: v for k, v in raw.items() if k != "source"}
        item = QAItem.model_validate(raw)
        if item.id is None:
            item.id = f"q{i}"
        items.append(item)
    return name, items


def source_matches(expected: str, source: SourceRef) -> bool:
    """Does an expected file name refer to this passage's document?"""
    exp = expected.replace("\\", "/").strip().lstrip("./")
    candidates = (source.source, source.path.replace("\\", "/"))
    return any(
        c == exp or c.endswith("/" + exp) or PurePosixPath(c).name == PurePosixPath(exp).name
        for c in candidates
    )


def retrieval_metrics(
    expected: list[str], retrieved: list[SourceRef], ks: list[int]
) -> tuple[int | None, dict[int, float]]:
    """``(rank of first relevant passage, {k: recall@k})`` for one question."""
    first: int | None = None
    for rank, src in enumerate(retrieved, start=1):
        if any(source_matches(e, src) for e in expected):
            first = rank
            break
    recall: dict[int, float] = {}
    for k in ks:
        top = retrieved[:k]
        found = sum(1 for e in expected if any(source_matches(e, s) for s in top))
        recall[k] = found / len(expected) if expected else 0.0
    return first, recall


async def run_eval(
    atlas: Atlas,
    items: list[QAItem],
    *,
    collection: str = DEFAULT_COLLECTION,
    k: int = 6,
    mode: Mode = "simple",
    answer: bool = True,
    verify: bool = True,
    concurrency: int = 2,
    name: str = "",
) -> EvalReport:
    """Run the QA set. With ``answer=False`` only retrieval is measured (no generation calls)."""
    ks = sorted({1, 3, k})
    sem = asyncio.Semaphore(max(1, concurrency))
    t0 = time.perf_counter()

    async def one(item: QAItem) -> ItemResult:
        async with sem:
            start = time.perf_counter()
            result = ItemResult(
                id=item.id or "?",
                question=item.question,
                expected=item.sources,
                retrieved=[],
                first_rank=None,
            )
            try:
                if answer:
                    ans = await atlas.ask(
                        item.question, collection, mode=mode, k=k, verify=verify
                    )
                    sources = ans.sources
                    result.answer = ans.text
                    if ans.verification:
                        result.groundedness = ans.verification.groundedness
                        result.unsupported_claims = ans.verification.unsupported
                elif mode == "agentic":
                    ans = await atlas.ask(item.question, collection, mode=mode, k=k)
                    sources = ans.sources
                else:
                    sources = await atlas.retrieve(item.question, collection, k=k)
            except (AtlasError, NIMAuthError):
                raise
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                result.latency_s = time.perf_counter() - start
                return result
            result.retrieved = [s.source for s in sources]
            if item.sources:
                result.first_rank, result.recall = retrieval_metrics(item.sources, sources, ks)
            if item.keywords:
                blob = " ".join(s.text for s in sources[:k]).lower()
                result.keyword_coverage = sum(kw.lower() in blob for kw in item.keywords) / len(
                    item.keywords
                )
            result.latency_s = time.perf_counter() - start
            return result

    results = list(await asyncio.gather(*(one(i) for i in items)))
    scored = [r for r in results if r.expected and r.error is None]
    n = len(scored)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    recall_at_k = {j: (mean([r.recall.get(j, 0.0) for r in scored]) or 0.0) for j in ks}
    mrr = mean([1.0 / r.first_rank if r.first_rank else 0.0 for r in scored]) or 0.0
    hit_rate = (sum(1 for r in scored if r.hit) / n) if n else 0.0
    grounded = [r.groundedness for r in results if r.groundedness is not None]
    kw = [r.keyword_coverage for r in results if r.keyword_coverage is not None]
    stats: dict[str, Any] = {
        "name": name,
        "collection": collection,
        "mode": mode,
        "k": k,
        "n_questions": len(items),
        "ks": ks,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "hit_rate": hit_rate,
        "keyword_coverage": mean(kw),
        "mean_groundedness": mean([g for g in grounded if g is not None]),
        "answered": answer,
        "mock": atlas.mock,
        "elapsed_s": time.perf_counter() - t0,
        "items": results,
    }
    return EvalReport(**stats)
