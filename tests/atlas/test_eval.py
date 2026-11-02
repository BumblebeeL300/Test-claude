"""Evaluation harness: QA loading, retrieval metrics and a full run over the example corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nimlab.atlas import Atlas, AtlasError, SourceRef, load_qa, run_eval
from nimlab.atlas.evaluation import QAItem, retrieval_metrics, source_matches


def ref(n: int, source: str, path: str | None = None) -> SourceRef:
    return SourceRef(
        n=n, chunk_id=n, source=source, path=path or f"/data/{source}", text=f"text {n}"
    )


def test_shipped_qa_file_is_valid_and_points_at_real_files(qa_file: Path, corpus_dir: Path):
    name, items = load_qa(qa_file)
    assert name and len(items) >= 10
    assert len({i.id for i in items}) == len(items)
    on_disk = {p.name for p in corpus_dir.glob("*.md")}
    assert all(i.question and i.sources and set(i.sources) <= on_disk for i in items)
    assert any(len(i.sources) > 1 for i in items)  # includes multi-source questions


def test_load_qa_accepts_several_shapes(tmp_path: Path):
    f = tmp_path / "qa.json"
    f.write_text(
        json.dumps(
            [
                {"question": "q1", "source": "a.md"},
                "plain question",
                {"question": "q3", "sources": ["b.md"], "id": "x"},
            ]
        )
    )
    name, items = load_qa(f)
    assert name == "" and [i.id for i in items] == ["q1", "q2", "x"]
    assert items[0].sources == ["a.md"] and items[1].sources == []
    f.write_text(json.dumps({"name": "set", "questions": [{"question": "q"}]}))
    assert load_qa(f)[0] == "set"


@pytest.mark.parametrize("content", ["not json", "[]", "{}", "[42]", '{"items": []}'])
def test_load_qa_rejects_bad_files(tmp_path: Path, content: str):
    f = tmp_path / "bad.json"
    f.write_text(content)
    with pytest.raises(AtlasError):
        load_qa(f)
    with pytest.raises(AtlasError):
        load_qa(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("expected", "source", "path", "match"),
    [
        ("api.md", "api.md", "/x/api.md", True),
        ("docs/api.md", "docs/api.md", "/x/docs/api.md", True),
        ("api.md", "docs/api.md", "/x/docs/api.md", True),  # basename match
        ("./api.md", "api.md", "/x/api.md", True),
        ("api.md", "other.md", "/x/other.md", False),
        ("guide.md", "myguide.md", "/x/myguide.md", False),
    ],
)
def test_source_matching(expected: str, source: str, path: str, match: bool):
    assert source_matches(expected, ref(1, source, path)) is match


def test_retrieval_metrics_rank_and_recall():
    retrieved = [ref(1, "x.md"), ref(2, "a.md"), ref(3, "y.md"), ref(4, "b.md")]
    first, recall = retrieval_metrics(["a.md", "b.md"], retrieved, [1, 2, 3, 4])
    assert first == 2
    assert recall == {1: 0.0, 2: 0.5, 3: 0.5, 4: 1.0}
    assert retrieval_metrics(["zzz.md"], retrieved, [4]) == (None, {4: 0.0})
    assert retrieval_metrics(["x.md"], retrieved, [1]) == (1, {1: 1.0})
    assert retrieval_metrics([], retrieved, [3]) == (None, {3: 0.0})


async def test_full_eval_run_over_the_example_corpus(indexed: Atlas, qa_file: Path):
    name, items = load_qa(qa_file)
    report = await run_eval(indexed, items, k=6, name=name)
    assert report.n_questions == len(items) and report.answered and report.mock
    assert report.ks == [1, 3, 6]
    assert (
        report.recall_at_k[6] >= 0.9
        and report.recall_at_k[1] <= report.recall_at_k[3] <= report.recall_at_k[6]
    )
    assert report.mrr >= 0.8 and report.hit_rate >= 0.9
    assert report.keyword_coverage is not None and report.keyword_coverage > 0.5
    assert report.mean_groundedness is not None and 0.0 <= report.mean_groundedness <= 1.0
    assert all(r.error is None and r.latency_s > 0 and r.answer for r in report.items)
    json.loads(report.model_dump_json())  # serialisable


async def test_retrieval_only_eval_makes_no_generation_calls(indexed: Atlas, qa_file: Path):
    _, items = load_qa(qa_file)
    chat_calls = (
        indexed.client.usage.summary()["models"].get(indexed.models.chat, {}).get("calls", 0)
    )
    report = await run_eval(indexed, items, k=6, answer=False)
    after = indexed.client.usage.summary()["models"].get(indexed.models.chat, {}).get("calls", 0)
    assert after == chat_calls and not report.answered
    assert report.mean_groundedness is None and report.recall_at_k[6] >= 0.9
    assert all(r.answer is None for r in report.items)


async def test_agentic_eval_run(indexed: Atlas):
    items = [
        QAItem(
            question=(
                "What must I complete before on-call? "
                "Who is paged when the primary does not acknowledge?"
            ),
            sources=["onboarding.md", "incident_runbook.md"],
        )
    ]
    chat_before = (
        indexed.client.usage.summary()["models"].get(indexed.models.chat, {}).get("calls", 0)
    )
    report = await run_eval(indexed, items, k=6, mode="agentic", answer=False)
    assert report.recall_at_k[6] == 1.0 and report.mode == "agentic"
    chat_after = (
        indexed.client.usage.summary()["models"].get(indexed.models.chat, {}).get("calls", 0)
    )
    assert chat_after == chat_before  # retrieval-only: the answer model is never called


async def test_questions_without_expected_sources_are_not_scored(indexed: Atlas):
    items = [
        QAItem(question="What is a tote?"),
        QAItem(question="What is the API key header?", sources=["api_guide.md"]),
    ]
    report = await run_eval(indexed, items, k=3, answer=False)
    assert report.n_questions == 2 and report.hit_rate == 1.0
    assert report.items[0].first_rank is None and report.items[0].recall == {}


async def test_one_failing_question_is_recorded_not_fatal(
    indexed: Atlas, monkeypatch: pytest.MonkeyPatch
):
    original = indexed.retrieve

    async def flaky(query, *args, **kwargs):
        if "explode" in query:
            raise RuntimeError("boom")
        return await original(query, *args, **kwargs)

    monkeypatch.setattr(indexed, "retrieve", flaky)
    items = [
        QAItem(question="explode please", sources=["api_guide.md"]),
        QAItem(question="API key header", sources=["api_guide.md"]),
    ]
    report = await run_eval(indexed, items, k=3, answer=False)
    assert report.items[0].error == "RuntimeError: boom" and report.items[1].hit
    assert (
        report.n_questions == 2 and report.hit_rate == 1.0
    )  # failed items are excluded from the metrics
