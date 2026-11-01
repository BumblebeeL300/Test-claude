"""End to end ingest -> ask: offline mock backend over the example corpus, and scripted live-style runs."""

from __future__ import annotations

import re

import pytest

from nimlab.atlas import Atlas, AtlasError
from nimlab.atlas.engine import MOCK_WARNING

from .conftest import FakeClient  # noqa: TID252

HEARTBEAT_Q = "How often do robots send heartbeats, and how many missed heartbeats mark a robot as lost?"


# ----------------------------------------------------------------------------- mock backend


async def test_mock_ask_retrieves_the_right_document_and_reports_the_pipeline(indexed: Atlas):
    answer = await indexed.ask(HEARTBEAT_Q)
    assert answer.sources[0].source == "architecture.md"
    assert "Messaging" in answer.sources[0].heading_path
    assert "3 missed heartbeats" in answer.sources[0].text
    assert answer.text and answer.mode == "simple" and answer.collection == "default"
    assert [s.kind for s in answer.trace] == ["retrieve", "generate"]
    assert MOCK_WARNING in answer.warnings
    assert answer.model == indexed.models.chat
    assert answer.usage["calls"] >= 3 and answer.latency_s > 0
    assert answer.verification is None and answer.guard == [] and not answer.blocked


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the API rate limit?", "api_guide.md"),
        ("What first-response time is required for a Sev1 incident?", "incident_runbook.md"),
        ("How many approvals does firmware code need?", "onboarding.md"),
        ("What does the term tote mean?", "glossary.md"),
    ],
)
async def test_mock_ask_finds_the_relevant_file_for_varied_questions(indexed: Atlas, question: str, expected: str):
    answer = await indexed.ask(question, k=4)
    assert expected in {s.source for s in answer.sources[:2]}


async def test_mock_ask_with_verify_uses_the_offline_lexical_judge(indexed: Atlas):
    answer = await indexed.ask(HEARTBEAT_Q, verify=True)
    report = answer.verification
    assert report is not None and report.method == "lexical"
    assert report.claims and all(c.method == "lexical" for c in report.claims)
    assert report.groundedness is not None and 0.0 <= report.groundedness <= 1.0
    # the mock model's prose is generic filler, so the honest verdict is low groundedness
    assert report.groundedness < 0.5
    assert [s.kind for s in answer.trace][-2:] == ["verify", "verify"]


async def test_mock_ask_with_revise_never_crashes_and_keeps_the_original_if_not_better(indexed: Atlas):
    answer = await indexed.ask(HEARTBEAT_Q, revise=True)
    assert answer.verification is not None and answer.text


async def test_mock_agentic_mode_end_to_end(indexed: Atlas):
    answer = await indexed.ask(
        "What must I complete before primary on-call? Who is paged if the primary does not acknowledge?",
        mode="agentic",
        verify=True,
        k=4,
    )
    kinds = [s.kind for s in answer.trace]
    assert kinds[0] == "plan" and "search" in kinds and "merge" in kinds and "verify" in kinds
    assert {"onboarding.md", "incident_runbook.md"} <= {s.source for s in answer.sources}
    assert answer.mode == "agentic" and answer.verification is not None


async def test_unknown_collection_or_empty_collection_returns_a_helpful_message(indexed: Atlas):
    answer = await indexed.ask("anything at all", "nope")
    assert "No documents are indexed" in answer.text and answer.sources == []
    assert any("empty" in w for w in answer.warnings)


async def test_query_without_any_overlap_finds_nothing_relevant(indexed: Atlas):
    answer = await indexed.ask("qxqxqx zzvvzz")
    assert answer.sources == [] and "could not find anything relevant" in answer.text


async def test_invalid_arguments(indexed: Atlas):
    with pytest.raises(AtlasError, match="empty"):
        await indexed.ask("   ")
    with pytest.raises(AtlasError, match="mode"):
        await indexed.ask("hi", mode="turbo")  # type: ignore[arg-type]


async def test_stream_events_arrive_in_order_and_tokens_rebuild_the_answer(indexed: Atlas):
    events = [e async for e in indexed.ask_stream(HEARTBEAT_Q, verify=True)]
    types = [e.type for e in events]
    assert types[-1] == "final" and types.count("final") == 1 and types.count("sources") == 1
    assert types.index("trace") < types.index("sources") < types.index("token") < types.index("final")
    assert types[types.index("token") :].count("trace") >= 2  # verification steps follow the tokens
    streamed = "".join(e.text for e in events if e.type == "token")
    assert streamed.strip() == events[-1].answer.text  # type: ignore[union-attr]
    assert events[-1].answer.verification is not None  # type: ignore[union-attr]


async def test_non_streaming_ask_emits_no_tokens(indexed: Atlas):
    events = [e async for e in indexed.ask_stream(HEARTBEAT_Q, stream_tokens=False)]
    assert not [e for e in events if e.type == "token"] and events[-1].type == "final"


async def test_answer_serialises_to_json(indexed: Atlas):
    answer = await indexed.ask(HEARTBEAT_Q, verify=True)
    data = answer.model_dump(mode="json")
    assert data["sources"][0]["label"].startswith("architecture.md")
    assert {"question", "text", "citations", "sources", "verification", "trace", "warnings"} <= set(data)


async def test_stats_describe_database_and_models(indexed: Atlas):
    stats = indexed.stats()
    assert stats["documents"] == 5 and stats["chunks"] > 20 and stats["mock"] is True
    assert stats["models"]["embed"] == indexed.models.embed
    assert stats["collections"][0]["dim"] == 1024


async def test_open_context_manager_owns_client_and_store(tmp_path, corpus_dir):
    async with Atlas.open(tmp_path / "own.db", mock=True) as atlas:
        assert atlas.mock
        await atlas.ingest(corpus_dir / "glossary.md")
        answer = await atlas.ask("What is a tote?")
        assert answer.sources[0].source == "glossary.md"


# ----------------------------------------------------------------------------- scripted "live" client


def source_number(system_prompt: str, filename: str) -> int:
    match = re.search(rf"^\[(\d+)\] {re.escape(filename)}", system_prompt, re.MULTILINE)
    assert match, f"{filename} not among the sources"
    return int(match.group(1))


def citing_chat(template: str, filename: str = "architecture.md"):
    def chat(model, messages):
        n = source_number(messages[0]["content"], filename)
        return template.format(n=n)

    return chat


async def make_atlas(client, db_path, corpus_dir) -> Atlas:
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    return atlas


async def test_citations_are_mapped_to_chunks_and_hallucinated_ones_removed(db_path, corpus_dir):
    client = FakeClient(chat=citing_chat("Robots send heartbeats every 2 seconds [{n}]. Invented detail [9]."))
    atlas = await make_atlas(client, db_path, corpus_dir)
    answer = await atlas.ask(HEARTBEAT_Q)
    assert answer.text == "Robots send heartbeats every 2 seconds [%d]. Invented detail." % answer.citations[0].n
    (citation,) = answer.citations
    assert citation.source == "architecture.md" and "Messaging" in citation.heading_path
    assert citation in answer.sources and citation.text.startswith("Robots and Hive exchange")
    assert any("non-existent sources: [9]" in w for w in answer.warnings)
    atlas.close()


async def test_answer_without_citations_gets_a_warning_unless_it_abstains(db_path, corpus_dir):
    atlas = await make_atlas(FakeClient(chat="Heartbeats happen regularly."), db_path, corpus_dir)
    answer = await atlas.ask(HEARTBEAT_Q)
    assert answer.citations == [] and any("no inline citations" in w for w in answer.warnings)
    atlas.close()
    atlas = await make_atlas(FakeClient(chat="The provided sources do not cover this."), db_path, corpus_dir)
    answer = await atlas.ask(HEARTBEAT_Q)
    assert not any("no inline citations" in w for w in answer.warnings)
    atlas.close()


async def test_prompt_contains_numbered_sources_and_the_bare_question(db_path, corpus_dir):
    client = FakeClient(chat="Fine [1].")
    atlas = await make_atlas(client, db_path, corpus_dir)
    await atlas.ask(HEARTBEAT_Q, k=3)
    messages = client.chat_calls[-1]["messages"]
    assert messages[-1] == {"role": "user", "content": HEARTBEAT_Q}
    assert len(re.findall(r"^\[\d+\] ", messages[0]["content"], re.MULTILINE)) == 3
    assert client.chat_calls[-1]["temperature"] == atlas.config.temperature
    assert client.chat_calls[-1]["model"] == atlas.models.chat
    atlas.close()


async def test_streaming_uses_stream_chat_and_final_event_has_citations(db_path, corpus_dir):
    client = FakeClient(chat=citing_chat("Heartbeats every 2 seconds [{n}]. Lost after 3 missed [{n}]."))
    atlas = await make_atlas(client, db_path, corpus_dir)
    events = [e async for e in atlas.ask_stream(HEARTBEAT_Q)]
    assert client.chat_calls[-1].get("stream") is True
    tokens = [e.text for e in events if e.type == "token"]
    assert len(tokens) > 5
    final = events[-1].answer
    assert final is not None and "".join(tokens).strip() == final.text
    assert [c.source for c in final.citations] == ["architecture.md"]
    atlas.close()


async def test_truncated_answers_are_flagged(db_path, corpus_dir):
    client = FakeClient(chat="Cut off mid [1]")
    client.finish_reason = "length"
    atlas = await make_atlas(client, db_path, corpus_dir)
    answer = await atlas.ask(HEARTBEAT_Q)
    assert any("cut off" in w for w in answer.warnings)
    atlas.close()


async def test_verification_with_a_scripted_judge_reports_groundedness(db_path, corpus_dir):
    def judge(messages):
        claim = messages[-1]["content"].split("CLAIM\n", 1)[1]
        ok = "2 seconds" in claim
        return {"verdict": "supported" if ok else "unsupported", "evidence": "", "reason": "scripted"}

    client = FakeClient(
        chat=citing_chat("Heartbeats are sent every 2 seconds [{n}]. Robots communicate with telepathy [{n}]."),
        json={"ClaimJudgement": judge},
    )
    atlas = await make_atlas(client, db_path, corpus_dir)
    answer = await atlas.ask(HEARTBEAT_Q, verify=True)
    report = answer.verification
    assert report is not None and report.method == "llm"
    assert report.groundedness == pytest.approx(0.5) and report.unsupported == 1
    assert [c.text for c in report.unsupported_claims] == [
        "Robots communicate with telepathy [%d]." % answer.citations[0].n
    ]
    assert answer.usage["calls"] >= 5  # embed + rerank + answer + two judgements
    atlas.close()


async def test_sources_contain_every_context_passage_even_if_uncited(db_path, corpus_dir):
    atlas = await make_atlas(FakeClient(chat=citing_chat("Only one cite [{n}].")), db_path, corpus_dir)
    answer = await atlas.ask(HEARTBEAT_Q, k=5)
    assert len(answer.sources) == 5 and len(answer.citations) == 1
    atlas.close()
