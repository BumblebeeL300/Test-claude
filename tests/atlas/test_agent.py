"""Agentic mode: planning, HyDE, reflection with follow-up searches, offline heuristics."""

from __future__ import annotations

import pytest

from nimlab.atlas import Atlas, AtlasConfig
from nimlab.atlas.agent import AgenticRetriever, AgentState, heuristic_subquestions
from nimlab.core import NIMError

from .conftest import FakeClient


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the rate limit?", ["What is the rate limit?"]),
        (
            "What must I finish before on-call? Who gets paged without acknowledgement?",
            ["What must I finish before on-call?", "Who gets paged without acknowledgement?"],
        ),
        (
            "Explain the heartbeat timeout; describe the aisle lock epoch",
            ["Explain the heartbeat timeout?", "describe the aisle lock epoch?"],
        ),
        ("What is X? ok", ["What is X? ok"]),
    ],
)
def test_heuristic_subquestions(question: str, expected: list[str]):
    assert heuristic_subquestions(question) == expected


def test_heuristic_subquestions_are_capped():
    q = " ".join(f"What about topic number {i}?" for i in range(10))
    assert len(heuristic_subquestions(q, 3)) == 3


def agent_client(*, plan=None, reflect=None, **kw) -> FakeClient:
    scripts = {}
    if plan is not None:
        scripts["QueryPlan"] = plan
    if reflect is not None:
        scripts["Reflection"] = reflect
    return FakeClient(chat="Answer with citation [1].", json=scripts, **kw)


async def test_agentic_mode_searches_the_question_and_every_sub_question(db_path, corpus_dir):
    client = agent_client(
        plan={
            "sub_questions": [
                "on-call prerequisites shadow rotations",
                "PagerDuty escalation secondary",
            ],
            "rationale": "two hops",
        },
        reflect={"sufficient": True},
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask(
        "What do I need before on-call, and who is paged if there is no ack?", mode="agentic"
    )
    kinds = [s.kind for s in answer.trace]
    assert kinds[0] == "plan" and kinds.count("search") == 3 and "reflect" in kinds
    assert kinds.index("merge") > max(i for i, k in enumerate(kinds) if k == "search")
    assert kinds[-1] == "generate" and kinds.index("generate") > kinds.index("merge")
    searched = [s.data["query"] for s in answer.trace if s.kind == "search"]
    assert searched[0].startswith("What do I need before on-call")
    assert "PagerDuty escalation secondary" in searched
    files = {s.source for s in answer.sources}
    assert {"onboarding.md", "incident_runbook.md"} <= files
    assert [c["schema"] for c in client.json_calls] == ["QueryPlan", "Reflection"]
    # the planner and reflector use the cheap model, the answer uses the chat model
    assert client.json_calls[0]["model"] == atlas.models.fast
    assert client.chat_calls[-1]["model"] == atlas.models.chat
    system = client.chat_calls[-1]["messages"][0]["content"]
    assert "several parts" in system and "PagerDuty escalation secondary" in system
    atlas.close()


async def test_reflection_can_request_follow_up_searches(db_path, corpus_dir):
    reflections = [
        {
            "sufficient": False,
            "missing": "who is paged",
            "follow_up_queries": ["PagerDuty hive-secondary escalation"],
        },
        {"sufficient": True},
    ]
    client = agent_client(
        plan={"sub_questions": ["shadow rotation requirements"]},
        reflect=lambda m: reflections.pop(0),
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("What about on-call?", mode="agentic")
    searches = [s for s in answer.trace if s.kind == "search"]
    hops = [s.data["hop"] for s in searches]
    assert hops == [0, 0, 1] and searches[-1].data["query"] == "PagerDuty hive-secondary escalation"
    assert "incident_runbook.md" in {s.source for s in answer.sources}
    reflect_steps = [s for s in answer.trace if s.kind == "reflect"]
    assert "incomplete" in reflect_steps[0].message and "sufficient" in reflect_steps[1].message
    atlas.close()


async def test_hop_limit_stops_the_loop(db_path, corpus_dir):
    client = agent_client(
        plan={"sub_questions": []},
        reflect=lambda m: {
            "sufficient": False,
            "follow_up_queries": [f"never satisfied {len(m[-1]['content'])}"],
        },
    )
    atlas = Atlas(client, db_path, config=AtlasConfig(max_hops=1))
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("heartbeat", mode="agentic")
    assert max(s.data["hop"] for s in answer.trace if s.kind == "search") == 1
    assert sum(c["schema"] == "Reflection" for c in client.json_calls) == 1
    atlas.close()


async def test_max_hops_zero_skips_reflection_entirely(db_path, corpus_dir):
    client = agent_client(plan={"sub_questions": []})
    atlas = Atlas(client, db_path, config=AtlasConfig(max_hops=0))
    await atlas.ingest(corpus_dir)
    await atlas.ask("heartbeat", mode="agentic")
    assert [c["schema"] for c in client.json_calls] == ["QueryPlan"]
    atlas.close()


async def test_follow_ups_that_repeat_earlier_queries_end_the_loop(db_path, corpus_dir):
    client = agent_client(
        plan={"sub_questions": []},
        reflect={"sufficient": False, "follow_up_queries": ["heartbeat"]},
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("heartbeat", mode="agentic")
    assert sum(s.kind == "search" for s in answer.trace) == 1
    atlas.close()


async def test_planner_failure_falls_back_to_the_question(db_path, corpus_dir):
    client = agent_client(plan=NIMError("planner down"), reflect=NIMError("reflector down"))
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("What is the API key header?", mode="agentic")
    assert answer.sources and answer.sources[0].source == "api_guide.md"
    assert "using question" in answer.trace[0].data["rationale"]
    atlas.close()


async def test_hyde_adds_a_trace_step_and_a_passage_embedding(db_path, corpus_dir):
    client = agent_client(plan={"sub_questions": []}, reflect={"sufficient": True})
    client._chat = lambda model, messages: (
        "The allocator is restarted with kubectl rollout restart."
        if "Write a short passage" in messages[0]["content"]
        else "Done [1]."
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    client.embed_calls.clear()
    answer = await atlas.ask("How do I restart the allocator?", mode="agentic", hyde=True)
    assert any(s.kind == "hyde" for s in answer.trace)
    assert ("passage", 1) in client.embed_calls
    simple = await atlas.ask("How do I restart the allocator?", hyde=True)
    assert any(s.kind == "hyde" for s in simple.trace)
    atlas.close()


async def test_hyde_failure_degrades_gracefully(db_path, corpus_dir):
    def chat(model, messages):
        if "Write a short passage" in messages[0]["content"]:
            raise NIMError("llm down")
        return "Fine [1]."

    atlas = Atlas(FakeClient(chat=chat), db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("How do I restart the allocator?", hyde=True)
    assert answer.sources and answer.text == "Fine [1]."
    assert any("HyDE generation failed" in w for w in answer.warnings)
    assert not any(s.kind == "hyde" for s in answer.trace)
    atlas.close()


async def test_mock_mode_uses_offline_heuristics_without_llm_planning(indexed: Atlas):
    before = indexed.client.usage.summary()["models"].get(indexed.models.fast, {}).get("calls", 0)
    answer = await indexed.ask(
        "What must I finish before primary on-call? Who is paged if nobody acknowledges?",
        mode="agentic",
    )
    after = indexed.client.usage.summary()["models"].get(indexed.models.fast, {}).get("calls", 0)
    assert after == before  # no planner / reflection calls offline
    plan = answer.trace[0]
    assert plan.kind == "plan" and len(plan.data["queries"]) == 3
    assert {"onboarding.md", "incident_runbook.md"} <= {s.source for s in answer.sources}
    assert len(answer.sources) >= 6


async def test_agent_state_collects_queries_and_hits(db_path, corpus_dir):
    client = agent_client(
        plan={"sub_questions": ["rate limit requests"]}, reflect={"sufficient": True}
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    agent = AgenticRetriever(client, atlas.retriever, atlas.models, atlas.config)
    state = AgentState()
    steps = [
        s
        async for s in agent.explore("API rate limit", "default", k=4, use_hyde=False, state=state)
    ]
    assert [s.kind for s in steps][-1] == "merge"
    assert state.queries == ["API rate limit", "rate limit requests"] and state.rounds == 1
    assert state.hits and state.hits[0].chunk.doc_name == "api_guide.md"
    atlas.close()
