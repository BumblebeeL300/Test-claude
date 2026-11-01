from __future__ import annotations

import pytest

from nimlab.core.errors import NIMResponseError
from nimlab.council import Council, JudgeVerdict, StepType
from nimlab.council.errors import NotEnoughModelsError

Q = "Which is best: ALPHA, BETA or GAMMA?"

# Initial answers point in orthogonal directions (see the keyword embedding in conftest).
FIRST = {
    "org-a/model-a": "I say ALPHA",
    "org-b/model-b": "I say BETA",
    "org-c/model-c": "I say GAMMA",
}


def divergent_then_agree(*, agree_after: int):
    """Models disagree initially; from revision number ``agree_after`` on they all say ALPHA."""
    revisions = {"n": 0}

    def chat(model: str, messages: list[dict], kind: str) -> str:
        if kind == "proposer":
            return FIRST[model]
        if kind == "debate_critique":
            return "Peers have gaps; ALPHA looks strongest."
        if kind == "debate_revise":
            revisions["n"] += 1
            round_no = (revisions["n"] - 1) // 3 + 1  # three models revise per round
            return "ALPHA is right" if round_no >= agree_after else f"still {FIRST[model][-5:]}"
        return "unexpected"

    return chat


def judge_pick(selected: int | None = 1, final: str = "", rationale: str = "because"):
    def fn(model, messages, schema, kind):
        assert kind == "judge", kind
        return JudgeVerdict(
            selected=selected, final_answer=final, rationale=rationale, confidence=0.8
        )

    return fn


async def test_early_stop_on_convergence(make_client, panel) -> None:
    client = make_client(chat=divergent_then_agree(agree_after=1), json=judge_pick())
    result = await Council(client).run(Q, "debate", models=panel, rounds=3)

    meta = result.trace.meta
    assert meta["rounds_run"] == 1 and meta["rounds_max"] == 3
    assert meta["stopped_early"] is True and meta["converged"] is True
    assert meta["agreement_by_round"][0] < 0.5 and meta["agreement_by_round"][-1] > 0.9
    # exactly: 3 proposals, 3 critiques, 3 revisions, 1 judge - nothing for rounds 2 and 3
    assert len(client.calls_of("chat")) == 9
    assert len(client.calls_of("json")) == 1


async def test_no_convergence_runs_all_rounds(make_client, panel) -> None:
    client = make_client(chat=divergent_then_agree(agree_after=99), json=judge_pick())
    result = await Council(client).run(Q, "debate", models=panel, rounds=2)

    meta = result.trace.meta
    assert meta["rounds_run"] == 2 and meta["stopped_early"] is False
    assert meta["converged"] is False
    n = len(panel)
    assert len(client.calls_of("chat")) == n + 2 * (2 * n)
    types = [s.type for s in result.trace.steps]
    assert types.count(StepType.CRITIQUE) == 2 * n
    assert types.count(StepType.REVISE) == 2 * n
    assert len(meta["agreement_by_round"]) == 3  # round 0, 1, 2


async def test_initial_consensus_skips_the_debate(make_client, panel) -> None:
    client = make_client(chat=lambda m, msgs, k: "ALPHA, obviously", json=judge_pick())
    result = await Council(client).run(Q, "debate", models=panel, rounds=3)
    assert result.trace.meta["rounds_run"] == 0
    assert result.trace.meta["stopped_early"] is True
    assert not [s for s in result.trace.steps if s.type in (StepType.CRITIQUE, StepType.REVISE)]
    assert len(client.calls_of("chat")) == len(panel)


async def test_early_stop_can_be_disabled(make_client, panel) -> None:
    client = make_client(chat=lambda m, msgs, k: "ALPHA, obviously", json=judge_pick())
    result = await Council(client).run(Q, "debate", models=panel, rounds=2, early_stop=False)
    assert result.trace.meta["rounds_run"] == 2


async def test_threshold_is_configurable(make_client, panel) -> None:
    def chat(model, messages, kind):
        return {
            "org-a/model-a": "ALPHA",
            "org-b/model-b": "ALPHA BETA",
            "org-c/model-c": "ALPHA",
        }.get(model, "x")

    # agreement is ~0.8: stops at threshold 0.7, debates at threshold 0.99
    loose = make_client(chat=chat, json=judge_pick())
    r1 = await Council(loose).run(Q, "debate", models=panel, rounds=2, convergence_threshold=0.7)
    assert r1.trace.meta["rounds_run"] == 0
    strict = make_client(chat=chat, json=judge_pick())
    r2 = await Council(strict).run(Q, "debate", models=panel, rounds=2, convergence_threshold=0.99)
    assert r2.trace.meta["rounds_run"] == 2


async def test_critique_phase_can_be_disabled(make_client, panel) -> None:
    client = make_client(chat=divergent_then_agree(agree_after=99), json=judge_pick())
    result = await Council(client).run(Q, "debate", models=panel, rounds=1, debate_critique=False)
    assert not result.trace.steps_of(StepType.CRITIQUE)
    assert len(result.trace.steps_of(StepType.REVISE)) == len(panel)


async def test_peers_only_in_critique_and_revise_prompts(make_client, panel) -> None:
    client = make_client(chat=divergent_then_agree(agree_after=99), json=judge_pick())
    await Council(client).run(Q, "debate", models=panel, rounds=1)

    crit = client.calls_of("chat", "debate_critique")
    assert len(crit) == len(panel)
    for call, own in zip(crit, panel, strict=True):
        assert call.model == own
        assert FIRST[own] not in call.user  # a model never critiques itself
        assert call.user.count("### Response") == len(panel) - 1

    revise = client.calls_of("chat", "debate_revise")
    for call in revise:
        assert f"Your previous answer:\n{FIRST[call.model]}" in call.user
        assert "Critiques written by your peers" in call.user
        assert call.user.count("### Critique") == len(panel) - 1  # others' critiques only


async def test_judge_selection_marks_candidate(make_client, panel) -> None:
    client = make_client(
        chat=lambda m, msgs, k: FIRST.get(m, "x"), json=judge_pick(2, "", "B wins")
    )
    result = await Council(client).run(Q, "debate", models=panel, rounds=0)
    assert result.answer == FIRST[panel[1]]
    assert result.rationale == "B wins"
    assert result.selected is not None and result.selected.model == panel[1]
    assert [c.selected for c in result.candidates] == [False, True, False]
    assert result.trace.meta["judge"] and result.trace.meta["confidence"] == pytest.approx(0.8)


async def test_judge_synthesis_overrides_selection(make_client, panel) -> None:
    client = make_client(
        chat=lambda m, msgs, k: FIRST.get(m, "x"),
        json=judge_pick(None, "A synthesis of all", "mixed"),
    )
    result = await Council(client).run(Q, "debate", models=panel, rounds=0)
    assert result.answer == "A synthesis of all"
    assert result.selected is None


async def test_invalid_selection_index_falls_back_to_medoid(make_client, panel) -> None:
    client = make_client(chat=lambda m, msgs, k: FIRST.get(m, "x"), json=judge_pick(7, "", "??"))
    result = await Council(client).run(Q, "debate", models=panel, rounds=0)
    assert result.answer in FIRST.values()
    assert "central" in (result.rationale or "")


async def test_judge_failure_uses_next_model_then_medoid(make_client, panel) -> None:
    judge_model = "org-z/judge"

    def judge(model, messages, schema, kind):
        if model == judge_model:
            return NIMResponseError("judge down", model=model)
        return JudgeVerdict(selected=3, final_answer="", rationale="backup judge")

    client = make_client(chat=lambda m, msgs, k: FIRST.get(m, "x"), json=judge)
    result = await Council(client).run(Q, "debate", models=panel, rounds=0, judge=judge_model)
    assert result.answer == FIRST[panel[2]]
    assert result.trace.meta["judge"] == panel[0]
    assert [c.model for c in client.calls_of("json")] == [judge_model, panel[0]]

    def all_down(model, messages, schema, kind):
        return NIMResponseError("down", model=model)

    client2 = make_client(chat=lambda m, msgs, k: FIRST.get(m, "x"), json=all_down)
    result2 = await Council(client2).run(Q, "debate", models=panel, rounds=0)
    assert result2.answer in FIRST.values()
    assert "no judge verdict" in (result2.rationale or "")


async def test_failed_revision_keeps_previous_answer(make_client, panel) -> None:
    def chat(model, messages, kind):
        if kind == "debate_revise" and model == panel[0]:
            return NIMResponseError("flaky", model=model)
        return divergent_then_agree(agree_after=99)(model, messages, kind)

    client = make_client(chat=chat, json=judge_pick())
    result = await Council(client).run(Q, "debate", models=panel, rounds=1)
    kept = next(c for c in result.candidates if c.model == panel[0])
    assert kept.text == FIRST[panel[0]]
    assert any(s.status == "error" and s.type == StepType.REVISE for s in result.trace.steps)


async def test_debate_needs_two_models(make_client) -> None:
    with pytest.raises(NotEnoughModelsError):
        await Council(make_client()).run(Q, "debate", models=["org/only"])


async def test_debate_in_mock_mode(mock_client) -> None:
    result = await Council(mock_client).run("Is a tomato a fruit?", "debate", rounds=2)
    assert result.answer.strip()
    assert result.trace.meta["rounds_run"] <= 2
    assert result.trace.steps_of(StepType.JUDGE)
    assert result.consensus is not None
