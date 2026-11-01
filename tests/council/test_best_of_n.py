from __future__ import annotations

import pytest

from nimlab.core.errors import NIMResponseError
from nimlab.council import Council, RubricScore, StepType
from nimlab.council.errors import AllModelsFailedError
from nimlab.council.strategies.best_of_n import sampling_plan, temperature_schedule

Q = "What is 2 + 2?"
REWARD_MODEL = "nvidia/llama-3.1-nemotron-70b-reward"


def answers(model: str, messages: list[dict], kind: str) -> str:
    """Each sample gets a distinct, recognisable answer (the temperature picks which)."""
    return f"answer-{model[-1]}"


def reward_by_text(table: dict[str, str]):
    """Reward model replies ``table[answer]`` for the assistant message it is asked to score."""

    def chat(model: str, messages: list[dict], kind: str) -> str:
        if kind == "reward":
            return table[messages[-1]["content"]]
        return answers(model, messages, kind)

    return chat


def rubric(value: float = 5.0):
    def fn(model, messages, schema, kind):
        assert kind == "rubric"
        return RubricScore(score=value, rationale="ok")

    return fn


def test_temperature_schedule_spreads_evenly() -> None:
    assert temperature_schedule(1, 0.3, 1.0) == [0.3]
    assert temperature_schedule(3, 0.0, 1.0) == [0.0, 0.5, 1.0]
    temps = temperature_schedule(5, 0.3, 1.0)
    assert temps == sorted(temps) and temps[0] == 0.3 and temps[-1] == 1.0


def test_sampling_plan_rotates_models_and_varies_temperature() -> None:
    plan = sampling_plan(["a", "b"], 5, 0.2, 1.0)
    assert [m for m, _ in plan] == ["a", "b", "a", "b", "a"]
    assert len({t for _, t in plan}) == 5
    assert len(set(plan)) == 5  # no (model, temperature) pair repeats


async def test_reward_model_picks_the_highest_reward(make_client, panel) -> None:
    table = {"answer-a": "reward:-12.3", "answer-b": "reward:-3.5", "answer-c": "reward:-8.0"}
    client = make_client(chat=reward_by_text(table))
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)

    assert result.answer == "answer-b"  # -3.5 is the greatest (least negative) reward
    assert result.selected is not None and result.selected.score == pytest.approx(-3.5)
    assert {c.score_source for c in result.candidates} == {"reward"}
    assert sorted(c.score for c in result.candidates) == [-12.3, -8.0, -3.5]
    assert result.trace.meta["scorer"] == "reward"
    assert "reward" in (result.rationale or "") and "-3.50" in (result.rationale or "")
    assert not client.calls_of("json")  # the judge was never needed


async def test_reward_model_receives_user_assistant_pair(make_client, panel) -> None:
    table = {"answer-a": "reward:1", "answer-b": "reward:2"}
    client = make_client(chat=reward_by_text(table))
    await Council(client).run(Q, "best-of-n", models=panel[:2], n=2, reward_model=REWARD_MODEL)

    calls = client.calls_of("chat", "reward")
    assert len(calls) == 2 and all(c.model == REWARD_MODEL for c in calls)
    for call in calls:
        assert [m["role"] for m in call.messages] == ["user", "assistant"]
        assert call.messages[0]["content"] == Q
        assert call.messages[1]["content"] in table
        assert call.kwargs["temperature"] == 0.0


async def test_malformed_reward_falls_back_to_judge_for_all(make_client, panel) -> None:
    # one candidate's reward is garbage -> mixed scales -> everything is regraded by the judge
    table = {"answer-a": "reward:5.0", "answer-b": "I think it is good", "answer-c": "reward:1.0"}
    scores = {"answer-a": 3.0, "answer-b": 9.0, "answer-c": 4.0}

    def judge(model, messages, schema, kind):
        text = messages[-1]["content"]
        return RubricScore(score=next(v for k, v in scores.items() if k in text), rationale="r")

    client = make_client(chat=reward_by_text(table), json=judge)
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)

    assert result.answer == "answer-b"
    assert {c.score_source for c in result.candidates} == {"judge"}
    assert result.trace.meta["scorer"] == "judge"
    warnings = result.trace.meta["warnings"]
    assert any("unparseable reward" in w for w in warnings)
    assert any("mixed" in w for w in warnings)
    # one judge call for the garbage reward, then all three regraded by the judge
    assert len(client.calls_of("json")) == 4


async def test_all_malformed_rewards_use_judge_once_each(make_client, panel) -> None:
    client = make_client(
        chat=reward_by_text({"answer-a": "n/a", "answer-b": "n/a", "answer-c": "n/a"}),
        json=rubric(6.0),
    )
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)
    assert len(client.calls_of("json")) == 3
    assert {c.score_source for c in result.candidates} == {"judge"}
    assert all(c.score == 6.0 for c in result.candidates)


async def test_mock_mode_skips_reward_model_and_uses_judge(make_client, panel) -> None:
    client = make_client(mock=True, json=rubric(7.5))
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)
    assert not client.calls_of("chat", "reward")
    assert {c.score_source for c in result.candidates} == {"judge"}


async def test_reward_can_be_disabled(make_client, panel) -> None:
    client = make_client(json=rubric(4.0))
    await Council(client).run(Q, "best-of-n", models=panel, n=2, use_reward_model=False)
    assert not client.calls_of("chat", "reward")


async def test_reward_model_error_falls_back_to_judge(make_client, panel) -> None:
    def chat(model, messages, kind):
        if kind == "reward":
            return NIMResponseError("reward endpoint 500", model=model)
        return answers(model, messages, kind)

    client = make_client(chat=chat, json=rubric(8.0))
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)
    assert {c.score_source for c in result.candidates} == {"judge"}
    assert any(s.status == "error" and s.type == StepType.SCORE for s in result.trace.steps)


async def test_ties_break_toward_the_most_central_candidate(make_client, panel) -> None:
    # all scores are equal; two samples agree (ALPHA) and one is off-topic (BETA)
    texts = {panel[0]: "ALPHA one", panel[1]: "BETA two", panel[2]: "ALPHA three"}
    client = make_client(chat=lambda m, msgs, k: texts[m], json=rubric(5.0), mock=True)
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)
    assert result.selected is not None and result.selected.text.startswith("ALPHA")
    assert result.selected.label == "Candidate 1"  # earliest of the two central candidates
    assert result.consensus is not None and result.consensus.medoid in (0, 2)


async def test_exact_ties_pick_the_earliest(make_client, panel) -> None:
    client = make_client(chat=lambda m, msgs, k: "same", json=rubric(5.0), mock=True)
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)
    assert result.selected is not None and result.selected.label == "Candidate 1"


async def test_n_larger_than_panel_cycles_models(make_client, panel) -> None:
    client = make_client(json=rubric(1.0), mock=True)
    result = await Council(client).run(Q, "best-of-n", models=panel[:2], n=5)
    sampled = client.calls_of("chat", "proposer")
    assert [c.model for c in sampled] == [panel[0], panel[1], panel[0], panel[1], panel[0]]
    assert len({c.kwargs["temperature"] for c in sampled}) == 5
    assert len(result.candidates) == 5
    # temperature is recorded on the trace steps and on the candidates
    assert all(c.meta["temperature"] is not None for c in result.candidates)


async def test_samples_are_drawn_in_parallel(make_client, panel) -> None:
    client = make_client(json=rubric(1.0), mock=True, delay=0.01)
    await Council(client).run(Q, "best-of-n", models=panel, n=4)
    assert client.max_in_flight >= 4


async def test_failed_samples_are_dropped(make_client, panel) -> None:
    def chat(model, messages, kind):
        if model == panel[0]:
            return NIMResponseError("down", model=model)
        return answers(model, messages, kind)

    client = make_client(chat=chat, json=rubric(5.0), mock=True)
    result = await Council(client).run(Q, "best-of-n", models=panel, n=3)
    assert len(result.candidates) == 2
    assert result.trace.meta["n_requested"] == 3 and result.trace.meta["n_sampled"] == 2


async def test_no_scorer_available_raises(make_client, panel) -> None:
    def judge(model, messages, schema, kind):
        return NIMResponseError("down", model=model)

    client = make_client(mock=True, json=judge)
    with pytest.raises(AllModelsFailedError) as info:
        await Council(client).run(Q, "best-of-n", models=panel, n=2)
    assert info.value.trace is not None and info.value.trace.error


async def test_best_of_n_in_mock_mode(mock_client) -> None:
    result = await Council(mock_client).run("Name a prime number", "best-of-n", n=4)
    assert result.answer.strip()
    assert len(result.candidates) == 4
    assert sum(c.selected for c in result.candidates) == 1
    assert all(c.score is not None for c in result.candidates)
