from __future__ import annotations

import pytest

from nimlab.core.errors import NIMResponseError
from nimlab.council import Council, StepType
from nimlab.council.errors import AllModelsFailedError

Q = "Why is the sky blue?"


def scripted(model: str, messages: list[dict], kind: str) -> str:
    return f"[{kind}] answer by {model}"


async def test_call_structure_and_final_answer(make_client, panel) -> None:
    client = make_client(chat=scripted)
    result = await Council(client).run(Q, "moa", models=panel, layers=2, aggregator=panel[0])

    n = len(panel)
    assert len(client.calls_of("chat")) == n + 2 * n + 1  # proposers + 2 layers + final
    steps = result.trace.steps
    assert [s.type for s in steps if s.type != StepType.CONSENSUS][:n] == [StepType.PROPOSE] * n
    assert sum(s.type == StepType.AGGREGATE for s in steps) == 2 * n + 1
    # one consensus pass per stage (proposers + 2 layers)
    assert len(result.trace.consensus) == 3
    assert result.answer == f"[moa] answer by {panel[0]}"
    assert result.trace.meta["layers_run"] == 2
    assert result.trace.final_answer == result.answer
    assert result.strategy == "moa"


async def test_each_layer_sees_all_previous_answers(make_client, panel) -> None:
    def chat(model: str, messages: list[dict], kind: str) -> str:
        return f"[{kind}] take number {panel.index(model)}"  # no model id in the text

    client = make_client(chat=chat)
    await Council(client).run(Q, "moa", models=panel, layers=1)

    first_layer = client.calls_of("chat", "moa")[: len(panel)]
    proposals = [f"[proposer] take number {i}" for i in range(len(panel))]
    for call in first_layer:
        assert Q in call.user
        for proposal in proposals:
            assert proposal in call.user
        # responses are anonymised: model ids never reach the aggregators
        assert all(m not in call.user for m in panel)

    # the final aggregator sees the layer-1 outputs, not the raw proposals
    final = client.calls_of("chat", "moa")[-1]
    assert "[moa] take number" in final.user
    assert "[proposer] take number" not in final.user


async def test_proposers_run_in_parallel(make_client, panel) -> None:
    client = make_client(chat=scripted, delay=0.01)
    await Council(client).run(Q, "moa", models=panel, layers=0)
    assert client.max_in_flight >= len(panel)


async def test_layers_zero_is_proposers_plus_final(make_client, panel) -> None:
    client = make_client(chat=scripted)
    result = await Council(client).run(Q, "moa", models=panel, layers=0)
    assert len(client.calls_of("chat")) == len(panel) + 1
    assert result.trace.meta["layers_run"] == 0


async def test_failed_proposer_is_recorded_and_skipped(make_client, panel) -> None:
    def chat(model, messages, kind):
        if model == panel[1] and kind == "proposer":
            return NIMResponseError("rate limited", model=model)
        return scripted(model, messages, kind)

    client = make_client(chat=chat)
    result = await Council(client).run(Q, "moa", models=panel, layers=1)

    failed = [s for s in result.trace.steps if s.status == "error"]
    assert len(failed) == 1 and failed[0].model == panel[1]
    assert "rate limited" in (failed[0].error or "")
    # only two answers reach layer 1
    layer_prompt = client.calls_of("chat", "moa")[0].user
    assert layer_prompt.count("### Response") == 2
    assert result.answer


async def test_empty_response_counts_as_failure(make_client, panel) -> None:
    def chat(model, messages, kind):
        return (
            "   " if model == panel[2] and kind == "proposer" else scripted(model, messages, kind)
        )

    result = await Council(make_client(chat=chat)).run(Q, "moa", models=panel, layers=0)
    errors = [s for s in result.trace.steps if s.status == "error"]
    assert len(errors) == 1 and "empty response" in (errors[0].error or "")


async def test_all_proposers_failing_raises_with_trace(make_client, panel) -> None:
    client = make_client(chat=lambda m, msgs, k: NIMResponseError("down", model=m))
    with pytest.raises(AllModelsFailedError) as info:
        await Council(client).run(Q, "moa", models=panel)
    trace = info.value.trace
    assert trace is not None and trace.error and "proposers" in trace.error
    assert len(trace.steps) == len(panel)
    assert all(s.status == "error" for s in trace.steps)


async def test_aggregator_falls_back_to_other_models(make_client, panel) -> None:
    bad = "org-z/aggregator"

    def chat(model, messages, kind):
        if model == bad:
            return NIMResponseError("nope", model=model)
        return scripted(model, messages, kind)

    client = make_client(chat=chat)
    result = await Council(client).run(Q, "moa", models=panel, layers=0, aggregator=bad)
    assert result.trace.meta["aggregator"] == panel[0]
    final_steps = [s for s in result.trace.steps if s.meta.get("final")]
    assert [s.status for s in final_steps] == ["error", "ok"]


async def test_whole_layer_failure_keeps_previous_answers(make_client, panel) -> None:
    calls = {"moa": 0}

    def chat(model, messages, kind):
        if kind == "moa":
            calls["moa"] += 1
            if calls["moa"] <= len(panel):  # the entire first layer fails
                return NIMResponseError("down", model=model)
        return scripted(model, messages, kind)

    result = await Council(make_client(chat=chat)).run(Q, "moa", models=panel, layers=1)
    assert result.trace.meta["layers_run"] == 0
    assert any("Layer 1" in w for w in result.trace.meta["warnings"])
    assert result.answer


async def test_single_model_panel_works(make_client) -> None:
    result = await Council(make_client(chat=scripted)).run(Q, "moa", models=["org/solo"], layers=1)
    assert result.answer and result.consensus is None  # nothing to compare


async def test_moa_end_to_end_in_mock_mode(mock_client) -> None:
    result = await Council(mock_client).run(Q, "moa", layers=2)
    assert result.answer.strip()
    assert len(result.trace.models) == 3
    assert result.trace.mock is True
    assert result.consensus is not None and 0.0 <= result.consensus.agreement <= 1.0
    assert result.trace.total_tokens > 0
    assert result.trace.wall_time_s > 0
