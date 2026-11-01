"""Pairwise battles: position-swapped judging, bias cancellation and Elo integration."""

from __future__ import annotations

from typing import Any

import pytest
from arena_fakes import ScriptedClient
from pydantic import BaseModel

from nimlab.arena import Arena
from nimlab.arena.battle import (
    ANSWER_A_MARK,
    ANSWER_B_MARK,
    BattleEngine,
    PairwiseVerdict,
    build_pairwise_messages,
    position_bias,
)
from nimlab.arena.leaderboard import Leaderboard
from nimlab.arena.results import ItemResult, RunResult, RunStore
from nimlab.arena.suites import Suite
from nimlab.core.errors import NIMAuthError, NIMResponseError


def summ_suite(n: int = 3) -> Suite:
    return Suite.model_validate(
        {
            "name": "summ",
            "category": "summarization",
            "open_ended": True,
            "battle_rubric": "Prefer the longer, more detailed answer.",
            "grader": {"type": "llm_judge", "rubric": "good summary"},
            "items": [
                {"id": f"i{k}", "prompt": f"Summarize text {k}", "reference": f"ref {k}"}
                for k in range(n)
            ],
        }
    )


def run_with_answers(suite: Suite, answers: dict[str, str]) -> RunResult:
    run = RunResult.new(models=list(answers))
    for model, text in answers.items():
        for item in suite.items:
            run.merge_item(
                ItemResult(
                    model=model,
                    suite=suite.name,
                    item_id=item.id,
                    output=f"{text} {item.id}",
                    score=0.5,
                )
            )
    return run


def parse_slots(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user = messages[1]["content"]
    a = user.split(ANSWER_A_MARK + "\n")[1].split("\n\n")[0]
    b = user.split(ANSWER_B_MARK + "\n")[1].split("\n\n")[0]
    return a, b


def longer_wins(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
    a, b = parse_slots(messages)
    return PairwiseVerdict(winner="A" if len(a) > len(b) else "B" if len(b) > len(a) else "tie")


def always_a(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
    return PairwiseVerdict(winner="A", rationale="first looks better")


async def test_every_pair_is_judged_twice_with_swapped_positions() -> None:
    suite = summ_suite(1)
    client = ScriptedClient(judge=longer_wins)
    engine = BattleEngine(client, "judge")
    rec = await engine.compare(suite, suite.items[0], "m1", "short", "m2", "much longer answer")
    assert len(client.chat_json_calls) == 2
    firsts = [parse_slots(msgs) for _, msgs, _ in client.chat_json_calls]
    assert sorted(firsts) == sorted(
        [("short", "much longer answer"), ("much longer answer", "short")]
    )
    assert [(v.slot_a, v.slot_b) for v in rec.verdicts] == [("m1", "m2"), ("m2", "m1")]
    assert rec.score_a == 0.0 and rec.winner == "m2" and rec.consistent
    assert rec.judge_model == "judge"


async def test_position_bias_cancels_to_a_tie() -> None:
    """A judge that always says 'A' must yield 0.5 for both models in every battle."""
    suite = summ_suite(2)
    run = run_with_answers(suite, {"m1": "short", "m2": "a much longer answer than the other"})
    client = ScriptedClient(judge=always_a)
    records = await BattleEngine(client, "judge").run(run, suite, ["m1", "m2"])
    assert len(records) == 2 and len(client.chat_json_calls) == 4
    for rec in records:
        assert rec.score_a == 0.5 and rec.winner is None and not rec.consistent
    assert position_bias(records) == 1.0  # every decisive verdict picked slot A
    board = Leaderboard.from_run(_with(run, records), n_boot=30)
    elos = {s.model: s.elo for s in board.standings}
    assert elos["m1"] == pytest.approx(elos["m2"], abs=1e-6) == pytest.approx(1000, abs=1e-6)
    assert board.position_bias == 1.0


def _with(run: RunResult, records: list) -> RunResult:
    run.battles = records
    return run


async def test_consistent_judge_decides_battles_and_elo_follows() -> None:
    suite = summ_suite(6)
    run = run_with_answers(suite, {"best": "x" * 60, "mid": "x" * 30, "worst": "x" * 5})
    client = ScriptedClient(judge=longer_wins)
    records = await BattleEngine(client, "judge", concurrency=3).run(
        run, suite, ["best", "mid", "worst"]
    )
    assert len(records) == 3 * 6 and len(client.chat_json_calls) == 3 * 6 * 2
    wins = {}
    for rec in records:
        assert rec.consistent and rec.winner
        wins[rec.winner] = wins.get(rec.winner, 0) + 1
    assert wins == {"best": 12, "mid": 6}
    board = Leaderboard.from_run(_with(run, records), n_boot=60)
    assert board.elo_source == "battles" and board.elo_games == 18
    by = {s.model: s for s in board.standings}
    assert by["best"].elo > by["mid"].elo > by["worst"].elo
    assert by["best"].elo_lo <= by["best"].elo <= by["best"].elo_hi
    assert position_bias(records) == pytest.approx(0.5)  # no bias: A and B each win half the time


async def test_mixed_verdicts_give_fractional_scores() -> None:
    suite = summ_suite(1)
    calls = {"n": 0}

    def judge(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
        calls["n"] += 1
        # first ordering: slot A wins (m1); second ordering: tie
        return PairwiseVerdict(winner="A" if calls["n"] == 1 else "tie")

    engine = BattleEngine(ScriptedClient(judge=judge), "judge")
    rec = await engine.compare(suite, suite.items[0], "m1", "x", "m2", "y")
    assert rec.score_a == pytest.approx(0.75)


async def test_battle_uses_rubric_and_reference_in_the_prompt() -> None:
    suite = summ_suite(1)
    msgs = build_pairwise_messages(
        prompt="P", answer_a="AA", answer_b="BB", criteria=suite.battle_rubric, reference="REF"
    )
    text = msgs[1]["content"]
    assert "Prefer the longer" in text and "REF" in text
    assert text.index("AA") < text.index("BB")
    assert (
        "REF" not in build_pairwise_messages(prompt="P", answer_a="A", answer_b="B")[1]["content"]
    )
    client = ScriptedClient(judge=longer_wins)
    await BattleEngine(client, "j", use_reference=False).compare(
        suite, suite.items[0], "a", "x", "b", "y"
    )
    assert all("ref 0" not in m[1]["content"] for _, m, _ in client.chat_json_calls)


async def test_judge_failures_skip_the_battle_and_are_reported() -> None:
    suite = summ_suite(2)
    run = run_with_answers(suite, {"m1": "aaa", "m2": "bbbb"})

    def flaky(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
        a, _ = parse_slots(messages)
        if "i0" in a:
            return NIMResponseError("judge exploded", status_code=400)  # type: ignore[return-value]
        return PairwiseVerdict(winner="tie")

    engine = BattleEngine(ScriptedClient(judge=flaky), "judge", max_retries=0)
    records = await engine.run(run, suite, ["m1", "m2"])
    assert [r.item_id for r in records] == ["i1"]
    assert len(engine.failures) == 1 and "i0" in engine.failures[0]


async def test_auth_errors_propagate_from_battles() -> None:
    suite = summ_suite(1)
    run = run_with_answers(suite, {"m1": "a", "m2": "b"})
    client = ScriptedClient(judge=lambda *a: NIMAuthError("nope"))  # type: ignore[arg-type,return-value]
    with pytest.raises(NIMAuthError):
        await BattleEngine(client, "judge").run(run, suite, ["m1", "m2"])


async def test_max_items_max_pairs_and_missing_answers() -> None:
    suite = summ_suite(5)
    run = run_with_answers(suite, {"a": "1", "b": "22", "c": "333"})
    client = ScriptedClient(judge=longer_wins)
    engine = BattleEngine(client, "j")
    assert len(await engine.run(run, suite, ["a", "b", "c"], max_items=2)) == 2 * 3
    assert len(await engine.run(run, suite, ["a", "b", "c"], max_pairs=1, seed=3)) == 5
    run.items = [r for r in run.items if not (r.model == "c" and r.item_id == "i4")]
    assert len(await engine.run(run, suite, ["a", "b", "c"])) == 5 * 3 - 2


async def test_retries_transient_judge_errors() -> None:
    suite = summ_suite(1)
    state = {"n": 0}

    def judge(model: str, messages: list[Any], schema: type[BaseModel]) -> BaseModel:
        state["n"] += 1
        if state["n"] <= 2:
            return NIMResponseError("503", status_code=503)  # type: ignore[return-value]
        return PairwiseVerdict(winner="tie")

    engine = BattleEngine(ScriptedClient(judge=judge), "j", max_retries=3, backoff_s=0.0)
    rec = await engine.compare(suite, suite.items[0], "a", "x", "b", "y")
    assert rec.score_a == 0.5 and not engine.failures


async def test_arena_battle_end_to_end_on_the_simulator(runs_dir: Any) -> None:
    models = [
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.2-1b-instruct",
    ]
    async with Arena(mock=True, runs_dir=runs_dir, concurrency=8) as arena:
        result = await arena.battle(models, "summarization", max_items=6)
        assert len(result.battles) == 3 * 6
        assert all(len(b.verdicts) == 2 for b in result.battles)
        board = arena.leaderboard(result, n_boot=40)
    assert board.elo_source == "battles"
    elo = {s.model: s.elo for s in board.standings}
    assert elo[models[0]] > elo[models[2]]  # the 405B outranks the 1B in the simulation
    assert RunStore(runs_dir).load(result.run_id).battles  # persisted
    assert result.judge_model == arena.judge_model


async def test_arena_battle_needs_two_distinct_models(runs_dir: Any) -> None:
    async with Arena(mock=True, runs_dir=runs_dir) as arena:
        with pytest.raises(ValueError, match="at least two"):
            await arena.battle(["only-one", "only-one"])
