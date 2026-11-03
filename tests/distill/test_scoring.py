"""Reward parsing and the reward -> judge fallback."""

from __future__ import annotations

from typing import Any

import pytest

from nimlab.core import ChatResult, NIMAuthError, NIMResponseError
from nimlab.distill import parse_reward
from nimlab.distill.costs import Tally
from nimlab.distill.gateway import Gateway
from nimlab.distill.prompts import JudgeScore
from nimlab.distill.scoring import Scorer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reward:-12.5", -12.5),
        ("reward: 3", 3.0),
        ("Reward = +1.25e1", 12.5),
        ("the final reward:  -0.75 (low)", -0.75),
        ("  4.5 ", 4.5),
        ("-7", -7.0),
        ("reward:nan", None),
        ("reward:inf", None),
        ("no score here", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_reward(text: str | None, expected: float | None) -> None:
    assert parse_reward(text) == expected


class FakeClient:
    """Scripted client: ``reward_replies`` feed the reward model, ``judge`` the rubric."""

    def __init__(self, reward_replies: list[Any], judge: float | Exception = 8.0) -> None:
        self.reward_replies = list(reward_replies)
        self.judge = judge
        self.reward_calls = 0
        self.judge_calls = 0

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        self.reward_calls += 1
        reply = self.reward_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(model=model, content=reply)

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[JudgeScore], **kw: Any
    ) -> JudgeScore:
        self.judge_calls += 1
        if isinstance(self.judge, Exception):
            raise self.judge
        return schema(score=self.judge, rationale="ok")


def scorer_for(client: FakeClient, *, use_reward: bool = True) -> Scorer:
    return Scorer(Gateway(client, Tally()), "rm", "judge", use_reward=use_reward)


async def test_reward_model_score_is_used_first() -> None:
    client = FakeClient(["reward:-3.5"])
    score = await scorer_for(client).score("q", "a")
    assert score is not None and (score.value, score.source) == (-3.5, "reward")
    assert client.judge_calls == 0


async def test_unparseable_reward_falls_back_to_judge() -> None:
    client = FakeClient(["I cannot rate this"], judge=7.5)
    score = await scorer_for(client).score("q", "a")
    assert score is not None and (score.value, score.source) == (7.5, "judge")


async def test_reward_error_falls_back_to_judge_but_auth_error_propagates() -> None:
    client = FakeClient([NIMResponseError("boom", status_code=500)], judge=6.0)
    score = await scorer_for(client).score("q", "a")
    assert score is not None and score.source == "judge"
    with pytest.raises(NIMAuthError):
        await scorer_for(FakeClient([NIMAuthError("bad key", status_code=401)])).score("q", "a")


async def test_both_failing_returns_none_and_reward_can_be_disabled() -> None:
    both = FakeClient(["nope"], judge=NIMResponseError("down", status_code=500))
    assert await scorer_for(both).score("q", "a") is None
    judge_only = FakeClient([], judge=9.0)
    score = await scorer_for(judge_only, use_reward=False).score("q", "a")
    assert score is not None and score.source == "judge" and judge_only.reward_calls == 0


async def test_pair_scores_share_one_scale() -> None:
    mixed = FakeClient(["reward:2.0", "garbled"], judge=5.0)
    pair = await scorer_for(mixed).score_pair("q", "a", "b")
    assert pair is not None
    assert {s.source for s in pair} == {"judge"}  # a reward score is never compared with a judge's
    same = FakeClient(["reward:2.0", "reward:-1.0"])
    pair = await scorer_for(same).score_pair("q", "a", "b")
    assert pair is not None and [s.value for s in pair] == [2.0, -1.0]
    assert await scorer_for(FakeClient(["x", "x"], judge=NIMResponseError("d", status_code=500))).score_pair("q", "a", "b") is None
