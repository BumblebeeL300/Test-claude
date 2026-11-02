from __future__ import annotations

import pytest

from nimlab.council.reward import parse_reward


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reward:-3.5", -3.5),
        ("reward:-12.3", -12.3),
        ("reward: 4", 4.0),
        ("Reward = 0.25", 0.25),
        ("REWARD:+7.5", 7.5),
        ("reward:-1.2e-1", -0.12),
        ("  -7.1  ", -7.1),  # bare number
        ("0", 0.0),
        ("The model said reward:2.0 for this one", 2.0),
        ("reward:1 reward:2", 1.0),  # first match wins
    ],
)
def test_parse_reward_valid(text: str, expected: float) -> None:
    assert parse_reward(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "reward:",
        "reward: great",
        "this answer is good",
        "reward:nan",
        "7 out of 10",  # a number inside prose is not a reward
        "reward:--3",
    ],
)
def test_parse_reward_malformed_returns_none(text: str | None) -> None:
    assert parse_reward(text) is None
