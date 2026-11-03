"""Scoring: the Nemotron reward model first, an LLM-judge rubric as the fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nimlab.core import NIMAuthError, NIMError
from nimlab.council.reward import parse_reward

from .gateway import Gateway
from .prompts import JudgeScore, judge_messages, reward_messages

__all__ = ["Score", "Scorer", "parse_reward"]


@dataclass(frozen=True)
class Score:
    """A score and its source. Reward and judge scales differ: never compare across sources."""

    value: float
    source: Literal["reward", "judge"]


class Scorer:
    """Scores (prompt, response) pairs.

    The reward model is asked for ``reward:<float>``; if it errors or answers with something
    :func:`parse_reward` cannot read, the judge model grades the answer 0-10 instead.
    """

    def __init__(
        self, gateway: Gateway, reward_model: str, judge_model: str, *, use_reward: bool = True
    ) -> None:
        self.gateway = gateway
        self.reward_model = reward_model
        self.judge_model = judge_model
        self.use_reward = use_reward

    async def reward(self, prompt: str, response: str) -> Score | None:
        if not self.use_reward:
            return None
        try:
            result = await self.gateway.chat(
                self.reward_model, reward_messages(prompt, response), temperature=0.0, max_tokens=32
            )
        except NIMAuthError:
            raise
        except NIMError:
            return None
        value = parse_reward(result.content)
        return None if value is None else Score(value, "reward")

    async def judge(self, prompt: str, response: str) -> Score | None:
        try:
            graded = await self.gateway.chat_json(
                self.judge_model, judge_messages(prompt, response), JudgeScore, temperature=0.0
            )
        except NIMAuthError:
            raise
        except NIMError:
            return None
        return Score(float(graded.score), "judge")

    async def score(self, prompt: str, response: str) -> Score | None:
        """Reward score if available, else judge score, else ``None``."""
        return await self.reward(prompt, response) or await self.judge(prompt, response)

    async def score_pair(self, prompt: str, first: str, second: str) -> tuple[Score, Score] | None:
        """Score two responses on one scale (judging both when the sources would differ)."""
        a = await self.score(prompt, first)
        b = await self.score(prompt, second)
        if a is None or b is None:
            return None
        if a.source != b.source:
            ja = await self.judge(prompt, first)
            jb = await self.judge(prompt, second)
            return (ja, jb) if ja is not None and jb is not None else None
        return a, b
