"""Scoring candidate answers: NVIDIA reward model first, LLM-as-judge rubric as the fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .context import RunContext
from .prompts import reward_messages, rubric_messages
from .records import StepType
from .schemas import RubricScore

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LABELLED = re.compile(rf"reward\s*[:=]\s*({_NUMBER})", re.IGNORECASE)
_BARE = re.compile(rf"\s*({_NUMBER})\s*")


def parse_reward(text: str | None) -> float | None:
    """Extract the scalar from a reward model reply such as ``reward:-12.3``.

    Accepts ``reward: 1.5`` / ``reward=1.5`` (any case, anywhere in the text) and a bare number.
    Returns ``None`` for anything else (including ``nan`` / ``inf``), signalling the caller to
    fall back to the rubric judge.
    """
    if not text:
        return None
    match = _LABELLED.search(text) or _BARE.fullmatch(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:  # pragma: no cover - regex guarantees a float literal
        return None
    return value if value == value and abs(value) != float("inf") else None


@dataclass(frozen=True)
class Score:
    """A candidate's score and where it came from (scales differ: don't mix sources)."""

    value: float
    source: Literal["reward", "judge"]
    rationale: str = ""


async def reward_score(ctx: RunContext, question: str, answer: str, label: str) -> Score | None:
    """Score with the reward model; ``None`` if unavailable, failed, or the reply is unparseable."""
    model = ctx.roles.reward
    if not model or not ctx.config.use_reward_model or ctx.mock:
        return None
    result = await ctx.try_chat(
        model,
        reward_messages(question, answer),
        type=StepType.SCORE,
        label=label,
        temperature=0.0,
        max_tokens=64,
        meta={"scorer": "reward"},
    )
    if result is None:
        return None
    value = parse_reward(result.content)
    if value is None:
        ctx.trace.meta.setdefault("warnings", []).append(
            f"{label}: unparseable reward reply {result.content[:60]!r}; falling back to judge"
        )
        return None
    return Score(value, "reward", result.content.strip())


async def judge_score(ctx: RunContext, question: str, answer: str, label: str) -> Score | None:
    """LLM-as-judge rubric score on a 0-10 scale (tries the judge, then the panel)."""
    msgs = rubric_messages(question, answer)
    for model in dict.fromkeys([ctx.roles.judge, *ctx.models]):
        rubric = await ctx.try_json(
            model,
            msgs,
            RubricScore,
            type=StepType.SCORE,
            label=label,
            temperature=0.0,
            meta={"scorer": "judge"},
        )
        if rubric is not None:
            return Score(rubric.score, "judge", rubric.rationale)
    return None
