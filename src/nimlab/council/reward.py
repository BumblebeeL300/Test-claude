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
# Nemotron-4-340B-Reward replies with per-attribute scores:
# "helpfulness:3.2,correctness:3.5,coherence:3.7,complexity:1.4,verbosity:1.8"
_ATTRIBUTE = re.compile(
    rf"(helpfulness|correctness|coherence|complexity|verbosity)\s*[:=]\s*({_NUMBER})",
    re.IGNORECASE,
)
_QUALITY = ("helpfulness", "correctness", "coherence")


def parse_reward(text: str | None) -> float | None:
    """Extract the scalar from a reward model reply.

    Accepts ``reward: 1.5`` / ``reward=1.5`` (Llama-3.1-Nemotron-70B-Reward; any case, anywhere in
    the text), a bare number, and the per-attribute reply of Nemotron-4-340B-Reward
    (``helpfulness:3.2,correctness:3.5,coherence:3.7,...``), scored as the mean of helpfulness,
    correctness and coherence. Returns ``None`` for anything else (including ``nan`` / ``inf``),
    signalling the caller to fall back to the rubric judge.
    """
    if not text:
        return None
    match = _LABELLED.search(text) or _BARE.fullmatch(text)
    if not match:
        attrs = {m.group(1).lower(): float(m.group(2)) for m in _ATTRIBUTE.finditer(text)}
        parts = [attrs[k] for k in _QUALITY if k in attrs]
        if not parts:
            return None
        mean = sum(parts) / len(parts)
        return mean if mean == mean and abs(mean) != float("inf") else None
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
