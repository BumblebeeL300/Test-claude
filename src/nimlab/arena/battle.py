"""Pairwise LLM-judged battles with position-swap debiasing."""

from __future__ import annotations

import asyncio
import itertools
import random
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from nimlab.core.errors import NIMAuthError

from .llm import is_transient
from .results import BattleRecord, RunResult, SlotVerdict
from .suites import EvalItem, Suite

__all__ = [
    "ANSWER_A_MARK",
    "ANSWER_B_MARK",
    "REFERENCE_MARK",
    "BattleEngine",
    "PairwiseVerdict",
    "build_pairwise_messages",
    "position_bias",
]

ANSWER_A_MARK = "### Answer A"
ANSWER_B_MARK = "### Answer B"
REFERENCE_MARK = "### Reference answer (guidance only)"

PAIRWISE_SYSTEM = (
    "You are an impartial judge comparing two answers to the same task. Decide which answer is "
    "better according to the criteria. Do not favour an answer because of its position, length or "
    "style. Say 'tie' only if they are genuinely equal in quality. Reply with JSON only."
)
DEFAULT_CRITERIA = (
    "Prefer the answer that is more accurate, more complete and follows the task's constraints."
)


class PairwiseVerdict(BaseModel):
    """What the judge must return for one ordering."""

    winner: Literal["A", "B", "tie"] = Field(description="Which answer is better")
    rationale: str = Field(default="", description="One or two sentences")


def build_pairwise_messages(
    *,
    prompt: str,
    answer_a: str,
    answer_b: str,
    criteria: str | None = None,
    reference: str | None = None,
) -> list[dict[str, Any]]:
    parts = [f"## Task\n{prompt.strip()}", f"## Criteria\n{(criteria or DEFAULT_CRITERIA).strip()}"]
    if reference:
        parts.append(f"{REFERENCE_MARK}\n{reference.strip()}")
    parts.append(f"{ANSWER_A_MARK}\n{answer_a.strip() or '(empty)'}")
    parts.append(f"{ANSWER_B_MARK}\n{answer_b.strip() or '(empty)'}")
    parts.append('Return JSON: {"winner": "A" | "B" | "tie", "rationale": "<why>"}')
    return [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _slot_score(pick: str) -> float:
    return {"A": 1.0, "B": 0.0}.get(pick, 0.5)


def position_bias(records: list[BattleRecord]) -> float | None:
    """Share of decisive verdicts that picked slot A (0.5 = no position bias)."""
    picks = [v.pick for r in records for v in r.verdicts if v.pick != "tie"]
    return picks.count("A") / len(picks) if picks else None


class BattleEngine:
    """Runs pairwise comparisons of already generated answers.

    Every pair of answers is judged **twice with the slots swapped**; a model's score for the
    battle is the mean of its two verdict scores (win 1, tie 0.5, loss 0). A judge that merely
    prefers the first slot therefore produces 0.5, cancelling position bias.
    """

    def __init__(
        self,
        client: Any,
        judge_model: str,
        *,
        concurrency: int = 4,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        use_reference: bool = True,
    ) -> None:
        self.client = client
        self.judge_model = judge_model
        self.concurrency = max(1, concurrency)
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.use_reference = use_reference
        self.failures: list[str] = []

    async def _judge(
        self, suite: Suite, item: EvalItem, first: tuple[str, str], second: tuple[str, str]
    ) -> SlotVerdict:
        """Judge with ``first`` in slot A and ``second`` in slot B (each ``(model, answer)``)."""
        messages = build_pairwise_messages(
            prompt=item.prompt,
            answer_a=first[1],
            answer_b=second[1],
            criteria=suite.battle_rubric,
            reference=item.reference if self.use_reference else None,
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                verdict = await self.client.chat_json(
                    self.judge_model, messages, PairwiseVerdict, temperature=0.0, max_tokens=300
                )
                break
            except NIMAuthError:
                raise
            except Exception as exc:
                if attempt > self.max_retries or not is_transient(exc):
                    raise
                await asyncio.sleep(
                    self.backoff_s * 2 ** (attempt - 1) * (0.5 + random.random() / 2)
                )
        winner = first[0] if verdict.winner == "A" else second[0] if verdict.winner == "B" else None
        return SlotVerdict(
            slot_a=first[0],
            slot_b=second[0],
            pick=verdict.winner,
            winner=winner,
            rationale=verdict.rationale[:300],
        )

    async def compare(
        self, suite: Suite, item: EvalItem, model_a: str, answer_a: str, model_b: str, answer_b: str
    ) -> BattleRecord:
        """Judge both orderings and combine them into one :class:`BattleRecord`."""
        first, second = await asyncio.gather(
            self._judge(suite, item, (model_a, answer_a), (model_b, answer_b)),
            self._judge(suite, item, (model_b, answer_b), (model_a, answer_a)),
        )

        def score_for_a(v: SlotVerdict) -> float:
            s = _slot_score(v.pick)
            return s if v.slot_a == model_a else 1.0 - s

        return BattleRecord(
            suite=suite.name,
            item_id=item.id,
            model_a=model_a,
            model_b=model_b,
            score_a=(score_for_a(first) + score_for_a(second)) / 2,
            verdicts=[first, second],
            judge_model=self.judge_model,
        )

    async def run(
        self,
        run: RunResult,
        suite: Suite,
        models: list[str],
        *,
        max_items: int | None = None,
        max_pairs: int | None = None,
        seed: int = 0,
        on_progress: Callable[[int, int, BattleRecord], None] | None = None,
    ) -> list[BattleRecord]:
        """Battle all pairs of ``models`` on ``suite`` using the answers stored in ``run``."""
        answers = {r.key: r.output for r in run.items if r.scored and r.output.strip()}
        pairs = list(itertools.combinations(dict.fromkeys(models), 2))
        if max_pairs is not None and len(pairs) > max_pairs:
            pairs = random.Random(seed).sample(pairs, max_pairs)
        items = suite.items[:max_items] if max_items else suite.items
        jobs = [
            (item, a, b)
            for item in items
            for a, b in pairs
            if (a, suite.name, item.id) in answers and (b, suite.name, item.id) in answers
        ]
        sem = asyncio.Semaphore(self.concurrency)
        records: list[BattleRecord] = []

        async def one(item: EvalItem, a: str, b: str) -> None:
            async with sem:
                try:
                    rec = await self.compare(
                        suite,
                        item,
                        a,
                        answers[(a, suite.name, item.id)],
                        b,
                        answers[(b, suite.name, item.id)],
                    )
                except NIMAuthError:
                    raise
                except Exception as exc:
                    self.failures.append(f"{suite.name}/{item.id} {a} vs {b}: {exc}")
                    return
            records.append(rec)
            if on_progress:
                on_progress(len(records), len(jobs), rec)

        tasks = [asyncio.ensure_future(one(*j)) for j in jobs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return records
