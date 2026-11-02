"""Best-of-N: sample candidates, score them with a reward model, keep the best."""

from __future__ import annotations

from ..context import Answer, RunContext, agent_labels
from ..errors import AllModelsFailedError
from ..prompts import proposer_messages
from ..records import ConsensusReport, StepType
from ..reward import Score, judge_score, reward_score
from .base import BaseStrategy, StrategyOutcome, register_strategy


def temperature_schedule(n: int, low: float, high: float) -> list[float]:
    """``n`` temperatures spread evenly over ``[low, high]`` (a single sample uses ``low``)."""
    if n <= 1:
        return [round(low, 3)]
    return [round(low + (high - low) * i / (n - 1), 3) for i in range(n)]


def sampling_plan(models: list[str], n: int, low: float, high: float) -> list[tuple[str, float]]:
    """(model, temperature) for each of ``n`` samples: models rotate, temperatures sweep.

    Samples are laid out so that the same model is never asked at the same temperature twice and
    cheap diversity comes from both axes.
    """
    temps = temperature_schedule(n, low, high)
    return [(models[i % len(models)], temps[i]) for i in range(n)]


@register_strategy
class BestOfN(BaseStrategy):
    """Generate ``n`` candidates across models/temperatures and pick the highest-scoring one.

    Scoring uses the NVIDIA reward model (the ``[user, assistant]`` pair is sent as a chat and the
    reply ``reward:<float>`` is parsed). If the reward model is missing, errors, replies with
    something unparseable, or the client is in mock mode, the candidate is graded by an
    LLM-as-judge rubric (0-10) instead. Reward and judge scores are on different scales, so if the
    sources end up mixed every candidate is re-graded by the judge. Ties break towards the answer
    most central in embedding space (i.e. the one the other samples agree with).
    """

    name = "best-of-n"
    aliases = ("best_of_n", "bon", "bestofn")
    description = (
        "Best-of-N: sample N candidates across models and temperatures, score each with the NVIDIA "
        "reward model (LLM-judge fallback) and return the highest-scoring answer."
    )
    options = ("models", "n", "reward_model", "temperature_range")
    min_models = 1

    async def run(self, ctx: RunContext) -> StrategyOutcome:
        self.check_models(ctx)
        cfg = ctx.config
        plan = sampling_plan(ctx.models, cfg.n, *cfg.temperature_range)
        labels = agent_labels(len(plan), prefix="Candidate")

        ctx.set_phase("Sampling")
        msgs = proposer_messages(ctx.question, cfg.system_prompt)
        sampled = await ctx.gather(
            *(
                ctx.ask(m, msgs, label=lab, type=StepType.PROPOSE, temperature=t)
                for (m, t), lab in zip(plan, labels, strict=True)
            )
        )
        candidates = [a for a in sampled if a is not None]
        ctx.require(candidates, "candidate sampling")
        report = await ctx.consensus("Candidates", candidates)

        ctx.set_phase("Scoring")
        scores = await self._score_all(ctx, candidates)

        best = self._pick(candidates, scores, report)
        out = self.candidates_from(candidates, report)
        for cand, score in zip(out, scores, strict=True):
            if score is not None:
                cand.score = round(score.value, 4)
                cand.score_source = score.source
            cand.meta["temperature"] = _temperature_of(ctx, cand.step_index)
        out[best].selected = True

        scored = [(i, s.value) for i, s in enumerate(scores) if s is not None]
        ranked = [i for i, _ in sorted(scored, key=lambda t: -t[1])]
        winner = scores[best]
        rationale = self._rationale(candidates, scores, best, ranked)
        return StrategyOutcome(
            answer=candidates[best].text,
            rationale=rationale,
            candidates=out,
            consensus=report,
            meta={
                "n_requested": cfg.n,
                "n_sampled": len(candidates),
                "scorer": winner.source if winner else None,
                "winner": candidates[best].label,
                "scores": {
                    c.label: (s.value if s else None)
                    for c, s in zip(candidates, scores, strict=True)
                },
            },
        )

    # ------------------------------------------------------------------ scoring
    async def _score_all(self, ctx: RunContext, candidates: list[Answer]) -> list[Score | None]:
        question = ctx.question

        async def score_one(cand: Answer) -> Score | None:
            return await reward_score(ctx, question, cand.text, cand.label) or await judge_score(
                ctx, question, cand.text, cand.label
            )

        scores = await ctx.gather(*(score_one(c) for c in candidates))
        sources = {s.source for s in scores if s is not None}
        if len(sources) > 1:
            # mixed scales (some reward, some judge): regrade everything with the judge
            ctx.trace.meta.setdefault("warnings", []).append(
                "reward and judge scores were mixed; regraded all candidates with the judge"
            )
            scores = await ctx.gather(
                *(judge_score(ctx, question, c.text, c.label) for c in candidates)
            )
        if all(s is None for s in scores):
            raise AllModelsFailedError(
                "no scorer produced a score for any candidate", trace=ctx.trace
            )
        return list(scores)

    @staticmethod
    def _pick(
        candidates: list[Answer], scores: list[Score | None], report: ConsensusReport | None
    ) -> int:
        """Index of the best candidate: highest score, then most central, then earliest."""
        centrality: dict[str, float] = {}
        if report is not None:
            centrality = dict(zip(report.labels, report.centrality, strict=True))

        def key(i: int) -> tuple[float, float, int]:
            s = scores[i]
            value = s.value if s is not None else float("-inf")
            return (value, centrality.get(candidates[i].label, 0.0), -i)

        return max(range(len(candidates)), key=key)

    @staticmethod
    def _rationale(
        candidates: list[Answer], scores: list[Score | None], best: int, ranked: list[int]
    ) -> str:
        winner = scores[best]
        if winner is None:
            return f"Picked {candidates[best].label}."
        unit = "reward" if winner.source == "reward" else "judge score (0-10)"
        pick = candidates[best]
        text = f"Picked {pick.label} ({pick.model}) with {unit} {winner.value:.2f}"
        runner = [i for i in ranked if i != best]
        if runner:
            second = scores[runner[0]]
            assert second is not None
            text += f"; margin over runner-up {winner.value - second.value:+.2f}"
        return text + f" out of {len(candidates)} candidates."


def _temperature_of(ctx: RunContext, step_index: int | None) -> float | None:
    if step_index is None or step_index >= len(ctx.trace.steps):
        return None
    return ctx.trace.steps[step_index].temperature
