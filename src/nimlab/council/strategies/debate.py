"""Debate: independent answers -> rounds of critique + revision -> judge."""

from __future__ import annotations

from ..context import Answer, RunContext, agent_labels, clip
from ..prompts import (
    debate_critique_messages,
    debate_judge_messages,
    debate_revise_messages,
    proposer_messages,
)
from ..records import ConsensusReport, StepType
from ..schemas import JudgeVerdict
from .base import BaseStrategy, StrategyOutcome, register_strategy


@register_strategy
class Debate(BaseStrategy):
    """Multi-round debate with early stopping on consensus and a final judge.

    1. every model answers independently;
    2. for up to ``rounds`` rounds: (a) each model critiques its peers' answers, then (b) each model
       sees its peers' answers and critiques and revises its own answer;
    3. before every round (and after the last) the answers' embedding agreement is measured; once
       it reaches ``convergence_threshold`` the debate stops early;
    4. a judge model sees the final (anonymised) answers plus the debate history and either picks
       the best answer or writes a synthesis, with a rationale.

    A model that fails mid-debate keeps its previous answer; the judge falls back to the most
    central (medoid) answer if no judge model can be reached.
    """

    name = "debate"
    description = (
        "Debate: models answer, then critique peers and revise for R rounds (stopping early when "
        "embedding consensus is reached); a judge picks or synthesises the final answer."
    )
    options = ("models", "rounds", "convergence_threshold", "judge")
    min_models = 2

    async def run(self, ctx: RunContext) -> StrategyOutcome:
        self.check_models(ctx)
        cfg = ctx.config
        labels = agent_labels(len(ctx.models))

        # --- round 0: independent answers
        ctx.set_phase("Round 0 · independent answers")
        msgs = proposer_messages(ctx.question, cfg.system_prompt)
        first = await ctx.gather(
            *(
                ctx.ask(m, msgs, label=lab, type=StepType.PROPOSE)
                for m, lab in zip(ctx.models, labels, strict=True)
            )
        )
        current = [a for a in first if a is not None]
        ctx.require(current, "initial answers")
        report = await ctx.consensus("Round 0", current)
        trajectory = [report.agreement if report else 1.0]
        history: list[str] = [self._digest(0, current, report)]

        rounds_run = 0
        stopped_early = False
        for rnd in range(1, cfg.rounds + 1):
            if self._converged(ctx, report):
                stopped_early = True
                break
            if len(current) < 2:  # nobody left to debate with
                break
            critiques: list[Answer] = []
            if cfg.debate_critique:
                ctx.set_phase(f"Round {rnd} · critique")
                crit = await ctx.gather(
                    *(
                        ctx.ask(
                            a.model,
                            debate_critique_messages(
                                ctx.question, [p for p in current if p is not a]
                            ),
                            label=a.label,
                            type=StepType.CRITIQUE,
                            temperature=cfg.aggregator_temperature,
                            meta={"round": rnd},
                        )
                        for a in current
                    )
                )
                critiques = [c for c in crit if c is not None]
                history.append(
                    f"Round {rnd} critiques:\n"
                    + "\n".join(f"- {c.label}: {clip(c.text, 500)}" for c in critiques)
                )

            ctx.set_phase(f"Round {rnd} · revise")
            revised = await ctx.gather(
                *(
                    ctx.ask(
                        a.model,
                        debate_revise_messages(
                            ctx.question,
                            a,
                            [p for p in current if p is not a],
                            [c for c in critiques if c.label != a.label],
                        ),
                        label=a.label,
                        type=StepType.REVISE,
                        temperature=cfg.temperature,
                        meta={"round": rnd},
                    )
                    for a in current
                )
            )
            # a model whose revision failed keeps its previous answer
            current = [new or old for old, new in zip(current, revised, strict=True)]
            rounds_run = rnd
            report = await ctx.consensus(f"Round {rnd}", current) or report
            trajectory.append(report.agreement if report else 1.0)
            history.append(self._digest(rnd, current, report))

        converged = self._converged(ctx, report)
        verdict, judge_model = await self._judge(ctx, current, history)
        answer, selected_idx, rationale = self._decide(verdict, current, report)

        candidates = self.candidates_from(current, report)
        if selected_idx is not None:
            candidates[selected_idx].selected = True
        return StrategyOutcome(
            answer=answer,
            rationale=rationale,
            candidates=candidates,
            consensus=report,
            meta={
                "rounds_run": rounds_run,
                "rounds_max": cfg.rounds,
                "stopped_early": stopped_early or (converged and rounds_run < cfg.rounds),
                "converged": converged,
                "agreement_by_round": [round(x, 4) for x in trajectory],
                "judge": judge_model,
                "confidence": verdict.confidence if verdict else None,
                "selected": selected_idx,
            },
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _converged(ctx: RunContext, report: ConsensusReport | None) -> bool:
        cfg = ctx.config
        return bool(
            cfg.early_stop and report is not None and report.agreement >= cfg.convergence_threshold
        )

    @staticmethod
    def _digest(rnd: int, answers: list[Answer], report: ConsensusReport | None) -> str:
        agreement = f"{report.agreement:.2f}" if report else "n/a"
        lines = [f"Round {rnd} answers (agreement {agreement}):"]
        lines += [f"- {a.label}: {clip(a.text, 400)}" for a in answers]
        return "\n".join(lines)

    async def _judge(
        self, ctx: RunContext, current: list[Answer], history: list[str]
    ) -> tuple[JudgeVerdict | None, str | None]:
        ctx.set_phase("Judge")
        msgs = debate_judge_messages(ctx.question, current, "\n\n".join(history))
        for model in dict.fromkeys([ctx.roles.judge, *ctx.models]):
            verdict = await ctx.try_json(
                model, msgs, JudgeVerdict, type=StepType.JUDGE, label="Judge", temperature=0.0
            )
            if verdict is not None:
                return verdict, model
        return None, None

    @staticmethod
    def _decide(
        verdict: JudgeVerdict | None, current: list[Answer], report: ConsensusReport | None
    ) -> tuple[str, int | None, str]:
        """(answer, index of the selected candidate or None, rationale)."""
        chosen: int | None = None
        if verdict is not None and verdict.selected is not None:
            idx = verdict.selected - 1
            if 0 <= idx < len(current):
                chosen = idx
        if verdict is not None and verdict.final_answer.strip():
            rationale = verdict.rationale.strip() or "The judge synthesised the final answer."
            return verdict.final_answer.strip(), chosen, rationale
        if chosen is not None:
            assert verdict is not None
            return (
                current[chosen].text,
                chosen,
                verdict.rationale.strip() or "Selected by the judge.",
            )
        # no usable verdict: fall back to the answer closest to everyone else's
        medoid = 0
        if report is not None and report.medoid is not None:
            label = report.labels[report.medoid]
            medoid = next((i for i, a in enumerate(current) if a.label == label), 0)
        note = "no judge verdict was available" if verdict is None else "the judge gave no answer"
        return (
            current[medoid].text,
            medoid,
            f"Fell back to the most central answer ({current[medoid].label}) because {note}.",
        )
