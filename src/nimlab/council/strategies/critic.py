"""Critic loop: drafter -> critic -> reviser, until no blocking issues remain."""

from __future__ import annotations

from ..context import Answer, RunContext
from ..prompts import critic_messages, draft_messages, reviser_messages
from ..records import Candidate, StepType
from ..schemas import Critique
from .base import BaseStrategy, StrategyOutcome, register_strategy


@register_strategy
class CriticLoop(BaseStrategy):
    """Iterative self-refinement with an independent reviewer.

    1. the drafter writes a first answer;
    2. the critic (preferably a different model) returns a structured critique via ``chat_json``:
       a list of issues, each with a severity (low / medium / high);
    3. if no issue reaches ``critic_stop_severity`` (default ``high``) the loop stops;
       otherwise the reviser rewrites the draft addressing the issues and we go back to 2.

    At most ``max_iterations`` critique/revise cycles run. The final consensus report compares the
    successive versions (how far the answer drifted). If the critic or reviser is unreachable the
    best draft so far is returned.
    """

    name = "critic"
    aliases = ("critic-loop", "critic_loop", "refine")
    description = (
        "Critic loop: a drafter writes, a critic returns structured issues (low/medium/high), a "
        "reviser fixes them; repeat until no high-severity issues or max iterations."
    )
    options = ("models", "max_iterations", "critic_stop_severity", "critic")
    min_models = 1

    async def run(self, ctx: RunContext) -> StrategyOutcome:
        self.check_models(ctx)
        cfg = ctx.config
        roles = ctx.roles

        ctx.set_phase("Draft")
        draft = await ctx.chat_first(
            [roles.drafter, *ctx.models],
            draft_messages(ctx.question, cfg.system_prompt),
            type=StepType.DRAFT,
            label="Draft",
        )
        drafter_model = draft.model
        versions = [Answer("Draft", drafter_model, draft.content.strip(), len(ctx.trace.steps) - 1)]
        critiques: list[Critique] = []
        stop_reason = "max_iterations"

        for it in range(1, cfg.max_iterations + 1):
            current = versions[-1]
            ctx.set_phase(f"Iteration {it} · critique")
            critique = await ctx.try_json(
                roles.critic,
                critic_messages(ctx.question, current.text, it - 1),
                Critique,
                type=StepType.CRITIQUE,
                label=f"Critique {it}",
                temperature=0.0,
                meta={"iteration": it},
            )
            if critique is None:
                stop_reason = "critic_failed"
                break
            critiques.append(critique)
            blocking = critique.blocking(cfg.critic_stop_severity)
            ctx.trace.steps[-1].meta["issues"] = len(critique.issues)
            if not blocking:
                stop_reason = "no_blocking_issues"
                break

            ctx.set_phase(f"Iteration {it} · revise")
            revised = await ctx.ask(
                roles.reviser,
                reviser_messages(ctx.question, current.text, critique),
                label=f"Revision {it}",
                type=StepType.REVISE,
                meta={"iteration": it, "addressing": len(blocking)},
            )
            if revised is None:
                stop_reason = "reviser_failed"
                break
            versions.append(revised)
        else:
            stop_reason = "max_iterations" if cfg.max_iterations else "no_iterations"

        final = versions[-1]
        report = await ctx.consensus("Revisions", versions)
        unresolved = 0
        if critiques and stop_reason == "max_iterations":
            unresolved = len(critiques[-1].blocking(cfg.critic_stop_severity))
        rationale = {
            "no_blocking_issues": (
                f"The critic found no {cfg.critic_stop_severity}-severity issues after "
                f"{len(versions) - 1} revision(s)."
            ),
            "max_iterations": (
                f"Stopped after {cfg.max_iterations} iteration(s); the last revision addressed "
                f"{unresolved} {cfg.critic_stop_severity}-severity issue(s) "
                "but was not re-reviewed."
            ),
            "no_iterations": "No critique iterations were requested; this is the first draft.",
            "critic_failed": "The critic was unavailable; returning the latest version.",
            "reviser_failed": "The reviser was unavailable; returning the latest version.",
        }[stop_reason]

        candidates = [
            Candidate(
                label=v.label,
                model=v.model,
                text=v.text,
                step_index=v.step_index,
                selected=v is final,
            )
            for v in versions
        ]
        return StrategyOutcome(
            answer=final.text,
            rationale=rationale,
            candidates=candidates,
            consensus=report,
            meta={
                "iterations": len(critiques),
                "revisions": len(versions) - 1,
                "stop_reason": stop_reason,
                "issues_per_iteration": [len(c.issues) for c in critiques],
                "unreviewed_blocking": unresolved,
                "drafter": drafter_model,
                "critic": roles.critic,
                "reviser": roles.reviser,
            },
        )
