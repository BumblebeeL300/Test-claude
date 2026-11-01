"""Mixture-of-Agents: proposers -> L aggregation layers -> final aggregator."""

from __future__ import annotations

from ..context import RunContext, agent_labels
from ..prompts import moa_messages, proposer_messages
from ..records import StepType
from .base import BaseStrategy, StrategyOutcome, register_strategy


@register_strategy
class MixtureOfAgents(BaseStrategy):
    """Layered synthesis (Wang et al., 2024).

    1. every panel model answers the question independently (in parallel);
    2. for each of ``layers`` layers every panel model sees *all* answers of the previous layer and
       writes a refined answer (in parallel);
    3. the aggregator model synthesises the last layer's answers into the final answer.

    Models that fail are skipped for that stage; the run only aborts if a whole stage fails.
    """

    name = "moa"
    aliases = ("mixture-of-agents", "mixture_of_agents")
    description = (
        "Mixture-of-Agents: N proposers answer in parallel, L layers of refinement where each "
        "model sees all previous answers, then a final aggregator synthesises the result."
    )
    options = ("models", "layers", "aggregator")
    min_models = 1

    async def run(self, ctx: RunContext) -> StrategyOutcome:
        self.check_models(ctx)
        cfg = ctx.config
        labels = agent_labels(len(ctx.models))

        # --- layer 0: independent proposals
        ctx.set_phase("Proposers")
        msgs = proposer_messages(ctx.question, cfg.system_prompt)
        proposals = await ctx.gather(
            *(
                ctx.ask(m, msgs, label=lab, type=StepType.PROPOSE)
                for m, lab in zip(ctx.models, labels, strict=True)
            )
        )
        current = [a for a in proposals if a is not None]
        ctx.require(current, "proposers")
        report = await ctx.consensus("Proposers", current)
        history = [report.agreement] if report else []

        # --- aggregation layers: every model refines using all previous answers
        layers_run = 0
        for layer in range(1, cfg.layers + 1):
            phase = f"Layer {layer}"
            ctx.set_phase(phase)
            msgs = moa_messages(ctx.question, current)
            refined = await ctx.gather(
                *(
                    ctx.ask(
                        m,
                        msgs,
                        label=lab,
                        type=StepType.AGGREGATE,
                        temperature=cfg.aggregator_temperature,
                        meta={"layer": layer, "inputs": len(current)},
                    )
                    for m, lab in zip(ctx.models, labels, strict=True)
                )
            )
            survivors = [a for a in refined if a is not None]
            if not survivors:  # whole layer failed: keep the previous answers, go to the end
                ctx.trace.meta.setdefault("warnings", []).append(f"{phase}: every model failed")
                break
            current = survivors
            layers_run = layer
            report = await ctx.consensus(phase, current) or report
            if report:
                history.append(report.agreement)

        # --- final synthesis
        ctx.set_phase("Final aggregation")
        final = await ctx.chat_first(
            [ctx.roles.aggregator, *ctx.models],
            moa_messages(ctx.question, current),
            type=StepType.AGGREGATE,
            label="Aggregator",
            temperature=cfg.aggregator_temperature,
            meta={"final": True, "inputs": len(current)},
        )
        return StrategyOutcome(
            answer=final.content.strip(),
            rationale=(
                f"Synthesised {len(current)} answers from {layers_run} aggregation layer(s) "
                f"using {final.model}."
            ),
            candidates=self.candidates_from(current, report),
            consensus=report,
            meta={
                "layers_run": layers_run,
                "aggregator": final.model,
                "agreement_by_stage": [round(x, 4) for x in history],
            },
        )
