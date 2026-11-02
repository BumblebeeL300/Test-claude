"""The orchestrator: a small state machine that drives the phases and always produces a report."""

from __future__ import annotations

import asyncio
import logging

from nimlab.core import NIMAuthError

from .context import TRANSITIONS, IllegalTransition, RunContext, State
from .models import ForgeReport
from .phases import Phase, default_phases

__all__ = ["Orchestrator"]

_log = logging.getLogger("nimlab.forge")

#: Hard cap on state transitions: a safety net against a buggy phase looping forever.
MAX_TRANSITIONS = 400

#: Phases that are optional: if one crashes the run continues in the mapped state.
RECOVERABLE: dict[State, State] = {
    State.EXPLORE: State.PLAN,
    State.ESCALATE: State.DEBUG,
    State.REVIEW: State.REPORT,
}


class Orchestrator:
    """Runs ``INIT -> ... -> REPORT -> DONE`` over a :class:`RunContext`.

    Phase exceptions never escape: they are recorded, and the machine jumps to ``REPORT`` so the
    caller always gets a :class:`ForgeReport` (``status="error"`` for fatal problems such as a
    rejected API key).
    """

    def __init__(self, ctx: RunContext, phases: dict[State, Phase] | None = None) -> None:
        self.ctx = ctx
        self.phases = phases or default_phases()

    async def run(self) -> ForgeReport:
        ctx = self.ctx
        state = State.INIT
        ctx.bus.emit("state", state.value, "start", **{"from": None, "to": state.value})
        for _ in range(MAX_TRANSITIONS):
            phase = self.phases[state]
            ctx.timer.start(phase.name)
            ctx.bus.emit("phase_start", phase.name, phase.name)
            try:
                next_state = await phase.run(ctx)
            except asyncio.CancelledError:
                ctx.timer.stop()
                raise
            except NIMAuthError as exc:
                ctx.fatal_error = f"authentication failed: {exc}"
                ctx.error(ctx.fatal_error, phase.name)
                next_state = State.DONE if state is State.REPORT else State.REPORT
            except Exception as exc:  # phases must not take the run down; report it instead
                _log.debug("forge phase %s crashed", phase.name, exc_info=True)
                message = f"{phase.name} phase crashed: {type(exc).__name__}: {exc}"
                ctx.error(message, phase.name)
                if state in RECOVERABLE:
                    next_state = RECOVERABLE[state]  # optional phase: carry on without it
                else:
                    ctx.ended_reason = ctx.ended_reason or "phase_error"
                    if state is not State.REPORT:
                        ctx.fatal_error = ctx.fatal_error or message
                    next_state = State.DONE if state is State.REPORT else State.REPORT
            finally:
                ctx.timer.stop()
            ctx.bus.emit("phase_end", phase.name, phase.name)
            if next_state not in TRANSITIONS[state]:
                raise IllegalTransition(f"{state.value} -> {next_state.value}")
            ctx.bus.emit(
                "state",
                next_state.value,
                f"{state.value} -> {next_state.value}",
                **{"from": state.value, "to": next_state.value},
            )
            state = next_state
            if state is State.DONE:
                break
        else:  # pragma: no cover - defensive
            ctx.error("state machine exceeded its transition budget")
            ctx.fatal_error = ctx.fatal_error or "state machine exceeded its transition budget"
        if ctx.report is None:
            # REPORT crashed before it could build anything: synthesise a minimal report.
            ctx.report = ForgeReport(
                task=ctx.task,
                workspace=str(ctx.workspace),
                status="error",
                summary=f"Run aborted: {ctx.fatal_error or 'no report was produced'}",
                errors=ctx.errors,
                models=ctx.models,
                mock=ctx.mock,
            )
        return ctx.report
