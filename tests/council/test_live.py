from __future__ import annotations

import io
import logging

from rich.console import Console

from nimlab.council import (
    CallFinished,
    CallStarted,
    ConsensusComputed,
    Council,
    CouncilLive,
    PhaseStarted,
    RunFinished,
    RunStarted,
    StepType,
    Trace,
)


def text_of(view: CouncilLive, now: float | None = None) -> str:
    console = Console(record=True, width=120, force_terminal=False, file=io.StringIO())
    console.print(view.render(now))
    return console.export_text()


def started(index: int, model: str, label: str, phase: str) -> CallStarted:
    return CallStarted(index, model, label, phase, StepType.PROPOSE)


def test_render_shows_models_working_in_parallel() -> None:
    view = CouncilLive(Console(file=io.StringIO()))
    view(RunStarted("moa", "Why is the sky blue?", ("meta/llama-a", "nvidia/nemo-b")))
    view(PhaseStarted("Proposers"))
    view(started(0, "meta/llama-a", "Agent 1", "Proposers"))
    view(started(1, "nvidia/nemo-b", "Agent 2", "Proposers"))
    out = text_of(view, now=0.0)
    assert "Council" in out and "moa" in out and "Why is the sky blue?" in out
    assert "Proposers" in out and "0/2" in out
    assert "llama-a" in out and "nemo-b" in out  # org prefix stripped
    assert "2 running" in out


def test_render_shows_completion_failure_tokens_and_consensus() -> None:
    view = CouncilLive(Console(file=io.StringIO()))
    view(RunStarted("debate", "q", ("a/m1", "b/m2")))
    view(started(0, "a/m1", "Agent 1", "Round 0"))
    view(started(1, "b/m2", "Agent 2", "Round 0"))
    view(CallFinished(0, True, 1.25, 321, None))
    view(CallFinished(1, False, 0.5, None, "HTTP 500 upstream"))
    view(ConsensusComputed("Round 0", 0.83, ("Agent 2",)))
    out = text_of(view)
    assert "✔" in out and "✘" in out
    assert "321 tok" in out and "1.25s" in out
    assert "HTTP 500 upstream" in out
    assert "agreement 0.83" in out and "outliers: Agent 2" in out
    assert "1 failed" in out and "1 done" in out
    assert "finished" not in out
    view(RunFinished(Trace(question="q", strategy="debate")))
    assert "finished" in text_of(view)


def test_render_marks_failed_runs_and_empty_state() -> None:
    view = CouncilLive(Console(file=io.StringIO()))
    assert "waiting for the first model call" in text_of(view)
    view(RunFinished(Trace(question="q", strategy="moa"), error="boom"))
    assert "failed" in text_of(view)


def test_phases_are_grouped_and_ordered() -> None:
    view = CouncilLive(Console(file=io.StringIO()))
    view(PhaseStarted("Layer 1"))
    view(started(0, "a/m", "A", "Layer 1"))
    view(PhaseStarted("Layer 2"))
    view(started(1, "a/m", "A", "Layer 2"))
    view(started(2, "a/m", "B", "Layer 2"))
    assert [(p.name, len(p.calls)) for p in view.phases] == [("Layer 1", 1), ("Layer 2", 2)]
    out = text_of(view)
    assert out.index("Layer 1") < out.index("Layer 2")


async def test_live_context_manager_over_a_full_run(mock_client, panel, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="nimlab")
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, width=120)
    with CouncilLive(console, refresh_per_second=50) as view:
        result = await Council(mock_client).run(
            "Name a colour", "debate", models=panel, rounds=1, observers=[view]
        )
    assert view.finished and view.error is None
    assert len([c for p in view.phases for c in p.calls]) == len([s for s in result.trace.steps])
    final = text_of(view)
    assert "Judge" in final and "finished" in final
    assert "observer failed" not in caplog.text  # the view handled every event type


async def test_council_run_live_flag(monkeypatch, mock_client, panel) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(
        "nimlab.council.live.get_console",
        lambda: Console(file=sink, force_terminal=False, width=100),
    )
    result = await Council(mock_client).run("hi", "moa", models=panel, layers=0, live=True)
    assert result.answer
    assert "Council" in sink.getvalue()


def test_consensus_note_attaches_to_the_latest_phase() -> None:
    view = CouncilLive(Console(file=io.StringIO()))
    view(PhaseStarted("Round 1 · revise"))
    view(started(0, "a/m", "Agent 1", "Round 1 · revise"))
    view(ConsensusComputed("Round 1", 0.77))  # phase name differs from the call phase
    assert [p.name for p in view.phases] == ["Round 1 · revise"]
    assert "agreement 0.77" in text_of(view)
