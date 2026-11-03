"""HTML scorecard with a radar chart per model, plus the terminal table."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console
from sentinel_fakes import ScriptedClient, small_suites
from test_runner import BAD, GOOD, behave, fast

from nimlab.sentinel.models import Scorecard, SentinelRun
from nimlab.sentinel.report import (
    build_context,
    radar_axes,
    radar_svg,
    render_html,
    render_terminal,
    write_html,
)
from nimlab.sentinel.runner import Runner
from nimlab.sentinel.scoring import build_scorecard


@pytest.fixture
async def run() -> SentinelRun:
    return await Runner(ScriptedClient(behave), settings=fast()).run(
        [GOOD, BAD], small_suites(), run_id="r1"
    )


def test_radar_axes_and_svg(run: SentinelRun) -> None:
    good = next(c for c in run.scorecards if c.model == GOOD)
    keys = [k for k, _, _ in radar_axes(good)]
    assert keys == ["prompt_injection", "harmful_refusal", "helpfulness"]
    svg = radar_svg(good)
    assert svg.startswith("<svg") and svg.count("<polygon") == 4 + 1  # four rings and the shape
    assert svg.count("<circle") == 3 and "Injection" in svg and "Helpful" in svg
    # a model that fails everything still draws a valid (collapsed) shape
    bad = next(c for c in run.scorecards if c.model == BAD)
    assert 'class="shape"' in radar_svg(bad)


def test_radar_needs_three_axes() -> None:
    card = build_scorecard("m", [])
    assert radar_svg(Scorecard(model="m")) == "" and radar_svg(card) == ""


def test_html_is_self_contained(run: SentinelRun) -> None:
    html = render_html(run)
    assert html.startswith("<!doctype html>") and "<svg" in html
    assert html.count('class="radar"') == 2  # one radar per model
    assert GOOD in html and BAD in html
    assert not re.search(r"""(?:src|href)=["']https?://""", html)  # no external assets
    assert "<script" not in html and "@import" not in html
    assert "prefers-color-scheme: dark" in html
    assert "How the grade is computed" in html and "Responsible use" in html
    assert "Simulated results" not in html  # the scripted client is not a simulator


def test_html_orders_models_by_score_and_shows_grades(run: SentinelRun) -> None:
    ctx = build_context(run)
    assert [m["name"] for m in ctx["models"]] == [GOOD, BAD]
    assert [m["grade"] for m in ctx["models"]] == ["A", "F"]
    html = render_html(run)
    assert 'class="grade big g-A"' in html and 'class="grade big g-F"' in html


def test_html_escapes_model_output(run: SentinelRun) -> None:
    run.results[0].response = "<script>alert(1)</script>"
    run.results[0].breached = True
    run.results[0].model = BAD
    html = render_html(run)
    assert "<script>alert(1)</script>" not in html


async def test_simulated_runs_carry_a_banner(tmp_path: Path) -> None:
    from nimlab.sentinel.simulate import SimulatedClient

    async with SimulatedClient() as client:
        sim_run = await Runner(client).run(["meta/llama-3.3-70b-instruct"], small_suites())
    assert "Simulated results" in render_html(sim_run)
    caption = Console(width=200, record=True)
    caption.print(render_terminal(sim_run))
    assert "SIMULATED" in caption.export_text()


def test_write_html_and_terminal_table(run: SentinelRun, tmp_path: Path) -> None:
    out = write_html(run, tmp_path / "deep" / "card.html", title="My scorecard")
    assert out.is_file() and "<title>My scorecard</title>" in out.read_text(encoding="utf-8")
    console = Console(width=200, record=True)
    console.print(render_terminal(run))
    text = console.export_text()
    assert "good/model" not in text and "model" in text  # short names
    assert "Inject" in text and "Harmful" in text
