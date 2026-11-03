"""The ``nimlab arena`` CLI, run through Typer's CliRunner in mock mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nimlab.arena.cli import app
from nimlab.arena.results import RunStore

runner = CliRunner()
MODELS = (
    "nvidia/nemotron-3-super-120b-a12b,nvidia/nemotron-3.5-lightning-30b-a3b,poolside/laguna-xs-2.1"
)
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "arena" / "custom_suite.json"


def invoke(*args: str, expect: int = 0):
    result = runner.invoke(app, list(args))
    assert result.exit_code == expect, result.output + repr(result.exception)
    return result


@pytest.fixture
def runs(tmp_path: Path) -> str:
    return str(tmp_path / "runs")


def test_no_args_shows_help_with_all_commands() -> None:
    result = runner.invoke(app, [])
    for cmd in ("suites", "run", "battle", "report", "route", "serve"):
        assert cmd in result.output


def test_every_command_has_help() -> None:
    for cmd in ("suites", "run", "battle", "report", "route", "serve"):
        out = invoke(cmd, "--help").output
        assert "Usage" in out
    assert (
        "--concurrency" in invoke("run", "--help").output
        and "--mock" in invoke("run", "--help").output
    )


def test_suites_lists_bundled_suites() -> None:
    out = invoke("suites").output
    for name in (
        "math_reasoning",
        "coding",
        "factual_qa",
        "instruction_following",
        "json_compliance",
        "summarization",
    ):
        assert name in out
    assert "code_exec" in out and "llm_judge" in out


def test_suites_json_show_and_custom_dirs(tmp_path: Path) -> None:
    data = json.loads(invoke("suites", "--json").output)
    assert {d["name"] for d in data} >= {"coding", "summarization"} and all(
        d["items"] >= 8 for d in data
    )
    shown = invoke("suites", "--show", "coding").output
    assert "palindrome" in shown and "code_exec" in shown
    invoke("suites", "--show", "nope", expect=1)
    (tmp_path / "x.json").write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    assert "support_triage" in invoke("suites", "--suites-dir", str(tmp_path)).output
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    bad = invoke("suites", "--suites-dir", str(tmp_path), expect=1)
    assert "invalid JSON" in bad.output


def test_run_benchmarks_models_and_persists(runs: str) -> None:
    out = invoke(
        "--mock",
        "run",
        "--suite",
        "math_reasoning,factual_qa",
        "--models",
        MODELS,
        "--concurrency",
        "6",
        "--limit",
        "4",
        "--runs-dir",
        runs,
    ).output
    assert "Mock mode" in out and "Arena leaderboard" in out and "SIMULATED" in out
    for model in MODELS.split(","):
        assert model in out
    store = RunStore(runs)
    (run_id,) = store.list_ids()
    run = store.load(run_id)
    assert run.status == "complete" and run.mock and len(run.items) == 3 * 2 * 4
    assert f"{run_id}.json" in out
    assert {r.suite for r in run.items} == {"math_reasoning", "factual_qa"}


def test_run_accepts_mock_flag_after_the_command_and_html_out(runs: str, tmp_path: Path) -> None:
    html = tmp_path / "lb.html"
    out = invoke(
        "run",
        "--mock",
        "--suite",
        "instruction_following",
        "--models",
        MODELS,
        "--limit",
        "3",
        "--runs-dir",
        runs,
        "--html-out",
        str(html),
    ).output
    assert html.is_file() and "Overall ranking" in html.read_text(encoding="utf-8")
    assert str(html) in out


def test_run_default_models_and_all_suites_complete_quickly(runs: str) -> None:
    invoke("--mock", "run", "--limit", "2", "--runs-dir", runs, "-c", "8")
    run = RunStore(runs).latest()
    assert run is not None and len(run.models) == 4 and len(run.suites) == 6
    assert all(r.scored for r in run.items)


def test_run_custom_suite_file_and_resume(runs: str) -> None:
    invoke("--mock", "run", "--suite", str(EXAMPLE), "--models", MODELS, "--runs-dir", runs)
    store = RunStore(runs)
    run = store.latest()
    assert run is not None and set(run.suites) == {"support_triage"} and len(run.items) == 15
    before = len(store.list_ids())
    invoke(
        "--mock",
        "run",
        "--suite",
        str(EXAMPLE),
        "--models",
        MODELS,
        "--runs-dir",
        runs,
        "--resume",
        run.run_id,
    )
    assert len(store.list_ids()) == before  # resumed in place, no new file


def test_run_errors_are_friendly(runs: str) -> None:
    out = invoke("--mock", "run", "--suite", "nope", "--runs-dir", runs, expect=1).output
    assert "unknown suite" in out and "Traceback" not in out
    out = invoke(
        "--mock", "run", "--resume", "missing", "--runs-dir", runs, "--suite", "coding", expect=1
    ).output
    assert "not found" in out


def test_battle_prints_elo_with_confidence_intervals(runs: str) -> None:
    models = (
        "nvidia/nemotron-3-ultra-550b-a55b,nvidia/nemotron-3.5-lightning-30b-a3b,"
        "nvidia/nemotron-3.5-bolt-e4b"
    )
    out = invoke(
        "--mock",
        "battle",
        "--models",
        models,
        "--suite",
        "summarization",
        "--max-items",
        "5",
        "--runs-dir",
        runs,
        "--judge-model",
        "nvidia/nemotron-3-super-120b-a12b",
    ).output
    assert "Elo" in out and "95% CI" in out and "W-T-L" in out and "judge battles" in out
    assert "position bias" in out
    run = RunStore(runs).latest()
    assert (
        run is not None
        and len(run.battles) == 3 * 5
        and run.judge_model == "nvidia/nemotron-3-super-120b-a12b"
    )
    assert "also a contestant" not in out


def test_battle_warns_when_the_judge_is_a_contestant_and_validates(runs: str) -> None:
    out = invoke(
        "--mock", "battle", "--models", MODELS, "--max-items", "2", "--runs-dir", runs
    ).output
    assert "also a contestant" in out
    err = invoke("--mock", "battle", "--models", "only-one", "--runs-dir", runs, expect=1).output
    assert "at least two" in err


def test_report_renders_latest_run_and_writes_html(runs: str, tmp_path: Path) -> None:
    invoke("--mock", "run", "--models", MODELS, "--limit", "3", "--runs-dir", runs)
    html = tmp_path / "leaderboard.html"
    out = invoke(
        "--mock", "report", "--runs-dir", runs, "--html-out", str(html), "--bootstrap", "30"
    ).output
    assert "Arena leaderboard" in out and "Elo" in out
    page = html.read_text(encoding="utf-8")
    for section in ("Overall ranking", "Scores by suite", "Elo ratings", "Quality vs latency"):
        assert section in page
    run_id = RunStore(runs).list_ids()[0]
    assert run_id in invoke("report", run_id, "--runs-dir", runs, "--bootstrap", "0").output
    path = RunStore(runs).path_for(run_id)
    assert "Arena leaderboard" in invoke("report", str(path), "--bootstrap", "0").output


def test_report_without_runs_fails_cleanly(runs: str) -> None:
    out = invoke("report", "--runs-dir", runs, expect=1).output
    assert "no runs found" in out


def test_route_explains_a_decision_without_any_runs(runs: str) -> None:
    out = invoke(
        "--mock", "route", "Write a Python function that parses a CSV", "--runs-dir", runs
    ).output
    assert "Routing decision" in out and "coding" in out and "registry" in out
    assert "poolside/laguna-xs-2.1" in out


def test_route_uses_leaderboard_data_and_prints_candidates(runs: str) -> None:
    invoke("--mock", "run", "--models", MODELS, "--limit", "4", "--runs-dir", runs)
    out = invoke(
        "--mock", "route", "Solve 3x + 7 = 28", "--runs-dir", runs, "--profile", "quality"
    ).output
    assert (
        "math_reasoning" in out
        and "leaderboard" in out
        and "Candidates" in out
        and "Utility" in out
    )
    assert "utility = quality" in out
    data = json.loads(
        invoke(
            "--mock", "route", "What is the capital of Peru?", "--runs-dir", runs, "--json"
        ).output
    )
    assert data["category"] == "factual_qa" and data["source"] == "leaderboard"
    assert data["candidates"] and data["reasons"] and data["model"] in MODELS.split(",")


def test_route_options(runs: str) -> None:
    data = json.loads(
        invoke(
            "--mock",
            "route",
            "anything",
            "--category",
            "coding",
            "--latency-weight",
            "0.5",
            "--cost-weight",
            "0",
            "--runs-dir",
            runs,
            "--json",
        ).output
    )
    assert (
        data["category"] == "coding" and data["latency_weight"] == 0.5 and data["cost_weight"] == 0
    )
    bad = invoke("--mock", "route", "x", "--profile", "warp", "--runs-dir", runs, expect=1).output
    assert "unknown profile" in bad


def test_serve_builds_the_app_and_hands_it_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch, runs: str
) -> None:
    import uvicorn

    captured: dict = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda application, **kw: captured.update(app=application, **kw)
    )
    out = invoke(
        "--mock", "serve", "--port", "9123", "--host", "0.0.0.0", "--runs-dir", runs
    ).output
    assert captured["port"] == 9123 and captured["host"] == "0.0.0.0"
    assert "http://0.0.0.0:9123/v1" in out and "nimlab/auto" in out and "Mock mode" in out
    assert captured["app"].title.startswith("NIM Arena")


def test_cli_is_a_typer_app_mountable_by_the_root_cli() -> None:
    import typer

    assert isinstance(app, typer.Typer)
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    assert names >= {"suites", "run", "battle", "report", "route", "serve"}
