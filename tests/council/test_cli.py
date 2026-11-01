from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nimlab.council.cli import app
from nimlab.council.records import Trace

runner = CliRunner()
Q = "Why is the sky blue?"


def invoke(*args: str):
    return runner.invoke(app, list(args), catch_exceptions=False)


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "ask" in result.output and "strategies" in result.output


def test_strategies_lists_all_four() -> None:
    result = invoke("strategies")
    assert result.exit_code == 0
    for name in ("moa", "debate", "best-of-n", "critic"):
        assert name in result.output
    assert "Mixture-of-Agents" in result.output and "--rounds" in result.output


@pytest.mark.parametrize("strategy", ["moa", "debate", "best-of-n", "critic"])
def test_ask_each_strategy_in_mock_mode(strategy: str, tmp_path: Path) -> None:
    json_out, html_out = tmp_path / "trace.json", tmp_path / "trace.html"
    result = invoke(
        "ask", Q, "--strategy", strategy, "--mock", "--no-live",
        "--rounds", "1", "--layers", "1", "--n", "3",
        "--json-out", str(json_out), "--html-out", str(html_out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Council answer" in result.output and strategy in result.output
    assert "Token usage" in result.output

    trace = Trace.load(json_out)
    assert trace.strategy == strategy and trace.question == Q
    assert trace.final_answer.strip() and trace.mock is True
    assert trace.steps and trace.usage
    assert json.loads(json_out.read_text())["strategy"] == strategy

    html = html_out.read_text(encoding="utf-8")
    assert "<details" in html and trace.final_answer.splitlines()[0][:30] in html


def test_ask_models_flag_overrides_the_panel(tmp_path: Path) -> None:
    out = tmp_path / "t.json"
    invoke(
        "ask", Q, "-s", "moa", "--mock", "--no-live", "--layers", "0",
        "--models", " org/alpha , org/beta ,", "--json-out", str(out),
    )  # fmt: skip
    trace = Trace.load(out)
    assert trace.models == ["org/alpha", "org/beta"]
    assert {s.model for s in trace.steps if s.type.value == "propose"} == {"org/alpha", "org/beta"}
    assert trace.config["resolved"]["aggregator"] == "org/alpha"


def test_ask_options_reach_the_config(tmp_path: Path) -> None:
    out = tmp_path / "t.json"
    invoke(
        "ask", Q, "-s", "best-of-n", "--mock", "--no-live", "--n", "5", "--temperature", "0.6",
        "--reward-model", "nvidia/custom-reward", "--json-out", str(out),
    )  # fmt: skip
    trace = Trace.load(out)
    assert trace.config["n"] == 5 and trace.config["temperature"] == 0.6
    assert trace.config["resolved"]["reward"] == "nvidia/custom-reward"
    assert len(trace.candidates) == 5


def test_verbose_prints_steps() -> None:
    quiet = invoke("ask", Q, "--mock", "--no-live", "--layers", "0")
    loud = invoke("ask", Q, "--mock", "--no-live", "--layers", "0", "-v")
    assert "Steps" not in quiet.output and "Steps" in loud.output


def test_mock_flag_restores_environment(monkeypatch) -> None:
    monkeypatch.delenv("NIMLAB_MOCK", raising=False)
    invoke("ask", Q, "--mock", "--no-live", "--layers", "0")
    import os

    assert "NIMLAB_MOCK" not in os.environ


def test_ask_works_without_flag_when_no_api_key() -> None:
    result = invoke("ask", Q, "--no-live", "--layers", "0")  # no NVIDIA_API_KEY => mock
    assert result.exit_code == 0 and "Council answer" in result.output


def test_unknown_strategy_is_a_usage_error() -> None:
    result = runner.invoke(app, ["ask", Q, "--strategy", "telepathy", "--mock"])
    assert result.exit_code == 2
    assert "telepathy" in result.output


def test_invalid_numeric_option_is_rejected() -> None:
    assert runner.invoke(app, ["ask", Q, "--n", "0", "--mock"]).exit_code == 2
    assert runner.invoke(app, ["ask", Q, "--threshold", "1.5", "--mock"]).exit_code == 2


def test_council_failure_exits_nonzero_but_still_writes_the_trace(tmp_path: Path) -> None:
    out, html = tmp_path / "fail.json", tmp_path / "fail.html"
    result = runner.invoke(
        app,
        ["ask", Q, "-s", "debate", "--mock", "--no-live", "--models", "solo/model",
         "--json-out", str(out), "--html-out", str(html)],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "council failed" in result.output and "at least 2 models" in result.output
    assert Trace.load(out).error and "Run failed" in html.read_text(encoding="utf-8")


def test_render_command_rebuilds_html_from_json(tmp_path: Path) -> None:
    trace_json = tmp_path / "t.json"
    invoke("ask", Q, "--mock", "--no-live", "--layers", "0", "--json-out", str(trace_json))
    out = tmp_path / "again.html"
    result = invoke("render", str(trace_json), "-o", str(out))
    assert result.exit_code == 0 and out.exists()
    assert "<details" in out.read_text(encoding="utf-8")


def test_render_command_rejects_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    result = runner.invoke(app, ["render", str(bad)])
    assert result.exit_code == 1 and "not a valid council trace" in result.output
    assert runner.invoke(app, ["render", str(tmp_path / "missing.json")]).exit_code == 2


def test_mounted_under_the_root_cli() -> None:
    from nimlab.cli import app as root

    result = runner.invoke(root, ["--mock", "council", "strategies"])
    assert result.exit_code == 0 and "best-of-n" in result.output
