"""The ``distill`` CLI through Typer's CliRunner, always with ``--mock``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from distill_helpers import DOCS, EXAMPLES
from typer.testing import CliRunner

from nimlab.distill.cli import app
from nimlab.distill.export import to_json, Record

runner = CliRunner()


def invoke(*args: str, expect: int = 0):  # type: ignore[no-untyped-def]
    result = runner.invoke(app, list(args))
    assert result.exit_code == expect, result.output + repr(result.exception)
    return result


@pytest.fixture
def runs(tmp_path: Path) -> str:
    return str(tmp_path / "runs")


def test_help_lists_every_command() -> None:
    out = runner.invoke(app, []).output
    for cmd in ("recipes", "run", "resume", "report", "inspect", "dedup"):
        assert cmd in out
        assert "Usage" in invoke(cmd, "--help").output


def test_recipes_table_json_and_show() -> None:
    out = invoke("recipes").output
    for name in ("qa_from_docs", "instruction_diversity", "preference_pairs", "eval_set"):
        assert name in out
    data = json.loads(invoke("recipes", "--json").output)
    assert set(data) >= {"qa_from_docs", "eval_set"}
    assert json.loads(invoke("recipes", "--show", "eval_set").output)["kind"] == "eval_set"
    assert "unknown recipe" in invoke("recipes", "--show", "zzz", expect=1).output


def test_run_report_inspect_dedup_workflow(tmp_path: Path, runs: str) -> None:
    out_dir = tmp_path / "data"
    result = invoke(
        "run", "instruction_diversity", "--n", "60", "--mock", "--runs-dir", runs,
        "--out", str(out_dir), "--format", "alpaca", "--run-id", "r1",
        "--models", "a/x,b/y", "--domain", "cooking",
    )  # fmt: skip
    assert "Simulated" in result.output and "examples kept of 60" in result.output
    assert "Passed filters" in result.output
    data = out_dir / "r1" / "instruction_diversity.alpaca.jsonl"
    assert data.is_file() and (out_dir / "r1" / "README.md").is_file()
    assert "instruction" in json.loads(data.read_text().splitlines()[0])

    html = tmp_path / "rep.html"
    rep = invoke("report", "r1", "--runs-dir", runs, "--html-out", str(html))
    assert "Distill funnel" in rep.output and "Models" in rep.output
    assert "Simulated" in html.read_text()
    assert json.loads(invoke("report", "r1", "--runs-dir", runs, "--json").output)["run_id"] == "r1"

    ins = invoke("inspect", str(data), "--sample", "2")
    assert "example 2" in ins.output and "records" in ins.output

    dup = tmp_path / "dups.jsonl"
    rows = [Record("Explain unit testing to a beginner", "r1"), Record("explain unit testing to a beginner!", "r2"), Record("Bake bread", "r3")]
    dup.write_text("\n".join(json.dumps(to_json(r, "openai-chat")) for r in rows) + "\n")
    out = tmp_path / "unique.jsonl"
    res = invoke("dedup", str(dup), "--threshold", "0.92", "--mock", "--out", str(out))
    assert "3 rows -> 2 kept" in res.output
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert [k["messages"][0]["content"] for k in kept] == [
        "Explain unit testing to a beginner",
        "Bake bread",
    ]


def test_qa_with_docs_eval_set_and_custom_recipe(tmp_path: Path, runs: str) -> None:
    custom = EXAMPLES / "recipe_faq_strict.json"
    res = invoke(
        "run", str(custom), "--n", "40", "--docs", str(DOCS), "--mock", "--runs-dir", runs,
        "--out", str(tmp_path / "o"),
    )  # fmt: skip
    assert "examples kept of 40" in res.output
    ev = invoke(
        "run", "instruction_diversity", "--n", "80", "--mock", "--runs-dir", runs,
        "--eval-set", str(EXAMPLES / "eval_holdout.jsonl"),
    )  # fmt: skip
    assert "contaminated" in ev.output


def test_resume_a_finished_run_and_unknown_run(runs: str) -> None:
    invoke("run", "eval_set", "--n", "20", "--mock", "--runs-dir", runs, "--run-id", "x")
    assert "examples kept" in invoke("resume", "x", "--mock", "--runs-dir", runs).output
    assert "no such run" in invoke("resume", "ghost", "--mock", "--runs-dir", runs, expect=1).output
    assert "no such run" in invoke("report", "ghost", "--runs-dir", runs, expect=1).output


def test_errors_are_reported_cleanly(runs: str, tmp_path: Path) -> None:
    assert "unknown recipe" in invoke("run", "nope", "--mock", "--runs-dir", runs, expect=1).output
    bad = invoke("run", "instruction_diversity", "--format", "dpo", "--mock", "--runs-dir", runs, expect=1)
    assert "dpo format needs" in bad.output
    assert "does not exist" in invoke("run", "qa_from_docs", "--docs", str(tmp_path / "x"), "--mock", "--runs-dir", runs, expect=1).output
    assert "cannot read" in invoke("inspect", str(tmp_path / "missing.jsonl"), expect=1).output
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert "no records" in invoke("inspect", str(empty), expect=1).output


def test_global_mock_flag_and_env_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_DISTILL_RUNS", str(tmp_path / "envruns"))
    res = runner.invoke(app, ["--mock", "run", "eval_set", "--n", "10", "--run-id", "g"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "envruns" / "g" / "run.json").is_file()
