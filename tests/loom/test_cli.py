from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nimlab.loom.cli import app
from nimlab.loom.library import list_pipelines

runner = CliRunner()


def invoke(*args: str, input: str | None = None):
    return runner.invoke(app, list(args), input=input)


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    for command in ("list", "run", "lint", "graph", "export", "new"):
        assert command in result.output


def test_list_shows_bundled() -> None:
    result = invoke("list", "--json")
    assert result.exit_code == 0
    items = json.loads(result.stdout)
    assert {i["id"] for i in items} >= {"research-brief", "support-triage", "doc-to-quiz", "translate-backcheck"}
    assert all(i["readonly"] for i in items if i["source"] == "bundled")
    assert "research-brief" in invoke("list").output


@pytest.mark.parametrize("name", [i.id for i in list_pipelines(include_user=False)])
def test_run_each_bundled_pipeline(name: str) -> None:
    result = invoke("run", name, "--mock", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["ok"] is True and data["outputs"] and data["tokens"] > 0


def test_run_with_inputs_and_pretty_output() -> None:
    result = invoke("run", "research-brief", "--mock", "--set", "topic=volcanoes")
    assert result.exit_code == 0, result.output
    assert "brief" in result.output and "ok" in result.output and "decompose" in result.output


def test_run_reads_input_from_file(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("Alpha beta gamma. " * 80)
    result = invoke("run", "doc-to-quiz", "--mock", "--json", "--set", f"document=@{tmp_path / 'doc.txt'}")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["outputs"]["question_count"] >= 1


def test_run_caches_between_invocations() -> None:
    first = json.loads(invoke("run", "research-brief", "--mock", "--json").stdout)
    second = json.loads(invoke("run", "research-brief", "--mock", "--json").stdout)
    assert first["cache_hits"] == 0 and second["cache_hits"] > 0 and second["tokens"] == 0
    fresh = json.loads(invoke("run", "research-brief", "--mock", "--json", "--no-cache").stdout)
    assert fresh["cache_hits"] == 0
    assert "removed" in invoke("cache-clear").output


def test_run_errors_exit_2() -> None:
    assert invoke("run", "no-such-pipeline", "--mock").exit_code == 2
    bad_set = invoke("run", "research-brief", "--mock", "--set", "oops")
    assert bad_set.exit_code == 2 and "key=value" in bad_set.output
    unknown = invoke("run", "research-brief", "--mock", "--set", "nope=1")
    assert unknown.exit_code == 2 and "unknown input" in unknown.output


def test_mock_flag_does_not_leak_into_the_environment() -> None:
    os.environ.pop("NIMLAB_MOCK", None)
    invoke("run", "research-brief", "--mock", "--json")
    assert "NIMLAB_MOCK" not in os.environ


def test_lint_clean_and_failing(tmp_path: Path) -> None:
    ok = invoke("lint", "doc-to-quiz")
    assert ok.exit_code == 0 and "clean" in ok.output
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "name": "bad",
        "nodes": [{"id": "a", "type": "prompt", "params": {"template": "{{who}}"}}],
    }))  # fmt: skip
    result = invoke("lint", str(bad))
    assert result.exit_code == 1 and "unknown variable" in result.output and "no output" in result.output
    data = json.loads(invoke("lint", str(bad), "--json").stdout)
    assert data["ok"] is False and data["errors"] >= 2


def test_lint_strict_fails_on_warnings(tmp_path: Path) -> None:
    path = tmp_path / "warn.json"
    path.write_text(json.dumps({
        "name": "w",
        "nodes": [
            {"id": "q", "type": "input", "params": {"default": "x"}},
            {"id": "dead", "type": "prompt", "params": {"template": "{{q}}"}},
            {"id": "o", "type": "output"},
        ],
        "edges": [{"from": "q.value", "to": "dead.q"}, {"from": "q.value", "to": "o.value"}],
    }))  # fmt: skip
    assert invoke("lint", str(path)).exit_code == 0
    assert invoke("lint", str(path), "--strict").exit_code == 1


def test_graph_prints_and_writes_mermaid(tmp_path: Path) -> None:
    result = invoke("graph", "translate-backcheck")
    assert result.exit_code == 0 and result.stdout.startswith("%% Translate and back-check\nflowchart LR")
    target = tmp_path / "g.mmd"
    assert invoke("graph", "translate-backcheck", "-o", str(target)).exit_code == 0
    assert "flowchart LR" in target.read_text()


def test_export_writes_a_script_that_compiles(tmp_path: Path) -> None:
    target = tmp_path / "script.py"
    result = invoke("export", "support-triage", "-o", str(target))
    assert result.exit_code == 0
    compile(target.read_text(), str(target), "exec")
    assert os.access(target, os.X_OK)


def test_new_scaffolds_a_runnable_pipeline(tmp_path: Path) -> None:
    target = tmp_path / "my-flow.json"
    result = invoke("new", "My Flow", "-o", str(target))
    assert result.exit_code == 0 and target.exists()
    data = json.loads(target.read_text())
    assert data["name"] == "my-flow" and data["title"] == "My Flow" and all("pos" in n for n in data["nodes"])
    assert invoke("lint", str(target)).exit_code == 0
    run = invoke("run", str(target), "--mock", "--json")
    assert run.exit_code == 0 and json.loads(run.stdout)["outputs"]["answer"]
    assert invoke("new", "My Flow", "-o", str(target)).exit_code == 2  # refuses to overwrite
    assert invoke("new", "My Flow", "-o", str(target), "--force").exit_code == 0


def test_new_defaults_to_the_working_directory() -> None:
    assert invoke("new", "Hello World").exit_code == 0
    assert Path("hello-world.json").exists()


def test_cli_is_discoverable_as_a_nimlab_plugin() -> None:
    from nimlab.cli import register_plugins

    import typer

    target = typer.Typer()
    assert register_plugins(target, names=("loom",)) == ["loom"]
    assert sys.modules["nimlab.loom.cli"].app is app
