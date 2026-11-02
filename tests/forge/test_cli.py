"""The ``nimlab forge`` CLI, driven through typer's CliRunner (always offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nimlab.forge import ForgeReport
from nimlab.forge.cli import app

runner = CliRunner()


def invoke(*args: str, input: str | None = None):
    return runner.invoke(app, list(args), input=input)


class TestHelp:
    def test_help_lists_commands(self) -> None:
        res = invoke("--help")
        assert res.exit_code == 0
        assert "run" in res.output and "inspect" in res.output
        assert "test-driven" in res.output

    def test_run_help_documents_all_options(self) -> None:
        res = invoke("run", "--help")
        assert res.exit_code == 0
        for flag in (
            "--workspace", "--max-iters", "--planner", "--coder", "--reviewer",
            "--dry-run", "--rollback-on-fail", "--mock", "--task-file", "--json",
        ):  # fmt: skip
            assert flag in res.output

    def test_no_args_shows_help(self) -> None:
        res = invoke()
        assert "Usage" in res.output


class TestRunDryRun:
    def test_mock_dry_run(self, ws: Path) -> None:
        res = invoke("run", "Write fizzbuzz with tests", "-w", str(ws), "--mock", "--dry-run")
        assert res.exit_code == 0, res.output
        assert "MOCK MODE" in res.output
        assert "DRY_RUN" in res.output and "Plan" in res.output
        assert "No files were written" in res.output
        assert list(ws.iterdir()) == []  # a dry run writes nothing, not even .forge/

    def test_dry_run_json(self, ws: Path) -> None:
        res = invoke("run", "Write fizzbuzz", "-w", str(ws), "--mock", "--dry-run", "--json")
        assert res.exit_code == 0, res.output
        data = json.loads(res.output)
        assert data["status"] == "dry_run" and data["plan"]["steps"] and data["mock"] is True
        assert list(ws.iterdir()) == []

    def test_dry_run_on_missing_workspace(self, tmp_path: Path) -> None:
        target = tmp_path / "later"
        res = invoke("run", "Idea", "-w", str(target), "--mock", "--dry-run", "-q")
        assert res.exit_code == 0 and not target.exists()
        assert res.output.strip() == ""


class TestRun:
    def test_mock_run_writes_report_and_exits_zero(self, ws: Path) -> None:
        res = invoke(
            "run", "Write fizzbuzz with tests", "-w", str(ws), "--mock", "--max-iters", "2"
        )
        assert res.exit_code == 0, res.output  # in mock mode a graceful finish is success
        assert "Forge report" in res.output and "Phase timings" in res.output
        assert "NIM usage" in res.output
        report = ForgeReport.load(ws)
        assert report.mock and report.task == "Write fizzbuzz with tests"
        assert report.models is not None

    def test_live_output_lists_phases_and_events(self, ws: Path) -> None:
        res = invoke("run", "Write fizzbuzz", "-w", str(ws), "--mock", "--max-iters", "1")
        for needle in ("> explore", "> plan", "> implement", "> verify", "> report", "list_dir"):
            assert needle in res.output

    def test_model_overrides_reach_the_report(self, ws: Path) -> None:
        res = invoke(
            "run", "t", "-w", str(ws), "--mock", "--dry-run", "--json",
            "--planner", "x/planner", "--coder", "y/coder", "--reviewer", "z/reviewer",
        )  # fmt: skip
        assert json.loads(res.output)["models"] == {
            "planner": "x/planner", "coder": "y/coder", "reviewer": "z/reviewer",
        }  # fmt: skip

    def test_rollback_flag(self, ws: Path) -> None:
        (ws / "keep.txt").write_text("keep")
        res = invoke("run", "t", "-w", str(ws), "--mock", "--rollback-on-fail", "--max-iters", "1")
        assert res.exit_code == 0, res.output
        assert ForgeReport.load(ws).status == "failed"
        assert (ws / "keep.txt").read_text() == "keep"

    def test_task_from_file(self, ws: Path, tmp_path: Path) -> None:
        task_file = tmp_path / "task.md"
        task_file.write_text("# Task\nBuild an LRU cache.\n")
        res = invoke("run", "-f", str(task_file), "-w", str(ws), "--mock", "--dry-run", "--json")
        assert res.exit_code == 0, res.output
        assert "Build an LRU cache" in json.loads(res.output)["task"]

    def test_task_from_stdin(self, ws: Path) -> None:
        res = invoke(
            "run", "-", "-w", str(ws), "--mock", "--dry-run", "--json", input="from stdin\n"
        )
        assert res.exit_code == 0 and json.loads(res.output)["task"] == "from stdin"

    def test_missing_task_is_a_usage_error(self, ws: Path) -> None:
        res = invoke("run", "-w", str(ws), "--mock")
        assert res.exit_code == 2
        assert "TASK" in res.output

    def test_unreadable_task_file(self, ws: Path, tmp_path: Path) -> None:
        res = invoke("run", "-f", str(tmp_path / "nope.md"), "-w", str(ws), "--mock")
        assert res.exit_code != 0

    def test_invalid_max_iters(self, ws: Path) -> None:
        assert invoke("run", "t", "-w", str(ws), "--mock", "--max-iters", "-1").exit_code == 2

    def test_diff_flag_prints_unified_diff(self, ws: Path) -> None:
        (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        (ws / "test_calc.py").write_text(
            "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        res = invoke("run", "t", "-w", str(ws), "--mock", "--diff")
        assert res.exit_code == 0, res.output

    def test_env_without_key_falls_back_to_mock_without_flag(self, ws: Path) -> None:
        # No NVIDIA_API_KEY in the (scrubbed) test env: the core switches to mock on its own.
        res = invoke("run", "t", "-w", str(ws), "--dry-run", "--json")
        assert res.exit_code == 0, res.output
        assert json.loads(res.output)["mock"] is True


class TestInspect:
    @pytest.fixture
    def finished(self, ws: Path) -> Path:
        res = invoke("run", "Implement a thing", "-w", str(ws), "--mock", "--max-iters", "1", "-q")
        assert res.exit_code == 0, res.output
        return ws

    def test_inspect_renders_last_report(self, finished: Path) -> None:
        res = invoke("inspect", str(finished))
        assert res.exit_code == 0, res.output
        assert "Task:" in res.output and "Implement a thing" in res.output
        assert "FAILED" in res.output or "SUCCESS" in res.output
        assert "Phase timings" in res.output

    def test_inspect_json_and_markdown(self, finished: Path) -> None:
        raw = invoke("inspect", str(finished), "--json")
        assert json.loads(raw.output)["task"] == "Implement a thing"
        md = invoke("inspect", str(finished), "--markdown")
        assert md.output.startswith("# Forge report:") and "Implement a thing" in md.output

    def test_inspect_accepts_the_forge_dir_itself(self, finished: Path) -> None:
        assert invoke("inspect", str(finished / ".forge"), "--json").exit_code == 0

    def test_inspect_without_report(self, ws: Path) -> None:
        res = invoke("inspect", str(ws))
        assert res.exit_code == 1 and "No Forge report" in res.output

    def test_inspect_corrupt_report(self, ws: Path) -> None:
        (ws / ".forge").mkdir()
        (ws / ".forge" / "forge_report.json").write_text("{not json")
        res = invoke("inspect", str(ws))
        assert res.exit_code == 1 and "Unreadable" in res.output
