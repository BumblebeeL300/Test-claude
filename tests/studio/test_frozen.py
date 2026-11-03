"""The PyInstaller helper: a bundled app acts as ``python`` for Forge's sandbox commands."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nimlab.studio import frozen


@pytest.fixture(autouse=True)
def _restore_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", list(sys.argv))


def test_not_frozen_does_nothing(tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("raise SystemExit(9)")
    assert frozen.is_frozen() is False
    assert frozen.maybe_run_interpreter(["app", "-B", str(script)]) is None


def test_runs_a_script_like_python_dash_b(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    script = tmp_path / "s.py"
    script.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(out)!r}).write_text(','.join(sys.argv) + '|' + __name__)\n"
    )
    code = frozen.maybe_run_interpreter(["app", "-B", str(script), "a", "b"], frozen=True)
    assert code == 0
    assert out.read_text() == f"{script},a,b|__main__"


def test_exit_codes_are_returned(tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("raise SystemExit(3)")
    assert frozen.maybe_run_interpreter(["app", str(script)], frozen=True) == 3
    script.write_text("raise SystemExit('message')")
    assert frozen.maybe_run_interpreter(["app", str(script)], frozen=True) == 1
    script.write_text("raise SystemExit")
    assert frozen.maybe_run_interpreter(["app", str(script)], frozen=True) == 0


def test_module_mode_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_module(name: str, **kw: object) -> None:
        seen.update(name=name, argv=list(sys.argv), **kw)
        raise SystemExit(5)

    monkeypatch.setattr(frozen.runpy, "run_module", fake_run_module)
    code = frozen.maybe_run_interpreter(
        ["app", "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"], frozen=True
    )
    assert code == 5
    assert seen["name"] == "pytest" and seen["run_name"] == "__main__"
    assert seen["argv"] == ["pytest", "-p", "no:cacheprovider", "-q"]


def test_failing_script_reports_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "s.py"
    script.write_text("raise RuntimeError('boom')")
    assert frozen.maybe_run_interpreter(["app", str(script)], frozen=True) == 1
    assert "boom" in capsys.readouterr().err
    assert frozen.maybe_run_interpreter(["app", "-m", "no_such_module_xyz"], frozen=True) == 1


@pytest.mark.parametrize(
    "argv", [["app"], ["app", "--flag"], ["app", "-psn_0_12345"], ["app", "missing.py"]]
)
def test_normal_launches_start_the_app(argv: list[str]) -> None:
    assert frozen.maybe_run_interpreter(argv, frozen=True) is None
