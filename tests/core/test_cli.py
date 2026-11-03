from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import typer
from typer.testing import CliRunner

from nimlab import __version__
from nimlab.cli import PLUGIN_NAMES, app, main, probe_connectivity, register_plugins
from nimlab.core import NIMConfig
from nimlab.core.logging import get_console, set_console

runner = CliRunner()
BASE = "https://integrate.api.nvidia.com/v1"


def invoke(*args: str, env: dict[str, str] | None = None) -> Any:
    return runner.invoke(app, list(args), env={"COLUMNS": "200", **(env or {})})


def test_version() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert f"nimlab {__version__}" in result.output


def test_no_args_shows_help() -> None:
    result = invoke()
    assert "doctor" in result.output and "models" in result.output and "--mock" in result.output


def test_main_is_a_callable_entry_point() -> None:
    assert callable(main) and isinstance(app, typer.Typer)


def test_console_script_target_declared() -> None:
    import importlib.metadata as md

    eps = [e for e in md.entry_points(group="console_scripts") if e.name == "nimlab"]
    assert eps and eps[0].value == "nimlab.cli:main"


# ------------------------------------------------------------------------------- doctor
def test_doctor_without_key_reports_mock() -> None:
    result = invoke("doctor", "--offline")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "not set" in out
    assert "ON" in out and "no NVIDIA_API_KEY" in out
    assert "https://integrate.api.nvidia.com/v1" in out
    assert "skipped (--offline)" in out
    assert "nvapi" not in out


def test_doctor_with_key_never_prints_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-SUPER-SECRET")
    result = invoke("doctor", "--offline")
    assert result.exit_code == 0
    assert "SUPER-SECRET" not in result.output
    assert " set" in result.output and "real NIM calls" in result.output


def test_global_mock_flag_sets_env_for_the_command_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setenv("NIMLAB_MOCK", "0")
    result = invoke("--mock", "doctor", "--offline")
    assert result.exit_code == 0
    assert "NIMLAB_MOCK / --mock" in result.output  # the command saw NIMLAB_MOCK=1
    assert os.environ["NIMLAB_MOCK"] == "0"  # ...and the previous value is back afterwards


def test_global_mock_flag_does_not_leak_when_previously_unset() -> None:
    assert "NIMLAB_MOCK" not in os.environ
    assert invoke("--mock", "doctor", "--offline").exit_code == 0
    assert "NIMLAB_MOCK" not in os.environ


def test_cli_logging_handler_can_be_reset() -> None:
    import logging

    from nimlab.core.logging import reset_logging

    assert invoke("doctor", "--offline").exit_code == 0
    logger = logging.getLogger("nimlab")
    assert any(getattr(h, "_nimlab_handler", False) for h in logger.handlers)
    reset_logging()
    assert not any(getattr(h, "_nimlab_handler", False) for h in logger.handlers)
    assert logger.level == logging.NOTSET


def test_doctor_connectivity_ok(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE}/models").respond(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "reachable, 2 models listed" in result.output


def test_doctor_real_mode_unreachable_fails(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    respx_mock.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("blocked"))
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "FAILED" in result.output and "unreachable" in result.output


def test_doctor_mock_mode_unreachable_is_informational(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("blocked"))
    result = invoke("doctor")
    assert result.exit_code == 0
    assert "unavailable" in result.output


def test_doctor_auth_failure(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-bad")
    respx_mock.get(f"{BASE}/models").respond(401, json={"detail": "nope"})
    result = invoke("doctor")
    assert result.exit_code == 1 and "authentication failed" in result.output


def test_probe_connectivity_states(respx_mock: respx.MockRouter) -> None:
    cfg = NIMConfig(api_key="k", mock=False)
    route = respx_mock.get(f"{BASE}/models")
    route.respond(200, text="not json")
    assert probe_connectivity(cfg) == (True, "reachable (HTTP 200)")
    route.respond(503)
    assert probe_connectivity(cfg) == (False, "HTTP 503")
    assert route.calls.last.request.headers["authorization"] == "Bearer k"


# ------------------------------------------------------------------------------- models
def test_models_lists_catalog() -> None:
    result = invoke("models")
    assert result.exit_code == 0, result.output
    for model_id in (
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-rerank-8b",
    ):
        assert model_id in result.output
    assert "Default for" in result.output


@pytest.mark.parametrize("flag", ["--role", "-r"])
def test_models_role_filter(flag: str) -> None:
    result = invoke("models", flag, "coder")
    assert result.exit_code == 0
    assert "poolside/laguna-xs-2.1" in result.output
    assert "nvidia/nemotron-3-super-120b-a12b" not in result.output


def test_models_role_is_case_insensitive() -> None:
    assert "nvidia/nemotron-3.5-embed-1b" in invoke("models", "--role", "EMBED").output


def test_models_invalid_role() -> None:
    result = invoke("models", "--role", "wizard")
    assert result.exit_code == 2


def test_models_shows_env_override_as_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MODEL_CODER", "mistralai/codestral-22b-instruct-v0.1")
    out = invoke("models", "--role", "coder").output
    lines = [ln for ln in out.splitlines() if "codestral" in ln]
    assert lines and "coder" in lines[0].split("│")[-3]  # "Default for" column


def test_models_live_in_mock_mode() -> None:
    result = invoke("models", "--live")
    assert result.exit_code == 0, result.output
    assert "Live" in result.output and "yes" in result.output
    assert "mock mode" in result.output


def test_models_live_marks_availability(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    respx_mock.get(f"{BASE}/models").respond(
        200,
        json={"data": [{"id": "nvidia/nemotron-3-super-120b-a12b"}, {"id": "brand/new-model-1"}]},
    )
    result = invoke("models", "--live", "--role", "chat")
    assert result.exit_code == 0, result.output
    row_ok = next(
        ln for ln in result.output.splitlines() if "nvidia/nemotron-3-super-120b-a12b" in ln
    )
    row_no = next(
        ln for ln in result.output.splitlines() if "nvidia/nemotron-3-ultra-550b-a55b" in ln
    )
    assert "yes" in row_ok and "no" in row_no
    assert "1 live model(s) not in the catalog: brand/new-model-1" in result.output


def test_models_live_failure_still_shows_table_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    respx_mock.get(f"{BASE}/models").respond(401, json={"detail": "bad key"})
    result = invoke("models", "--live")
    assert result.exit_code == 1
    assert "Could not fetch live models" in result.output
    assert "nvidia/nemotron-3-super-120b-a12b" in result.output


# ------------------------------------------------------------------------------- plugins
def make_plugin(name: str, *, with_app: bool = True) -> types.ModuleType:
    mod = types.ModuleType(f"nimlab.{name}.cli")
    if with_app:
        sub = typer.Typer(help=f"{name} app")

        @sub.command()
        def hello(who: str = "world") -> None:
            typer.echo(f"{name} says hello to {who}")

        @sub.command()
        def other() -> None:
            typer.echo("other")

        mod.app = sub  # type: ignore[attr-defined]
    return mod


def test_register_plugins_adds_typer_subapps() -> None:
    root = typer.Typer()

    @root.command()
    def base() -> None: ...

    @root.command()
    def base2() -> None: ...

    modules = {
        "nimlab.council.cli": make_plugin("council"),
        "nimlab.forge.cli": make_plugin("forge"),
    }

    def importer(name: str) -> types.ModuleType:
        if name in modules:
            return modules[name]
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    assert register_plugins(root, importer=importer) == ["council", "forge"]
    result = runner.invoke(root, ["council", "hello", "--who", "Ada"])
    assert result.exit_code == 0 and "council says hello to Ada" in result.output
    assert "other" in runner.invoke(root, ["forge", "--help"]).output
    # idempotent: registering twice does not duplicate groups
    register_plugins(root, importer=importer)
    assert [g.name for g in root.registered_groups] == ["council", "forge"]


def test_missing_plugin_package_or_cli_module_is_skipped_silently() -> None:
    def importer(name: str) -> types.ModuleType:
        pkg = name.split(".")[1]
        missing = f"nimlab.{pkg}" if pkg in ("council", "atlas") else name
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    assert register_plugins(typer.Typer(), importer=importer) == []


def test_broken_plugin_import_is_reraised() -> None:
    def importer(name: str) -> types.ModuleType:
        raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")

    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        register_plugins(typer.Typer(), importer=importer)


def test_other_errors_inside_a_plugin_are_reraised() -> None:
    def importer(name: str) -> types.ModuleType:
        raise ImportError("cannot import name 'thing' from 'nimlab.core'")

    with pytest.raises(ImportError, match="cannot import name"):
        register_plugins(typer.Typer(), importer=importer)

    def importer2(name: str) -> types.ModuleType:
        raise SyntaxError("bad plugin")

    with pytest.raises(SyntaxError):
        register_plugins(typer.Typer(), importer=importer2)


def test_module_without_typer_app_is_ignored() -> None:
    modules = {
        "nimlab.arena.cli": make_plugin("arena", with_app=False),
        "nimlab.atlas.cli": types.ModuleType("nimlab.atlas.cli"),
    }
    modules["nimlab.atlas.cli"].app = object()  # type: ignore[attr-defined]

    def importer(name: str) -> types.ModuleType:
        if name in modules:
            return modules[name]
        raise ModuleNotFoundError(name=name)

    assert register_plugins(typer.Typer(), importer=importer) == []


def test_real_importlib_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default importer finds plugins through sys.modules / the import system."""
    pkg = types.ModuleType("nimlab.arena")
    pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nimlab.arena", pkg)
    monkeypatch.setitem(sys.modules, "nimlab.arena.cli", make_plugin("arena"))
    root = typer.Typer()

    @root.command()
    def a() -> None: ...

    @root.command()
    def b() -> None: ...

    assert register_plugins(root, names=("arena", "doesnotexist")) == ["arena"]
    assert runner.invoke(root, ["arena", "hello"]).output.strip() == "arena says hello to world"


def test_plugin_names_contract() -> None:
    assert PLUGIN_NAMES == (
        "council",
        "atlas",
        "forge",
        "arena",
        "prism",
        "sentinel",
        "loom",
        "distill",
        "studio",
    )


def test_doctor_lists_unloaded_apps() -> None:
    out = invoke("doctor", "--offline").output
    assert "apps not installed" in out


# ------------------------------------------------------------------------------- console
def test_get_console_is_shared() -> None:
    set_console(None)
    assert get_console() is get_console()


# ------------------------------------------------------------------------------- models --live
LIVE_LIST = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "google/gemma-4-31b-it",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3.5-embed-1b",
    "acme/zeta-guard-8b",
    "brand/new-model-1",
]


def live_listing() -> dict[str, object]:
    return {"data": [{"id": i} for i in LIVE_LIST]}


def test_models_live_shows_each_roles_choice_and_suggests_replacements(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    respx_mock.get(f"{BASE}/models").respond(200, json=live_listing())
    out = invoke("models", "--live").output
    assert "Chosen model per role" in out
    # chat default (nemotron-3-super) is not served here: gemma 4 is substituted
    chat_row = next(ln for ln in out.splitlines() if ln.strip().startswith("│ chat"))
    assert "google/gemma-4-31b-it" in chat_row and "auto-substituted" in chat_row
    reasoning_row = next(ln for ln in out.splitlines() if ln.strip().startswith("│ reasoning"))
    assert "nvidia/nemotron-3-ultra-550b-a55b" in reasoning_row and "ok" in reasoning_row
    # safety falls back to the heuristic match (name contains "guard")
    safety_row = next(ln for ln in out.splitlines() if ln.strip().startswith("│ safety"))
    assert "acme/zeta-guard-8b" in safety_row
    assert "Curated ids missing from the live catalog" in out
    assert "nvidia/nemotron-3-super-120b-a12b" in out and "suggested:" in out
    rerank_row = next(ln for ln in out.splitlines() if ln.strip().startswith("│ rerank"))
    assert "retrieval API" in rerank_row  # not listed by /models: not flagged as missing


def test_models_live_respects_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setenv("NIMLAB_MODEL_CHAT", "mine/not-listed")
    respx_mock.get(f"{BASE}/models").respond(200, json=live_listing())
    out = invoke("models", "--live").output
    chat_row = next(ln for ln in out.splitlines() if ln.strip().startswith("│ chat"))
    assert (
        "mine/not-listed" in chat_row and "env" in chat_row and "not in the live list" in chat_row
    )


def test_models_refresh_ignores_the_cache(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    route = respx_mock.get(f"{BASE}/models").respond(200, json=live_listing())
    invoke("models", "--live")
    invoke("models", "--live")
    assert route.call_count == 1  # second run served by the 24 h cache
    result = invoke("models", "--refresh")
    assert result.exit_code == 0, result.output
    assert route.call_count == 2 and "Chosen model per role" in result.output


def test_models_refresh_failure_exits_1(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    respx_mock.get(f"{BASE}/models").mock(side_effect=httpx.ConnectError("blocked"))
    result = invoke("models", "--refresh")
    assert result.exit_code == 1 and "Could not fetch live models" in result.output


# ------------------------------------------------------------------------------- uninstall
def test_uninstall_dry_run_changes_nothing(tmp_path: Path) -> None:
    home = tmp_path / "nimlab-home"
    (home / "Projects" / "p").mkdir(parents=True)
    (home / "studio_history.json").write_text("{}")
    result = invoke("uninstall", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output and "Forge projects" in result.output
    assert (home / "studio_history.json").exists() and (home / "Projects" / "p").exists()


def test_uninstall_asks_for_the_word(tmp_path: Path) -> None:
    home = tmp_path / "nimlab-home"
    home.mkdir()
    (home / "studio_history.json").write_text("{}")
    runner = CliRunner()
    cancelled = runner.invoke(app, ["uninstall"], input="nope\n")
    assert cancelled.exit_code == 1 and "Cancelled" in cancelled.output
    assert (home / "studio_history.json").exists()
    done = runner.invoke(app, ["uninstall"], input="UNINSTALL\n")
    assert done.exit_code == 0, done.output
    assert not (home / "studio_history.json").exists()


def test_uninstall_keeps_projects_unless_asked(tmp_path: Path) -> None:
    home = tmp_path / "nimlab-home"
    (home / "Projects" / "p").mkdir(parents=True)
    (home / "Projects" / "p" / "main.py").write_text("x")
    (home / "studio_history.json").write_text("{}")
    assert invoke("uninstall", "--yes").exit_code == 0
    assert (home / "Projects" / "p" / "main.py").exists()
    assert invoke("uninstall", "--yes", "--remove-projects").exit_code == 0
    assert not home.exists()


# ------------------------------------------------------------------------------- update
def test_update_check_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from nimlab.studio import updater

    monkeypatch.setattr(
        updater,
        "check_for_updates",
        lambda **kw: updater.UpdateStatus(
            kind="git",
            state="behind",
            behind=2,
            can_update=True,
            message="2 new commits available on origin/claude/nimlab.",
            commits=["abc1234 add things"],
        ),
    )
    result = invoke("update", "--check")
    assert result.exit_code == 0 and "2 new commits" in result.output and "abc1234" in result.output


def test_update_applies_when_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    from nimlab.studio import updater

    monkeypatch.setattr(
        updater,
        "check_for_updates",
        lambda **kw: updater.UpdateStatus(
            kind="git", state="behind", behind=1, can_update=True, message="1 new commit"
        ),
    )
    monkeypatch.setattr(
        updater,
        "apply_update",
        lambda **kw: {
            "ok": True,
            "restart": True,
            "steps": [{"step": "git pull --ff-only", "ok": True, "output": ""}],
            "message": "Updated. Restart NIM Lab to use the new version.",
        },
    )
    result = invoke("update", "--yes")
    assert result.exit_code == 0 and "Restart NIM Lab" in result.output


@pytest.mark.parametrize("state", ["offline", "error", "diverged"])
def test_update_problems_exit_1(monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    from nimlab.studio import updater

    monkeypatch.setattr(
        updater,
        "check_for_updates",
        lambda **kw: updater.UpdateStatus(kind="git", state=state, message="a clear message"),
    )
    result = invoke("update")
    assert result.exit_code == 1 and "a clear message" in result.output
