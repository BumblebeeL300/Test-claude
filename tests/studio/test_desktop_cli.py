"""The desktop shell and the ``nimlab studio`` command (no display, no pywebview needed)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from studio_helpers import make_settings
from typer.testing import CliRunner

from nimlab.cli import app, register_plugins
from nimlab.studio import desktop

runner = CliRunner()


def interrupt_once(monkeypatch: pytest.MonkeyPatch, running: desktop.RunningServer) -> None:
    """Make the first ``thread.join`` (launch's wait loop) raise Ctrl+C; later joins are real."""
    real_join = running.thread.join
    state = {"fired": False}

    def join(timeout: float | None = None) -> None:
        if not state["fired"]:
            state["fired"] = True
            raise KeyboardInterrupt
        real_join(timeout)

    monkeypatch.setattr(running.thread, "join", join)


class TestCli:
    def test_studio_is_registered_on_the_root_cli(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0 and "studio" in result.output

    def test_help(self) -> None:
        result = runner.invoke(app, ["studio", "--help"])
        assert result.exit_code == 0
        for text in ("--browser", "--port", "--dev", "native window"):
            assert text in result.output

    def test_plugin_discovery_loads_studio(self) -> None:
        import typer

        root = typer.Typer()

        @root.command()
        def a() -> None: ...

        @root.command()
        def b() -> None: ...

        assert "studio" in register_plugins(root)

    def test_invocation_launches_with_the_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_launch(**kw: Any) -> int:
            seen.update(kw)
            return 0

        monkeypatch.setattr(desktop, "launch", fake_launch)
        result = runner.invoke(app, ["studio", "--browser", "--port", "8123", "--dev"])
        assert result.exit_code == 0
        assert seen == {"browser": True, "port": 8123, "dev": True}
        seen.clear()
        assert runner.invoke(app, ["studio"]).exit_code == 0
        assert seen == {"browser": False, "port": 0, "dev": False}

    def test_exit_code_is_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(desktop, "launch", lambda **kw: 3)
        assert runner.invoke(app, ["studio"]).exit_code == 3

    def test_bad_port_is_a_usage_error(self) -> None:
        assert runner.invoke(app, ["studio", "--port", "99999"]).exit_code == 2

    def test_python_dash_m_help(self) -> None:
        out = subprocess.run(
            [sys.executable, "-m", "nimlab", "studio", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0 and "--browser" in out.stdout


class TestWithoutPywebview:
    def test_module_imports_and_reports_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "webview", None)  # `import webview` -> ImportError
        mod = importlib.reload(desktop)
        try:
            assert mod.webview_available() is False
            assert mod.WINDOW_TITLE == "NIM Lab Studio"
        finally:
            importlib.reload(desktop)

    def test_launch_falls_back_to_the_browser(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        opened: list[str] = []
        monkeypatch.setitem(sys.modules, "webview", None)
        monkeypatch.setattr(desktop.webbrowser, "open", lambda url: opened.append(url) or True)

        real_start = desktop.start_server

        def start(settings: Any = None, **kw: Any) -> desktop.RunningServer:
            running = real_start(make_settings(tmp_path), **kw)
            # emulate Ctrl+C right after the browser opened
            interrupt_once(monkeypatch, running)
            return running

        monkeypatch.setattr(desktop, "start_server", start)
        assert desktop.launch(browser=False) == 0
        assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:")

    def test_launch_uses_the_browser_when_asked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        opened: list[str] = []
        monkeypatch.setattr(desktop.webbrowser, "open", lambda url: opened.append(url) or True)
        monkeypatch.setattr(
            desktop, "open_window", lambda *a, **k: pytest.fail("no window with --browser")
        )
        monkeypatch.setattr(desktop, "webview_available", lambda: True)
        real_start = desktop.start_server

        def start(settings: Any = None, **kw: Any) -> desktop.RunningServer:
            running = real_start(make_settings(tmp_path, token="tok123"), **kw)
            interrupt_once(monkeypatch, running)
            return running

        monkeypatch.setattr(desktop, "start_server", start)
        assert desktop.launch(browser=True) == 0
        assert opened[0].endswith("/?t=tok123")


class TestWindowOptions:
    def test_macos_uses_hidden_titlebar_styling(self) -> None:
        opts = desktop.window_options("darwin")
        assert (opts["width"], opts["height"]) == (1280, 820)
        assert opts["min_size"] == (960, 640)
        assert opts["background_color"] == "#0B0A12"
        assert opts["frameless"] is True and opts["easy_drag"] is False

    def test_other_platforms_keep_the_native_frame(self) -> None:
        opts = desktop.window_options("linux")
        assert "frameless" not in opts and opts["background_color"] == "#0B0A12"

    def test_unsupported_options_are_dropped(self) -> None:
        def create_window(title, url, width=0, height=0, min_size=(0, 0)): ...  # type: ignore[no-untyped-def]

        kept = desktop._supported(create_window, desktop.window_options("darwin"))
        assert set(kept) == {"width", "height", "min_size"}
        assert desktop._supported(lambda **kw: None, {"x": 1}) == {"x": 1}

    def test_window_title(self) -> None:
        assert desktop.WINDOW_TITLE == "NIM Lab Studio"


class TestJsBridge:
    def test_public_api_and_private_window_handle(self) -> None:
        api = desktop.DesktopApi()
        public = {n for n in dir(api) if not n.startswith("_")}
        assert {"pick_folder", "open_path", "bind"} <= public
        assert api._window is None and api.pick_folder() is None

    def test_pick_folder_uses_the_native_dialog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeWebview:
            FOLDER_DIALOG = 20

        class Window:
            def create_file_dialog(self, kind: int) -> tuple[str, ...] | None:
                assert kind == 20
                return ("/Users/me/Documents",)

        monkeypatch.setitem(sys.modules, "webview", FakeWebview)
        api = desktop.DesktopApi()
        api.bind(Window())
        assert api.pick_folder() == "/Users/me/Documents"
        api.bind(type("W", (), {"create_file_dialog": lambda self, k: None})())
        assert api.pick_folder() is None

    def test_open_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        api = desktop.DesktopApi()
        assert api.open_path(str(tmp_path / "missing")) is False
        launched: list[list[str]] = []
        monkeypatch.setattr(
            desktop.subprocess, "Popen", lambda cmd, **kw: launched.append(list(cmd))
        )
        monkeypatch.setattr(desktop.sys, "platform", "darwin")
        assert api.open_path(str(tmp_path)) is True
        assert launched == [["open", str(tmp_path)]]
        monkeypatch.setattr(desktop.sys, "platform", "linux")
        assert api.open_path(str(tmp_path)) is True
        assert launched[-1] == ["xdg-open", str(tmp_path)]


class TestRealServer:
    """Start the actual uvicorn thread on a free loopback port."""

    def test_start_serve_and_stop(self, tmp_path: Path) -> None:
        running = desktop.start_server(make_settings(tmp_path, token="s3cret-token"), port=0)
        try:
            assert running.port > 0 and running.base_url == f"http://127.0.0.1:{running.port}"
            assert running.url == f"{running.base_url}/?t=s3cret-token"
            assert httpx.get(f"{running.base_url}/api/health").json()["ok"] is True
            assert httpx.get(f"{running.base_url}/api/status").status_code == 401
            ok = httpx.get(
                f"{running.base_url}/api/status", headers={"X-Studio-Token": "s3cret-token"}
            )
            assert ok.status_code == 200 and ok.json()["mode"] == "mock"
            # DNS rebinding: right socket, wrong Host
            evil = httpx.get(f"{running.base_url}/api/health", headers={"Host": "evil.example.com"})
            assert evil.status_code == 403
            assert httpx.get(f"{running.base_url}/").status_code == 200
        finally:
            running.stop()
        assert not running.thread.is_alive()
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"{running.base_url}/api/health", timeout=1)

    def test_binds_loopback_only(self, tmp_path: Path) -> None:
        running = desktop.start_server(make_settings(tmp_path), port=0)
        try:
            (sock,) = running.server.servers[0].sockets
            assert sock.getsockname()[0] == "127.0.0.1"
        finally:
            running.stop()

    def test_two_instances_get_different_ports_and_tokens(self, tmp_path: Path) -> None:
        a = desktop.start_server(make_settings(tmp_path / "a", token="aaaa"), port=0)
        b = desktop.start_server(make_settings(tmp_path / "b", token="bbbb"), port=0)
        try:
            assert a.port != b.port
        finally:
            a.stop()
            b.stop()

    def test_wait_for_health_times_out(self) -> None:
        with pytest.raises(RuntimeError, match="did not become healthy"):
            desktop.wait_for_health("http://127.0.0.1:9", timeout=0.3)
