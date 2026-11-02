"""The desktop shell: run the Studio server in a background thread and show it in a window.

* With ``pywebview`` installed (``pip install 'nimlab[desktop]'``) Studio opens a native window.
* Otherwise, or with ``--browser``, it opens your default browser.

The server binds ``127.0.0.1`` only, on a free port, and every ``/api`` call needs the random token
that is passed to the page in the URL (``?t=...``). Importing this module never requires pywebview.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .settings import StudioSettings

__all__ = [
    "BACKGROUND_COLOR",
    "WINDOW_TITLE",
    "DesktopApi",
    "RunningServer",
    "launch",
    "start_server",
    "webview_available",
]

_log = logging.getLogger("nimlab.studio")

WINDOW_TITLE = "NIM Lab Studio"
BACKGROUND_COLOR = "#0B0A12"
WINDOW_SIZE = (1280, 820)
MIN_SIZE = (960, 640)
HOST = "127.0.0.1"


def _say(message: str) -> None:
    """``print`` that tolerates a windowed app without a stdout."""
    if sys.stdout is not None:
        print(message, flush=True)


def webview_available() -> bool:
    """True when pywebview can be imported (it is an optional dependency)."""
    try:
        import webview  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- server thread


@dataclass
class RunningServer:
    """A Studio server running in a background thread."""

    server: Any  # uvicorn.Server
    thread: threading.Thread
    port: int
    token: str | None
    app: Any

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    @property
    def url(self) -> str:
        """The URL to open: carries the auth token for the web UI to keep in sessionStorage."""
        return f"{self.base_url}/?t={self.token}" if self.token else f"{self.base_url}/"

    def stop(self, timeout: float = 8.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)


def _bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, port))
    sock.listen(128)
    sock.setblocking(False)
    return sock


def wait_for_health(base_url: str, timeout: float = 20.0) -> None:
    """Block until ``/api/health`` answers (the only unauthenticated API route)."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/api/health", timeout=1.5).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(0.05)
    raise RuntimeError(f"Studio server did not become healthy within {timeout:.0f}s ({last})")


def start_server(
    settings: StudioSettings | None = None, *, port: int = 0, dev: bool = False
) -> RunningServer:
    """Start uvicorn on ``127.0.0.1:<port>`` (0 = any free port) and wait until it is healthy."""
    import uvicorn

    from .server import create_app

    app = create_app(settings)
    cfg = app.state.studio.settings
    sock = _bind(port)
    bound = sock.getsockname()[1]
    config = uvicorn.Config(
        app,
        host=HOST,
        port=bound,
        log_level="info" if dev else "warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, name="studio-server", daemon=True
    )
    thread.start()
    running = RunningServer(server, thread, bound, cfg.token, app)
    try:
        wait_for_health(running.base_url)
    except Exception:
        running.stop()
        raise
    return running


# --------------------------------------------------------------------------- js bridge


class DesktopApi:
    """Methods the web UI can call as ``window.pywebview.api.<name>()``.

    Underscore attributes are not exposed to JavaScript, so the window handle lives in ``_window``.
    """

    def __init__(self) -> None:
        self._window: Any = None

    def bind(self, window: Any) -> None:
        self._window = window

    def pick_folder(self) -> str | None:
        """Show the native folder dialog; the chosen path, or ``None`` if cancelled."""
        if self._window is None:
            return None
        import webview

        dialog = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
        if dialog is None:  # older pywebview
            dialog = webview.FOLDER_DIALOG
        chosen = self._window.create_file_dialog(dialog)
        if not chosen:
            return None
        first = chosen[0] if isinstance(chosen, (list, tuple)) else chosen
        return str(first)

    def open_path(self, path: str) -> bool:
        """Open a file or folder with the OS default app (Finder / Explorer / xdg-open)."""
        target = Path(str(path)).expanduser()
        if not target.exists():
            return False
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            elif sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined,unused-ignore]
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError:
            return False
        return True


def window_options(platform: str | None = None) -> dict[str, Any]:
    """Keyword arguments for ``webview.create_window`` (before filtering by what it supports)."""
    platform = platform or sys.platform
    opts: dict[str, Any] = {
        "width": WINDOW_SIZE[0],
        "height": WINDOW_SIZE[1],
        "min_size": MIN_SIZE,
        "background_color": BACKGROUND_COLOR,
        "text_select": True,
    }
    if platform == "darwin":
        # Hidden title bar: the web UI draws its own (class ``pywebview-drag-region``) and leaves
        # room for the traffic lights.
        opts["frameless"] = True
        opts["easy_drag"] = False
    return opts


def _supported(func: Any, options: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return options
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return options
    return {k: v for k, v in options.items() if k in params}


def force_dark_appearance() -> bool:  # pragma: no cover - macOS only
    """Best effort: make the app (title bar, scrollbars, dialogs) use macOS's dark appearance."""
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import (  # type: ignore[import-not-found,unused-ignore]
            NSAppearance,
            NSApplication,
        )

        NSApplication.sharedApplication().setAppearance_(
            NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua")
        )
    except Exception:
        return False
    return True


def open_window(url: str, *, dev: bool = False) -> None:  # pragma: no cover - needs a display
    """Create the native window and block until it is closed."""
    import webview

    force_dark_appearance()
    api = DesktopApi()
    options = _supported(webview.create_window, window_options())
    window = webview.create_window(WINDOW_TITLE, url, js_api=api, **options)
    api.bind(window)
    webview.start(debug=dev)


# --------------------------------------------------------------------------- entry point


def launch(
    *,
    browser: bool = False,
    port: int = 0,
    dev: bool = False,
    settings: StudioSettings | None = None,
) -> int:  # pragma: no cover - blocks until the window/terminal is closed
    """Start the server and the UI; returns a process exit code."""
    use_window = not browser and webview_available()
    running = start_server(settings, port=port, dev=dev)
    _say(f"NIM Lab Studio is running at {running.base_url}/")
    try:
        if use_window:
            try:
                open_window(running.url, dev=dev)
                return 0
            except Exception as exc:
                _log.warning("native window failed (%s); falling back to the browser", exc)
        elif not browser:
            _say(
                "pywebview is not installed (pip install 'nimlab[desktop]'); opening your browser."
            )
        webbrowser.open(running.url)
        _say(f"If the browser did not open, visit: {running.url}")
        _say("Press Ctrl+C to stop.")
        with contextlib.suppress(KeyboardInterrupt):
            while running.thread.is_alive():
                running.thread.join(0.5)
        return 0
    finally:
        running.stop()
