"""PyInstaller entry point for NIM Lab Studio.app.

Two jobs:

1. Behave like ``python`` when Forge's sandbox re-invokes the app as
   ``<app> -m pytest ...`` / ``<app> -B script.py`` (``sys.executable`` is the app in a bundle).
2. Otherwise start Studio: server thread + native window.
"""

from __future__ import annotations

import multiprocessing
import os
import sys


def _ensure_streams() -> None:
    # A windowed .app may have no stdout/stderr; uvicorn's logging and print() need real streams.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def main() -> int:
    multiprocessing.freeze_support()
    _ensure_streams()
    from nimlab.studio.frozen import maybe_run_interpreter

    code = maybe_run_interpreter(sys.argv)
    if code is not None:
        return code

    from nimlab.studio.desktop import launch

    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
