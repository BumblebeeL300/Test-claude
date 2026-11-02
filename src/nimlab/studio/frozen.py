"""Support for running inside a PyInstaller bundle (the macOS ``NIM Lab Studio.app``).

Forge's sandbox runs the user's tests as ``[sys.executable, "-m", "pytest", ...]`` and scripts as
``[sys.executable, "-B", "script.py"]``. In a frozen app ``sys.executable`` is the app itself, so
without help those child processes would start a second Studio. :func:`maybe_run_interpreter`
makes the bundled executable behave like ``python`` for exactly those two invocation shapes.
"""

from __future__ import annotations

import runpy
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

__all__ = ["is_frozen", "maybe_run_interpreter"]

_FLAGS = frozenset({"-B", "-u", "-E", "-s", "-S", "-I", "-O", "-OO", "-q"})


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def maybe_run_interpreter(
    argv: Sequence[str] | None = None, *, frozen: bool | None = None
) -> int | None:
    """Run ``-m module`` / ``script.py`` if ``argv`` asks for it; ``None`` means "start the app".

    Only active in a frozen build (or when ``frozen=True``), so a normal ``python`` process is never
    affected. Returns the exit code the caller should use.
    """
    if not (is_frozen() if frozen is None else frozen):
        return None
    args = list(sys.argv if argv is None else argv)[1:]
    while args and args[0] in _FLAGS:
        args.pop(0)
    if not args:
        return None
    if args[0] == "-m" and len(args) >= 2:
        module, rest = args[1], args[2:]
        sys.argv = [module, *rest]
        runner = lambda: runpy.run_module(module, run_name="__main__", alter_sys=True)  # noqa: E731
    elif args[0].endswith(".py") and Path(args[0]).is_file():
        script, rest = args[0], args[1:]
        sys.argv = [script, *rest]
        runner = lambda: runpy.run_path(script, run_name="__main__")  # noqa: E731
    else:
        return None
    try:
        runner()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    except Exception:  # like `python -m missing`: report on stderr, exit 1
        traceback.print_exc()
        return 1
    return 0
