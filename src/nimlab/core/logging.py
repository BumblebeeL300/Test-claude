"""Shared rich console and logging helpers."""

from __future__ import annotations

import logging as _logging
import os
import sys

from rich.console import Console

_console: Console | None = None
_HANDLER_MARK = "_nimlab_handler"


def get_console() -> Console:
    """Process-wide rich ``Console`` (stdout).

    Output is wide (140 cols) when stdout is not a terminal and ``COLUMNS`` is unset, so piped or
    captured tables are not truncated.
    """
    global _console
    if _console is None:
        width = None
        if not sys.stdout.isatty() and "COLUMNS" not in os.environ:
            width = 140
        _console = Console(width=width, highlight=False)
    return _console


def set_console(console: Console | None) -> None:
    """Replace (or reset with ``None``) the shared console. Mostly useful in tests."""
    global _console
    _console = console


def get_logger(name: str = "nimlab") -> _logging.Logger:
    return _logging.getLogger(name)


def configure_logging(level: int | str | None = None) -> None:
    """Install a rich stderr handler on the ``nimlab`` logger (idempotent).

    ``level`` defaults to ``$NIMLAB_LOG_LEVEL`` or WARNING.
    """
    from rich.logging import RichHandler

    logger = _logging.getLogger("nimlab")
    resolved = level if level is not None else os.environ.get("NIMLAB_LOG_LEVEL", "WARNING")
    logger.setLevel(resolved.upper() if isinstance(resolved, str) else resolved)
    if any(getattr(h, _HANDLER_MARK, False) for h in logger.handlers):
        return
    handler = RichHandler(
        console=Console(stderr=True, highlight=False),
        show_path=False,
        show_time=False,
        markup=False,
    )
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)


def reset_logging() -> None:
    """Undo :func:`configure_logging`: drop our stderr handler and reset the ``nimlab`` level.

    ``configure_logging`` mutates process-global state (a handler on the ``nimlab`` logger), so
    anything that runs the CLI in-process more than once (tests, notebooks) can call this to get
    a clean slate.
    """
    logger = _logging.getLogger("nimlab")
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_MARK, False)]:
        logger.removeHandler(handler)
    logger.setLevel(_logging.NOTSET)
