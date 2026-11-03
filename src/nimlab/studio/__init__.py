"""NIM Lab Studio: the desktop app (FastAPI backend + web UI + native window shell).

Importing this package is cheap on purpose: ``nimlab.core.config`` imports
:mod:`nimlab.studio.credentials` lazily, and neither FastAPI nor pywebview is touched until
:func:`nimlab.studio.server.create_app` / :func:`nimlab.studio.desktop.launch` are used.
"""

from __future__ import annotations

__all__ = ["credentials"]
