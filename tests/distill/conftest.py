"""Shared fixtures for the Distill tests (everything runs on the offline simulator)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# The suite runs with --import-mode=importlib, which does not put test directories on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from nimlab.distill import Distiller


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def distiller(runs_dir: Path) -> Distiller:
    return Distiller(mock=True, runs_dir=runs_dir)


@pytest.fixture
def events() -> list[tuple[str, dict[str, Any]]]:
    return []


@pytest.fixture
def make_distiller(
    runs_dir: Path, events: list[tuple[str, dict[str, Any]]]
) -> Callable[..., Distiller]:
    def factory(**kw: Any) -> Distiller:
        kw.setdefault("mock", True)
        return Distiller(runs_dir=runs_dir, on_event=lambda t, d: events.append((t, d)), **kw)

    return factory
