"""Shared fixtures for the Arena test-suite (everything runs offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The suite runs with --import-mode=importlib, which does not put test directories on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from arena_fakes import ScriptedClient, make_suite

from nimlab.arena.suites import Suite


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def small_suite() -> Suite:
    """Three numeric items with known answers 4, 9 and 16."""
    return make_suite(
        "small",
        [
            ("two-plus-two", "What is 2+2?", 4),
            ("three-sq", "3 squared?", 9),
            ("four-sq", "4 squared?", 16),
        ],
    )


@pytest.fixture
def scripted() -> type[ScriptedClient]:
    return ScriptedClient
