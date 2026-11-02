"""Shared fixtures for the Forge test-suite (everything runs offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The suite runs with --import-mode=importlib, which does not put test directories on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from forge_fakes import (
    FIZZBUZZ_BUGGY,
    FIZZBUZZ_PATCH,
    FIZZBUZZ_TESTS,
    FakeClient,
)

from nimlab.forge import Sandbox


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """An empty workspace directory."""
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def sandbox(ws: Path) -> Sandbox:
    return Sandbox(ws)


@pytest.fixture
def fizzbuzz_sources() -> dict[str, str]:
    return {"buggy": FIZZBUZZ_BUGGY, "tests": FIZZBUZZ_TESTS, "patch": FIZZBUZZ_PATCH}


@pytest.fixture
def fake_client_cls() -> type[FakeClient]:
    return FakeClient
