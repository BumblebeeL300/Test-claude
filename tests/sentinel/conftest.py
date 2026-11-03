"""Fixtures for the Sentinel suite (everything runs offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The suite runs with --import-mode=importlib, which does not put test directories on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from sentinel_fakes import ScriptedClient

from nimlab.sentinel.simulate import SimulatedClient


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def scripted() -> type[ScriptedClient]:
    return ScriptedClient


@pytest.fixture
async def sim():
    async with SimulatedClient() as client:
        yield client


@pytest.fixture(autouse=True)
def _nemoguard_template_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fakes here script the NemoGuard instruction-template exchange; pin that model."""
    monkeypatch.setenv("NIMLAB_MODEL_SAFETY", "nvidia/llama-3.1-nemoguard-8b-content-safety")
