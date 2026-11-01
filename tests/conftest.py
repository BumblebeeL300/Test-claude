"""Shared pytest fixtures for the whole monorepo (app test suites get these for free).

Fixtures
--------
mock_config   NIMConfig(mock=True)
mock_client   async NIMClient on the offline mock backend
real_config   NIMConfig(api_key="nvapi-test", mock=False, max_retries=3) for respx tests
sleeps        list of backoff delays the client "slept" (sleeping is faked, tests run instantly)

An autouse fixture scrubs NVIDIA_*/NIM_*/NIMLAB_* env vars, disables ./.env loading, and runs each
test in an empty temp cwd so no real key, nimlab.toml or .env can leak into tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from rich.console import Console

from nimlab.core import NIMClient, NIMConfig
from nimlab.core import config as config_module
from nimlab.core.logging import set_console
from nimlab.core.registry import reset_default_registry

_ENV_PREFIXES = ("NVIDIA_", "NIM_", "NIMLAB_")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NIMLAB_NO_DOTENV", "1")
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    config_module._reset_warning()
    reset_default_registry()
    set_console(Console(width=200, color_system=None, force_terminal=False, highlight=False))


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("nimlab.core.client._sleep", fake_sleep)
    return recorded


@pytest.fixture
def mock_config() -> NIMConfig:
    return NIMConfig(mock=True)


@pytest.fixture
async def mock_client(mock_config: NIMConfig) -> AsyncIterator[NIMClient]:
    async with NIMClient(mock_config) as client:
        yield client


@pytest.fixture
def real_config() -> NIMConfig:
    return NIMConfig(api_key="nvapi-test", mock=False, max_retries=3)
