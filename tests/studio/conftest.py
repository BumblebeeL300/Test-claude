"""Fixtures for the Studio test-suite: an in-process app with no auth, no pauses, tmp data dirs."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The suite runs with --import-mode=importlib, which does not put test directories on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from studio_helpers import HOST, make_settings

from nimlab.studio.server import create_app
from nimlab.studio.settings import StudioSettings


@pytest.fixture
def settings(tmp_path: Path) -> StudioSettings:
    return make_settings(tmp_path)


@pytest.fixture
def client(settings: StudioSettings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app, base_url=HOST) as c:
        yield c
