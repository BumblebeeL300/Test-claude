"""Fixtures for the opt-in live tests (real NVIDIA NIM calls)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def sleeps() -> list[float]:
    """Override the root fixture that fakes backoff sleeping: live calls must really back off."""
    return []
