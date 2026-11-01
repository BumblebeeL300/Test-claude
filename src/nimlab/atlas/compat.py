"""Small helpers that keep the rest of Atlas independent of client implementation details."""

from __future__ import annotations

from typing import Any


def is_mock(client: Any) -> bool:
    """True when ``client`` is the offline mock backend (``client.mock`` / ``client.config.mock``)."""
    flag = getattr(client, "mock", None)
    if isinstance(flag, bool):
        return flag
    return bool(getattr(getattr(client, "config", None), "mock", False))


def usage_totals(client: Any) -> tuple[int, int]:
    """``(calls, total_tokens)`` recorded so far by ``client.usage`` (``(0, 0)`` if unavailable)."""
    try:
        total = client.usage.summary()["total"]
        return int(total["calls"]), int(total["total_tokens"])
    except Exception:
        return 0, 0
