"""Bookkeeping helpers: stall detection, per-phase timing and LLM usage deltas."""

from __future__ import annotations

import time
from typing import Any

from .models import PhaseTiming

__all__ = ["PhaseTimer", "StallDetector", "UsageProbe"]


class StallDetector:
    """Detects "no progress": the same failure signature keeps coming back.

    A signature seen ``threshold`` or more times among the last ``window`` verification rounds
    counts as a stall. Looking at a window (not just consecutive rounds) also catches oscillation
    (A, B, A).
    """

    def __init__(self, threshold: int = 2, window: int = 4) -> None:
        if threshold < 2:
            raise ValueError("threshold must be >= 2")
        self.threshold = threshold
        self.window = max(window, threshold)
        self.signatures: list[str] = []

    def record(self, signature: str) -> None:
        self.signatures.append(signature)

    @property
    def repeats(self) -> int:
        """How often the latest signature occurs in the window (0 if nothing recorded)."""
        if not self.signatures:
            return 0
        recent = self.signatures[-self.window :]
        return recent.count(recent[-1])

    @property
    def stalled(self) -> bool:
        return self.repeats >= self.threshold


def _totals(client: Any) -> dict[str, Any]:
    usage = getattr(client, "usage", None)
    if usage is None or not hasattr(usage, "summary"):
        return {"models": {}, "total": {}}
    try:
        return usage.summary()
    except Exception:  # a broken tracker must not break a run
        return {"models": {}, "total": {}}


class UsageProbe:
    """Reads ``client.usage`` (if the client has one) and computes deltas since creation."""

    _FIELDS = ("calls", "errors", "tokens_in", "tokens_out", "total_tokens")

    def __init__(self, client: Any) -> None:
        self._client = client
        self._base = _totals(client)

    def totals(self) -> dict[str, int]:
        total = _totals(self._client).get("total") or {}
        return {k: int(total.get(k, 0)) for k in ("calls", "tokens_in", "tokens_out")}

    def delta(self) -> dict[str, Any]:
        """Usage accrued since this probe was created, in ``UsageTracker.summary()`` shape."""
        now = _totals(self._client)
        models: dict[str, dict[str, Any]] = {}
        for name, stats in (now.get("models") or {}).items():
            before = (self._base.get("models") or {}).get(name, {})
            row = {k: int(stats.get(k, 0)) - int(before.get(k, 0)) for k in self._FIELDS}
            row["latency_s"] = round(
                float(stats.get("latency_s", 0)) - float(before.get("latency_s", 0)), 6
            )
            if row["calls"] or row["tokens_in"] or row["tokens_out"]:
                models[name] = row
        total = {k: sum(m[k] for m in models.values()) for k in self._FIELDS}
        total["latency_s"] = round(sum(m["latency_s"] for m in models.values()), 6)
        return {"models": models, "total": total}


class PhaseTimer:
    """Accumulates wall time and LLM usage per phase name (repeated phases are summed)."""

    def __init__(self, probe: UsageProbe) -> None:
        self._probe = probe
        self._acc: dict[str, PhaseTiming] = {}
        self._current: tuple[str, float, dict[str, int]] | None = None

    def start(self, name: str) -> None:
        self.stop()
        self._current = (name, time.monotonic(), self._probe.totals())

    def stop(self) -> None:
        if self._current is None:
            return
        name, t0, base = self._current
        self._accumulate(name, time.monotonic() - t0, base, count_run=True)
        self._current = None

    def _accumulate(
        self, name: str, seconds: float, base: dict[str, int], *, count_run: bool
    ) -> None:
        now = self._probe.totals()
        entry = self._acc.setdefault(name, PhaseTiming(phase=name))
        entry.seconds += seconds
        entry.runs += 1 if count_run else 0
        entry.llm_calls += now["calls"] - base["calls"]
        entry.tokens_in += now["tokens_in"] - base["tokens_in"]
        entry.tokens_out += now["tokens_out"] - base["tokens_out"]

    def snapshot(self) -> list[PhaseTiming]:
        """Timings so far, including the phase that is still running."""
        rows = {k: v.model_copy() for k, v in self._acc.items()}
        if self._current is not None:
            name, t0, base = self._current
            now = self._probe.totals()
            entry = rows.setdefault(name, PhaseTiming(phase=name))
            entry.seconds += time.monotonic() - t0
            entry.runs += 1
            entry.llm_calls += now["calls"] - base["calls"]
            entry.tokens_in += now["tokens_in"] - base["tokens_in"]
            entry.tokens_out += now["tokens_out"] - base["tokens_out"]
        for row in rows.values():
            row.seconds = round(row.seconds, 4)
        return list(rows.values())
