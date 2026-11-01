"""Thread- and async-safe usage aggregation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from rich.table import Table

from .results import Usage


@dataclass
class ModelStats:
    calls: int = 0
    errors: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "retries": self.retries,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 6),
            "avg_latency_s": round(self.latency_s / self.calls, 6) if self.calls else 0.0,
        }


class UsageTracker:
    """Aggregates calls / tokens / latency / errors per model.

    ``calls`` counts every logical API call (chat, stream, embed, rerank, list), including
    failed ones; ``errors`` counts the failed subset; ``retries`` counts extra HTTP attempts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, ModelStats] = {}

    def record(
        self,
        model: str,
        usage: Usage | None = None,
        latency_s: float = 0.0,
        *,
        error: bool = False,
    ) -> None:
        with self._lock:
            s = self._stats.setdefault(model, ModelStats())
            s.calls += 1
            s.latency_s += latency_s
            if error:
                s.errors += 1
            if usage is not None:
                s.tokens_in += usage.prompt_tokens
                s.tokens_out += usage.completion_tokens

    def record_retry(self, model: str) -> None:
        with self._lock:
            self._stats.setdefault(model, ModelStats()).retries += 1

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()

    def _snapshot(self) -> dict[str, ModelStats]:
        with self._lock:
            return {k: ModelStats(**vars(v)) for k, v in self._stats.items()}

    def _total(self, snap: dict[str, ModelStats]) -> ModelStats:
        total = ModelStats()
        for s in snap.values():
            total.calls += s.calls
            total.errors += s.errors
            total.retries += s.retries
            total.tokens_in += s.tokens_in
            total.tokens_out += s.tokens_out
            total.latency_s += s.latency_s
        return total

    @property
    def total_calls(self) -> int:
        return self._total(self._snapshot()).calls

    @property
    def total_tokens(self) -> int:
        return self._total(self._snapshot()).total_tokens

    def summary(self) -> dict[str, Any]:
        """``{"models": {model: stats}, "total": stats}``; stats keys: calls, errors, retries,
        tokens_in, tokens_out, total_tokens, latency_s, avg_latency_s."""
        snap = self._snapshot()
        return {
            "models": {m: s.as_dict() for m, s in sorted(snap.items())},
            "total": self._total(snap).as_dict(),
        }

    def rich_table(self, title: str = "NIM usage") -> Table:
        snap = self._snapshot()
        table = Table(title=title, title_justify="left")
        table.add_column("Model", overflow="fold")
        for col in (
            "Calls",
            "Errors",
            "Retries",
            "Tokens in",
            "Tokens out",
            "Latency (s)",
            "Avg (s)",
        ):
            table.add_column(col, justify="right")

        def row(name: str, s: ModelStats) -> list[str]:
            avg = s.latency_s / s.calls if s.calls else 0.0
            return [
                name,
                str(s.calls),
                str(s.errors),
                str(s.retries),
                f"{s.tokens_in:,}",
                f"{s.tokens_out:,}",
                f"{s.latency_s:.2f}",
                f"{avg:.2f}",
            ]

        for model, s in sorted(snap.items()):
            table.add_row(*row(model, s))
        if len(snap) != 1:
            table.add_section()
            table.add_row(*row("TOTAL", self._total(snap)), style="bold")
        return table
