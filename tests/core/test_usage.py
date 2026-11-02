from __future__ import annotations

import threading

from rich.console import Console
from rich.table import Table

from nimlab.core import Usage, UsageTracker


def test_empty_summary() -> None:
    s = UsageTracker().summary()
    assert s["models"] == {}
    assert s["total"]["calls"] == 0 and s["total"]["avg_latency_s"] == 0.0


def test_record_and_aggregate() -> None:
    t = UsageTracker()
    t.record("a", Usage(prompt_tokens=10, completion_tokens=5), 1.0)
    t.record("a", Usage(prompt_tokens=20, completion_tokens=10), 3.0)
    t.record("b", Usage(prompt_tokens=1, completion_tokens=1), 0.5)
    s = t.summary()
    assert s["models"]["a"] == {
        "calls": 2,
        "errors": 0,
        "retries": 0,
        "tokens_in": 30,
        "tokens_out": 15,
        "total_tokens": 45,
        "latency_s": 4.0,
        "avg_latency_s": 2.0,
    }
    assert s["total"]["calls"] == 3
    assert s["total"]["total_tokens"] == 47
    assert t.total_calls == 3 and t.total_tokens == 47


def test_errors_and_retries() -> None:
    t = UsageTracker()
    t.record("m", None, 0.2, error=True)
    t.record_retry("m")
    t.record_retry("m")
    m = t.summary()["models"]["m"]
    assert (m["calls"], m["errors"], m["retries"], m["total_tokens"]) == (1, 1, 2, 0)


def test_reset() -> None:
    t = UsageTracker()
    t.record("m", Usage(prompt_tokens=1), 0.1)
    t.reset()
    assert t.summary()["models"] == {}


def test_thread_safety() -> None:
    t = UsageTracker()

    def work() -> None:
        for _ in range(1000):
            t.record("m", Usage(prompt_tokens=1, completion_tokens=2), 0.001)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    m = t.summary()["models"]["m"]
    assert m["calls"] == 8000
    assert m["tokens_in"] == 8000 and m["tokens_out"] == 16000


def test_rich_table_renders() -> None:
    t = UsageTracker()
    t.record("meta/llama-3.3-70b-instruct", Usage(prompt_tokens=1200, completion_tokens=300), 1.5)
    t.record("nvidia/nv-embedqa-e5-v5", Usage(prompt_tokens=50), 0.2)
    table = t.rich_table()
    assert isinstance(table, Table)
    console = Console(width=160, record=True, color_system=None)
    console.print(table)
    out = console.export_text()
    assert "meta/llama-3.3-70b-instruct" in out
    assert "1,200" in out
    assert "TOTAL" in out


def test_usage_total_filled() -> None:
    assert Usage(prompt_tokens=3, completion_tokens=4).total_tokens == 7
    assert Usage(prompt_tokens=3, completion_tokens=4, total_tokens=10).total_tokens == 10
