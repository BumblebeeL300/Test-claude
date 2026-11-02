# Task: a retry decorator with exponential backoff

Create `retry.py` with a `retry` decorator that works on both sync and async functions, plus tests in
`test_retry.py` (async tests may use `pytest-asyncio` in auto mode, or `asyncio.run`).

## Requirements

- `@retry(attempts=3, base_delay=0.1, factor=2.0, max_delay=5.0, jitter=0.0, exceptions=(Exception,))`
- Calls the function up to `attempts` times, sleeping `min(max_delay, base_delay * factor**n)` between
  tries (plus a random `0..jitter` fraction when `jitter > 0`). Re-raises the last exception.
- Only exceptions listed in `exceptions` are retried; anything else propagates immediately.
- Accept an optional `on_retry(attempt, exception, delay)` callback.
- Make the sleep function injectable (`sleep=` parameter) so tests run instantly and can assert the
  exact delays. Preserve `__name__`/`__doc__` with `functools.wraps`.
- `attempts < 1` raises `ValueError` when the decorator is created.
- Standard library only.

## Edge cases the tests must cover

- succeeds on first try (no sleep), succeeds on the last try, exhausts attempts, non-retryable
  exception, `max_delay` cap, async function support, decorator used without parentheses is **not**
  required.
