# Task: a generic LRU cache

Create `lru_cache.py` with a class `LRUCache` that evicts the least recently used entry, plus a
thorough pytest suite in `test_lru_cache.py`.

## Requirements

- `LRUCache(capacity: int)`; `capacity` must be a positive integer, otherwise raise `ValueError`.
- `get(key, default=None)` returns the value (or `default`) and marks the key as most recently used.
- `put(key, value)` inserts or updates a key and marks it as most recently used. When the cache is
  over capacity the least recently used entry is evicted.
- `__len__`, `__contains__` (membership test must **not** change recency), `clear()`.
- `keys()` returns the keys ordered from least to most recently used.
- `pop(key, default=...)` removes and returns a key; a missing key without a default raises `KeyError`.
- O(1) `get` and `put`. Use only the standard library.

## Edge cases the tests must cover

- capacity 1, updating an existing key does not evict anything, reading refreshes recency,
  `None` stored as a value is distinguishable from a missing key, unhashable keys raise `TypeError`.
