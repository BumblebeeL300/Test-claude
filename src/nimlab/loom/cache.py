"""On-disk result cache for node runs, keyed by node config + inputs.

Layout: ``<dir>/<sha256>.json`` holding ``{"outputs", "tokens", "model", "created"}``.
Default directory: ``$NIMLAB_LOOM_CACHE`` or ``.nimlab/loom/cache`` under the working directory.
Writes are atomic (temp file + rename); unreadable entries are treated as misses.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

__all__ = ["CacheEntry", "NodeCache", "cache_key", "default_cache_dir"]

CACHE_VERSION = 1


def default_cache_dir() -> Path:
    env = os.environ.get("NIMLAB_LOOM_CACHE", "").strip()
    return Path(env).expanduser() if env else Path(".nimlab") / "loom" / "cache"


def cache_key(node_type: str, params: dict[str, Any], inputs: dict[str, Any], backend: str) -> str:
    """Stable digest of everything that can change a node's result."""
    blob = json.dumps(
        {"v": CACHE_VERSION, "type": node_type, "params": params, "inputs": inputs, "backend": backend},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CacheEntry:
    def __init__(self, outputs: dict[str, Any], tokens: int = 0, model: str | None = None) -> None:
        self.outputs = outputs
        self.tokens = tokens
        self.model = model


class NodeCache:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.dir = Path(directory) if directory else default_cache_dir()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            entry = CacheEntry(dict(data["outputs"]), int(data.get("tokens", 0)), data.get("model"))
        except (OSError, ValueError, KeyError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, outputs: dict[str, Any], tokens: int = 0, model: str | None = None) -> None:
        payload = {
            "outputs": outputs,
            "tokens": tokens,
            "model": model,
            "created": time.time(),
        }
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
            self.dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=self.dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                os.replace(tmp, self._path(key))
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
        except (OSError, TypeError, ValueError):
            pass  # a cache must never fail a run

    def clear(self) -> int:
        removed = 0
        if self.dir.is_dir():
            for f in self.dir.glob("*.json"):
                with contextlib.suppress(OSError):
                    f.unlink()
                    removed += 1
        return removed
