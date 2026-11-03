"""Live model discovery: ``GET /v1/models`` cached on disk, plus name heuristics per role.

NVIDIA's hosted catalog changes every few weeks and its ``/v1/models`` listing is only a superset
of what actually answers (retired models can stay listed), so discovery is an *optimisation*:
``ModelRegistry`` uses it to skip preferred defaults that are clearly gone, and the client still
reacts to a 404 at call time (see ``NIMClient``).

This module has no dependency on ``registry`` (roles are plain strings such as ``"chat"``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from .config import NIMConfig
from .logging import get_logger

__all__ = [
    "CACHE_FILENAME",
    "CACHE_TTL_S",
    "CatalogSnapshot",
    "LiveCatalog",
    "heuristic_score",
    "parse_size_b",
    "rank_for_role",
    "studio_home",
]

CACHE_FILENAME = "models_cache.json"
CACHE_TTL_S = 24 * 3600.0
RETRY_AFTER_FAILURE_S = 600.0  # do not hammer an unreachable endpoint
FETCH_TIMEOUT_S = 8.0

_log = get_logger("nimlab.catalog")


def studio_home(env: Mapping[str, str] | None = None) -> Path:
    """``$NIMLAB_STUDIO_HOME`` or ``~/NIMLab`` (the folder Studio keeps its data in)."""
    source = env if env is not None else os.environ
    value = source.get("NIMLAB_STUDIO_HOME", "").strip()
    return Path(value).expanduser() if value else Path.home() / "NIMLab"


# --------------------------------------------------------------------------- heuristics
_MOE = re.compile(r"(?<![a-z0-9])(\d+)x(\d+(?:\.\d+)?)b(?![a-z0-9])")
_SIZE = re.compile(r"(?<![a-z0-9])e?(\d+(?:\.\d+)?)([bmt])(?![a-z0-9])")


def parse_size_b(model_id: str) -> float | None:
    """Parameter count in billions parsed from the name (``...-120b-a12b`` -> 120), else ``None``.

    Active-parameter suffixes (``a12b``) are ignored; ``8x22b`` MoE names multiply out; ``1t`` is
    1000 and ``800m`` is 0.8.
    """
    low = model_id.lower().rsplit("/", 1)[-1]
    moe = _MOE.search(low)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    m = _SIZE.search(low)
    if not m:
        return None
    value = float(m.group(1))
    return value * {"b": 1.0, "m": 0.001, "t": 1000.0}[m.group(2)]


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


#: Names that are never general chat/completions models.
_NOT_CHAT = (
    r"embed|rerank|reward|guard|safety|moderat|topic-control|parse|ocr|clip|translate|"
    r"calibration|detector|retriev|riva|whisper|parakeet|canary|tts|page-elements|"
    r"table-structure|deplot|diffusion|synthetic|isaac|cosmos|fourcastnet|esm|molmim"
)
_VISION = (
    r"vision|(?<![a-z0-9])vl(?![a-z])|vlm|omni|vila|neva|fuyu|kosmos|paligemma|pixtral|llava|"
    r"gemma-[34]-|muse-glimmer|phi-3\.?5?-vision"
)
_CODER = r"coder|codestral|codegemma|codellama|starcoder|devstral|laguna|(?<![a-z])code[-_]"
_REASONING = (
    r"(?<![a-z0-9])r1(?![0-9a-z])|reason|thinking|(?<![a-z])think(?![a-z])|qwq|"
    r"nemotron-3(?:\.5)?-(?:ultra|super)|nemotron-(?:super|ultra)|gpt-oss|kimi|glm-|"
    r"deepseek-v[34]|minimax-m|step-\d|laguna|muse-"
)
_FAST = r"mini|lite|flash|lightning|bolt|nano|small|tiny|(?<![a-z0-9])e\d+b"


def _size_term(role: str, size: float | None) -> float:
    """Role-specific preference for a model size (billions); 0 when the size is unknown."""
    if size is None or size <= 0:
        return 0.0
    log = math.log(size)
    if role == "reasoning" or role == "reward":
        return log  # bigger is better
    if role == "coder" or role == "vision":
        return -abs(log - math.log(40.0)) * 0.6
    if role == "chat":
        return -abs(log - math.log(70.0)) * 0.8
    if role == "fast":
        return -log  # smaller is faster
    if role == "embed" or role == "rerank":
        return -abs(log - math.log(2.0)) * 0.3
    if role == "safety":
        return -abs(log - math.log(8.0)) * 0.3
    return 0.0


def heuristic_score(role: str, model_id: str) -> float | None:
    """How well ``model_id`` fits ``role`` by name alone; ``None`` if it does not match at all.

    Patterns: ``embed``, ``rerank``, ``reward``, ``guard``/``safety``, ``vision``/``vl``,
    ``coder``, ``r1``/``reasoning``/``thinking``, ``instruct`` (chat) and small-model markers
    (fast), with the size parsed from the name used as a role-specific tie-break.
    """
    role = role.lower()
    low = model_id.lower()
    name = low.rsplit("/", 1)[-1]
    size = parse_size_b(low)
    base: float | None
    if role == "embed":
        base = 100.0 if _has(r"embed", name) else None
    elif role == "rerank":
        base = 100.0 if _has(r"rerank", name) else None
    elif role == "reward":
        base = 100.0 if _has(r"reward", name) else None
    elif role == "safety":
        base = 100.0 if _has(r"guard|safety|moderat", name) and "topic" not in name else None
    elif _has(_NOT_CHAT, name):
        base = None
    elif role == "vision":
        base = 100.0 if _has(_VISION, name) else None
    elif role == "coder":
        base = 100.0 if _has(_CODER, name) else None
    elif role == "reasoning":
        base = 100.0 if _has(_REASONING, name) else None
    elif role == "fast":
        small = size is not None and size <= 40.0
        base = 100.0 if (_has(_FAST, name) or small) else None
        if base is not None and size is not None and size > 40.0:
            base = 90.0
    elif role == "chat":
        base = 100.0 if _has(r"instruct|(?<![a-z0-9])it$|-it-|chat", name) else 60.0
    else:
        return None
    if base is None:
        return None
    return base + _size_term(role, size)


def rank_for_role(role: str, model_ids: Iterable[str]) -> list[str]:
    """Live ids matching ``role`` by name, best first (ties broken by id, newest-looking first)."""
    scored = [(heuristic_score(role, i), i) for i in set(model_ids)]
    matched = [(s, i) for s, i in scored if s is not None]
    matched.sort(key=lambda p: (-p[0], p[1]))
    return [i for _, i in matched]


# --------------------------------------------------------------------------- the live catalog
@dataclass(frozen=True)
class CatalogSnapshot:
    """Model ids the endpoint listed, and where that list came from."""

    ids: frozenset[str]
    fetched_at: float
    source: Literal["network", "cache", "stale-cache"]
    base_url: str = ""

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.fetched_at)


def _key_fingerprint(api_key: str | None) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


class LiveCatalog:
    """``GET {base_url}/models`` with a 24 h disk cache (``~/NIMLab/models_cache.json``).

    The cache is keyed by endpoint URL and a fingerprint of the API key (never the key itself), so
    switching accounts or endpoints refetches. A failed fetch falls back to a stale cache when one
    exists; otherwise ``load`` returns ``None`` and callers use the curated defaults.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        ttl_s: float = CACHE_TTL_S,
        clock: Callable[[], float] = time.time,
        timeout: float = FETCH_TIMEOUT_S,
    ) -> None:
        self.path = Path(path) if path is not None else studio_home() / CACHE_FILENAME
        self.ttl_s = ttl_s
        self.timeout = timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._failed_at: float | None = None
        self.last_error: str | None = None

    # -- cache file
    def _read_cache(self, base_url: str, fp: str) -> tuple[frozenset[str], float] | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("base_url") != base_url or data.get("key_fp", "") != fp:
                return None
            ids = frozenset(str(i) for i in data["models"])
            return ids, float(data["fetched_at"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def _write_cache(self, base_url: str, fp: str, ids: Iterable[str], at: float) -> None:
        payload = {
            "version": 1,
            "fetched_at": at,
            "base_url": base_url,
            "key_fp": fp,
            "models": sorted(ids),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:  # a read-only home must never break the app
            _log.debug("could not write model cache %s: %s", self.path, exc)

    def clear_cache(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink()

    # -- network
    def fetch(self, config: NIMConfig) -> frozenset[str]:
        """One uncached ``GET /models`` (no retries); raises ``httpx.HTTPError``/``ValueError``."""
        headers = {"Accept": "application/json", "User-Agent": "nimlab"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        resp = httpx.get(f"{config.base_url}/models", headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data: Any = resp.json()
        return frozenset(str(m["id"]) for m in data["data"])

    def load(self, config: NIMConfig, *, force: bool = False) -> CatalogSnapshot | None:
        """Fresh cache, else the network, else a stale cache, else ``None``. Never raises."""
        with self._lock:
            fp = _key_fingerprint(config.api_key)
            now = self._clock()
            cached = self._read_cache(config.base_url, fp)
            if cached and not force and now - cached[1] < self.ttl_s:
                return CatalogSnapshot(cached[0], cached[1], "cache", config.base_url)
            if (
                not force
                and self._failed_at is not None
                and now - self._failed_at < RETRY_AFTER_FAILURE_S
                and not cached
            ):
                return None
            try:
                ids = self.fetch(config)
            except Exception as exc:  # network, auth, bad JSON: all just mean "no discovery"
                self._failed_at = now
                self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                _log.warning(
                    "could not list models at %s (%s); using %s",
                    config.base_url,
                    self.last_error,
                    "the cached list" if cached else "the built-in defaults",
                )
                if cached:
                    return CatalogSnapshot(cached[0], cached[1], "stale-cache", config.base_url)
                return None
            self._failed_at = None
            self.last_error = None
            self._write_cache(config.base_url, fp, ids, now)
            return CatalogSnapshot(ids, now, "network", config.base_url)
