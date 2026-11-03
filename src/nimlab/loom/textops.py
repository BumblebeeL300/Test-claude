"""Pure helpers shared by the engine and by exported scripts (stdlib only, no package imports)."""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from .errors import LoomError

__all__ = ["check_http_url", "dedupe_indices", "http_allowlist", "regex_apply"]

MAX_REGEX_TEXT = 1_000_000
_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def regex_apply(
    text: str,
    pattern: str,
    mode: str = "search",
    group: int = 0,
    flags: str = "",
    replacement: str = "",
) -> dict[str, Any]:
    """Run a regex node: returns ``{"match", "matches", "found"}``."""
    if len(text) > MAX_REGEX_TEXT:
        raise LoomError(f"text is too long for regex ({len(text)} > {MAX_REGEX_TEXT} characters)")
    bits = 0
    for ch in flags or "":
        bits |= _FLAGS.get(ch, 0)
    try:
        rx = re.compile(pattern, bits)
    except re.error as exc:
        raise LoomError(f"invalid regex: {exc}") from exc
    if group > rx.groups:
        raise LoomError(f"pattern has {rx.groups} group(s); cannot read group {group}")
    matches: list[Any]
    if mode == "findall":
        matches = [m.group(group) for m in rx.finditer(text)]
    elif mode == "split":
        matches = rx.split(text)
    elif mode == "sub":
        matches = [rx.sub(replacement, text)]
    else:
        m = rx.search(text)
        matches = [m.group(group)] if m else []
    first = matches[0] if matches else ""
    return {"match": first or "", "matches": matches, "found": rx.search(text) is not None}


def dedupe_indices(
    vectors: Sequence[Sequence[float]], threshold: float, cosine: Callable[[Any, Any], float]
) -> list[int]:
    """Indices to keep: a vector is dropped when it is >= ``threshold`` similar to a kept one."""
    kept: list[int] = []
    for i, vec in enumerate(vectors):
        if all(cosine(vec, vectors[j]) < threshold for j in kept):
            kept.append(i)
    return kept


def http_allowlist(extra: Any = None) -> list[str]:
    """Allowed hosts: ``NIMLAB_LOOM_HTTP_ALLOW`` (comma separated, ``*.example.com`` ok) + extras."""
    hosts = [h.strip().lower() for h in os.environ.get("NIMLAB_LOOM_HTTP_ALLOW", "").split(",")]
    hosts += [str(h).strip().lower() for h in (extra or [])]
    return [h for h in hosts if h]


def check_http_url(url: str, allow: Sequence[str]) -> str:
    """Return the host if ``url`` may be fetched, else raise :class:`LoomError`."""
    if not allow:
        raise LoomError(
            "http_get is disabled. Allow hosts with NIMLAB_LOOM_HTTP_ALLOW=example.com,... "
            "(or the http_allow run option)"
        )
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise LoomError(f"only http(s) URLs are allowed, got {url!r}")
    if parts.username or parts.password:
        raise LoomError("URLs with credentials are not allowed")
    host = parts.hostname.lower()
    if not any(host == a or fnmatch.fnmatchcase(host, a) for a in allow):
        raise LoomError(f"host {host!r} is not on the allowlist ({', '.join(allow)})")
    return host
