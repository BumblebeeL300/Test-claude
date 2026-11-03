"""Request guards for the local server: Host check (DNS rebinding), Origin check and auth token.

Pure ASGI middleware so streaming (SSE) responses are never buffered or wrapped.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .settings import StudioSettings

__all__ = ["SecurityMiddleware", "host_allowed", "token_ok"]

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: ``/api`` paths that never need the token (liveness probe used while the server starts).
OPEN_API_PATHS = frozenset({"/api/health"})

SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cross-origin-resource-policy", b"same-origin"),
)


def _hostname(value: str) -> str:
    """Host name of a ``Host`` header value (port and IPv6 brackets handled)."""
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[: end + 1] if end != -1 else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def host_allowed(host_header: str | None, allowed: tuple[str, ...]) -> bool:
    if not host_header:
        return False
    return _hostname(host_header) in {h.lower() for h in allowed}


def token_ok(expected: str | None, header: str | None, query: str | None) -> bool:
    if expected is None:
        return True
    for candidate in (header, query):
        if candidate and hmac.compare_digest(candidate.encode(), expected.encode()):
            return True
    return False


class SecurityMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], settings: StudioSettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        s = self.settings
        if s.check_host and not host_allowed(headers.get("host"), s.allowed_hosts):
            await self._deny(scope, receive, send, 403, "host not allowed")
            return
        origin = headers.get("origin")
        if s.check_host and origin and origin != "null":
            parts = urlsplit(origin)
            if parts.hostname is None or not host_allowed(parts.netloc, s.allowed_hosts):
                await self._deny(scope, receive, send, 403, "origin not allowed")
                return
        path: str = scope.get("path", "")
        if path.startswith("/api") and path not in OPEN_API_PATHS and scope["type"] == "http":
            query = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("t", [None])[0]
            if not token_ok(s.token, headers.get("x-studio-token"), query):
                await self._deny(
                    scope, receive, send, 401, "missing or invalid Studio token (X-Studio-Token)"
                )
                return

        async def send_with_headers(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in SECURITY_HEADERS if k not in existing]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    async def _deny(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
