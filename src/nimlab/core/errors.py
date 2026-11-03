"""Exception hierarchy for the NIM client."""

from __future__ import annotations

from typing import Any


class NIMError(Exception):
    """Base class for every error raised by :mod:`nimlab.core`."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: Any = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.model = model

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"(HTTP {self.status_code})")
        if self.model:
            parts.append(f"[model={self.model}]")
        return " ".join(parts)


class NIMAuthError(NIMError):
    """401/403: missing, invalid or unauthorised API key. Never retried."""


class NIMRateLimitError(NIMError):
    """429 that persisted after all retries (or asked for an unreasonably long wait)."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class NIMResponseError(NIMError):
    """Non-retryable 4xx, 5xx after retries, or a malformed/unexpected response body."""


class NIMModelUnavailableError(NIMResponseError):
    """404 / "model not found" / "not available for this account": the model id is not served.

    The client marks the model unavailable for the session and, when the caller used a role pick,
    retries once with the next candidate for that role.
    """


class NIMTimeoutError(NIMError):
    """Request timed out or the connection failed on every attempt."""


class NIMJSONError(NIMResponseError):
    """``chat_json`` could not obtain schema-valid JSON from the model."""

    def __init__(self, message: str, *, raw: str = "", **kw: Any) -> None:
        super().__init__(message, **kw)
        self.raw = raw
