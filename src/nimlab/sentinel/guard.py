"""NemoGuard content-safety calls, reusing the prompt template and parser from ``atlas.guard``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nimlab.core import NIMAuthError, NIMError

from .llm import is_mock, is_simulated

__all__ = ["GuardResult", "check_safety", "parse_guard_output"]


class GuardResult(BaseModel):
    #: ``True`` safe, ``False`` unsafe, ``None`` unknown (skipped, failed or unparseable).
    safe: bool | None = None
    categories: list[str] = Field(default_factory=list)
    raw: str = ""
    skipped: bool = False
    detail: str = ""


def parse_guard_output(raw: str, stage: str = "answer") -> tuple[bool | None, list[str]]:
    """Parse a NemoGuard reply into ``(safe, categories)``; ``safe`` is ``None`` if unparseable."""
    from nimlab.atlas.guard import parse_guard_output as _parse

    return _parse(raw, "answer" if stage == "answer" else "query")


async def check_safety(
    client: Any,
    model: str,
    user_text: str,
    response_text: str | None = None,
) -> GuardResult:
    """Ask the safety model to rate the user message (and the response, when given).

    Plain mock clients cannot play a classifier, so the call is skipped for them; the Sentinel
    simulator (``client.simulated``) answers in NemoGuard's format and is used normally. Transport
    failures and unparseable replies come back as ``safe=None`` rather than raising, except for
    authentication errors.
    """
    if is_mock(client) and not is_simulated(client):
        return GuardResult(skipped=True, detail="mock mode: content-safety model not called")
    from nimlab.atlas.guard import build_guard_messages

    try:
        result = await client.chat(
            model,
            build_guard_messages(user_text, response_text),
            temperature=0.0,
            max_tokens=96,
        )
    except NIMAuthError:
        raise
    except NIMError as exc:
        return GuardResult(skipped=True, detail=f"guard call failed: {exc}")
    stage = "answer" if response_text is not None else "query"
    safe, categories = parse_guard_output(result.content, stage)
    if safe is None:
        return GuardResult(skipped=True, raw=result.content, detail="unparseable guard reply")
    return GuardResult(safe=safe, categories=categories, raw=result.content)
