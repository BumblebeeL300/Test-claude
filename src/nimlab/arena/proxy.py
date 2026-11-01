"""OpenAI-compatible routing proxy.

``POST /v1/chat/completions`` accepts any OpenAI client. ``model: "nimlab/auto"`` (optionally
``nimlab/auto:speed|quality|economy|balanced``) routes the request with :class:`SmartRouter`;
any other model id is passed straight through to NIM. The model that actually served the request
is reported in the ``x-nimlab-routed-model`` response header (and ``x-nimlab-category`` when routed).
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from nimlab.core import Role, default_registry
from nimlab.core.errors import (
    NIMAuthError,
    NIMError,
    NIMRateLimitError,
    NIMTimeoutError,
)
from nimlab.core.registry import ModelRegistry
from nimlab.core.results import ChatResult

from .llm import make_client
from .router import PROFILES, SmartRouter

__all__ = ["AUTO_MODEL", "ChatCompletionRequest", "create_app"]

AUTO_MODEL = "nimlab/auto"
ROUTED_HEADER = "x-nimlab-routed-model"
CHAT_ROLES = (Role.CHAT, Role.REASONING, Role.CODER, Role.VISION, Role.FAST)
_KNOWN = {
    "model", "messages", "temperature", "max_tokens", "max_completion_tokens", "stream",
    "tools", "tool_choice", "response_format", "stream_options",
}  # fmt: skip


class ChatCompletionRequest(BaseModel):
    """The subset of the OpenAI chat request Arena interprets; other fields pass through."""

    model_config = ConfigDict(extra="allow")

    model: str = AUTO_MODEL
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None

    def chat_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        limit = self.max_tokens or self.max_completion_tokens
        if limit is not None:
            kw["max_tokens"] = limit
        if self.tools:
            kw["tools"] = self.tools
        if self.tool_choice is not None:
            kw["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            kw["response_format"] = self.response_format
        extra = {k: v for k, v in (self.model_extra or {}).items() if k not in _KNOWN}
        if extra:
            kw["extra"] = extra
        return kw


def _error(status: int, message: str, kind: str, headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"message": message, "type": kind, "code": status}}
    return JSONResponse(body, status_code=status, headers=headers)


def _chunk(cid: str, created: int, model: str, delta: dict[str, Any], finish: str | None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _completion_body(cid: str, created: int, model: str, result: ChatResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content or None}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in result.tool_calls
        ]
    elif message["content"] is None:
        message["content"] = ""
    usage = result.usage
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": result.finish_reason or "stop"}
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    }


def _split_auto(model: str) -> tuple[bool, str | None]:
    """``nimlab/auto`` -> (True, None); ``nimlab/auto:speed`` -> (True, 'speed')."""
    if model == AUTO_MODEL:
        return True, None
    if model.startswith(AUTO_MODEL + ":"):
        return True, model.split(":", 1)[1]
    return False, None


def create_app(
    client: Any = None,
    router: SmartRouter | None = None,
    *,
    runs_dir: str | Path | None = None,
    registry: ModelRegistry | None = None,
    mock: bool | None = None,
) -> FastAPI:
    """Build the proxy app.

    ``client`` defaults to the real NIM client (or the offline simulator in mock mode) and is
    closed on shutdown only if the app created it. ``router`` defaults to a :class:`SmartRouter`
    over the latest runs in ``runs_dir``.
    """
    reg = registry or default_registry()
    owns_client = client is None
    llm = client if client is not None else make_client(mock=mock)
    smart = router or SmartRouter(client=llm, registry=reg, runs_dir=runs_dir)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_client and hasattr(llm, "aclose"):
            await llm.aclose()

    app = FastAPI(
        title="NIM Arena routing proxy",
        version="0.1.0",
        description="OpenAI-compatible endpoint that routes `nimlab/auto` to the best model.",
        lifespan=lifespan,
    )
    app.state.client = llm
    app.state.router = smart

    @app.exception_handler(NIMError)
    async def _nim_error(_: Request, exc: NIMError) -> JSONResponse:
        if isinstance(exc, NIMAuthError):
            return _error(401, str(exc), "authentication_error")
        if isinstance(exc, NIMRateLimitError):
            headers = {"retry-after": str(int(exc.retry_after))} if exc.retry_after else None
            return _error(429, str(exc), "rate_limit_error", headers)
        if isinstance(exc, NIMTimeoutError):
            return _error(504, str(exc), "timeout_error")
        return _error(502, str(exc), "upstream_error")

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        return _error(400, f"invalid request: {detail}", "invalid_request_error")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        lb = smart.leaderboard
        return {
            "status": "ok",
            "mock": bool(getattr(llm, "mock", False)),
            "leaderboard_run": lb.run_id if lb else None,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = [AUTO_MODEL]
        lb = smart.leaderboard
        pool = [m.id for role in CHAT_ROLES for m in reg.by_role(role)]
        pool += lb.models() if lb else []
        ids += [m for m in dict.fromkeys(pool)]
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nimlab" if mid == AUTO_MODEL else (mid.split("/")[0] if "/" in mid else "nim"),
                }
                for mid in ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        is_auto, suffix = _split_auto(req.model)
        profile = request.headers.get("x-nimlab-profile") or suffix
        headers: dict[str, str] = {}
        model = req.model
        decision = None
        if is_auto:
            if profile is not None and profile not in PROFILES:
                return _error(
                    400,
                    f"unknown routing profile {profile!r}; choose from {', '.join(PROFILES)}",
                    "invalid_request_error",
                )
            try:
                decision = await smart.route(
                    req.messages, category=request.headers.get("x-nimlab-category"), profile=profile
                )
            except ValueError as exc:  # unknown category header
                return _error(400, str(exc), "invalid_request_error")
            model = decision.model
            headers["x-nimlab-category"] = decision.category
            headers["x-nimlab-routing-source"] = decision.source
        headers[ROUTED_HEADER] = model

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        kwargs = req.chat_kwargs()

        if not req.stream:
            result = await llm.chat(model, req.messages, **kwargs)
            body = _completion_body(cid, created, model, result)
            if decision is not None:
                body["nimlab"] = {
                    "routed_model": model,
                    "category": decision.category,
                    "classifier": decision.classifier,
                    "reasons": decision.reasons,
                }
            return JSONResponse(body, headers=headers)

        pieces = llm.stream_chat(model, req.messages, **kwargs)
        try:  # pull the first chunk now so upstream errors still become proper HTTP errors
            first: str | None = await pieces.__anext__()
        except StopAsyncIteration:
            first = None

        async def events() -> AsyncIterator[str]:
            yield _chunk(cid, created, model, {"role": "assistant", "content": ""}, None)
            try:
                if first:
                    yield _chunk(cid, created, model, {"content": first}, None)
                async for piece in pieces:
                    if piece:
                        yield _chunk(cid, created, model, {"content": piece}, None)
            except NIMError as exc:  # headers are gone: report in-band like OpenAI does
                err = {"error": {"message": str(exc), "type": "upstream_error"}}
                yield f"data: {json.dumps(err)}\n\n"
            else:
                yield _chunk(cid, created, model, {}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**headers, "cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.get("/router/explain")
    async def explain(
        prompt: str = Query(..., min_length=1, description="Prompt to route"),
        profile: str | None = Query(None),
        category: str | None = Query(None),
    ) -> Any:
        try:
            decision = await smart.route(prompt, category=category, profile=profile)
        except ValueError as exc:
            return _error(400, str(exc), "invalid_request_error")
        body = decision.model_dump(exclude={"classification"})
        body["classification"] = (
            decision.classification.model_dump() if decision.classification else None
        )
        body["explanation"] = decision.explain()
        return body

    @app.get("/router/leaderboard")
    async def leaderboard() -> Any:
        lb = smart.leaderboard
        if lb is None:
            return _error(404, "no benchmark runs found; run `nimlab arena run` first", "not_found")
        return lb.model_dump()

    return app
