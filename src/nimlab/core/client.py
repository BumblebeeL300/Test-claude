"""Async NVIDIA NIM client with retries, concurrency limiting, usage tracking and a mock backend."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from . import mock as _mock
from .config import NIMConfig
from .errors import (
    NIMAuthError,
    NIMError,
    NIMJSONError,
    NIMRateLimitError,
    NIMResponseError,
    NIMTimeoutError,
)
from .logging import get_logger
from .results import ChatResult, RerankResult, ToolCall, Usage
from .text import ThinkStreamFilter, extract_json, strip_think
from .usage import UsageTracker

__all__ = ["NIMClient"]

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

EMBED_BATCH_SIZE = 32
RERANK_BATCH_SIZE = 256
BACKOFF_BASE = 0.5
BACKOFF_MAX = 30.0
RETRY_AFTER_MAX = 120.0  # a longer Retry-After is surfaced instead of waited out
USER_AGENT = "nimlab/0.1"

_log = get_logger("nimlab.client")


async def _sleep(seconds: float) -> None:
    """Indirection so tests can observe/skip backoff delays."""
    await asyncio.sleep(seconds)


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with equal jitter; never shorter than ``Retry-After``."""
    ceiling = min(BACKOFF_BASE * (2**attempt), BACKOFF_MAX)
    delay = ceiling * (0.5 + random.random() * 0.5)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return delay


def parse_retry_after(value: str | None) -> float | None:
    """``Retry-After`` as seconds (delta-seconds or HTTP-date); ``None`` if absent/invalid."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300].strip() or resp.reason_phrase
    if isinstance(data, dict):
        for key in ("detail", "message", "error", "title"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, dict) and val.get("message"):
                return str(val["message"])
            if isinstance(val, list) and val:
                return json.dumps(val)[:300]
    return json.dumps(data)[:300]


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = extract_json(raw)
        except ValueError:
            return {"_raw": raw}
    return value if isinstance(value, dict) else {"_value": value}


def _sse_delta(line: str) -> tuple[bool, str]:
    """Parse one SSE line -> (done, content_delta)."""
    line = line.strip()
    if not line.startswith("data:"):
        return False, ""
    payload = line[5:].strip()
    if payload == "[DONE]":
        return True, ""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return False, ""
    if not isinstance(obj, dict):
        return False, ""
    if obj.get("error"):
        raise NIMResponseError(f"error in stream: {_truncate(obj['error'])}")
    choices = obj.get("choices") or []
    if not choices:
        return False, ""
    delta = choices[0].get("delta") or {}
    return False, _content_text(delta.get("content"))


def _truncate(obj: Any, n: int = 500) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return text if len(text) <= n else text[:n] + "..."


def _validation_hint(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        lines = [
            f"- {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}"
            for err in exc.errors()[:10]
        ]
        return "The JSON did not match the schema:\n" + "\n".join(lines)
    return f"The output was not parseable JSON: {exc}"


class NIMClient:
    """Async client for NVIDIA NIM (OpenAI-compatible chat/embeddings, NVIDIA reranking).

    Use as ``async with NIMClient() as c:``. With ``config.mock`` true no network is used.
    """

    def __init__(
        self, config: NIMConfig | None = None, *, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self.config: NIMConfig = config or NIMConfig.from_env()
        self.usage: UsageTracker = UsageTracker()
        self._http = http_client
        self._owns_http = http_client is None
        self._sem: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> NIMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        if self._owns_http:
            self._http = None

    @property
    def mock(self) -> bool:
        return self.config.mock

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop semaphore (the client may be reused across ``asyncio.run`` calls)."""
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._sem = asyncio.Semaphore(self.config.max_concurrency)
            if self._loop is not None and self._owns_http:
                self._http = None  # bound to a dead loop
            self._loop = loop
        return self._sem

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout), headers=self._headers()
            )
        return self._http

    # ------------------------------------------------------------------ transport
    @staticmethod
    def _classify(resp: httpx.Response, model: str) -> tuple[NIMError, bool, float | None]:
        """Map a non-2xx response to (error, retryable, retry_after)."""
        status = resp.status_code
        message = _error_message(resp)
        body = resp.text[:500]
        if status in (401, 403):
            err: NIMError = NIMAuthError(
                f"authentication failed: {message}", status_code=status, body=body, model=model
            )
            return err, False, None
        if status == 429:
            retry_after = parse_retry_after(resp.headers.get("retry-after"))
            err = NIMRateLimitError(
                f"rate limited: {message}",
                retry_after=retry_after,
                status_code=status,
                body=body,
                model=model,
            )
            return err, True, retry_after
        if status == 408 or status >= 500:
            err = NIMResponseError(
                f"server error: {message}", status_code=status, body=body, model=model
            )
            return err, True, None
        err = NIMResponseError(
            f"request rejected: {message}", status_code=status, body=body, model=model
        )
        return err, False, None

    async def _wait_before_retry(
        self, attempt: int, attempts: int, error: NIMError, retry_after: float | None, model: str
    ) -> None:
        """Sleep before the next attempt, or raise ``error`` if attempts are exhausted."""
        if attempt >= attempts - 1:
            raise error
        if retry_after is not None and retry_after > RETRY_AFTER_MAX:
            raise error
        delay = backoff_delay(attempt, retry_after)
        self.usage.record_retry(model)
        _log.debug(
            "retrying %s in %.2fs (attempt %d/%d): %s", model, delay, attempt + 1, attempts, error
        )
        await _sleep(delay)

    async def _request_json(
        self, method: str, url: str, *, model: str, body: dict[str, Any] | None = None
    ) -> Any:
        sem = self._semaphore()
        http = self._client()
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            retry_after: float | None = None
            try:
                async with sem:
                    resp = await http.request(method, url, json=body)
            except httpx.TimeoutException as exc:
                error: NIMError = NIMTimeoutError(
                    f"request timed out ({type(exc).__name__})", model=model
                )
            except httpx.UnsupportedProtocol as exc:
                raise NIMError(f"unsupported URL {url!r}: {exc}", model=model) from exc
            except httpx.TransportError as exc:
                error = NIMError(f"connection error: {type(exc).__name__}: {exc}", model=model)
            else:
                if resp.is_success:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise NIMResponseError(
                            "response was not valid JSON",
                            status_code=resp.status_code,
                            body=resp.text[:500],
                            model=model,
                        ) from exc
                error, retryable, retry_after = self._classify(resp, model)
                if not retryable:
                    raise error
            await self._wait_before_retry(attempt, attempts, error, retry_after, model)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _stream_deltas(self, model: str, body: dict[str, Any]) -> AsyncIterator[str]:
        """Yield raw content deltas of an SSE chat stream; retries only before the first byte."""
        sem = self._semaphore()
        http = self._client()
        url = f"{self.config.base_url}/chat/completions"
        attempts = self.config.max_retries + 1
        started = False
        for attempt in range(attempts):
            retry_after: float | None = None
            try:
                async with (
                    sem,
                    http.stream(
                        "POST", url, json=body, headers={"Accept": "text/event-stream"}
                    ) as resp,
                ):
                    if resp.is_success:
                        async for line in resp.aiter_lines():
                            done, delta = _sse_delta(line)
                            if done:
                                return
                            if delta:
                                started = True
                                yield delta
                        return
                    await resp.aread()
                    error, retryable, retry_after = self._classify(resp, model)
                    if not retryable:
                        raise error
            except httpx.TimeoutException as exc:
                if started:
                    raise NIMTimeoutError("stream interrupted by timeout", model=model) from exc
                error = NIMTimeoutError(f"request timed out ({type(exc).__name__})", model=model)
            except httpx.UnsupportedProtocol as exc:
                raise NIMError(f"unsupported URL {url!r}: {exc}", model=model) from exc
            except httpx.TransportError as exc:
                if started:
                    raise NIMError(f"stream interrupted: {exc}", model=model) from exc
                error = NIMError(f"connection error: {type(exc).__name__}: {exc}", model=model)
            await self._wait_before_retry(attempt, attempts, error, retry_after, model)

    async def _mock_call(self, fn: Callable[[], R]) -> R:
        async with self._semaphore():
            await asyncio.sleep(_mock.mock_delay())
            return fn()

    @staticmethod
    async def _gather(coros: list[Awaitable[R]]) -> list[R]:
        """Like ``asyncio.gather`` but cancels siblings when one fails (py3.10-safe)."""
        tasks = [asyncio.ensure_future(c) for c in coros]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    # ------------------------------------------------------------------ chat
    @staticmethod
    def _chat_body(
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        response_format: dict[str, Any] | None,
        extra: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if extra:
            body.update(extra)
        return body

    @staticmethod
    def _parse_chat(model: str, data: Any, latency_s: float) -> ChatResult:
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise NIMResponseError(
                f"unexpected chat response shape: {_truncate(data, 300)}", model=model
            ) from exc
        content, think = strip_think(_content_text(msg.get("content")))
        side = msg.get("reasoning_content") or msg.get("reasoning")
        reasoning_parts = [p.strip() for p in (side, think) if isinstance(p, str) and p.strip()]
        tool_calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(fn.get("name", "")),
                    arguments=_parse_arguments(fn.get("arguments")),
                )
            )
        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            total_tokens=int(raw_usage.get("total_tokens") or 0),
        )
        return ChatResult(
            model=model,
            content=content,
            reasoning="\n\n".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            latency_s=latency_s,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        """One chat completion. ``<think>`` blocks are moved from ``content`` to ``reasoning``."""
        t0 = time.perf_counter()
        try:
            if self.config.mock:
                raw = await self._mock_call(
                    lambda: _mock.mock_chat_response(
                        model,
                        messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                        max_tokens=max_tokens,
                    )
                )
            else:
                body = self._chat_body(
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    extra=extra,
                    stream=False,
                )
                raw = await self._request_json(
                    "POST", f"{self.config.base_url}/chat/completions", model=model, body=body
                )
            result = self._parse_chat(model, raw, time.perf_counter() - t0)
        except Exception:
            self.usage.record(model, None, time.perf_counter() - t0, error=True)
            raise
        self.usage.record(model, result.usage, result.latency_s)
        return result

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the assistant's text as chunks (SSE). ``<think>`` blocks are filtered out.

        Only text is yielded; tool-call deltas are ignored (use :meth:`chat` for tool calls).
        """
        t0 = time.perf_counter()
        error = False
        raw_chars = 0
        prompt_tokens = sum(_mock.estimate_tokens(_mock.message_text(m)) for m in messages)
        usage_override: Usage | None = None
        filt = ThinkStreamFilter()
        try:
            if self.config.mock:
                raw = await self._mock_call(
                    lambda: _mock.mock_chat_response(
                        model,
                        messages,
                        tool_choice="none",
                        response_format=response_format,
                        max_tokens=max_tokens,
                    )
                )
                text = _content_text(raw["choices"][0]["message"].get("content"))
                usage_override = Usage(**raw["usage"])
                for chunk in _mock.mock_stream_chunks(text):
                    raw_chars += len(chunk)
                    piece = filt.feed(chunk)
                    if piece:
                        yield piece
                    await asyncio.sleep(0)
            else:
                body = self._chat_body(
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    extra=extra,
                    stream=True,
                )
                async for delta in self._stream_deltas(model, body):
                    raw_chars += len(delta)
                    piece = filt.feed(delta)
                    if piece:
                        yield piece
            tail = filt.flush()
            if tail:
                yield tail
        except Exception:
            error = True
            raise
        finally:
            usage = usage_override or Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=_mock.estimate_tokens("x" * raw_chars),
            )
            self.usage.record(model, usage, time.perf_counter() - t0, error=error)

    # ------------------------------------------------------------------ structured output
    @staticmethod
    def _with_schema_prompt(
        messages: list[dict[str, Any]], schema: type[BaseModel]
    ) -> list[dict[str, Any]]:
        instruction = (
            "Respond with ONLY a single valid JSON value that conforms to the JSON Schema below. "
            "Do not add explanations, comments or markdown code fences.\n\n"
            f"{_mock.JSON_SCHEMA_OPEN}\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"{_mock.JSON_SCHEMA_CLOSE}"
        )
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            existing = _content_text(msgs[0].get("content"))
            msgs[0]["content"] = f"{existing}\n\n{instruction}" if existing else instruction
        else:
            msgs.insert(0, {"role": "system", "content": instruction})
        return msgs

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        retries: int = 2,
        **kw: Any,
    ) -> T:
        """Chat and return a validated ``schema`` instance.

        The JSON Schema is injected into the system message. JSON is extracted tolerantly (code
        fences, leading prose, ``<think>`` blocks). On parse/validation failure the model is
        re-prompted with the error, up to ``retries`` extra times; then :class:`NIMJSONError`.
        ``**kw`` is forwarded to :meth:`chat`.
        """
        msgs = self._with_schema_prompt(messages, schema)
        last_error: ValueError | None = None
        raw = ""
        for _ in range(retries + 1):
            result = await self.chat(model, msgs, **kw)
            raw = result.content
            try:
                return schema.model_validate(extract_json(raw))
            except ValueError as exc:  # JSONExtractionError and pydantic ValidationError
                last_error = exc
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": raw or "(empty response)"},
                    {
                        "role": "user",
                        "content": (
                            f"{_validation_hint(exc)}\n\nReturn ONLY the corrected JSON, "
                            "conforming to the schema."
                        ),
                    },
                ]
        raise NIMJSONError(
            f"model did not return valid {schema.__name__} JSON after {retries + 1} attempt(s): "
            f"{last_error}",
            raw=raw,
            model=model,
        )

    # ------------------------------------------------------------------ embeddings
    async def _embed_batch(
        self, model: str, batch: list[str], input_type: str
    ) -> tuple[list[list[float]], int]:
        body = {
            "model": model,
            "input": batch,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        data = await self._request_json(
            "POST", f"{self.config.base_url}/embeddings", model=model, body=body
        )
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise NIMResponseError(
                f"unexpected embeddings response: {_truncate(data, 300)}", model=model
            ) from exc
        if len(vectors) != len(batch):
            raise NIMResponseError(
                f"embeddings count mismatch: sent {len(batch)} texts, got {len(vectors)}",
                model=model,
            )
        tokens = int((data.get("usage") or {}).get("prompt_tokens") or 0)
        return vectors, tokens

    async def embed(
        self,
        model: str,
        texts: list[str],
        *,
        input_type: Literal["query", "passage"] = "passage",
    ) -> list[list[float]]:
        """Embed texts (batches of 32). Use ``input_type="query"`` for search queries."""
        if not texts:
            return []
        t0 = time.perf_counter()
        try:
            if self.config.mock:
                vectors = await self._mock_call(lambda: _mock.mock_embeddings(texts))
                tokens = sum(_mock.estimate_tokens(t) for t in texts)
            else:
                batches = [
                    texts[i : i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)
                ]
                parts = await self._gather(
                    [self._embed_batch(model, b, input_type) for b in batches]
                )
                vectors = [v for vecs, _ in parts for v in vecs]
                tokens = sum(t for _, t in parts)
        except Exception:
            self.usage.record(model, None, time.perf_counter() - t0, error=True)
            raise
        self.usage.record(model, Usage(prompt_tokens=tokens), time.perf_counter() - t0)
        return vectors

    # ------------------------------------------------------------------ rerank
    async def _rerank_batch(
        self, model: str, query: str, batch: list[str], offset: int
    ) -> tuple[list[tuple[int, float]], int]:
        body = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": p} for p in batch],
        }
        url = f"{self.config.rerank_base_url}/{model}/reranking"
        data = await self._request_json("POST", url, model=model, body=body)
        try:
            ranked = [(int(r["index"]), float(r["logit"])) for r in data["rankings"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise NIMResponseError(
                f"unexpected rerank response: {_truncate(data, 300)}", model=model
            ) from exc
        if any(not 0 <= i < len(batch) for i, _ in ranked):
            raise NIMResponseError("rerank response contained an out-of-range index", model=model)
        tokens = int((data.get("usage") or {}).get("prompt_tokens") or 0)
        return [(offset + i, s) for i, s in ranked], tokens

    async def rerank(
        self, model: str, query: str, passages: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Rerank passages for a query; returns results sorted by descending score.

        ``index`` is the passage's position in ``passages``. Scores are model logits (real) or
        overlap scores (mock): compare within one call only.
        """
        if not passages:
            return []
        t0 = time.perf_counter()
        try:
            if self.config.mock:
                scores = await self._mock_call(lambda: _mock.mock_rerank_scores(query, passages))
                pairs = list(enumerate(scores))
                tokens = _mock.estimate_tokens(query) + sum(
                    _mock.estimate_tokens(p) for p in passages
                )
            else:
                batches = [
                    (i, passages[i : i + RERANK_BATCH_SIZE])
                    for i in range(0, len(passages), RERANK_BATCH_SIZE)
                ]
                parts = await self._gather(
                    [self._rerank_batch(model, query, b, off) for off, b in batches]
                )
                pairs = [p for ranked, _ in parts for p in ranked]
                tokens = sum(t for _, t in parts)
        except Exception:
            self.usage.record(model, None, time.perf_counter() - t0, error=True)
            raise
        self.usage.record(model, Usage(prompt_tokens=tokens), time.perf_counter() - t0)
        pairs.sort(key=lambda p: (-p[1], p[0]))
        if top_n is not None:
            pairs = pairs[: max(top_n, 0)]
        return [RerankResult(index=i, score=s, text=passages[i]) for i, s in pairs]

    # ------------------------------------------------------------------ models
    async def list_models(self) -> list[str]:
        """Model IDs served by the endpoint (mock mode: the curated registry)."""
        label = "(list_models)"
        t0 = time.perf_counter()
        try:
            if self.config.mock:
                from .registry import default_registry

                ids = await self._mock_call(lambda: sorted(default_registry().ids()))
            else:
                data = await self._request_json(
                    "GET", f"{self.config.base_url}/models", model=label
                )
                try:
                    ids = sorted({str(m["id"]) for m in data["data"]})
                except (KeyError, TypeError) as exc:
                    raise NIMResponseError(
                        f"unexpected models response: {_truncate(data, 300)}", model=label
                    ) from exc
        except Exception:
            self.usage.record(label, None, time.perf_counter() - t0, error=True)
            raise
        self.usage.record(label, None, time.perf_counter() - t0)
        return ids
