"""Real HTTP code path of NIMClient, exercised with respx (no network)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from nimlab.core import (
    NIMAuthError,
    NIMClient,
    NIMConfig,
    NIMError,
    NIMJSONError,
    NIMRateLimitError,
    NIMResponseError,
    NIMTimeoutError,
)
from nimlab.core.client import RETRY_AFTER_MAX, parse_retry_after

BASE = "https://integrate.api.nvidia.com/v1"
RERANK = "https://ai.api.nvidia.com/v1/retrieval"
CHAT_URL = f"{BASE}/chat/completions"
CHAT = "nvidia/nemotron-3-super-120b-a12b"
MSGS = [{"role": "user", "content": "hello"}]


def completion(content: str | None = "hi there", **message_extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, **message_extra}
    finish = "tool_calls" if message_extra.get("tool_calls") else "stop"
    return {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def body_of(call: Any) -> dict[str, Any]:
    return json.loads(call.request.content)


@pytest.fixture
async def client(real_config: NIMConfig) -> Any:
    async with NIMClient(real_config) as c:
        yield c


# ------------------------------------------------------------------------------- chat basics
async def test_chat_request_and_response(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    route = respx_mock.post(CHAT_URL).respond(200, json=completion("hi there"))
    result = await client.chat(CHAT, MSGS, temperature=0.7, max_tokens=50)

    assert result.content == "hi there" and result.model == CHAT
    assert result.finish_reason == "stop" and result.reasoning is None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 7)
    assert result.usage.total_tokens == 18 and result.latency_s >= 0

    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer nvapi-test"
    body = body_of(route.calls.last)
    assert body == {
        "model": CHAT,
        "messages": MSGS,
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False,
    }
    s = client.usage.summary()["models"][CHAT]
    assert (s["calls"], s["tokens_in"], s["tokens_out"], s["errors"]) == (1, 11, 7, 0)


async def test_optional_chat_params_are_forwarded(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(CHAT_URL).respond(200, json=completion())
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    await client.chat(
        CHAT,
        MSGS,
        tools=tools,
        tool_choice="auto",
        response_format={"type": "json_object"},
        extra={"top_p": 0.9, "nvext": {"guided_json": {}}},
    )
    body = body_of(route.calls.last)
    assert body["tools"] == tools and body["tool_choice"] == "auto"
    assert body["response_format"] == {"type": "json_object"}
    assert body["top_p"] == 0.9 and body["nvext"] == {"guided_json": {}}


async def test_no_authorization_header_without_key(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(CHAT_URL).respond(200, json=completion())
    async with NIMClient(NIMConfig(mock=False)) as c:
        await c.chat(CHAT, MSGS)
    assert "authorization" not in route.calls.last.request.headers


async def test_custom_base_url(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("http://localhost:8000/v1/chat/completions").respond(
        200, json=completion()
    )
    cfg = NIMConfig(mock=False, base_url="http://localhost:8000/v1/")
    async with NIMClient(cfg) as c:
        assert (await c.chat(CHAT, MSGS)).content == "hi there"
    assert route.called


async def test_think_blocks_are_stripped_into_reasoning(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    raw = "<think>\nlet me think\nabout this\n</think>\n\nThe answer is 42."
    respx_mock.post(CHAT_URL).respond(200, json=completion(raw))
    r = await client.chat("nvidia/nemotron-3-ultra-550b-a55b", MSGS)
    assert r.content == "The answer is 42."
    assert r.reasoning == "let me think\nabout this"


async def test_reasoning_content_field_is_captured(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    respx_mock.post(CHAT_URL).respond(
        200, json=completion("final", reasoning_content="step by step")
    )
    r = await client.chat(CHAT, MSGS)
    assert (r.content, r.reasoning) == ("final", "step by step")


async def test_tool_calls_are_parsed(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Oslo", "days": 2}'},
        },
        {"id": "call_2", "type": "function", "function": {"name": "noargs", "arguments": ""}},
        {"id": "call_3", "type": "function", "function": {"name": "broken", "arguments": "{oops"}},
    ]
    respx_mock.post(CHAT_URL).respond(200, json=completion(None, tool_calls=calls))
    r = await client.chat(CHAT, MSGS)
    assert r.content == "" and r.finish_reason == "tool_calls"
    assert [(t.id, t.name) for t in r.tool_calls] == [
        ("call_1", "get_weather"),
        ("call_2", "noargs"),
        ("call_3", "broken"),
    ]
    assert r.tool_calls[0].arguments == {"city": "Oslo", "days": 2}
    assert r.tool_calls[1].arguments == {}
    assert r.tool_calls[2].arguments == {"_raw": "{oops"}
    msg = r.to_message()
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"city": "Oslo", "days": 2}'


@pytest.mark.parametrize(
    "payload", [{"choices": []}, {"nothing": 1}, {"choices": [None]}, ["not", "a", "dict"]]
)
async def test_malformed_chat_response(
    respx_mock: respx.MockRouter, client: NIMClient, payload: Any
) -> None:
    respx_mock.post(CHAT_URL).respond(200, json=payload)
    with pytest.raises(NIMResponseError, match="unexpected chat response"):
        await client.chat(CHAT, MSGS)
    assert client.usage.summary()["models"][CHAT]["errors"] == 1


async def test_non_json_success_body(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    respx_mock.post(CHAT_URL).respond(200, text="<html>oops</html>")
    with pytest.raises(NIMResponseError, match="not valid JSON"):
        await client.chat(CHAT, MSGS)


# ------------------------------------------------------------------------------- retries
async def test_retry_on_429_respects_retry_after(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "slow down"}),
            httpx.Response(200, json=completion("ok")),
        ]
    )
    assert (await client.chat(CHAT, MSGS)).content == "ok"
    assert route.call_count == 2
    assert len(sleeps) == 1 and sleeps[0] >= 7
    s = client.usage.summary()["models"][CHAT]
    assert (s["calls"], s["retries"], s["errors"]) == (1, 1, 0)


async def test_5xx_exponential_backoff_with_jitter(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(503, text="unavailable"),
            httpx.Response(502, text="bad gateway"),
            httpx.Response(500, json={"error": {"message": "boom"}}),
            httpx.Response(200, json=completion("finally")),
        ]
    )
    assert (await client.chat(CHAT, MSGS)).content == "finally"
    assert route.call_count == 4
    assert len(sleeps) == 3
    # delay_n in [0.5, 1.0] * 0.5 * 2**n  (equal jitter)
    for n, delay in enumerate(sleeps):
        ceiling = 0.5 * 2**n
        assert ceiling * 0.5 <= delay <= ceiling


async def test_retries_exhausted_on_5xx(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).respond(500, json={"detail": "internal"})
    with pytest.raises(NIMResponseError) as info:
        await client.chat(CHAT, MSGS)
    assert info.value.status_code == 500 and "internal" in str(info.value)
    assert route.call_count == 4  # 1 try + max_retries(3)
    assert len(sleeps) == 3
    s = client.usage.summary()["models"][CHAT]
    assert (s["calls"], s["errors"], s["retries"]) == (1, 1, 3)


async def test_retries_exhausted_on_429_raises_rate_limit_error(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    respx_mock.post(CHAT_URL).respond(429, headers={"Retry-After": "1"}, text="Too Many Requests")
    with pytest.raises(NIMRateLimitError) as info:
        await client.chat(CHAT, MSGS)
    assert info.value.status_code == 429 and info.value.retry_after == 1.0


async def test_huge_retry_after_is_not_waited_out(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).respond(
        429, headers={"Retry-After": str(int(RETRY_AFTER_MAX) + 1)}
    )
    with pytest.raises(NIMRateLimitError):
        await client.chat(CHAT, MSGS)
    assert route.call_count == 1 and sleeps == []


async def test_max_retries_zero_means_single_attempt(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(CHAT_URL).respond(503)
    async with NIMClient(NIMConfig(api_key="k", mock=False, max_retries=0)) as c:
        with pytest.raises(NIMResponseError):
            await c.chat(CHAT, MSGS)
    assert route.call_count == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_not_retried(
    respx_mock: respx.MockRouter, client: NIMClient, status: int, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).respond(status, json={"detail": "Invalid API key"})
    with pytest.raises(NIMAuthError) as info:
        await client.chat(CHAT, MSGS)
    assert route.call_count == 1 and sleeps == []
    assert info.value.status_code == status and "Invalid API key" in str(info.value)
    assert client.usage.summary()["models"][CHAT]["errors"] == 1


@pytest.mark.parametrize("status", [400, 404, 422])
async def test_client_errors_not_retried(
    respx_mock: respx.MockRouter, client: NIMClient, status: int
) -> None:
    route = respx_mock.post(CHAT_URL).respond(status, json={"detail": "bad request body"})
    with pytest.raises(NIMResponseError) as info:
        await client.chat(CHAT, MSGS)
    assert route.call_count == 1 and info.value.status_code == status
    assert not isinstance(info.value, (NIMAuthError, NIMRateLimitError))


async def test_timeout_then_success(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, json=completion("ok"))]
    )
    assert (await client.chat(CHAT, MSGS)).content == "ok"
    assert route.call_count == 2 and len(sleeps) == 1


async def test_timeouts_exhausted(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    route = respx_mock.post(CHAT_URL).mock(side_effect=httpx.ConnectTimeout("nope"))
    with pytest.raises(NIMTimeoutError):
        await client.chat(CHAT, MSGS)
    assert route.call_count == 4


async def test_connection_errors_are_retried(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=completion("ok"))]
    )
    assert (await client.chat(CHAT, MSGS)).content == "ok"
    assert route.call_count == 2


def test_error_hierarchy() -> None:
    for cls in (NIMAuthError, NIMRateLimitError, NIMResponseError, NIMTimeoutError):
        assert issubclass(cls, NIMError)
    assert issubclass(NIMJSONError, NIMResponseError)


def test_parse_retry_after() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("-4") == 0.0
    assert parse_retry_after(None) is None and parse_retry_after("") is None
    assert parse_retry_after("garbage") is None
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    parsed = parse_retry_after(future)
    assert parsed is not None and 20 <= parsed <= 31
    past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=30), usegmt=True)
    assert parse_retry_after(past) == 0.0


async def test_retry_after_http_date(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
    respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": when}),
            httpx.Response(200, json=completion()),
        ]
    )
    await client.chat(CHAT, MSGS)
    assert 5 <= sleeps[0] <= 11


# ------------------------------------------------------------------------------- concurrency
async def test_max_concurrency_limits_in_flight_requests(respx_mock: respx.MockRouter) -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json=completion("ok"))

    respx_mock.post(CHAT_URL).mock(side_effect=handler)
    async with NIMClient(NIMConfig(api_key="k", mock=False, max_concurrency=3)) as c:
        results = await asyncio.gather(*(c.chat(CHAT, MSGS) for _ in range(12)))
    assert len(results) == 12
    assert peak == 3


async def test_semaphore_released_during_backoff(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request backing off must not hold its concurrency slot.

    max_concurrency=1; request A gets a 503 and then "sleeps" until request B has completed.
    If A kept the slot while sleeping, B could never run and this would time out.
    """
    b_done = asyncio.Event()
    a_failed_once = False

    async def sleep_until_b_done(seconds: float) -> None:
        await asyncio.wait_for(b_done.wait(), timeout=2)

    monkeypatch.setattr("nimlab.core.client._sleep", sleep_until_b_done)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal a_failed_once
        who = json.loads(request.content)["messages"][0]["content"]
        if who == "A" and not a_failed_once:
            a_failed_once = True
            return httpx.Response(503, json={})
        if who == "B":
            b_done.set()
        return httpx.Response(200, json=completion(f"done {who}"))

    respx_mock.post(CHAT_URL).mock(side_effect=handler)
    async with NIMClient(NIMConfig(api_key="k", mock=False, max_concurrency=1)) as c:
        a_task = asyncio.create_task(c.chat(CHAT, [{"role": "user", "content": "A"}]))
        await asyncio.sleep(0.01)  # let A take the slot and fail first
        b = await c.chat(CHAT, [{"role": "user", "content": "B"}])
        a = await a_task
    assert (a.content, b.content) == ("done A", "done B")


# ------------------------------------------------------------------------------- streaming
def sse(*events: Any) -> str:
    lines = []
    for ev in events:
        if ev == "[DONE]":
            lines.append("data: [DONE]\n")
        elif isinstance(ev, str) and ev.startswith(":"):
            lines.append(f"{ev}\n")
        else:
            lines.append(f"data: {json.dumps(ev)}\n")
    return "\n".join(lines) + "\n"


def delta(content: str | None = None, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = dict(extra)
    if content is not None:
        d["content"] = content
    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}


async def test_stream_chat_parses_sse(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    payload = sse(
        ": keep-alive comment",
        delta(role="assistant", content=""),
        delta("Hello"),
        delta(", "),
        {"choices": []},  # usage-only chunk without choices
        delta("wor"),
        delta("ld"),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        "[DONE]",
        delta("IGNORED AFTER DONE"),
    )
    route = respx_mock.post(CHAT_URL).respond(
        200, headers={"content-type": "text/event-stream"}, text=payload
    )
    chunks = [c async for c in client.stream_chat(CHAT, MSGS, max_tokens=33)]
    assert chunks == ["Hello", ", ", "wor", "ld"]
    body = body_of(route.calls.last)
    assert body["stream"] is True and body["max_tokens"] == 33
    assert route.calls.last.request.headers["accept"] == "text/event-stream"
    s = client.usage.summary()["models"][CHAT]
    assert s["calls"] == 1 and s["errors"] == 0 and s["tokens_out"] > 0


async def test_stream_chat_tolerates_data_without_space_and_junk(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    payload = (
        'data:{"choices":[{"delta":{"content":"a"}}]}\n\n'
        "data: not-json\n\n"
        "event: ping\n\n"
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx_mock.post(CHAT_URL).respond(200, text=payload)
    assert "".join([c async for c in client.stream_chat(CHAT, MSGS)]) == "ab"


async def test_stream_chat_strips_think_split_across_chunks(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    pieces = ["<thi", "nk>secret ", "reasoning</th", "ink>\n\n", "The ", "answer."]
    payload = sse(*[delta(p) for p in pieces], "[DONE]")
    respx_mock.post(CHAT_URL).respond(200, text=payload)
    text = "".join([c async for c in client.stream_chat("nvidia/nemotron-3-ultra-550b-a55b", MSGS)])
    assert text == "The answer."


async def test_stream_chat_retries_before_first_byte(
    respx_mock: respx.MockRouter, client: NIMClient, sleeps: list[float]
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(503),
            httpx.Response(200, text=sse(delta("ok"), "[DONE]")),
        ]
    )
    assert [c async for c in client.stream_chat(CHAT, MSGS)] == ["ok"]
    assert route.call_count == 3 and len(sleeps) == 2 and sleeps[0] >= 2


async def test_stream_chat_auth_error(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    respx_mock.post(CHAT_URL).respond(401, json={"detail": "unauthorized"})
    with pytest.raises(NIMAuthError):
        async for _ in client.stream_chat(CHAT, MSGS):
            pass
    assert client.usage.summary()["models"][CHAT]["errors"] == 1


async def test_stream_chat_error_event(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    payload = sse(delta("partial"), {"error": {"message": "model crashed"}})
    respx_mock.post(CHAT_URL).respond(200, text=payload)
    got: list[str] = []
    with pytest.raises(NIMResponseError, match="model crashed"):
        async for c in client.stream_chat(CHAT, MSGS):
            got.append(c)
    assert got == ["partial"]


async def test_stream_chat_early_exit_still_records_usage(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    respx_mock.post(CHAT_URL).respond(
        200, text=sse(*[delta(f"w{i} ") for i in range(10)], "[DONE]")
    )
    agen = client.stream_chat(CHAT, MSGS)
    first = await agen.__anext__()
    await agen.aclose()
    assert first == "w0 "
    assert client.usage.summary()["models"][CHAT]["calls"] == 1


# ------------------------------------------------------------------------------- chat_json
class Verdict(BaseModel):
    answer: str
    score: int = Field(ge=1, le=10)


async def test_chat_json_repairs_after_invalid_output(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, json=completion("Sure, here you go: no json at all")),
            httpx.Response(200, json=completion('{"answer": "x", "score": 99}')),
            httpx.Response(200, json=completion('```json\n{"answer": "fixed", "score": 7}\n```')),
        ]
    )
    result = await client.chat_json(CHAT, MSGS, Verdict, retries=2)
    assert result == Verdict(answer="fixed", score=7)
    assert route.call_count == 3

    first = body_of(route.calls[0])["messages"]
    assert first[0]["role"] == "system" and "<json_schema>" in first[0]["content"]
    assert '"score"' in first[0]["content"]
    second = body_of(route.calls[1])["messages"]
    assert second[-2] == {"role": "assistant", "content": "Sure, here you go: no json at all"}
    assert "not parseable JSON" in second[-1]["content"]
    third = body_of(route.calls[2])["messages"]
    assert "score" in third[-1]["content"] and "less than or equal to 10" in third[-1]["content"]


async def test_chat_json_handles_reasoning_model_output(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    raw = (
        '<think>maybe {"answer": "wrong"}?</think>\nResult:\n'
        '```json\n{"answer": "ok", "score": 3}\n```'
    )
    respx_mock.post(CHAT_URL).respond(200, json=completion(raw))
    assert await client.chat_json("nvidia/nemotron-3-ultra-550b-a55b", MSGS, Verdict) == Verdict(
        answer="ok", score=3
    )


async def test_chat_json_exhausts_retries(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    route = respx_mock.post(CHAT_URL).respond(200, json=completion("never json"))
    with pytest.raises(NIMJSONError) as info:
        await client.chat_json(CHAT, MSGS, Verdict, retries=1)
    assert route.call_count == 2 and info.value.raw == "never json"
    assert "Verdict" in str(info.value)


async def test_chat_json_merges_into_existing_system_message(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(CHAT_URL).respond(200, json=completion('{"answer": "a", "score": 1}'))
    msgs = [{"role": "system", "content": "Be brief."}, *MSGS]
    await client.chat_json(CHAT, msgs, Verdict, temperature=0.0)
    sent = body_of(route.calls.last)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"].startswith("Be brief.")
    assert sent["temperature"] == 0.0


# ------------------------------------------------------------------------------- embeddings
EMBED_URL = f"{BASE}/embeddings"
EMBED = "nvidia/nemotron-3.5-embed-1b"


def embeddings_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    # vector encodes the text length; response order is deliberately reversed
    data = [
        {"index": i, "object": "embedding", "embedding": [float(len(t)), 1.0]}
        for i, t in enumerate(body["input"])
    ]
    return httpx.Response(
        200, json={"data": list(reversed(data)), "usage": {"prompt_tokens": len(body["input"])}}
    )


async def test_embed_batches_of_32_preserving_order(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(EMBED_URL).mock(side_effect=embeddings_handler)
    texts = ["x" * (i + 1) for i in range(70)]
    vecs = await client.embed(EMBED, texts)

    assert route.call_count == 3
    sizes = sorted(len(body_of(c)["input"]) for c in route.calls)
    assert sizes == [6, 32, 32]
    assert [v[0] for v in vecs] == [float(i + 1) for i in range(70)]
    s = client.usage.summary()["models"][EMBED]
    assert s["calls"] == 1 and s["tokens_in"] == 70


async def test_embed_request_shape(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    route = respx_mock.post(EMBED_URL).mock(side_effect=embeddings_handler)
    await client.embed(EMBED, ["hello"], input_type="query")
    assert body_of(route.calls.last) == {
        "model": EMBED,
        "input": ["hello"],
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "END",
    }
    await client.embed(EMBED, ["hello"])
    assert body_of(route.calls.last)["input_type"] == "passage"


async def test_embed_empty_makes_no_request(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(EMBED_URL).mock(side_effect=embeddings_handler)
    assert await client.embed(EMBED, []) == []
    assert not route.called


async def test_embed_count_mismatch(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    respx_mock.post(EMBED_URL).respond(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(NIMResponseError, match="count mismatch"):
        await client.embed(EMBED, ["a", "b"])


async def test_embed_malformed_response(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    respx_mock.post(EMBED_URL).respond(200, json={"oops": True})
    with pytest.raises(NIMResponseError, match="unexpected embeddings response"):
        await client.embed(EMBED, ["a"])


async def test_embed_failure_in_one_batch_fails_the_call(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    respx_mock.post(EMBED_URL).respond(400, json={"detail": "input too long"})
    with pytest.raises(NIMResponseError, match="input too long"):
        await client.embed(EMBED, [f"t{i}" for i in range(100)])
    assert client.usage.summary()["models"][EMBED]["errors"] == 1


# ------------------------------------------------------------------------------- rerank
RR = "nvidia/nemotron-3.5-rerank-8b"
RR_URL = f"{RERANK}/{RR}/reranking"


async def test_rerank_request_and_response_shape(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(RR_URL).respond(
        200,
        json={
            "rankings": [
                {"index": 2, "logit": 4.5},
                {"index": 0, "logit": -1.25},
                {"index": 1, "logit": 0.5},
            ]
        },
    )
    results = await client.rerank(RR, "what is nim", ["alpha", "beta", "gamma"])

    assert body_of(route.calls.last) == {
        "model": RR,
        "query": {"text": "what is nim"},
        "passages": [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}],
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer nvapi-test"
    assert [(r.index, r.score, r.text) for r in results] == [
        (2, 4.5, "gamma"),
        (1, 0.5, "beta"),
        (0, -1.25, "alpha"),
    ]


async def test_rerank_sorts_descending_even_if_api_does_not(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    respx_mock.post(RR_URL).respond(
        200, json={"rankings": [{"index": 0, "logit": 1.0}, {"index": 1, "logit": 3.0}]}
    )
    results = await client.rerank(RR, "q", ["a", "b"])
    assert [r.index for r in results] == [1, 0]


async def test_rerank_top_n(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    respx_mock.post(RR_URL).respond(
        200, json={"rankings": [{"index": i, "logit": float(i)} for i in range(5)]}
    )
    results = await client.rerank(RR, "q", list("abcde"), top_n=2)
    assert [r.index for r in results] == [4, 3]


async def test_rerank_many_passages_are_batched_with_offsets(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        passages = json.loads(request.content)["passages"]
        # score = numeric value in passage text, so ordering is checkable across batches
        rankings = [{"index": i, "logit": float(p["text"])} for i, p in enumerate(passages)]
        return httpx.Response(200, json={"rankings": rankings})

    route = respx_mock.post(RR_URL).mock(side_effect=handler)
    passages = [str(i) for i in range(300)]
    results = await client.rerank(RR, "q", passages, top_n=3)
    assert route.call_count == 2
    assert [(r.index, r.text) for r in results] == [(299, "299"), (298, "298"), (297, "297")]


async def test_rerank_empty_and_bad_responses(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    route = respx_mock.post(RR_URL).respond(200, json={"rankings": [{"index": 9, "logit": 1.0}]})
    assert await client.rerank(RR, "q", []) == []
    assert not route.called
    with pytest.raises(NIMResponseError, match="out-of-range"):
        await client.rerank(RR, "q", ["a"])
    route.respond(200, json={"nope": []})
    with pytest.raises(NIMResponseError, match="unexpected rerank response"):
        await client.rerank(RR, "q", ["a"])


async def test_rerank_uses_custom_rerank_base_url(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"http://rr.local/v1/retrieval/{RR}/reranking").respond(
        200, json={"rankings": [{"index": 0, "logit": 1.0}]}
    )
    cfg = NIMConfig(api_key="k", mock=False, rerank_base_url="http://rr.local/v1/retrieval/")
    async with NIMClient(cfg) as c:
        assert len(await c.rerank(RR, "q", ["a"])) == 1
    assert route.called


# ------------------------------------------------------------------------------- models
async def test_list_models(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    route = respx_mock.get(f"{BASE}/models").respond(
        200,
        json={
            "object": "list",
            "data": [{"id": "z/model"}, {"id": "a/model"}, {"id": "a/model"}],
        },
    )
    assert await client.list_models() == ["a/model", "z/model"]
    assert route.calls.last.request.headers["authorization"] == "Bearer nvapi-test"


async def test_list_models_malformed(respx_mock: respx.MockRouter, client: NIMClient) -> None:
    respx_mock.get(f"{BASE}/models").respond(200, json={"data": [{"no_id": 1}]})
    with pytest.raises(NIMResponseError):
        await client.list_models()


# ------------------------------------------------------------------------------- lifecycle
async def test_injected_http_client_is_used_and_not_closed(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(CHAT_URL).respond(200, json=completion())
    http = httpx.AsyncClient()
    async with NIMClient(NIMConfig(api_key="k", mock=False), http_client=http) as c:
        await c.chat(CHAT, MSGS)
    assert not http.is_closed
    await http.aclose()


async def test_client_is_usable_after_aclose_by_reopening(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(CHAT_URL).respond(200, json=completion())
    c = NIMClient(NIMConfig(api_key="k", mock=False))
    await c.chat(CHAT, MSGS)
    await c.aclose()
    assert (await c.chat(CHAT, MSGS)).content == "hi there"
    await c.aclose()
