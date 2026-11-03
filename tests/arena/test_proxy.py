"""The OpenAI-compatible routing proxy, exercised through FastAPI's TestClient."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from arena_fakes import BIG, CODER, FASTM, REASONER, ScriptedClient, make_run
from fastapi.testclient import TestClient

from nimlab.arena.leaderboard import Leaderboard
from nimlab.arena.proxy import AUTO_MODEL, create_app
from nimlab.arena.router import SmartRouter
from nimlab.core.errors import NIMAuthError, NIMRateLimitError, NIMResponseError
from nimlab.core.results import ChatResult, ToolCall, Usage


class ProxyClient(ScriptedClient):
    """Echoes which model answered and records the kwargs it was called with."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.kwargs: list[dict[str, Any]] = []
        self.stream_error: Exception | None = None
        self.mid_stream_error: Exception | None = None
        self.tool_call = False

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        self.kwargs.append(kw)
        if self.stream_error:
            raise self.stream_error
        calls = [ToolCall(id="call_1", name="get_weather", arguments={"city": "Oslo"})]
        return ChatResult(
            model=model,
            content="" if self.tool_call else f"answer from {model}",
            tool_calls=calls if self.tool_call else [],
            finish_reason="tool_calls" if self.tool_call else "stop",
            usage=Usage(prompt_tokens=5, completion_tokens=7),
        )

    async def stream_chat(
        self, model: str, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        self.kwargs.append(kw)
        if self.stream_error:
            raise self.stream_error
        for piece in ("Hel", "lo ", f"from {model}"):
            yield piece
        if self.mid_stream_error:
            raise self.mid_stream_error


@pytest.fixture
def client_double() -> ProxyClient:
    return ProxyClient()


@pytest.fixture
def api(client_double: ProxyClient):
    lb = Leaderboard.from_run(make_run(), n_boot=0)
    smart = SmartRouter(lb, runs_dir="/nonexistent")
    app = create_app(client=client_double, router=smart)
    with TestClient(app) as tc:
        yield tc


def chat(api: TestClient, model: str, content: str = "hi", **extra: Any):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return api.post("/v1/chat/completions", json=body)


def sse_events(text: str) -> list[str]:
    assert text.endswith("\n\n")
    return [block[len("data: ") :] for block in text.strip().split("\n\n")]


# ------------------------------------------------------------------ /v1/models


def test_models_endpoint_lists_auto_and_registry_models(api: TestClient) -> None:
    body = api.get("/v1/models").json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids[0] == AUTO_MODEL == "nimlab/auto"
    assert "nvidia/nemotron-3-super-120b-a12b" in ids and CODER in ids
    assert len(ids) == len(set(ids))
    first = body["data"][0]
    assert first["object"] == "model" and first["owned_by"] == "nimlab" and "created" in first
    assert "nvidia/nemotron-3.5-embed-1b" not in ids  # embedding models are not chat models


# ------------------------------------------------------------------ non-streaming


def test_auto_model_routes_and_reports_the_routed_model(
    api: TestClient, client_double: ProxyClient
) -> None:
    r = chat(api, "nimlab/auto", "Write a Python function that merges overlapping intervals")
    assert r.status_code == 200
    assert r.headers["x-nimlab-routed-model"] == CODER
    assert r.headers["x-nimlab-category"] == "coding"
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    assert body["model"] == CODER
    choice = body["choices"][0]
    assert choice["index"] == 0 and choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": f"answer from {CODER}"}
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert body["nimlab"]["routed_model"] == CODER and body["nimlab"]["category"] == "coding"
    assert body["nimlab"]["reasons"]


def test_other_categories_route_to_other_models(api: TestClient) -> None:
    # balanced utility on factual QA: the 405B (0.92 - 0.06 - 0.15) beats the cheaper models
    r = chat(api, "nimlab/auto", "What is the capital of Peru?")
    assert r.headers["x-nimlab-routed-model"] == BIG
    assert r.headers["x-nimlab-category"] == "factual_qa"
    # ... while math goes to the reasoning model (0.97 - 0.12 - 0.12)
    m = chat(api, "nimlab/auto", "Solve the equation 5x - 3 = 22 and show the working")
    assert m.headers["x-nimlab-routed-model"] == REASONER


def test_explicit_models_pass_through_untouched(
    api: TestClient, client_double: ProxyClient
) -> None:
    r = chat(api, "nvidia/nemotron-3.5-lightning-30b-a3b", "Write a python function")
    assert r.status_code == 200
    assert r.headers["x-nimlab-routed-model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert "x-nimlab-category" not in r.headers
    body = r.json()
    assert body["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b" and "nimlab" not in body
    assert (
        body["choices"][0]["message"]["content"]
        == "answer from nvidia/nemotron-3.5-lightning-30b-a3b"
    )


def test_unknown_model_ids_are_passed_through_not_rejected(api: TestClient) -> None:
    r = chat(api, "some-vendor/brand-new-model")
    assert (
        r.status_code == 200 and r.headers["x-nimlab-routed-model"] == "some-vendor/brand-new-model"
    )


def test_profile_suffix_and_header_change_the_decision(api: TestClient) -> None:
    prompt = "Solve the equation 5x - 3 = 22 and show the working"
    quality = chat(api, "nimlab/auto:quality", prompt)
    speed = chat(api, "nimlab/auto:speed", prompt)
    assert quality.headers["x-nimlab-routed-model"] == REASONER
    assert speed.headers["x-nimlab-routed-model"] != REASONER
    via_header = api.post(
        "/v1/chat/completions",
        json={"model": AUTO_MODEL, "messages": [{"role": "user", "content": prompt}]},
        headers={"x-nimlab-profile": "quality"},
    )
    assert via_header.headers["x-nimlab-routed-model"] == REASONER


def test_category_header_skips_classification(api: TestClient) -> None:
    r = api.post(
        "/v1/chat/completions",
        json={"model": AUTO_MODEL, "messages": [{"role": "user", "content": "hello"}]},
        headers={"x-nimlab-category": "coding"},
    )
    assert (
        r.headers["x-nimlab-category"] == "coding" and r.headers["x-nimlab-routed-model"] == CODER
    )
    bad = api.post(
        "/v1/chat/completions",
        json={"model": AUTO_MODEL, "messages": [{"role": "user", "content": "x"}]},
        headers={"x-nimlab-category": "poetry"},
    )
    assert bad.status_code == 400 and bad.json()["error"]["type"] == "invalid_request_error"


def test_sampling_tools_and_extra_params_are_forwarded(
    api: TestClient, client_double: ProxyClient
) -> None:
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    chat(
        api,
        "nvidia/nemotron-3-super-120b-a12b",
        temperature=0.7,
        max_tokens=33,
        top_p=0.9,
        seed=4,
        tools=tools,
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    kw = client_double.kwargs[-1]
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 33
    assert kw["tools"] == tools and kw["tool_choice"] == "auto"
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["extra"] == {"top_p": 0.9, "seed": 4}
    chat(api, "m", max_completion_tokens=50)
    assert client_double.kwargs[-1] == {"max_tokens": 50}
    chat(api, "m")
    assert client_double.kwargs[-1] == {}  # nothing invented: client defaults apply


def test_tool_calls_are_returned_in_openai_format(
    api: TestClient, client_double: ProxyClient
) -> None:
    client_double.tool_call = True
    body = chat(api, "nvidia/nemotron-3-super-120b-a12b", "weather in Oslo?").json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_1" and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Oslo"}


# ------------------------------------------------------------------ streaming (SSE)


def test_streaming_uses_openai_chunk_format(api: TestClient) -> None:
    with api.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": AUTO_MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "Write a python function"}],
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-nimlab-routed-model"] == CODER
        assert r.headers["cache-control"] == "no-cache"
        text = "".join(r.iter_text())
    events = sse_events(text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1 and chunks[0]["id"].startswith("chatcmpl-")
    assert all(c["model"] == CODER for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[0]["choices"][0]["finish_reason"] is None
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == f"Hello from {CODER}"
    last = chunks[-1]["choices"][0]
    assert last["delta"] == {} and last["finish_reason"] == "stop"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_streaming_passthrough_model(api: TestClient) -> None:
    with api.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": FASTM, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        assert r.headers["x-nimlab-routed-model"] == FASTM and "x-nimlab-category" not in r.headers
        events = sse_events("".join(r.iter_text()))
    assert json.loads(events[1])["choices"][0]["delta"]["content"] == "Hel"


def test_upstream_errors_before_the_first_chunk_become_http_errors(
    api: TestClient, client_double: ProxyClient
) -> None:
    client_double.stream_error = NIMRateLimitError("slow down", retry_after=7, status_code=429)
    r = chat(api, FASTM, stream=True)
    assert r.status_code == 429 and r.headers["retry-after"] == "7"
    assert r.json()["error"]["type"] == "rate_limit_error"
    client_double.stream_error = NIMAuthError("bad key", status_code=401)
    assert chat(api, FASTM, stream=True).status_code == 401


def test_mid_stream_errors_are_reported_in_band(
    api: TestClient, client_double: ProxyClient
) -> None:
    client_double.mid_stream_error = NIMResponseError("connection lost")
    with api.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": FASTM, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        events = sse_events("".join(r.iter_text()))
    assert events[-1] == "[DONE]"
    err = json.loads(events[-2])
    assert err["error"]["type"] == "upstream_error" and "connection lost" in err["error"]["message"]
    assert not any(json.loads(e)["choices"][0]["finish_reason"] == "stop" for e in events[:-2])


# ------------------------------------------------------------------ errors


def test_error_responses_follow_the_openai_shape(
    api: TestClient, client_double: ProxyClient
) -> None:
    client_double.stream_error = NIMAuthError("bad key", status_code=401)
    r = chat(api, FASTM)
    assert r.status_code == 401
    err = r.json()["error"]
    assert (err["type"], err["code"]) == ("authentication_error", 401) and "bad key" in err[
        "message"
    ]
    client_double.stream_error = NIMResponseError("boom", status_code=500)
    r = chat(api, FASTM)
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_error"


def test_invalid_requests_are_400s(api: TestClient) -> None:
    r = api.post("/v1/chat/completions", json={"model": "x", "messages": "oops"})
    assert r.status_code == 400 and "messages" in r.json()["error"]["message"]
    assert api.post("/v1/chat/completions", json={"model": "x"}).status_code == 400
    r = chat(api, "nimlab/auto:warp")
    assert r.status_code == 400 and "unknown routing profile" in r.json()["error"]["message"]


# ------------------------------------------------------------------ /router/explain & friends


def test_router_explain_endpoint(api: TestClient) -> None:
    r = api.get("/router/explain", params={"prompt": "Write a Python function that sorts a list"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == CODER and body["category"] == "coding" and body["classifier"] == "knn"
    assert body["source"] == "leaderboard" and body["profile"] == "balanced"
    assert body["candidates"][0]["model"] == CODER
    assert all(k in body["candidates"][0] for k in ("quality", "latency_s", "cost_tier", "utility"))
    assert any("utility = quality" in reason for reason in body["reasons"])
    assert body["explanation"].startswith(f"Routed to {CODER}")
    assert body["classification"]["neighbors"]
    assert api.get("/router/explain").status_code == 400  # prompt is required
    quality = api.get(
        "/router/explain", params={"prompt": "solve 2x+1=9", "profile": "quality"}
    ).json()
    assert quality["model"] == REASONER and quality["profile"] == "quality"
    assert api.get("/router/explain", params={"prompt": "x", "profile": "warp"}).status_code == 400
    forced = api.get("/router/explain", params={"prompt": "hello", "category": "coding"}).json()
    assert forced["classifier"] == "override"


def test_health_and_leaderboard_endpoints(api: TestClient) -> None:
    health = api.get("/health").json()
    assert health["status"] == "ok" and health["leaderboard_run"]
    lb = api.get("/router/leaderboard").json()
    assert {s["model"] for s in lb["standings"]} == {CODER, REASONER, FASTM, BIG}


def test_explain_without_leaderboard_data_uses_the_registry() -> None:
    app = create_app(client=ProxyClient(), router=SmartRouter(runs_dir="/nonexistent"))
    with TestClient(app) as tc:
        body = tc.get("/router/explain", params={"prompt": "Write a python function"}).json()
        assert body["source"] == "registry" and body["model"] == CODER
        assert tc.get("/router/leaderboard").status_code == 404
        assert (
            chat(tc, AUTO_MODEL, "Write a python function").headers["x-nimlab-routed-model"]
            == CODER
        )


def test_default_app_runs_fully_offline_in_mock_mode(tmp_path) -> None:
    app = create_app(mock=True, runs_dir=tmp_path / "none")
    with TestClient(app) as tc:
        assert tc.get("/health").json()["mock"] is True
        r = chat(tc, AUTO_MODEL, "What is the capital of France?")
        assert r.status_code == 200 and r.headers["x-nimlab-routed-model"]
        assert r.json()["choices"][0]["message"]["content"]
        with tc.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": AUTO_MODEL,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as s:
            events = sse_events("".join(s.iter_text()))
        assert events[-1] == "[DONE]" and len(events) > 3
