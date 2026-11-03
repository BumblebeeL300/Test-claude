"""Runtime fallback: 404 / "model not found" marks the model unavailable and retries once."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
import respx

from nimlab.core import (
    ModelRegistry,
    NIMClient,
    NIMConfig,
    NIMModelUnavailableError,
    NIMResponseError,
    Role,
)
from nimlab.core.registry import PREFERRED

BASE = "https://integrate.api.nvidia.com/v1"
CHAT_URL = f"{BASE}/chat/completions"
EMBED_URL = f"{BASE}/embeddings"
RERANK = "https://ai.api.nvidia.com/v1/retrieval"
MSGS = [{"role": "user", "content": "hello"}]

#: What NVIDIA answers for a retired / not-provisioned model (observed shape, 2025-2026).
NVIDIA_404 = {
    "status": 404,
    "title": "Not Found",
    "detail": "Function '3a4c0d0e-1d1c-4f0a-9d5d-b4b3c0f2a111': Not found for account 'acct'",
}


def completion(text: str = "ok") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(live=False, env={}, settings_path="/nonexistent/none.json")


@pytest.fixture
async def client(real_config: NIMConfig, registry: ModelRegistry) -> Any:
    async with NIMClient(real_config, registry=registry) as c:
        yield c


def served_models(route: respx.Route) -> list[str]:
    return [json.loads(c.request.content)["model"] for c in route.calls]


async def test_role_pick_retries_once_with_the_next_candidate_and_warns(
    respx_mock: respx.MockRouter,
    client: NIMClient,
    registry: ModelRegistry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first, second = PREFERRED[Role.CHAT][:2]
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=lambda req: (
            httpx.Response(404, json=NVIDIA_404)
            if json.loads(req.content)["model"] == first
            else httpx.Response(200, json=completion("hi"))
        )
    )
    model = registry.pick(Role.CHAT)
    assert model == first
    with caplog.at_level(logging.WARNING, logger="nimlab"):
        result = await client.chat(model, MSGS)
    assert result.model == second and result.content == "hi"
    assert served_models(route) == [first, second]
    warning = next(r.getMessage() for r in caplog.records if "not available" in r.getMessage())
    assert first in warning and second in warning and "chat" in warning
    # the model is now blacklisted for the session: later picks skip it, no more 404s
    assert registry.is_unavailable(first) and registry.pick(Role.CHAT) == second
    assert registry.substitutions()[-1]["requested"] == first


async def test_explicit_models_are_not_substituted(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    route = respx_mock.post(CHAT_URL).respond(404, json=NVIDIA_404)
    with pytest.raises(NIMModelUnavailableError) as info:
        await client.chat("acme/typed-by-hand", MSGS)
    assert route.call_count == 1 and info.value.status_code == 404
    assert isinstance(info.value, NIMResponseError)  # existing handlers keep working
    assert registry.is_unavailable("acme/typed-by-hand")


async def test_override_picks_are_explicit_too(
    respx_mock: respx.MockRouter, real_config: NIMConfig
) -> None:
    reg = ModelRegistry(live=False, env={"NIMLAB_MODEL_CHAT": "mine/chat"})
    route = respx_mock.post(CHAT_URL).respond(404, json=NVIDIA_404)
    async with NIMClient(real_config, registry=reg) as c:
        with pytest.raises(NIMModelUnavailableError):
            await c.chat(reg.pick(Role.CHAT), MSGS)
    assert route.call_count == 1


async def test_only_one_retry(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    route = respx_mock.post(CHAT_URL).respond(404, json=NVIDIA_404)
    with pytest.raises(NIMModelUnavailableError):
        await client.chat(registry.pick(Role.CHAT), MSGS)
    assert route.call_count == 2


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "The model `acme/x` does not exist"}},
        {"detail": "Model acme/x is not available for this account"},
        {"message": "Function 'abc': Not found for account"},
    ],
)
async def test_model_not_found_shapes_on_400(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry, body: dict[str, Any]
) -> None:
    first = registry.pick(Role.CHAT)
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=lambda req: (
            httpx.Response(400, json=body)
            if json.loads(req.content)["model"] == first
            else httpx.Response(200, json=completion())
        )
    )
    assert (await client.chat(first, MSGS)).model != first
    assert route.call_count == 2


async def test_other_client_errors_do_not_trigger_the_fallback(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    route = respx_mock.post(CHAT_URL).respond(400, json={"detail": "max_tokens is too large"})
    with pytest.raises(NIMResponseError) as info:
        await client.chat(registry.pick(Role.CHAT), MSGS)
    assert not isinstance(info.value, NIMModelUnavailableError) and route.call_count == 1
    assert registry.unavailable() == {}


async def test_listing_models_404_is_not_a_model_problem(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    respx_mock.get(f"{BASE}/models").respond(404, json={"detail": "no such route"})
    with pytest.raises(NIMResponseError) as info:
        await client.list_models()
    assert not isinstance(info.value, NIMModelUnavailableError)


async def test_embed_falls_back(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    first, second = PREFERRED[Role.EMBED][:2]

    def reply(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if body["model"] == first:
            return httpx.Response(404, json=NVIDIA_404)
        assert body["input_type"] == "query"  # still sent on the retry
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    route = respx_mock.post(EMBED_URL).mock(side_effect=reply)
    vecs = await client.embed(registry.pick(Role.EMBED), ["q"], input_type="query")
    assert vecs == [[0.1, 0.2]] and served_models(route) == [first, second]


async def test_rerank_falls_back_and_uses_the_models_own_path(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    first, second = PREFERRED[Role.RERANK][:2]
    bad = respx_mock.post(f"{RERANK}/{first}/reranking").respond(404, json=NVIDIA_404)
    good = respx_mock.post(f"{RERANK}/{second}/reranking").respond(
        200, json={"rankings": [{"index": 1, "logit": 2.0}, {"index": 0, "logit": -1.0}]}
    )
    out = await client.rerank(registry.pick(Role.RERANK), "q", ["a", "b"])
    assert [r.index for r in out] == [1, 0]
    assert bad.call_count == 1 and good.call_count == 1


async def test_rerank_accepts_both_response_shapes(
    respx_mock: respx.MockRouter, client: NIMClient
) -> None:
    model = "nvidia/nemotron-3.5-rerank-8b"
    route = respx_mock.post(f"{RERANK}/{model}/reranking")
    route.respond(200, json={"rankings": [{"index": 0, "logit": 3.5}, {"index": 1, "logit": 1.0}]})
    old = await client.rerank(model, "q", ["a", "b"])
    route.respond(
        200,
        json={
            "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        },
    )
    new = await client.rerank(model, "q", ["a", "b"])
    assert [r.index for r in old] == [0, 1] and [r.index for r in new] == [1, 0]
    body = json.loads(route.calls.last.request.content)
    assert body["query"] == {"text": "q"} and body["passages"] == [{"text": "a"}, {"text": "b"}]


async def test_stream_falls_back_before_the_first_token(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    first = registry.pick(Role.CHAT)
    sse = 'data: {"choices":[{"delta":{"content":"hey"}}]}\n\ndata: [DONE]\n\n'

    def reply(req: httpx.Request) -> httpx.Response:
        if json.loads(req.content)["model"] == first:
            return httpx.Response(404, json=NVIDIA_404)
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    route = respx_mock.post(CHAT_URL).mock(side_effect=reply)
    text = "".join([c async for c in client.stream_chat(first, MSGS)])
    assert text == "hey" and route.call_count == 2


async def test_chat_json_goes_through_the_fallback(
    respx_mock: respx.MockRouter, client: NIMClient, registry: ModelRegistry
) -> None:
    from pydantic import BaseModel

    class Out(BaseModel):
        n: int

    first = registry.pick(Role.CHAT)
    respx_mock.post(CHAT_URL).mock(
        side_effect=lambda req: (
            httpx.Response(404, json=NVIDIA_404)
            if json.loads(req.content)["model"] == first
            else httpx.Response(200, json=completion('{"n": 3}'))
        )
    )
    assert (await client.chat_json(first, MSGS, Out)).n == 3
