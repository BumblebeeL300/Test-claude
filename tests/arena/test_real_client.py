"""Arena on top of the real ``NIMClient`` HTTP code path (respx, no network)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from arena_fakes import make_suite
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from nimlab.arena import Arena
from nimlab.arena.cli import app as arena_cli
from nimlab.arena.proxy import create_app
from nimlab.arena.router import SmartRouter
from nimlab.arena.suites import Suite
from nimlab.core import NIMAuthError, NIMClient, NIMConfig

CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GOOD, BAD = "meta/llama-3.3-70b-instruct", "meta/llama-3.1-8b-instruct"
ANSWERS = {"What is 6*7?": "42", "What is 5+5?": "10"}


def sse(text: str) -> str:
    pieces = [text[i : i + 4] for i in range(0, len(text), 4)] or [""]
    events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
    events += [{"choices": [{"index": 0, "delta": {"content": p}}]} for p in pieces]
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def completion(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    }


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    model = body["model"]
    user = body["messages"][-1]["content"]
    if "Candidate response" in user:  # judge call from llm_judge
        return httpx.Response(200, json=completion('{"score": 9, "rationale": "solid"}'))
    if "### Answer A" in user:  # pairwise judge
        return httpx.Response(200, json=completion('{"winner": "A", "rationale": "first"}'))
    answer = f"Answer: {ANSWERS.get(user, '0')}" if model == GOOD else "Answer: 1"
    if body.get("stream"):
        return httpx.Response(200, text=sse(answer), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=completion(answer))


@pytest.fixture
def mock_nim(respx_mock: respx.MockRouter) -> respx.MockRouter:
    respx_mock.post(CHAT_URL).mock(side_effect=handler)
    return respx_mock


def suite() -> Suite:
    return make_suite("mini", [("a", "What is 6*7?", 42), ("b", "What is 5+5?", 10)])


async def test_runner_scores_real_streamed_responses(
    mock_nim: respx.MockRouter, real_config: NIMConfig, runs_dir
) -> None:
    async with Arena(NIMClient(real_config), runs_dir=runs_dir, concurrency=3) as arena:
        run = await arena.run([GOOD, BAD], [suite()])
        board = arena.leaderboard(run, n_boot=20)
    scores = {(r.model, r.item_id): r.score for r in run.items}
    assert scores == {(GOOD, "a"): 1.0, (GOOD, "b"): 1.0, (BAD, "a"): 0.0, (BAD, "b"): 0.0}
    for r in run.items:
        assert r.ttft_s is not None and 0 <= r.ttft_s <= r.latency_s and r.usage_estimated
        assert r.output.startswith("Answer:") and r.tokens_out > 0
    assert run.mock is False
    assert board.standings[0].model == GOOD and board.standings[0].overall == 1.0
    requests = [json.loads(c.request.content) for c in mock_nim.calls]
    assert all(r["stream"] is True and r["temperature"] == 0.0 for r in requests)
    assert all(c.request.headers["authorization"] == "Bearer nvapi-test" for c in mock_nim.calls)


async def test_llm_judge_and_battles_over_http(
    mock_nim: respx.MockRouter, real_config: NIMConfig, runs_dir
) -> None:
    summ = Suite.model_validate(
        {
            "name": "s",
            "category": "summarization",
            "open_ended": True,
            "grader": {"type": "llm_judge", "rubric": "faithful"},
            "items": [{"id": "x", "prompt": "What is 6*7?", "reference": "42"}],
        }
    )
    async with Arena(NIMClient(real_config), runs_dir=runs_dir, judge_model=GOOD) as arena:
        result = await arena.battle([GOOD, BAD], summ)
    judged = [r for r in result.items if r.model == GOOD]
    assert judged[0].score == pytest.approx(8 / 9) and "9/10" in judged[0].detail
    (battle,) = result.battles
    assert battle.score_a == 0.5  # the stub judge always says "A": position bias cancels
    assert [v.pick for v in battle.verdicts] == ["A", "A"]


async def test_auth_failure_aborts_with_nim_auth_error(
    respx_mock: respx.MockRouter, real_config: NIMConfig, runs_dir
) -> None:
    respx_mock.post(CHAT_URL).respond(401, json={"detail": "invalid api key"})
    async with Arena(NIMClient(real_config), runs_dir=runs_dir) as arena:
        with pytest.raises(NIMAuthError):
            await arena.run([GOOD], [suite()])


async def test_transient_http_errors_are_retried_by_the_client(
    respx_mock: respx.MockRouter, real_config: NIMConfig, sleeps: list[float]
) -> None:
    failures = [
        httpx.Response(503, json={"detail": "busy"}),
        httpx.Response(429, json={"detail": "slow"}),
    ]

    def flaky(request: httpx.Request) -> httpx.Response:
        return failures.pop(0) if failures else handler(request)

    route = respx_mock.post(CHAT_URL).mock(side_effect=flaky)
    async with Arena(NIMClient(real_config), persist=False) as arena:
        run = await arena.run([GOOD], [suite().limited(1)])
    assert run.items[0].score == 1.0 and run.status == "complete"
    assert route.call_count == 3 and len(sleeps) == 2  # the core client backed off twice


def test_cli_reports_auth_errors_cleanly(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NIMLAB_MOCK", "0")
    respx_mock.post(CHAT_URL).respond(401, json={"detail": "invalid api key"})
    result = CliRunner().invoke(
        arena_cli,
        [
            "run",
            "--suite",
            "factual_qa",
            "--limit",
            "1",
            "--models",
            GOOD,
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "authentication failed" in result.output and "NVIDIA_API_KEY" in result.output
    assert "Mock mode" not in result.output


def test_proxy_over_the_real_client_streams_sse_and_forwards_the_request(
    mock_nim: respx.MockRouter, real_config: NIMConfig
) -> None:
    client = NIMClient(real_config)
    smart = SmartRouter(runs_dir="/nonexistent", client=None)
    with TestClient(create_app(client=client, router=smart)) as api:
        plain = api.post(
            "/v1/chat/completions",
            json={
                "model": GOOD,
                "messages": [{"role": "user", "content": "What is 6*7?"}],
                "temperature": 0.1,
            },
        )
        assert plain.status_code == 200
        assert plain.json()["choices"][0]["message"]["content"] == "Answer: 42"
        assert plain.json()["usage"]["total_tokens"] == 13
        assert plain.headers["x-nimlab-routed-model"] == GOOD
        with api.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": GOOD,
                "stream": True,
                "messages": [{"role": "user", "content": "What is 6*7?"}],
            },
        ) as r:
            text = "".join(r.iter_text())
        chunks = [json.loads(b[6:]) for b in text.strip().split("\n\n")[:-1]]
        assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Answer: 42"
        assert text.rstrip().endswith("data: [DONE]")
        auto = api.post(
            "/v1/chat/completions",
            json={
                "model": "nimlab/auto",
                "messages": [{"role": "user", "content": "Write a python function"}],
            },
        )
        assert auto.headers["x-nimlab-category"] == "coding"
        assert auto.headers["x-nimlab-routed-model"] == "qwen/qwen2.5-coder-32b-instruct"
    sent = json.loads(mock_nim.calls[0].request.content)
    assert sent["model"] == GOOD and sent["temperature"] == 0.1 and sent["stream"] is False
