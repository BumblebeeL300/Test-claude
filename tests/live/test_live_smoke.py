"""Opt-in smoke tests against the real NVIDIA NIM API: one chat, one embed, one rerank call.

Skipped unless BOTH of these are set in the environment::

    NVIDIA_API_KEY=nvapi-...      # a key from https://build.nvidia.com
    NIMLAB_LIVE=1

Run them with::

    NIMLAB_LIVE=1 pytest tests/live -m live -v

They spend a few tokens of your free NIM credits. The environment is read once at import time,
because the shared autouse fixture in ``tests/conftest.py`` scrubs the ``NVIDIA_*``, ``NIM_*`` and
``NIMLAB_*`` variables before every test; ``NIM_BASE_URL``, ``NIM_RERANK_URL`` and
``NIMLAB_MODEL_<ROLE>`` overrides from the original environment are honoured. Model ids that
are missing from your endpoint fail the test on purpose: check ``nimlab models --live`` and
override the role.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from nimlab.core import NIMClient, NIMConfig, Role
from nimlab.core.registry import ModelRegistry

_ENV = dict(os.environ)
_API_KEY = _ENV.get("NVIDIA_API_KEY", "").strip()
_ENABLED = bool(_API_KEY) and _ENV.get("NIMLAB_LIVE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _ENABLED, reason="live tests need NVIDIA_API_KEY and NIMLAB_LIVE=1 in the environment"
    ),
]


def _live_config() -> NIMConfig:
    kwargs: dict[str, Any] = {"api_key": _API_KEY, "mock": False}
    if _ENV.get("NIM_BASE_URL", "").strip():
        kwargs["base_url"] = _ENV["NIM_BASE_URL"].strip()
    if _ENV.get("NIM_RERANK_URL", "").strip():
        kwargs["rerank_base_url"] = _ENV["NIM_RERANK_URL"].strip()
    return NIMConfig(**kwargs)


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(env=_ENV)


@pytest.fixture
async def live_client() -> AsyncIterator[NIMClient]:
    config = _live_config()
    assert not config.mock, "live tests must not run on the mock backend"
    async with NIMClient(config) as client:
        yield client


async def test_chat(live_client: NIMClient, registry: ModelRegistry) -> None:
    model = registry.pick(Role.FAST)
    result = await live_client.chat(
        model,
        [{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=32,
        temperature=0.0,
    )
    assert result.content.strip(), f"{model} returned an empty reply"
    assert result.usage.total_tokens > 0
    assert result.latency_s > 0


async def test_embed(live_client: NIMClient, registry: ModelRegistry) -> None:
    model = registry.pick(Role.EMBED)
    vectors = await live_client.embed(
        model, ["NVIDIA NIM serves models over an API.", "Bananas are yellow."]
    )
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
    assert any(x != 0.0 for x in vectors[0])


async def test_rerank(live_client: NIMClient, registry: ModelRegistry) -> None:
    model = registry.pick(Role.RERANK)
    passages = [
        "Bananas are a yellow fruit that grows in bunches.",
        "The capital of France is Paris.",
        "Paris is the capital and largest city of France.",
    ]
    results = await live_client.rerank(model, "What is the capital of France?", passages)
    assert len(results) == len(passages)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].index in (1, 2), "a Paris passage should outrank the banana passage"
