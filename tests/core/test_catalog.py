"""Live discovery (GET /v1/models + disk cache), role heuristics, override precedence."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from nimlab.core import ModelRegistry, NIMConfig, Role, model_scope
from nimlab.core.catalog import (
    CACHE_TTL_S,
    LiveCatalog,
    heuristic_score,
    parse_size_b,
    rank_for_role,
)
from nimlab.core.registry import DEFAULTS, PREFERRED

BASE = "https://integrate.api.nvidia.com/v1"
MODELS_URL = f"{BASE}/models"

# A trimmed copy of what https://integrate.api.nvidia.com/v1/models listed on 2026-11-03.
LIVE_2026_11_03 = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-embed-1b",
    "nvidia/nemotron-3.5-content-safety",
    "nvidia/nemotron-4-340b-reward",
    "poolside/laguna-xs-2.1",
    "google/gemma-4-31b-it",
    "meta/llama-guard-4-12b",
    "writer/palmyra-med-70b",
]


def listing(ids: list[str]) -> dict[str, object]:
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def cfg() -> NIMConfig:
    return NIMConfig(api_key="nvapi-test", mock=False)


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make_registry(tmp_path: Path, cfg: NIMConfig, clock: Clock, **kw: object) -> ModelRegistry:
    catalog = LiveCatalog(tmp_path / "models_cache.json", clock=clock)
    return ModelRegistry(live=True, nim_config=cfg, catalog=catalog, **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- heuristics
@pytest.mark.parametrize(
    ("name", "size"),
    [
        ("nvidia/nemotron-3-ultra-550b-a55b", 550.0),
        ("google/gemma-4-31b-it", 31.0),
        ("mistralai/mixtral-8x22b-instruct-v0.1", 176.0),
        ("mistralai/mistral-large-4-1t-a52b", 1000.0),
        ("nvidia/nemotron-3.5-bolt-e4b", 4.0),
        ("ibm/granite-3.0-3b-a800m-instruct", 3.0),
        ("nvidia/nvclip", None),
        ("nvidia/nemotron-3.5-embed-8b", 8.0),
    ],
)
def test_parse_size_from_name(name: str, size: float | None) -> None:
    assert parse_size_b(name) == size


@pytest.mark.parametrize(
    ("role", "name", "matches"),
    [
        ("embed", "acme/super-embed-v9", True),
        ("embed", "acme/chatty-70b-instruct", False),
        ("rerank", "acme/rerank-2b", True),
        ("reward", "acme/judge-reward-340b", True),
        ("safety", "acme/content-safety-8b", True),
        ("safety", "acme/foo-guard-1b", True),
        ("safety", "nvidia/llama-3.1-nemoguard-8b-topic-control", False),
        ("vision", "acme/foo-vl-7b", True),
        ("vision", "acme/foo-vision-instruct", True),
        ("vision", "acme/plain-70b-instruct", False),
        ("coder", "acme/foo-coder-32b-instruct", True),
        ("coder", "acme/plain-70b-instruct", False),
        ("reasoning", "acme/deepseek-r1-distill", True),
        ("reasoning", "acme/foo-thinking-32b", True),
        ("reasoning", "acme/foo-reasoning-8b", True),
        ("chat", "acme/foo-70b-instruct", True),
        ("chat", "acme/foo-embed-1b", False),  # embedding models are never chat models
        ("fast", "acme/foo-3b-instruct", True),
        ("fast", "acme/foo-400b-instruct", False),
    ],
)
def test_role_heuristics(role: str, name: str, matches: bool) -> None:
    assert (heuristic_score(role, name) is not None) is matches


def test_heuristics_use_size_as_the_tiebreak() -> None:
    ids = ["a/big-reasoning-500b", "a/small-reasoning-8b", "a/mid-reasoning-70b"]
    assert rank_for_role("reasoning", ids)[0] == "a/big-reasoning-500b"  # bigger is better
    assert (
        rank_for_role("fast", ["a/x-8b-instruct", "a/x-1b-instruct", "a/x-70b-instruct"])[0]
        == "a/x-1b-instruct"
    )  # smaller is faster


# --------------------------------------------------------------------------- cache + TTL
def test_discovery_fetches_once_then_serves_from_the_disk_cache(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    route = respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))
    reg = make_registry(tmp_path, cfg, clock)
    assert reg.pick(Role.CHAT) == "nvidia/nemotron-3-super-120b-a12b"
    assert reg.pick(Role.CODER) == "poolside/laguna-xs-2.1"
    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer nvapi-test"

    cached = json.loads((tmp_path / "models_cache.json").read_text())
    assert cached["base_url"] == BASE and "nvapi-test" not in json.dumps(cached)  # no key on disk
    assert sorted(cached["models"]) == sorted(LIVE_2026_11_03)

    # a second registry (a new process) reads the cache: still one request
    again = make_registry(tmp_path, cfg, clock)
    assert again.pick(Role.EMBED) == "nvidia/nemotron-3.5-embed-1b"
    assert route.call_count == 1


def test_cache_expires_after_24_hours(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    route = respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))
    make_registry(tmp_path, cfg, clock).pick(Role.CHAT)
    assert route.call_count == 1

    clock.now += CACHE_TTL_S - 60  # still fresh
    make_registry(tmp_path, cfg, clock).pick(Role.CHAT)
    assert route.call_count == 1

    clock.now += 120  # past the TTL
    make_registry(tmp_path, cfg, clock).pick(Role.CHAT)
    assert route.call_count == 2


def test_cache_is_keyed_by_endpoint_and_key(
    respx_mock: respx.MockRouter, tmp_path: Path, clock: Clock
) -> None:
    route = respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))
    make_registry(tmp_path, NIMConfig(api_key="nvapi-a", mock=False), clock).pick(Role.CHAT)
    make_registry(tmp_path, NIMConfig(api_key="nvapi-b", mock=False), clock).pick(Role.CHAT)
    assert route.call_count == 2  # another account may see other models


def test_refresh_ignores_the_cache(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    route = respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))
    reg = make_registry(tmp_path, cfg, clock)
    reg.pick(Role.CHAT)
    snap = reg.refresh_catalog()
    assert route.call_count == 2 and snap is not None and snap.source == "network"


def test_unreachable_endpoint_falls_back_to_curated_defaults_then_stale_cache(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    respx_mock.get(MODELS_URL).mock(side_effect=httpx.ConnectError("blocked"))
    reg = make_registry(tmp_path, cfg, clock)
    assert reg.pick(Role.CHAT) == DEFAULTS[Role.CHAT]  # no cache: curated default
    assert reg.snapshot(fetch=False) is None

    # with an expired cache on disk, the stale list is used instead of nothing
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "fetched_at": clock.now - 3 * CACHE_TTL_S,
                "base_url": BASE,
                "key_fp": __import__("hashlib").sha256(b"nvapi-test").hexdigest()[:12],
                "models": ["acme/only-model-70b-instruct"],
            }
        )
    )
    stale = make_registry(tmp_path, cfg, clock)
    snap = stale.snapshot()
    assert snap is not None and snap.source == "stale-cache"


# --------------------------------------------------------------------------- picking
def test_pick_returns_the_first_preferred_default_that_is_available(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    live = [i for i in LIVE_2026_11_03 if i != PREFERRED[Role.CHAT][0]]
    respx_mock.get(MODELS_URL).respond(200, json=listing(live))
    reg = make_registry(tmp_path, cfg, clock)
    assert reg.pick(Role.CHAT) == PREFERRED[Role.CHAT][1]  # google/gemma-4-31b-it
    assert reg.pick(Role.REASONING) == PREFERRED[Role.REASONING][0]
    stale = {s["role"]: s for s in reg.stale_defaults()}
    assert stale["chat"]["default"] == PREFERRED[Role.CHAT][0]
    assert stale["chat"]["substitute"] == PREFERRED[Role.CHAT][1]
    assert "reasoning" not in stale


def test_pick_falls_back_to_the_best_heuristic_match(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    live = [
        "acme/zeta-embed-v3",
        "acme/zeta-embed-large-7b",
        "acme/zeta-70b-instruct",
        "acme/zeta-coder-32b-instruct",
        "acme/zeta-vl-72b",
        "acme/zeta-reasoning-400b",
        "acme/zeta-guard-8b",
        "acme/zeta-reward-340b",
        "acme/zeta-1b-instruct",
    ]
    respx_mock.get(MODELS_URL).respond(200, json=listing(live))
    reg = make_registry(tmp_path, cfg, clock)
    assert reg.pick(Role.EMBED) == "acme/zeta-embed-v3"  # ~2B target beats 7B
    assert reg.pick(Role.CODER) == "acme/zeta-coder-32b-instruct"
    assert reg.pick(Role.VISION) == "acme/zeta-vl-72b"
    assert reg.pick(Role.REASONING) == "acme/zeta-reasoning-400b"
    assert reg.pick(Role.SAFETY) == "acme/zeta-guard-8b"
    assert reg.pick(Role.REWARD) == "acme/zeta-reward-340b"
    assert reg.pick(Role.FAST) == "acme/zeta-1b-instruct"
    assert reg.pick(Role.CHAT) == "acme/zeta-70b-instruct"
    # nothing rerank-like is listed, and reranking is not served by /v1/models anyway
    assert reg.pick(Role.RERANK) == DEFAULTS[Role.RERANK]


def test_curated_default_when_nothing_matches(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    respx_mock.get(MODELS_URL).respond(200, json=listing(["acme/unrelated-thing"]))
    reg = make_registry(tmp_path, cfg, clock)
    assert reg.pick(Role.CODER) == DEFAULTS[Role.CODER]


def test_reranking_is_not_judged_by_v1_models(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))  # no rerank models
    reg = make_registry(tmp_path, cfg, clock)
    assert reg.pick(Role.RERANK) == "nvidia/nemotron-3.5-rerank-8b"
    assert "rerank" not in {s["role"] for s in reg.stale_defaults()}


def test_no_discovery_in_mock_mode(respx_mock: respx.MockRouter, tmp_path: Path) -> None:
    route = respx_mock.get(MODELS_URL).respond(200, json=listing([]))
    reg = ModelRegistry(nim_config=NIMConfig(mock=True))
    assert reg.pick(Role.CHAT) == DEFAULTS[Role.CHAT] and route.call_count == 0


def test_discovery_can_be_switched_off_with_an_env_var(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(MODELS_URL).respond(200, json=listing([]))
    reg = ModelRegistry(
        nim_config=NIMConfig(api_key="k", mock=False), env={"NIMLAB_DISCOVERY": "0"}
    )
    assert reg.pick(Role.CHAT) == DEFAULTS[Role.CHAT] and route.call_count == 0


# --------------------------------------------------------------------------- precedence
def test_override_precedence_request_app_studio_env_toml_auto_default(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))
    toml = tmp_path / "nimlab.toml"
    env: dict[str, str] = {}
    reg = make_registry(
        tmp_path,
        cfg,
        clock,
        config_path=toml,
        env=env,
        settings_path=tmp_path / "studio_settings.json",
    )
    # 7. curated default / live auto-pick
    assert reg.pick(Role.CHAT) == PREFERRED[Role.CHAT][0]
    assert reg.describe(Role.CHAT)["source"] == "default"

    # 6. nimlab.toml beats the auto-pick
    toml.write_text('[models]\nchat = "toml/chat"\n')
    assert reg.pick(Role.CHAT) == "toml/chat" and reg.describe(Role.CHAT)["source"] == "toml"
    # 5. env beats toml
    env["NIMLAB_MODEL_CHAT"] = "env/chat"
    assert reg.pick(Role.CHAT) == "env/chat" and reg.describe(Role.CHAT)["source"] == "env"
    # 4. the global (Studio) override beats env and toml
    reg.set_studio_overrides({"chat": "studio/chat"})
    assert reg.pick(Role.CHAT) == "studio/chat"
    assert reg.describe(Role.CHAT)["source"] == "studio"
    # 3. a per-app override beats the global one, but only for that app
    reg.set_app_overrides("atlas", {"chat": "app/chat"})
    assert reg.pick(Role.CHAT, app="atlas") == "app/chat"
    assert reg.pick(Role.CHAT, app="council") == "studio/chat"
    assert reg.pick(Role.CHAT) == "studio/chat"
    # 2. a per-request override beats everything
    view = reg.scoped("atlas", {"chat": "request/chat"})
    assert view.pick(Role.CHAT) == "request/chat"
    assert view.describe(Role.CHAT)["source"] == "request"
    assert view.pick(Role.CODER) == PREFERRED[Role.CODER][0]  # other roles unaffected
    # clearing walks back down the ladder
    reg.set_app_overrides("atlas", {"chat": None})
    assert reg.pick(Role.CHAT, app="atlas") == "studio/chat"
    reg.set_studio_overrides({"chat": "auto"})
    assert reg.pick(Role.CHAT) == "env/chat"


def test_overrides_may_name_models_missing_from_the_live_catalog(
    respx_mock: respx.MockRouter, tmp_path: Path, cfg: NIMConfig, clock: Clock
) -> None:
    respx_mock.get(MODELS_URL).respond(200, json=listing(LIVE_2026_11_03))
    reg = make_registry(tmp_path, cfg, clock, env={"NIMLAB_MODEL_CODER": "nope/not-listed"})
    assert reg.pick(Role.CODER) == "nope/not-listed"


def test_studio_overrides_persist_and_ignore_garbage(tmp_path: Path) -> None:
    path = tmp_path / "studio_settings.json"
    reg = ModelRegistry(live=False, settings_path=path, env={})
    reg.set_studio_overrides({"coder": "x/coder", "vision": "x/vision"})
    reg.set_app_overrides("forge", {"coder": "forge/coder"})
    again = ModelRegistry(live=False, settings_path=path, env={})
    assert again.studio_overrides() == {Role.CODER: "x/coder", Role.VISION: "x/vision"}
    assert again.app_overrides("forge") == {"forge": {Role.CODER: "forge/coder"}}
    path.write_text('{"model_overrides": {"coder": 5, "bogus": "x", "chat": "ok/chat"}}')
    assert ModelRegistry(live=False, settings_path=path, env={}).studio_overrides() == {
        Role.CHAT: "ok/chat"
    }
    path.write_text("not json")
    assert ModelRegistry(live=False, settings_path=path, env={}).studio_overrides() == {}


def test_model_scope_applies_overrides_to_default_registry(tmp_path: Path) -> None:
    from nimlab.core import default_registry

    default_registry().use_settings_path(tmp_path / "s.json")
    default_registry().set_app_overrides("arena", {"coder": "arena/coder"})
    assert default_registry().pick(Role.CODER) == DEFAULTS[Role.CODER]
    with model_scope("arena", {"chat": "req/chat"}) as view:
        assert default_registry() is view
        assert default_registry().pick(Role.CODER) == "arena/coder"
        assert default_registry().pick(Role.CHAT) == "req/chat"
    assert view.picks is not None
    assert {(p["role"], p["model"], p["source"]) for p in view.picks} == {
        ("coder", "arena/coder", "app"),
        ("chat", "req/chat", "request"),
    }
    assert default_registry().pick(Role.CODER) == DEFAULTS[Role.CODER]
