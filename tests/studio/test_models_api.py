"""Model switching in Studio: role table, pickers, global + per-app overrides, run wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from studio_helpers import HOST, finish, make_settings

from nimlab.core import ModelRegistry, NIMConfig, Role, default_registry, set_default_registry
from nimlab.core.catalog import LiveCatalog
from nimlab.core.registry import DEFAULTS
from nimlab.studio.server import create_app

CORPUS = Path(__file__).resolve().parents[2] / "examples" / "atlas_corpus"
KEY = "nvapi-" + "m" * 60 + "c0de"
BASE = "https://integrate.api.nvidia.com/v1"


def roles(c: TestClient) -> dict[str, Any]:
    return c.get("/api/models/roles").json()


def by_role(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["role"]: r for r in payload["roles"]}


# ------------------------------------------------------------------------------- role table
def test_roles_payload_in_simulation(client: TestClient) -> None:
    data = roles(client)
    assert data["live"] is False and data["mock"] is True
    assert "live model list appears once a key is added" in data["note"]
    table = by_role(data)
    assert set(table) == {r.value for r in Role}
    assert table["coder"]["model"] == DEFAULTS[Role.CODER] and table["coder"]["source"] == "default"
    assert set(data["apps"]) == {
        "forge", "council", "atlas", "arena", "prism", "sentinel", "loom", "distill",
    }  # fmt: skip
    atlas = {e["label"]: e["role"] for e in data["apps"]["atlas"]["roles"]}
    assert atlas == {"EMBED": "embed", "RERANK": "rerank", "ANSWER": "chat"}
    forge = {e["label"]: e["role"] for e in data["apps"]["forge"]["roles"]}
    assert forge == {"PLANNER": "reasoning", "CODER": "coder", "REVIEWER": "chat"}
    assert [e["role"] for e in data["apps"]["prism"]["roles"]] == ["vision"]


def test_status_carries_the_stale_defaults_summary(client: TestClient) -> None:
    models = client.get("/api/status").json()["models"]
    assert models == {"live": False, "stale": [], "stale_count": 0}


def test_options_filter_by_role_with_an_all_models_toggle(client: TestClient) -> None:
    coder = client.get("/api/models/options", params={"role": "coder"}).json()
    ids = [m["id"] for m in coder["models"]]
    assert coder["live"] is False and "live model list appears" in coder["note"]
    assert DEFAULTS[Role.CODER] in ids and DEFAULTS[Role.EMBED] not in ids
    everything = client.get("/api/models/options", params={"role": "coder", "all": "1"}).json()
    assert len(everything["models"]) > len(ids)
    assert client.get("/api/models/options", params={"role": "wizard"}).status_code == 422


# ------------------------------------------------------------------------------- overrides
def test_global_overrides_persist_and_clear(client: TestClient, settings: Any) -> None:
    out = client.put("/api/models/overrides", json={"overrides": {"coder": "acme/coder-9b"}})
    assert out.status_code == 200
    row = by_role(out.json())["coder"]
    assert row["model"] == "acme/coder-9b" and row["source"] == "studio"
    assert row["override"] == "acme/coder-9b"
    saved = json.loads((settings.home / "studio_settings.json").read_text())
    assert saved["model_overrides"] == {"coder": "acme/coder-9b"}
    assert default_registry().pick("coder") == "acme/coder-9b"

    cleared = client.put("/api/models/overrides", json={"overrides": {"coder": None}}).json()
    assert by_role(cleared)["coder"]["source"] == "default"
    cleared = client.put("/api/models/overrides", json={"overrides": {"coder": "acme/x"}}).json()
    cleared = client.put("/api/models/overrides", json={"overrides": {"coder": "auto"}}).json()
    assert by_role(cleared)["coder"]["override"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"overrides": {"wizard": "x/y"}},
        {"overrides": {"chat": "has space"}},
        {"overrides": {"chat": "x" * 300}},
    ],
)
def test_invalid_overrides_are_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.put("/api/models/overrides", json=body).status_code == 422


def test_env_still_shows_up_as_pinned(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MODEL_FAST", "env/fast")
    row = by_role(roles(client))["fast"]
    assert row["model"] == "env/fast" and row["source"] == "env" and row["pinned_by"] == "env"
    # a global override saved in Studio outranks the environment
    client.put("/api/models/overrides", json={"overrides": {"fast": "studio/fast"}})
    assert by_role(roles(client))["fast"]["model"] == "studio/fast"


def test_app_overrides_win_over_global_ones_and_reset(client: TestClient) -> None:
    client.put("/api/models/overrides", json={"overrides": {"chat": "global/chat"}})
    out = client.put("/api/models/app/atlas", json={"overrides": {"chat": "atlas/chat"}}).json()
    atlas = {e["role"]: e for e in out["apps"]["atlas"]["roles"]}
    assert atlas["chat"]["model"] == "atlas/chat" and atlas["chat"]["override"] == "atlas/chat"
    assert atlas["chat"]["source"] == "app" and out["apps"]["atlas"]["overridden"] is True
    assert atlas["embed"]["source"] == "default"
    council = {e["role"]: e for e in out["apps"]["council"]["roles"]}
    assert council["chat"]["model"] == "global/chat"  # other apps follow the global setting
    assert by_role(out)["chat"]["model"] == "global/chat"

    reset = client.delete("/api/models/app/atlas").json()
    assert reset["apps"]["atlas"]["overridden"] is False
    assert {e["role"]: e for e in reset["apps"]["atlas"]["roles"]}["chat"]["model"] == "global/chat"


def test_unknown_app_is_404(client: TestClient) -> None:
    assert client.put("/api/models/app/nope", json={"overrides": {}}).status_code == 404
    assert client.delete("/api/models/app/nope").status_code == 404


def test_app_overrides_survive_a_restart(settings: Any) -> None:
    with TestClient(create_app(settings), base_url=HOST) as first:
        first.put("/api/models/app/prism", json={"overrides": {"vision": "acme/vision-7b"}})
    from nimlab.core.registry import reset_default_registry

    reset_default_registry()
    with TestClient(create_app(settings), base_url=HOST) as second:
        vision = roles(second)["apps"]["prism"]["roles"][0]
        assert vision["model"] == "acme/vision-7b" and vision["override"] == "acme/vision-7b"


# ------------------------------------------------------------------------------- run wiring
def arena_models(c: TestClient, **body: Any) -> tuple[list[str], dict[str, Any]]:
    resp = c.post("/api/arena/run", json={"suites": ["factual_qa"], **body})
    assert resp.status_code == 200, resp.text
    detail = finish(c, resp.json()["run_id"])
    assert detail["status"] == "done", detail["error"]
    return detail["result"]["models"], detail["result"]


def test_request_overrides_choose_the_models_and_are_reported(client: TestClient) -> None:
    models, result = arena_models(client, model_overrides={"chat": "req/chat-70b"})
    assert "req/chat-70b" in models and DEFAULTS[Role.CHAT] not in models
    info = result["model_info"]
    assert info["overrides"] == {"chat": "req/chat-70b"}
    assert {"role": "chat", "model": "req/chat-70b", "source": "request"} in info["picked"]
    assert info["substitutions"] == []


def test_precedence_request_over_app_over_global(client: TestClient) -> None:
    client.put("/api/models/overrides", json={"overrides": {"coder": "global/coder"}})
    models, _ = arena_models(client)
    assert "global/coder" in models

    client.put("/api/models/app/arena", json={"overrides": {"coder": "arena/coder"}})
    models, result = arena_models(client)
    assert "arena/coder" in models and "global/coder" not in models
    assert {"role": "coder", "model": "arena/coder", "source": "app"} in result["model_info"][
        "picked"
    ]

    models, result = arena_models(client, model_overrides={"coder": "req/coder"})
    assert "req/coder" in models and "arena/coder" not in models

    # another app is unaffected by arena's override
    assert by_role(roles(client))["coder"]["model"] == "global/coder"


def test_explicit_model_list_still_wins_over_every_override(client: TestClient) -> None:
    models, _ = arena_models(
        client, models=["me/picked-by-hand"], model_overrides={"chat": "req/chat"}
    )
    assert models == ["me/picked-by-hand"]


def test_bad_override_roles_are_422_on_every_run_endpoint(client: TestClient) -> None:
    bad = {"model_overrides": {"wizard": "x/y"}}
    assert client.post("/api/arena/run", json=bad).status_code == 422
    assert client.post("/api/council/run", json={"question": "q?", **bad}).status_code == 422
    assert client.post("/api/atlas/ask", json={"question": "q?", **bad}).status_code == 422
    assert client.post("/api/forge/build", json={"prompt": "x", **bad}).status_code == 422
    assert client.post("/api/sentinel/scan", json=bad).status_code == 422
    assert (
        client.post("/api/distill/run", json={"recipe": "qa_from_docs", **bad}).status_code == 422
    )
    assert client.post("/api/loom/run", json={"id": "doc-to-quiz", **bad}).status_code in (404, 422)


def test_forge_planner_coder_reviewer_fields(client: TestClient) -> None:
    resp = client.post(
        "/api/forge/build",
        json={
            "prompt": "fizzbuzz",
            "planner": "me/planner",
            "coder": "me/coder",
            "reviewer": "me/reviewer",
        },
    )
    detail = finish(client, resp.json()["run_id"])
    assert detail["status"] == "done", detail["error"]
    models = detail["result"]["models"]
    assert (models["planner"], models["coder"], models["reviewer"]) == (
        "me/planner",
        "me/coder",
        "me/reviewer",
    )
    assert detail["result"]["model_info"]["overrides"]["reasoning"] == "me/planner"


def test_atlas_embed_rerank_chat_overrides(client: TestClient) -> None:
    client.post("/api/atlas/ingest", json={"path": str(CORPUS)})
    resp = client.post(
        "/api/atlas/ask",
        json={
            "question": "How do I roll back a deploy?",
            "model_overrides": {"chat": "me/answerer", "embed": "me/embedder"},
        },
    )
    assert resp.status_code == 200, resp.text
    detail = finish(client, resp.json()["run_id"])
    assert detail["status"] == "done", detail["error"]
    assert detail["result"]["model"] == "me/answerer"
    picked = {p["role"]: p["model"] for p in detail["result"]["model_info"]["picked"]}
    assert picked["chat"] == "me/answerer" and picked["embed"] == "me/embedder"


def test_council_chat_override_changes_the_default_panel(client: TestClient) -> None:
    resp = client.post(
        "/api/council/run",
        json={"question": "Tabs or spaces?", "model_overrides": {"chat": "me/panelist"}},
    )
    detail = finish(client, resp.json()["run_id"])
    assert detail["status"] == "done", detail["error"]
    assert "me/panelist" in detail["result"]["models"]


def test_sentinel_guard_and_targets_follow_overrides(client: TestClient) -> None:
    resp = client.post(
        "/api/sentinel/scan",
        json={"suites": ["prompt_injection"], "limit": 1, "model_overrides": {"chat": "me/target"}},
    )
    assert resp.status_code == 200, resp.text
    detail = finish(client, resp.json()["run_id"])
    assert detail["status"] == "done", detail["error"]
    assert "me/target" in detail["result"]["models"]


def test_prism_vision_override_via_form_field(client: TestClient) -> None:
    from nimlab.atlas.server import parse_multipart  # noqa: F401  (same multipart shape as Prism)

    png = Path(__file__).resolve().parents[2] / "examples" / "prism" / "sample_ui.png"
    boundary = "----nimlabtest"
    parts = [
        ("mode", None, b"describe"),
        ("model_overrides", None, json.dumps({"vision": "me/eyes"}).encode()),
        ("images", "sample_ui.png", png.read_bytes()),
    ]
    body = b""
    for name, filename, data in parts:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        body += f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    resp = client.post(
        "/api/prism/run",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 200, resp.text
    detail = finish(client, resp.json()["run_id"])
    assert detail["status"] == "done", detail["error"]
    assert detail["result"]["items"][0]["model"] == "me/eyes"


# ------------------------------------------------------------------------------- live catalog
def live_registry(tmp_path: Path, ids: list[str]) -> ModelRegistry:
    cfg = NIMConfig(api_key=KEY, mock=False)
    return ModelRegistry(
        live=True,
        nim_config=cfg,
        catalog=LiveCatalog(tmp_path / "cache.json"),
        settings_path=tmp_path / "home" / "studio_settings.json",
    )


def test_stale_defaults_notice_and_picker_options_come_from_the_live_catalog(
    tmp_path: Path, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = [i for i in ["nvidia/nemotron-3-ultra-550b-a55b", "google/gemma-4-31b-it",
                        "acme/zeta-coder-32b-instruct", "nvidia/nemotron-3.5-embed-1b"]]  # fmt: skip
    respx_mock.get(f"{BASE}/models").respond(
        200, json={"data": [{"id": i, "object": "model"} for i in live]}
    )
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    reg = live_registry(tmp_path, live)
    set_default_registry(reg)
    with TestClient(create_app(make_settings(tmp_path)), base_url=HOST) as c:
        c.post("/api/models/refresh")
        data = roles(c)
        assert data["live"] is True and data["catalog"]["count"] == 4
        stale = {s["role"]: s for s in data["stale"]}
        # nemotron-3-super (chat) and laguna (coder) are not served here: substitutes are used
        assert stale["chat"]["substitute"] == "google/gemma-4-31b-it"
        assert stale["coder"]["substitute"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert by_role(data)["chat"]["source"] == "auto"
        assert c.get("/api/status").json()["models"]["stale_count"] == len(stale) >= 2
        coder = c.get("/api/models/options", params={"role": "coder"}).json()
        assert coder["live"] is True and coder["note"] == ""
        assert "acme/zeta-coder-32b-instruct" in [m["id"] for m in coder["models"]]
        assert all(m["live"] for m in coder["models"])
        every = c.get("/api/models/options", params={"role": "coder", "all": "1"}).json()
        assert {m["id"] for m in every["models"]} == set(live)


def test_saving_a_key_refreshes_the_catalog(
    tmp_path: Path, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = respx_mock.get(f"{BASE}/models").respond(
        200, json={"data": [{"id": "nvidia/nemotron-3-ultra-550b-a55b"}]}
    )
    monkeypatch.setenv("NIMLAB_CONFIG_DIR", str(tmp_path / "cfg"))
    cfg = NIMConfig(api_key=KEY, mock=False)
    reg = ModelRegistry(
        live=True,
        nim_config=cfg,
        catalog=LiveCatalog(tmp_path / "cache.json"),
        settings_path=tmp_path / "home" / "studio_settings.json",
    )
    set_default_registry(reg)
    with TestClient(create_app(make_settings(tmp_path)), base_url=HOST) as c:
        assert reg.snapshot(fetch=False) is None
        resp = c.post("/api/key", json={"api_key": KEY})
        assert resp.status_code == 200, resp.text
        snap = reg.snapshot(fetch=False)
        assert snap is not None and snap.source == "network"
        assert route.call_count >= 2  # one to verify the key, one to refresh the catalog
        assert roles(c)["live"] is True
        c.delete("/api/key")
        assert roles(c)["live"] is False


def test_models_endpoint_is_unchanged(client: TestClient) -> None:
    rows = client.get("/api/models").json()
    assert rows and {"id", "roles", "cost_tier", "context_window", "notes"} <= set(rows[0])
    assert isinstance(httpx.URL(BASE), httpx.URL)
