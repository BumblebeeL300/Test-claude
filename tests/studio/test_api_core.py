"""Status, key management, models, usage and the run endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from studio_helpers import HOST, events_of, finish, make_settings

from nimlab.studio import credentials as cred
from nimlab.studio.server import create_app

KEY = "nvapi-" + "b" * 60 + "c0de"
MODELS_URL = "https://integrate.api.nvidia.com/v1/models"


class TestStatus:
    def test_shape_in_mock_mode(self, client: TestClient) -> None:
        data = client.get("/api/status").json()
        assert set(data) == {"version", "mode", "key", "base_url", "network", "apps"}
        assert data["mode"] == "mock"
        assert data["key"] == {"set": False, "source": None, "masked": None}
        assert data["network"] == "unknown"
        assert data["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert [a["id"] for a in data["apps"]] == [
            "council",
            "atlas",
            "forge",
            "arena",
            "prism",
            "sentinel",
            "loom",
            "distill",
        ]
        for app in data["apps"]:
            assert set(app) == {
                "id",
                "name",
                "tagline",
                "color",
                "glyph",
                "status",
                "runs",
                "last_run_at",
            }
            assert app["color"].startswith("#") and len(app["color"]) == 7
            assert app["status"] == "ready" and app["runs"] == 0 and app["last_run_at"] is None

    def test_needs_key_is_opt_in(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, require_key=True))
        with TestClient(app, base_url=HOST) as c:
            assert {a["status"] for a in c.get("/api/status").json()["apps"]} == {"needs-key"}

    def test_health(self, client: TestClient) -> None:
        assert client.get("/api/health").json()["ok"] is True

    def test_busy_and_run_counts(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, sim_delay=0.05))
        with TestClient(app, base_url=HOST) as c:
            rid = c.post("/api/forge/build", json={"prompt": "fizzbuzz"}).json()["run_id"]
            apps = {a["id"]: a for a in c.get("/api/status").json()["apps"]}
            assert apps["forge"]["status"] == "busy" and apps["forge"]["runs"] == 1
            assert apps["forge"]["last_run_at"]
            c.post(f"/api/runs/{rid}/cancel")
            finish(c, rid)
            apps = {a["id"]: a for a in c.get("/api/status").json()["apps"]}
            assert apps["forge"]["status"] == "ready"


class TestKey:
    def test_save_and_verify_ok(self, client: TestClient) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.get(MODELS_URL).respond(
                200, json={"data": [{"id": "a/x"}, {"id": "b/y"}]}
            )
            body = client.post("/api/key", json={"api_key": KEY}).json()
        assert route.called
        assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
        assert body["ok"] is True and body["verified"] is True
        assert body["model_count"] == 2 and body["error"] is None
        assert body["key"] == {"set": True, "source": "file", "masked": "nvapi-…c0de"}
        assert cred.load_api_key() == KEY
        status = client.get("/api/status").json()
        assert status["mode"] == "live" and status["network"] == "ok"
        assert status["key"]["masked"] == "nvapi-…c0de"
        assert KEY not in json.dumps(status)

    def test_save_with_unreachable_network_keeps_the_key(self, client: TestClient) -> None:
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).mock(side_effect=httpx.ConnectError("blocked by proxy"))
            body = client.post("/api/key", json={"api_key": KEY}).json()
        assert body["ok"] is True and body["verified"] is False and body["model_count"] is None
        assert "integrate.api.nvidia.com" in body["error"] and "saved" in body["error"]
        assert body["key"]["set"] is True
        assert cred.load_api_key() == KEY
        assert client.get("/api/status").json()["network"] == "blocked"

    def test_rejected_key_is_saved_but_not_verified(self, client: TestClient) -> None:
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).respond(401, json={"detail": "bad key"})
            body = client.post("/api/key", json={"api_key": KEY}).json()
        assert body["verified"] is False and "rejected" in body["error"]
        assert client.get("/api/status").json()["network"] == "ok"

    def test_server_error_reports_reachable(self, client: TestClient) -> None:
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).respond(503, json={"detail": "down"})
            body = client.post("/api/key", json={"api_key": KEY}).json()
        assert body["verified"] is False and "answered with an error" in body["error"]

    def test_odd_format_warns(self, client: TestClient) -> None:
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).respond(200, json={"data": []})
            body = client.post("/api/key", json={"api_key": "not-an-nvidia-key-at-all"}).json()
        assert body["verified"] is True and body["model_count"] == 0
        assert "nvapi-" in body["warning"]

    def test_garbage_key_is_422(self, client: TestClient) -> None:
        assert (
            client.post("/api/key", json={"api_key": "has space inside key 123"}).status_code == 422
        )
        assert client.post("/api/key", json={"api_key": ""}).status_code == 422
        assert client.post("/api/key", json={}).status_code == 422
        assert cred.load_api_key() is None

    def test_test_endpoint_without_key(self, client: TestClient) -> None:
        body = client.post("/api/key/test").json()
        assert body["verified"] is False and "No API key" in body["error"]
        assert body["key"]["set"] is False

    def test_test_endpoint_reverifies_stored_key(self, client: TestClient) -> None:
        cred.save_api_key(KEY)
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).respond(200, json={"data": [{"id": "m"}]})
            body = client.post("/api/key/test").json()
        assert set(body) == {"ok", "verified", "model_count", "error", "warning", "key"}
        assert body["verified"] is True and body["model_count"] == 1

    def test_delete_returns_to_mock(self, client: TestClient) -> None:
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).respond(200, json={"data": []})
            client.post("/api/key", json={"api_key": KEY})
        assert client.get("/api/status").json()["mode"] == "live"
        assert client.delete("/api/key").json()["ok"] is True
        assert cred.load_api_key() is None
        status = client.get("/api/status").json()
        assert status["mode"] == "mock" and status["key"]["set"] is False

    def test_env_key_takes_precedence_and_is_reported(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-" + "e" * 40)
        with respx.mock(assert_all_called=False) as router:
            router.get(MODELS_URL).respond(200, json={"data": []})
            body = client.post("/api/key", json={"api_key": KEY}).json()
        assert body["key"]["source"] == "env"
        assert "environment variable" in body["warning"]


class TestModelsAndUsage:
    def test_models_from_the_registry(self, client: TestClient) -> None:
        rows = client.get("/api/models").json()
        assert len(rows) > 20
        first = rows[0]
        assert set(first) == {"id", "roles", "cost_tier", "context_window", "notes"}
        assert isinstance(first["roles"], list) and first["roles"] == sorted(first["roles"])
        ids = {r["id"] for r in rows}
        assert "nvidia/nemotron-3-super-120b-a12b" in ids

    def test_usage_starts_empty_then_accumulates(self, client: TestClient) -> None:
        empty = client.get("/api/usage").json()
        assert empty["total"]["calls"] == 0 and empty["models"] == {} and empty["rows"] == []
        rid = client.post("/api/council/run", json={"question": "Why is the sky blue?"}).json()[
            "run_id"
        ]
        finish(client, rid)
        usage = client.get("/api/usage").json()
        assert usage["total"]["calls"] > 0 and usage["total"]["total_tokens"] > 0
        model, stats = next(iter(usage["models"].items()))
        assert {"calls", "tokens_in", "tokens_out", "latency_s", "avg_latency_s"} <= set(stats)
        assert usage["rows"][0]["model"] in usage["models"]
        assert usage["runs"]["council"] == 1
        assert model


class TestRuns:
    def test_list_detail_and_404(self, client: TestClient) -> None:
        assert client.get("/api/runs").json() == []
        rid = client.post("/api/council/run", json={"question": "What is a monad?"}).json()[
            "run_id"
        ]
        detail = finish(client, rid)
        assert (
            detail["status"] == "done"
            and detail["app"] == "council"
            and detail["simulated"] is True
        )
        assert detail["finished_at"] and detail["result"]["answer"]
        listing = client.get("/api/runs").json()
        assert listing[0] == {
            k: detail[k]
            for k in ("id", "app", "title", "status", "started_at", "finished_at", "simulated")
        }
        assert client.get("/api/runs/nope").status_code == 404
        assert client.get("/api/runs/nope/events").status_code == 404
        assert client.post("/api/runs/nope/cancel").status_code == 404

    def test_sse_replay_for_late_subscribers(self, client: TestClient) -> None:
        rid = client.post("/api/council/run", json={"question": "Explain TCP slow start"}).json()[
            "run_id"
        ]
        first = events_of(client, rid)
        second = events_of(client, rid)  # the run is over: full replay
        assert first == second
        assert [e["seq"] for e in first] == list(range(1, len(first) + 1))
        assert first[0]["type"] == "run.start" and first[-1]["type"] == "run.done"
        for e in first:
            assert set(e) >= {"seq", "ts", "run_id", "type", "data"} and e["run_id"] == rid

    def test_sse_resume_with_last_event_id(self, client: TestClient) -> None:
        rid = client.post("/api/council/run", json={"question": "Explain TCP slow start"}).json()[
            "run_id"
        ]
        full = events_of(client, rid)
        resumed = events_of(client, rid, headers={"Last-Event-ID": "3"})
        assert [e["seq"] for e in resumed] == [e["seq"] for e in full if e["seq"] > 3]
        # fully caught up on a finished run: 204 tells EventSource to stop reconnecting
        done = client.get(
            f"/api/runs/{rid}/events", headers={"Last-Event-ID": str(full[-1]["seq"])}
        )
        assert done.status_code == 204
        assert client.get(f"/api/runs/{rid}/events?after={full[-1]['seq']}").status_code == 204

    def test_sse_live_stream_while_running(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, sim_delay=0.03))
        with TestClient(app, base_url=HOST) as c:
            rid = c.post("/api/council/run", json={"question": "Why do cats purr?"}).json()[
                "run_id"
            ]
            assert c.get(f"/api/runs/{rid}").json()["status"] in ("queued", "running")
            events = events_of(c, rid)  # subscribes mid-run: replay + live
            assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
            assert events[-1]["type"] == "run.done"
            assert any(e["type"] == "model.end" for e in events)
            assert c.get(f"/api/runs/{rid}").json()["status"] == "done"

    def test_cancel_a_running_run(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, sim_delay=0.2))
        with TestClient(app, base_url=HOST) as c:
            rid = c.post("/api/forge/build", json={"prompt": "todo list"}).json()["run_id"]
            body = c.post(f"/api/runs/{rid}/cancel").json()
            assert body["ok"] is True
            events = events_of(c, rid)
            assert events[-1]["type"] == "run.failed" and events[-1]["data"]["cancelled"] is True
            detail = c.get(f"/api/runs/{rid}").json()
            assert detail["status"] == "cancelled" and detail["error"] == "cancelled"
            again = c.post(f"/api/runs/{rid}/cancel").json()
            assert again["ok"] is False and again["status"] == "cancelled"

    def test_failed_run_reports_the_error(self, client: TestClient) -> None:
        # a question that the council cannot answer with a single model for debate
        rid = client.post(
            "/api/council/run",
            json={
                "question": "x?",
                "strategy": "debate",
                "models": ["nvidia/nemotron-3-super-120b-a12b"],
            },
        ).json()["run_id"]
        detail = finish(client, rid)
        assert detail["status"] == "failed" and detail["error"]
        last = events_of(client, rid)[-1]
        assert last["type"] == "run.failed" and last["data"]["error"] == detail["error"]


class TestHistory:
    def test_runs_survive_a_restart(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        with TestClient(create_app(settings), base_url=HOST) as c:
            rid = c.post("/api/council/run", json={"question": "Why is the sky blue?"}).json()[
                "run_id"
            ]
            finish(c, rid)
        history = json.loads(settings.history_path.read_text())  # type: ignore[union-attr]
        assert history["counts"]["council"] == 1 and history["runs"][0]["id"] == rid
        with TestClient(create_app(settings), base_url=HOST) as c2:
            apps = {a["id"]: a for a in c2.get("/api/status").json()["apps"]}
            assert apps["council"]["runs"] == 1 and apps["council"]["last_run_at"]
            listing = c2.get("/api/runs").json()
            assert [r["id"] for r in listing] == [rid]
            detail = c2.get(f"/api/runs/{rid}").json()
            assert detail["status"] == "done" and detail["result"]["answer"]
            events = events_of(c2, rid)
            assert events[-1]["type"] == "run.done" and events[-1]["data"]["restored"] is True

    def test_interrupted_runs_are_marked_failed(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        assert settings.history_path is not None
        settings.history_path.parent.mkdir(parents=True)
        settings.history_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "counts": {"forge": 3},
                    "runs": [
                        {
                            "id": "abc",
                            "app": "forge",
                            "title": "t",
                            "status": "running",
                            "started_at": "2026-01-01T00:00:00+00:00",
                            "simulated": True,
                        }
                    ],
                }
            )
        )
        with TestClient(create_app(settings), base_url=HOST) as c:
            detail: dict[str, Any] = c.get("/api/runs/abc").json()
            assert detail["status"] == "failed" and "interrupted" in detail["error"]
            assert {a["id"]: a["runs"] for a in c.get("/api/status").json()["apps"]}["forge"] == 3

    def test_corrupt_history_is_ignored(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        assert settings.history_path is not None
        settings.history_path.parent.mkdir(parents=True)
        settings.history_path.write_text("{ nope")
        with TestClient(create_app(settings), base_url=HOST) as c:
            assert c.get("/api/runs").json() == []
