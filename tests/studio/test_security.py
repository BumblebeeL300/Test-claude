"""Auth token, Host/Origin checks and static file serving."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from studio_helpers import HOST, make_settings, parse_sse

from nimlab.studio.security import host_allowed, token_ok
from nimlab.studio.server import create_app

TOKEN = "t0ken-for-tests-1234567890"


@pytest.fixture
def web(tmp_path: Path) -> Path:
    root = tmp_path / "web"
    (root / "js").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>shell</title>")
    (root / "js" / "app.js").write_text("console.log('hi')")
    (tmp_path / "secret.txt").write_text("OUTSIDE")
    return root


@pytest.fixture
def secured(tmp_path: Path, web: Path) -> Iterator[TestClient]:
    app = create_app(make_settings(tmp_path, token=TOKEN, web_dir=web))
    with TestClient(app, base_url=HOST) as c:
        yield c


class TestToken:
    def test_api_requires_the_token(self, secured: TestClient) -> None:
        for path in ("/api/status", "/api/runs", "/api/models", "/api/usage", "/api/projects"):
            resp = secured.get(path)
            assert resp.status_code == 401, path
            assert "token" in resp.json()["detail"]
        assert secured.post("/api/forge/build", json={"prompt": "x"}).status_code == 401
        assert secured.delete("/api/key").status_code == 401

    def test_header_and_query_param_work(self, secured: TestClient) -> None:
        assert secured.get("/api/status", headers={"X-Studio-Token": TOKEN}).status_code == 200
        assert secured.get("/api/status", params={"t": TOKEN}).status_code == 200

    @pytest.mark.parametrize("bad", ["", "nope", TOKEN[:-1], TOKEN + "x", TOKEN.upper()])
    def test_wrong_tokens_are_refused(self, secured: TestClient, bad: str) -> None:
        assert secured.get("/api/status", headers={"X-Studio-Token": bad}).status_code == 401
        assert secured.get("/api/status", params={"t": bad}).status_code == 401

    def test_health_is_open(self, secured: TestClient) -> None:
        resp = secured.get("/api/health")
        assert resp.status_code == 200 and resp.json()["ok"] is True

    def test_sse_accepts_the_query_param(self, secured: TestClient) -> None:
        rid = secured.post(
            "/api/council/run",
            json={"question": "Why is the sky blue?"},
            headers={"X-Studio-Token": TOKEN},
        ).json()["run_id"]
        assert secured.get(f"/api/runs/{rid}/events").status_code == 401
        with secured.stream("GET", f"/api/runs/{rid}/events", params={"t": TOKEN}) as resp:
            events = parse_sse(resp.iter_lines())
        assert events[-1]["type"] == "run.done"

    def test_unknown_api_paths_need_the_token_too(self, secured: TestClient) -> None:
        assert secured.get("/api/nope").status_code == 401
        assert secured.get("/api/nope", headers={"X-Studio-Token": TOKEN}).status_code == 404

    def test_token_can_be_disabled_for_tests(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, token=None))
        with TestClient(app, base_url=HOST) as c:
            assert c.get("/api/status").status_code == 200

    def test_each_app_gets_a_distinct_random_token(self, tmp_path: Path) -> None:
        from nimlab.studio.settings import StudioSettings

        a, b = StudioSettings(home=tmp_path), StudioSettings(home=tmp_path)
        assert a.token and b.token and a.token != b.token and len(a.token) >= 40


class TestHostAndOrigin:
    @pytest.mark.parametrize(
        "host",
        [
            "evil.example.com",
            "evil.example.com:8765",
            "127.0.0.1.evil.com",
            "10.0.0.5:80",
            "localhost.evil.com",
        ],
    )
    def test_rebinding_hosts_are_refused(self, tmp_path: Path, host: str) -> None:
        app = create_app(make_settings(tmp_path, token=None))
        with TestClient(app, base_url=f"http://{host}") as c:
            assert c.get("/api/health").status_code == 403
            assert c.get("/").status_code == 403

    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "127.0.0.1:51234", "localhost", "localhost:8765", "[::1]:8765"]
    )
    def test_loopback_hosts_are_fine(self, tmp_path: Path, host: str) -> None:
        app = create_app(make_settings(tmp_path, token=None))
        with TestClient(app, base_url=f"http://{host}") as c:
            assert c.get("/api/health").status_code == 200

    def test_cross_origin_requests_are_refused_even_with_the_token(
        self, secured: TestClient
    ) -> None:
        resp = secured.get(
            "/api/status", headers={"X-Studio-Token": TOKEN, "Origin": "https://evil.example.com"}
        )
        assert resp.status_code == 403
        ok = secured.get(
            "/api/status", headers={"X-Studio-Token": TOKEN, "Origin": "http://127.0.0.1:8765"}
        )
        assert ok.status_code == 200

    def test_no_cors_headers_are_ever_sent(self, secured: TestClient) -> None:
        resp = secured.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert not any(h.lower().startswith("access-control-") for h in resp.headers)
        pre = secured.options(
            "/api/status",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert pre.status_code in (401, 405)

    def test_host_check_can_be_disabled(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, token=None, check_host=False))
        with TestClient(app, base_url="http://testserver") as c:
            assert c.get("/api/health").status_code == 200

    def test_helpers(self) -> None:
        assert host_allowed("127.0.0.1:80", ("127.0.0.1",)) and not host_allowed(
            None, ("127.0.0.1",)
        )
        assert host_allowed("LOCALHOST", ("localhost",))
        assert token_ok(None, None, None) and not token_ok("a", None, None)
        assert token_ok("a", "a", None) and token_ok("a", None, "a") and not token_ok("a", "b", "c")


class TestStatic:
    def test_index_assets_and_spa_fallback(self, secured: TestClient) -> None:
        assert "<title>shell</title>" in secured.get("/").text  # no token needed for the shell
        js = secured.get("/js/app.js")
        assert js.status_code == 200 and "console.log" in js.text
        assert "javascript" in js.headers["content-type"]
        assert "<title>shell</title>" in secured.get("/settings/profile").text
        assert secured.get("/js/missing.js").status_code == 404

    def test_security_headers(self, secured: TestClient) -> None:
        resp = secured.get("/")
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["referrer-policy"] == "no-referrer"
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["cache-control"] == "no-cache"

    @pytest.mark.parametrize(
        "path",
        [
            "/../secret.txt",
            "/%2e%2e/secret.txt",
            "/js/../../secret.txt",
            "/..%2fsecret.txt",
            "/%2e%2e%2fsecret.txt",
        ],
    )
    def test_static_path_traversal(self, secured: TestClient, path: str) -> None:
        resp = secured.get(path)
        assert "OUTSIDE" not in resp.text

    def test_placeholder_when_the_ui_is_missing(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, web_dir=tmp_path / "nowhere"))
        with TestClient(app, base_url=HOST) as c:
            assert "NIM Lab Studio" in c.get("/").text
