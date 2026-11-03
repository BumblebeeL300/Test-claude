"""Prism's Studio routes, mounted on a bare FastAPI app with the real run manager."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from studio_helpers import HOST, make_settings, parse_sse

from nimlab.core import NIMClient, NIMConfig
from nimlab.prism import check_html
from nimlab.studio import routes_prism
from nimlab.studio.routes_prism import APP_INFO, router
from nimlab.studio.state import StudioState

EXAMPLES = Path(__file__).parents[2] / "examples" / "prism"
UI = EXAMPLES / "sample_ui.png"
CHART = EXAMPLES / "sample_chart.png"


def multipart(
    files: list[tuple[str, bytes]], fields: dict[str, str] | None = None
) -> tuple[bytes, dict[str, str]]:
    boundary = "----prism-test"
    out = b""
    for k, v in (fields or {}).items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for name, data in files:
        out += (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="images"; filename="{name}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    out += f"--{boundary}--\r\n".encode()
    return out, {"content-type": f"multipart/form-data; boundary={boundary}"}


@pytest.fixture
def state(tmp_path: Path) -> StudioState:
    return StudioState(make_settings(tmp_path), client=NIMClient(NIMConfig(mock=True)))


@pytest.fixture
def api(state: StudioState) -> Iterator[TestClient]:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.studio = state
    app.include_router(router)

    @app.get("/api/runs/{run_id}")
    async def detail(run_id: str) -> dict[str, Any]:
        run = state.runs.get(run_id)
        if run is None:
            raise HTTPException(404)
        return run.detail()

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        run = state.runs.get(run_id)
        if run is None:
            raise HTTPException(404)
        return StreamingResponse(run.stream(0, 5.0), media_type="text/event-stream")

    with TestClient(app, base_url=HOST) as client:
        yield client


def start(
    api: TestClient,
    mode: str,
    files: list[Path] | list[tuple[str, bytes]] | None = None,
    **fields: str,
) -> Any:
    chosen = files if files is not None else [UI]
    prepared = [(f.name, f.read_bytes()) if isinstance(f, Path) else f for f in chosen]
    body, headers = multipart(prepared, {"mode": mode, **fields})
    return api.post("/api/prism/run", content=body, headers=headers)


def finish(api: TestClient, resp: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    with api.stream("GET", f"/api/runs/{run_id}/events") as r:
        events = parse_sse(r.iter_lines())
    return api.get(f"/api/runs/{run_id}").json(), events


class TestAppInfo:
    def test_card_fields(self) -> None:
        assert APP_INFO["id"] == "prism" and APP_INFO["name"] == "Prism"
        assert APP_INFO["color"] == "#B07CFF" and APP_INFO["tagline"] and APP_INFO["glyph"]
        taken = {"#FF3D9A", "#22E1FF", "#C6FF3D", "#FF8A3D", "#8B7BFF"}
        assert APP_INFO["color"].upper() not in taken

    def test_router_prefix(self) -> None:
        assert router.prefix == "/api/prism"
        assert {r.path for r in router.routes} == {"/api/prism/modes", "/api/prism/run"}  # type: ignore[attr-defined]


class TestModes:
    def test_lists_modes_options_and_schemas(self, api: TestClient) -> None:
        data = api.get("/api/prism/modes").json()
        names = [m["name"] for m in data["modes"]]
        assert names == ["describe", "ocr", "extract", "ui2code", "compare", "chart", "ask"]
        by = {m["name"]: m for m in data["modes"]}
        assert by["compare"]["min_images"] == 2 and by["ui2code"]["options"][0]["name"] == "refine"
        assert {s["name"] for s in data["schemas"]} >= {"receipt", "invoice", "business_card"}
        assert data["simulated"] is True and data["model"].endswith("vision-instruct")
        assert isinstance(data["pillow"], bool) and data["vision_models"]


class TestRun:
    def test_event_stream_for_describe(self, api: TestClient) -> None:
        detail, events = finish(api, start(api, "describe"))
        assert detail["status"] == "done" and detail["app"] == "prism" and detail["simulated"]
        types = [e["type"] for e in events]
        assert types[0] == "run.start" and types[-1] == "run.done"
        assert types.count("stage.start") == types.count("stage.end") == 2
        assert types.index("stage.start") < types.index("result") < types.index("run.done")
        stages = [e["data"]["stage"] for e in events if e["type"] == "stage.start"]
        assert stages == ["prepare", "describe"]
        done = events[-1]["data"]["result"]
        assert done["mode"] == "describe" and done["batch"] is False and done["simulated"] is True
        item = done["items"][0]
        assert item["output"]["alt_text"] and item["images"][0]["name"] == "sample_ui.png"
        assert next(e["data"] for e in events if e["type"] == "result")["mode"] == "describe"
        assert detail["result"] == done

    def test_ocr_streams_tokens(self, api: TestClient) -> None:
        _, events = finish(api, start(api, "ocr"))
        tokens = [e["data"] for e in events if e["type"] == "token"]
        assert len(tokens) > 10 and {t["stage"] for t in tokens} == {"read"}
        text = events[-1]["data"]["result"]["items"][0]["output"]["text"]
        assert "".join(t["text"] for t in tokens) == text

    def test_ui2code_with_refine(self, api: TestClient) -> None:
        detail, events = finish(api, start(api, "ui2code", refine="2"))
        stages = [e["data"]["stage"] for e in events if e["type"] == "stage.start"]
        assert stages == ["prepare", "draft", "critique 1", "refine 1", "critique 2", "refine 2"]
        out = detail["result"]["items"][0]["output"]
        assert out["valid"] and out["drafts"] == 3 and check_html(out["html"]) == []
        assert len(out["critiques"]) == 2
        phases = {e.get("phase") for e in events if e["type"] == "token"}
        assert phases == {"draft", "refine 1", "refine 2"}

    def test_extract_with_builtin_and_inline_schema(self, api: TestClient) -> None:
        detail, _ = finish(api, start(api, "extract", schema="invoice"))
        out = detail["result"]["items"][0]["output"]
        assert out["schema_name"] == "invoice" and out["valid"] and out["data"]["line_items"]
        inline = json.dumps(
            {"type": "object", "properties": {"who": {"type": "string"}}, "required": ["who"]}
        )
        detail, _ = finish(api, start(api, "extract", schema=inline))
        assert detail["result"]["items"][0]["output"]["data"]["who"]

    def test_chart_ask_and_compare(self, api: TestClient) -> None:
        chart, _ = finish(api, start(api, "chart", [CHART]))
        assert chart["result"]["items"][0]["output"]["csv"].startswith("Period,")
        ask, _ = finish(api, start(api, "ask", question="What colours?"))
        assert "navy" in ask["result"]["items"][0]["output"]["answer"]
        cmp_, _ = finish(api, start(api, "compare", [UI, CHART]))
        item = cmp_["result"]["items"][0]
        assert cmp_["result"]["batch"] is False and len(item["images"]) == 2
        assert {d["area"] for d in item["output"]["differences"]} >= {"dimensions"}

    def test_several_images_become_a_batch(self, api: TestClient) -> None:
        detail, events = finish(api, start(api, "ocr", [UI, CHART]))
        res = detail["result"]
        assert res["batch"] is True and detail["status"] == "done"
        assert [i["images"][0]["name"] for i in res["items"]] == [
            "sample_ui.png",
            "sample_chart.png",
        ]
        assert all(i["mode"] == "ocr" and i["output"]["text"] for i in res["items"])
        results = [e["data"] for e in events if e["type"] == "result"]
        assert sorted(r["index"] for r in results) == [0, 1]
        assert {r["name"] for r in results} == {"sample_ui.png", "sample_chart.png"}

    def test_a_failure_in_one_image_is_kept_as_an_item(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = routes_prism.Prism.run

        async def flaky(self: Any, mode: str, images: Any, **kw: Any) -> Any:
            if getattr(images, "name", "") == "sample_chart.png":
                raise RuntimeError("rate limited")
            return await real(self, mode, images, **kw)

        monkeypatch.setattr(routes_prism.Prism, "run", flaky)
        detail, _ = finish(api, start(api, "describe", [UI, CHART]))
        ok, bad = detail["result"]["items"]
        assert ok["mode"] == "describe" and bad == {
            "name": "sample_chart.png",
            "error": "rate limited",
        }

    def test_run_is_counted_and_titled(self, api: TestClient, state: StudioState) -> None:
        finish(api, start(api, "describe"))
        assert state.runs.counts["prism"] == 1
        assert state.runs.recent(1)[0].title == "describe: sample_ui.png"

    def test_model_field_is_used(self, api: TestClient) -> None:
        detail, _ = finish(api, start(api, "describe", model="microsoft/phi-3.5-vision-instruct"))
        assert detail["result"]["items"][0]["model"] == "microsoft/phi-3.5-vision-instruct"

    def test_failing_run_ends_with_run_failed(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: Any, *a: Any, **k: Any) -> Any:
            raise RuntimeError("model fell over")

        monkeypatch.setattr(routes_prism.Prism, "run", boom)
        detail, events = finish(api, start(api, "describe"))
        assert detail["status"] == "failed" and "model fell over" in detail["error"]
        assert events[-1]["type"] == "run.failed"


class TestValidation:
    def test_unknown_mode(self, api: TestClient) -> None:
        r = start(api, "paint")
        assert r.status_code == 422 and "unknown mode" in r.json()["detail"]

    def test_missing_mode(self, api: TestClient) -> None:
        body, headers = multipart([("a.png", UI.read_bytes())])
        assert api.post("/api/prism/run", content=body, headers=headers).status_code == 422

    def test_no_image(self, api: TestClient) -> None:
        body, headers = multipart([], {"mode": "describe"})
        r = api.post("/api/prism/run", content=body, headers=headers)
        assert r.status_code == 400 and "no image" in r.json()["detail"]

    def test_not_multipart(self, api: TestClient) -> None:
        assert api.post("/api/prism/run", json={"mode": "describe"}).status_code == 415

    def test_bad_image(self, api: TestClient) -> None:
        r = start(api, "describe", [("note.png", b"%PDF-1.4")])
        assert r.status_code == 422 and "unsupported image type" in r.json()["detail"]

    def test_one_bad_image_rejects_the_whole_upload(self, api: TestClient) -> None:
        r = start(api, "ocr", [("good.png", UI.read_bytes()), ("bad.png", b"junk")])
        assert r.status_code == 422 and "bad.png" in r.json()["detail"]

    @pytest.mark.parametrize(
        ("mode", "files", "fields", "match"),
        [
            ("ask", [UI], {}, "'question' is required"),
            ("compare", [UI], {}, "exactly 2"),
            ("compare", [UI, CHART, UI], {}, "exactly 2"),
            ("ui2code", [UI], {"refine": "99"}, "between 0 and 5"),
            ("extract", [UI], {"schema": "bogus"}, "schema must be one of"),
            ("extract", [UI], {"schema": "/etc/passwd"}, "schema must be one of"),
            ("extract", [UI], {"schema": "{broken"}, "not valid JSON"),
        ],
    )
    def test_option_errors_are_422(
        self, api: TestClient, mode: str, files: list[Path], fields: dict[str, str], match: str
    ) -> None:
        r = start(api, mode, files, **fields)
        assert r.status_code == 422 and match in r.json()["detail"]

    def test_too_many_images(self, api: TestClient) -> None:
        files = [(f"{i}.png", UI.read_bytes()) for i in range(routes_prism.MAX_IMAGES + 1)]
        assert start(api, "describe", files).status_code == 422

    def test_upload_limit(self, api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(routes_prism, "MAX_UPLOAD_BYTES", 300)
        assert start(api, "describe").status_code == 413

    def test_nothing_is_started_on_a_rejected_request(
        self, api: TestClient, state: StudioState
    ) -> None:
        start(api, "ask")
        start(api, "nope")
        assert state.runs.counts.get("prism", 0) == 0 and not state.runs.runs
