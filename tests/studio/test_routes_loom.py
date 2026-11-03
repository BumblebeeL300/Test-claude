"""Loom's Studio routes, mounted on a bare FastAPI app (and once on the real Studio app)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from studio_helpers import HOST, events_of, finish, make_settings

from nimlab.studio import routes_loom
from nimlab.studio.routes_loom import APP_INFO, router
from nimlab.studio.server import create_app

TAKEN = {"#FF3D9A", "#22E1FF", "#C6FF3D", "#FF8A3D", "#8B7BFF", "#FF3EA5", "#2DE2E6", "#FFA62B"}


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("NIMLAB_LOOM_DIR", str(tmp_path / "saved"))
    monkeypatch.setenv("NIMLAB_LOOM_CACHE", str(tmp_path / "cache"))
    bare = FastAPI()
    bare.include_router(router)
    return bare


@pytest.fixture
def c(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


def wait(app: FastAPI, run_id: str, timeout: float = 15.0) -> Any:
    run = app.state.loom.runs.get(run_id)
    deadline = time.monotonic() + timeout
    while not run.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    assert run.finished, "run did not finish"
    return run


def simple(template: str = "Say {{word}}") -> dict[str, Any]:
    return {
        "name": "simple",
        "nodes": [
            {"id": "word", "type": "input", "params": {"type": "string", "default": "hi"}},
            {"id": "a", "type": "prompt", "params": {"template": template}},
            {"id": "out", "type": "output", "params": {"name": "answer"}},
        ],
        "edges": [{"from": "word.value", "to": "a.word"}, {"from": "a.text", "to": "out.value"}],
    }


# ------------------------------------------------------------------ APP_INFO
def test_app_info() -> None:
    assert APP_INFO["id"] == "loom" and APP_INFO["name"] == "Loom" and APP_INFO["glyph"]
    color = APP_INFO["color"].upper()
    assert len(color) == 7 and color.startswith("#") and color not in TAKEN
    assert APP_INFO["tagline"]


# ------------------------------------------------------------------ node types
def test_node_types_have_param_schemas(c: TestClient) -> None:
    types = c.get("/api/loom/node-types").json()
    by = {t["type"]: t for t in types}
    wanted = {"input", "prompt", "json", "embed", "rerank", "branch", "map", "merge", "join", "template",
              "regex", "http_get", "python_fn", "council", "atlas_ask", "output"}  # fmt: skip
    assert wanted <= set(by)
    for t in types:
        assert t["params"]["type"] == "object" and isinstance(t["params"]["properties"], dict)
        assert {"title", "category", "description", "inputs", "outputs"} <= set(t)
    prompt = by["prompt"]
    assert prompt["params"]["required"] == ["template"]
    assert prompt["params"]["properties"]["template"]["format"] == "textarea"
    assert prompt["params"]["properties"]["role"]["enum"][0] == "chat"
    assert prompt["input_vars"][0] == {"path": "template", "kind": "template"}
    assert by["branch"]["outputs"] == [
        {"name": "then", "type": "any"},
        {"name": "else", "type": "any"},
    ]
    assert by["branch"]["input_vars"] == [{"path": "condition", "kind": "expr"}]
    assert "flatten" in by["python_fn"]["params"]["properties"]["fn"]["enum"]
    assert by["http_get"]["enabled"] is False


# ------------------------------------------------------------------ saved pipelines
def test_list_and_get_bundled(c: TestClient) -> None:
    items = c.get("/api/loom/pipelines").json()
    ids = {i["id"] for i in items}
    assert {"research-brief", "support-triage", "doc-to-quiz", "translate-backcheck"} <= ids
    assert all(i["source"] == "bundled" and i["readonly"] for i in items)
    got = c.get("/api/loom/pipelines/support-triage").json()
    assert got["readonly"] is True and got["source"] == "bundled"
    assert all("pos" in n for n in got["pipeline"]["nodes"])
    assert c.get("/api/loom/pipelines/ghost").status_code == 404


def test_save_get_list_delete_roundtrip(c: TestClient, tmp_path: Path) -> None:
    saved = c.put("/api/loom/pipelines/my-flow", json={"pipeline": simple()})
    assert saved.status_code == 200 and saved.json()["created"] is True
    assert (tmp_path / "saved" / "my-flow.json").is_file()  # honours NIMLAB_LOOM_DIR
    listing = c.get("/api/loom/pipelines").json()
    mine = next(i for i in listing if i["id"] == "my-flow")
    assert mine["source"] == "user" and mine["readonly"] is False and mine["nodes"] == 3
    got = c.get("/api/loom/pipelines/my-flow").json()
    assert got["pipeline"]["nodes"][1]["params"]["template"] == "Say {{word}}"
    again = c.put(
        "/api/loom/pipelines/my-flow", json=simple("Shout {{word}}")
    )  # bare document works too
    assert again.status_code == 200 and again.json()["created"] is False
    assert (
        c.get("/api/loom/pipelines/my-flow").json()["pipeline"]["nodes"][1]["params"]["template"]
        == "Shout {{word}}"
    )
    assert c.delete("/api/loom/pipelines/my-flow").json() == {"ok": True, "id": "my-flow"}
    assert c.get("/api/loom/pipelines/my-flow").status_code == 404
    assert c.delete("/api/loom/pipelines/my-flow").status_code == 404


def test_save_rejects_bad_requests(c: TestClient) -> None:
    assert (
        c.put("/api/loom/pipelines/research-brief", json=simple()).status_code == 409
    )  # bundled id
    assert c.put("/api/loom/pipelines/Bad Id", json=simple()).status_code == 422
    assert c.put("/api/loom/pipelines/../x", json=simple()).status_code in (404, 405, 422)
    broken = c.put("/api/loom/pipelines/ok-id", json={"name": "x", "nodes": [{"id": "a"}]})
    assert broken.status_code == 422 and "invalid pipeline" in broken.json()["detail"]
    assert c.delete("/api/loom/pipelines/research-brief").status_code == 403
    # a pipeline with lint errors may still be saved (work in progress) and reports them
    wip = simple("Say {{word}} {{missing}}")
    result = c.put("/api/loom/pipelines/wip", json=wip).json()
    assert result["saved"] is True and result["errors"] >= 1


def test_saved_pipelines_default_to_the_studio_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NIMLAB_LOOM_DIR", raising=False)
    studio = create_app(make_settings(tmp_path))
    studio.router.routes[0:0] = _routes_of(router)
    with TestClient(studio, base_url=HOST) as client:
        assert client.put("/api/loom/pipelines/home-flow", json=simple()).status_code == 200
    assert (tmp_path / "home" / "loom" / "home-flow.json").is_file()


# ------------------------------------------------------------------ lint & export
def test_lint_endpoint(c: TestClient) -> None:
    clean = c.post("/api/loom/lint", json={"pipeline": simple()}).json()
    assert clean["ok"] is True and clean["issues"] == []
    assert clean["ports"]["a"]["inputs"] == [{"name": "word", "type": "any"}]
    bad = c.post(
        "/api/loom/lint", json=simple("Say {{word}} {{missing}}")
    ).json()  # bare document also accepted
    assert (
        bad["ok"] is False
        and bad["issues"][0]["code"] == "unbound"
        and bad["issues"][0]["node"] == "a"
    )
    schema = c.post(
        "/api/loom/lint", json={"pipeline": {"name": "x", "nodes": [{"id": "9", "type": "input"}]}}
    ).json()
    assert schema["ok"] is False and schema["issues"][0]["code"] == "schema"


def test_export_endpoint(c: TestClient) -> None:
    mer = c.post("/api/loom/export", json={"pipeline": simple(), "format": "mermaid"}).json()
    assert mer["text"].startswith("%% ") and "flowchart LR" in mer["text"]
    py = c.post("/api/loom/export", json={"pipeline": simple(), "format": "python"}).json()
    compile(py["text"], "x.py", "exec")
    assert (
        c.post("/api/loom/export", json={"pipeline": simple(), "format": "pdf"}).status_code == 422
    )


# ------------------------------------------------------------------ runs
def test_run_by_id_streams_engine_events(c: TestClient, app: FastAPI) -> None:
    res = c.post("/api/loom/run", json={"id": "support-triage"})
    assert res.status_code == 200
    run = wait(app, res.json()["run_id"])
    assert run.app == "loom" and run.status == "done" and run.simulated is True
    types = [e["type"] for e in run.events]
    assert types[0] == "run.start" and types[1] == "run.plan" and types[-1] == "run.done"
    assert types.count("run.done") == 1  # the manager's terminal event, not a duplicate
    assert {"node.start", "node.end", "node.skipped", "node.token"} <= set(types)
    end = next(e for e in run.events if e["type"] == "node.end" and e["data"]["node"] == "classify")
    assert end["data"]["preview"] and end["data"]["latency_ms"] >= 0 and "tokens" in end["data"]
    result = run.result
    assert (
        result["ok"] is True
        and result["outputs"]["reply"]
        and result["nodes"]["classify"]["status"] == "done"
    )


def test_run_inline_pipeline_with_inputs(c: TestClient, app: FastAPI) -> None:
    res = c.post("/api/loom/run", json={"pipeline": simple(), "inputs": {"word": "zebra"}})
    run = wait(app, res.json()["run_id"])
    assert run.status == "done" and run.result["outputs"]["answer"].strip()
    second = wait(
        app,
        c.post("/api/loom/run", json={"pipeline": simple(), "inputs": {"word": "zebra"}}).json()[
            "run_id"
        ],
    )
    assert second.result["cache_hits"] >= 1  # the on-disk cache is on for Studio runs


def test_run_rejects_bad_input_before_starting(c: TestClient, app: FastAPI) -> None:
    unknown = c.post("/api/loom/run", json={"pipeline": simple(), "inputs": {"nope": 1}})
    assert unknown.status_code == 422 and "unknown input" in unknown.json()["detail"]
    lint_errors = c.post("/api/loom/run", json={"pipeline": simple("Say {{word}} {{missing}}")})
    assert lint_errors.status_code == 422 and lint_errors.json()["issues"][0]["code"] == "unbound"
    cyc = simple()
    cyc["edges"].append({"from": "a.text", "to": "a.word"})
    assert c.post("/api/loom/run", json={"pipeline": cyc}).status_code == 422
    assert c.post("/api/loom/run", json={}).status_code == 422
    assert c.post("/api/loom/run", json={"id": "ghost"}).status_code == 404
    assert c.post("/api/loom/run", json={"pipeline": simple(), "inputs": [1]}).status_code == 422
    assert not hasattr(app.state, "loom") or not app.state.loom.runs.runs  # nothing was started


def test_a_failing_node_fails_the_run(c: TestClient, app: FastAPI) -> None:
    broken = {
        "name": "broken",
        "nodes": [
            {"id": "t", "type": "input", "params": {"type": "string", "default": "abc"}},
            {"id": "rx", "type": "regex", "params": {"pattern": "("}},
            {"id": "out", "type": "output"},
        ],
        "edges": [{"from": "t.value", "to": "rx.text"}, {"from": "rx.match", "to": "out.value"}],
    }
    run = wait(app, c.post("/api/loom/run", json={"pipeline": broken}).json()["run_id"])
    assert run.status == "failed" and "rx" in run.error and "invalid regex" in run.error
    assert any(e["type"] == "node.error" and e["data"]["node"] == "rx" for e in run.events)
    assert run.events[-1]["type"] == "run.failed"


# ------------------------------------------------------------------ real Studio integration
def _routes_of(r: Any) -> list[Any]:
    """The routes ``include_router`` adds (version independent), to insert before the catch-all."""
    scratch = FastAPI()
    before = len(scratch.router.routes)
    scratch.include_router(r)
    return list(scratch.router.routes[before:])


def test_mounted_on_studio_uses_its_run_manager_and_sse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NIMLAB_LOOM_CACHE", raising=False)
    studio = create_app(make_settings(tmp_path))
    studio.router.routes[0:0] = _routes_of(router)  # before the /api catch-all
    with TestClient(studio, base_url=HOST) as client:
        run_id = client.post("/api/loom/run", json={"id": "translate-backcheck"}).json()["run_id"]
        events = events_of(client, run_id)
        assert events[0]["type"] == "run.start" and events[-1]["type"] == "run.done"
        assert "node.end" in {e["type"] for e in events}
        detail = finish(client, run_id)
        assert detail["app"] == "loom" and detail["result"]["ok"] is True
        assert any(r["id"] == run_id for r in client.get("/api/runs").json())
    assert (tmp_path / "home" / "loom" / "cache").is_dir()  # the Studio cache lives under its home


def test_module_exports() -> None:
    assert routes_loom.router.prefix == "/api/loom"
