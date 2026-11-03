"""The Distill router mounted on a bare FastAPI app (no Studio server needed)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nimlab.studio.routes_distill import APP_INFO, router

DOCS = Path(__file__).parents[2] / "examples" / "distill" / "docs"


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    application = FastAPI()
    application.include_router(router)
    application.state.distill_runs_dir = tmp_path / "distill-runs"
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def wait_done(c: TestClient, run_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = c.get(f"/api/distill/runs/{run_id}")
        if resp.status_code == 200 and resp.json()["status"] in ("done", "failed"):
            body: dict[str, Any] = resp.json()
            return body
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def start(c: TestClient, recipe: str = "instruction_diversity", n: int = 40, **kw: Any) -> str:
    resp = c.post("/api/distill/run", json={"recipe": recipe, "n": n, **kw})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["simulated"] is True
    return str(body["run_id"])


def test_app_info_shape() -> None:
    assert APP_INFO["id"] == "distill"
    assert APP_INFO["color"].lower() == "#ffc83d"
    taken = {"#ff3d9a", "#22e1ff", "#c6ff3d", "#ff8a3d", "#8b7bff"}
    assert APP_INFO["color"].lower() not in taken
    assert {"name", "tagline", "glyph"} <= set(APP_INFO)
    assert router.prefix == "/api/distill"


def test_recipes_endpoint(client: TestClient) -> None:
    recipes = client.get("/api/distill/recipes").json()
    by_name = {r["name"]: r for r in recipes}
    assert set(by_name) == {"qa_from_docs", "instruction_diversity", "preference_pairs", "eval_set"}
    assert by_name["qa_from_docs"]["needs_docs"] is True
    assert "dpo" in by_name["preference_pairs"]["formats"]
    assert "dpo" not in by_name["eval_set"]["formats"]
    assert by_name["preference_pairs"]["min_models"] == 2


def test_run_lifecycle_and_events(app: FastAPI, client: TestClient) -> None:
    run_id = start(client, n=40, models=["meta/llama-3.3-70b-instruct", "meta/llama-3.1-8b-instruct"])
    detail = wait_done(client, run_id)
    assert detail["status"] == "done" and detail["simulated"] is True
    assert detail["result"]["n_final"] > 0 and len(detail["result"]["funnel"]) == 5

    # events went through the (fallback) Studio run manager
    run = app.state.distill_manager.get(run_id)
    types = [e["type"] for e in run.events]
    assert types[0] == "run.start" and types[-1] == "run.done"
    for expected in (
        "stage.start", "stage.end", "sample.generated", "sample.scored", "sample.dropped", "funnel",
    ):  # fmt: skip
        assert expected in types
    dropped = next(e for e in run.events if e["type"] == "sample.dropped")
    assert dropped["data"]["reason"] and dropped["data"]["stage"]
    scored = next(e for e in run.events if e["type"] == "sample.scored")
    assert isinstance(scored["data"]["score"], float)
    final = run.events[-1]["data"]["result"]
    assert final["run_id"] == run_id and final["histograms"] and final["files"]
    assert run.app == "distill" and run.status == "done"

    runs = client.get("/api/distill/runs").json()
    assert [r["id"] for r in runs] == [run_id] and runs[0]["status"] == "done"


def test_dedup_cluster_events(app: FastAPI, client: TestClient) -> None:
    run_id = start(client, "qa_from_docs", n=60, options={"docs": str(DOCS)})
    wait_done(client, run_id)
    events = app.state.distill_manager.get(run_id).events
    clusters = [e["data"] for e in events if e["type"] == "dedup.cluster"]
    assert clusters and all(c["size"] >= 2 and c["kept"] in c["members"] for c in clusters)


def test_samples_endpoint_stages_and_paging(client: TestClient) -> None:
    run_id = start(client, n=50)
    detail = wait_done(client, run_id)
    kept = detail["result"]["n_final"]

    final = client.get(f"/api/distill/runs/{run_id}/samples", params={"limit": 5}).json()
    assert final["total"] == kept and len(final["items"]) == 5 and final["stage"] == "final"
    item = final["items"][0]
    assert item["alive"] and item["score"] is not None and item["prompt"] and item["response"]

    page2 = client.get(
        f"/api/distill/runs/{run_id}/samples", params={"limit": 5, "offset": 5}
    ).json()
    assert {i["id"] for i in page2["items"]}.isdisjoint({i["id"] for i in final["items"]})

    dropped = client.get(f"/api/distill/runs/{run_id}/samples", params={"stage": "dropped"}).json()
    assert dropped["total"] == 50 - kept
    assert all(not i["alive"] and i["reason"] and i["dropped_at"] for i in dropped["items"])
    reason = dropped["items"][0]["reason"]
    only = client.get(
        f"/api/distill/runs/{run_id}/samples", params={"stage": "dropped", "reason": reason}
    ).json()
    assert 0 < only["total"] <= dropped["total"]
    assert all(i["reason"] == reason for i in only["items"])

    scored = client.get(f"/api/distill/runs/{run_id}/samples", params={"stage": "score"}).json()
    assert scored["total"] >= kept
    assert client.get(f"/api/distill/runs/{run_id}/samples", params={"stage": "all"}).json()["total"] == 50
    bad = client.get(f"/api/distill/runs/{run_id}/samples", params={"stage": "nope"})
    assert bad.status_code == 422
    assert client.get(f"/api/distill/runs/{run_id}/samples", params={"limit": 0}).status_code == 422


@pytest.mark.parametrize("fmt", ["openai-chat", "alpaca", "sharegpt"])
def test_export_download(client: TestClient, fmt: str) -> None:
    run_id = start(client, n=40)
    kept = wait_done(client, run_id)["result"]["n_final"]
    resp = client.get(f"/api/distill/runs/{run_id}/export", params={"format": fmt})
    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"] and fmt in resp.headers["content-disposition"]
    lines = resp.text.splitlines()
    assert len(lines) == kept
    first = json.loads(lines[0])
    assert {"openai-chat": "messages", "alpaca": "instruction", "sharegpt": "conversations"}[fmt] in first


def test_dpo_export_for_preference_runs(client: TestClient) -> None:
    run_id = start(client, "preference_pairs", n=40)
    wait_done(client, run_id)
    resp = client.get(f"/api/distill/runs/{run_id}/export", params={"format": "dpo"})
    assert resp.status_code == 200
    row = json.loads(resp.text.splitlines()[0])
    assert set(row) == {"prompt", "chosen", "rejected"}
    default = client.get(f"/api/distill/runs/{run_id}/export")  # defaults to the run's format
    assert default.status_code == 200 and default.text == resp.text


def test_report_and_card_endpoints(client: TestClient) -> None:
    run_id = start(client, n=30)
    wait_done(client, run_id)
    report = client.get(f"/api/distill/runs/{run_id}/report")
    assert report.status_code == 200 and "Pipeline funnel" in report.text
    assert report.headers["content-type"].startswith("text/html")
    card = client.get(f"/api/distill/runs/{run_id}/card")
    assert "Dataset card" in card.text and "SIMULATED" in card.text


def test_validation_errors(client: TestClient) -> None:
    def post(**body: Any) -> Any:
        return client.post("/api/distill/run", json=body)

    assert post(recipe="nope").status_code == 422
    assert "unknown recipe" in post(recipe="nope").json()["detail"]
    assert post(recipe="eval_set", n=0).status_code == 422
    assert post(recipe="eval_set", n=100000).status_code == 422
    assert post(recipe="eval_set", options={"bogus": 1}).status_code == 422
    assert post(recipe="eval_set", format="dpo").status_code == 422
    assert post(recipe="eval_set", format="parquet").status_code == 422
    assert post(recipe="eval_set", options={"concurrency": 0}).status_code == 422
    assert post(n=5).status_code == 422  # recipe missing


def test_unknown_run_and_unsafe_ids(client: TestClient) -> None:
    for path in ("", "/samples", "/export", "/report", "/card"):
        assert client.get(f"/api/distill/runs/ghost{path}").status_code == 404
    assert client.get("/api/distill/runs/..%2f..%2fetc/samples").status_code in (404, 422)


def test_export_of_an_unfinished_run_is_refused(app: FastAPI, client: TestClient, tmp_path: Path) -> None:
    import asyncio

    from nimlab.distill import Distiller

    runs = app.state.distill_runs_dir
    asyncio.run(
        Distiller(mock=True, runs_dir=runs).run(
            "instruction_diversity", n=10, run_id="half", stop_after="generate"
        )
    )
    assert client.get("/api/distill/runs/half/export").status_code == 409
    assert client.get("/api/distill/runs/half").json()["status"] == "interrupted"


def test_runs_list_is_empty_initially_and_newest_first(client: TestClient) -> None:
    assert client.get("/api/distill/runs").json() == []
    first = start(client, n=10)
    wait_done(client, first)
    second = start(client, "eval_set", n=10)
    wait_done(client, second)
    ids = [r["id"] for r in client.get("/api/distill/runs").json()]
    assert set(ids) == {first, second}
