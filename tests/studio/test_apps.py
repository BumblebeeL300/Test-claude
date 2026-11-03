"""Council, Atlas and Arena through the Studio API (mock backend + simulation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from studio_helpers import HOST, events_of, finish, make_settings

from nimlab.studio.server import create_app

CORPUS = Path(__file__).parents[2] / "examples" / "atlas_corpus"
PANEL = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "poolside/laguna-xs-2.1",
]


def multipart(
    files: list[tuple[str, bytes]], fields: dict[str, str] | None = None
) -> tuple[bytes, str]:
    boundary = "----studio-test"
    out = b""
    for k, v in (fields or {}).items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for name, data in files:
        out += (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    out += f"--{boundary}--\r\n".encode()
    return out, f"multipart/form-data; boundary={boundary}"


class TestCouncil:
    @pytest.mark.parametrize("strategy", ["moa", "debate", "best-of-n", "critic"])
    def test_every_strategy_runs_in_simulation(self, client: TestClient, strategy: str) -> None:
        rid = client.post(
            "/api/council/run",
            json={
                "question": "Is a monorepo right for a small team?",
                "strategy": strategy,
                "models": PANEL,
            },
        ).json()["run_id"]
        detail = finish(client, rid)
        assert detail["status"] == "done", detail["error"]
        events = events_of(client, rid)
        types = [e["type"] for e in events]
        assert types[0] == "run.start" and types[-1] == "run.done"
        assert {"stage.start", "stage.end", "model.start", "model.end"} <= set(types)
        assert types.index("stage.start") < types.index("model.start") < types.index("model.end")
        assert types.count("stage.start") == types.count("stage.end")
        result = detail["result"]
        assert result["strategy"] == strategy and result["simulated"] is True
        assert len(result["answer"]) > 80 and "Final answer regarding" not in result["answer"]
        assert result["candidates"] and all(
            {"label", "model", "text", "score", "selected", "outlier"} <= set(c)
            for c in result["candidates"]
        )
        assert result["rationale"]
        assert result["agreement"] is None or 0 <= result["agreement"] <= 1

    def test_event_payloads(self, client: TestClient) -> None:
        rid = client.post(
            "/api/council/run",
            json={"question": "Why is the sky blue?", "strategy": "moa", "models": PANEL},
        ).json()["run_id"]
        events = events_of(client, rid)
        starts = [e["data"] for e in events if e["type"] == "model.start"]
        ends = [e["data"] for e in events if e["type"] == "model.end"]
        assert {s["model"] for s in starts} == set(PANEL)
        assert len(starts) == len(ends) and all(e["ok"] for e in ends)
        assert all(len(e["preview"]) > 40 for e in ends)  # a readable preview of each reply
        stages = [e["data"]["stage"] for e in events if e["type"] == "stage.start"]
        assert stages[0] == "Proposers" and "Final" in " ".join(stages)
        consensus = [e["data"] for e in events if e["type"] == "consensus"]
        assert consensus and all(0 <= c["agreement"] <= 1 and "stage" in c for c in consensus)

    def test_model_replies_differ_and_are_not_core_filler(self, client: TestClient) -> None:
        rid = client.post(
            "/api/council/run", json={"question": "How should we version an API?", "models": PANEL}
        ).json()["run_id"]
        result = finish(client, rid)["result"]
        texts = [c["text"] for c in result["candidates"]]
        assert len(set(texts)) == len(texts)
        assert all("handful of key points" not in t for t in texts)

    def test_unknown_strategy_is_422(self, client: TestClient) -> None:
        resp = client.post("/api/council/run", json={"question": "hi?", "strategy": "vibes"})
        assert resp.status_code == 422 and "moa" in resp.json()["detail"]
        assert client.post("/api/council/run", json={"question": ""}).status_code == 422

    def test_strategies_listing(self, client: TestClient) -> None:
        names = {s["name"] for s in client.get("/api/council/strategies").json()}
        assert {"moa", "debate", "best-of-n", "critic"} <= names


class TestAtlas:
    def test_empty_collections(self, client: TestClient) -> None:
        assert client.get("/api/atlas/collections").json() == []

    def test_ingest_a_folder_by_path(self, client: TestClient, settings: Any) -> None:
        body = client.post(
            "/api/atlas/ingest", json={"path": str(CORPUS), "collection": "docs"}
        ).json()
        assert body["collection"] == "docs" and body["added"] == 5 and body["errors"] == 0
        assert body["chunks"] > 5 and body["ok"] is True
        (coll,) = client.get("/api/atlas/collections").json()
        assert (
            coll["name"] == "docs" and coll["documents"] == 5 and coll["chunks"] == body["chunks"]
        )
        assert {f["name"] for f in coll["files"]} >= {"api_guide.md", "glossary.md"}
        assert settings.atlas_db.is_file()  # lives under the Studio home, not the cwd
        again = client.post(
            "/api/atlas/ingest", json={"path": str(CORPUS), "collection": "docs"}
        ).json()
        assert again["unchanged"] == 5 and again["added"] == 0

    def test_ingest_multipart_uploads(self, client: TestClient) -> None:
        data, ctype = multipart(
            [
                (
                    "notes.md",
                    b"# Notes\n\nThe deploy window is Tuesday 10:00 UTC. Rollbacks need approval.",
                ),
                ("image.png", b"\x89PNG"),
            ],
            {"collection": "up"},
        )
        body = client.post(
            "/api/atlas/ingest", content=data, headers={"content-type": ctype}
        ).json()
        assert body["collection"] == "up" and body["added"] == 1
        names = {f["name"]: f["status"] for f in body["files"]}
        assert names == {"notes.md": "added", "image.png": "skipped"}

    def test_ingest_errors(self, client: TestClient) -> None:
        assert (
            client.post("/api/atlas/ingest", json={"path": "/definitely/not/here"}).status_code
            == 404
        )
        assert client.post("/api/atlas/ingest", json={}).status_code == 400
        assert (
            client.post(
                "/api/atlas/ingest", content=b"x", headers={"content-type": "text/plain"}
            ).status_code
            == 415
        )
        data, ctype = multipart([("a.exe", b"MZ")])
        resp = client.post("/api/atlas/ingest", content=data, headers={"content-type": ctype})
        assert resp.status_code == 415 and ".md" in resp.json()["detail"]
        data, ctype = multipart([("a.md", b"# hi\n\ntext text text")], {"collection": "bad name!"})
        assert (
            client.post(
                "/api/atlas/ingest", content=data, headers={"content-type": ctype}
            ).status_code
            == 422
        )

    def test_ask_streams_tokens_then_sources_then_result(self, client: TestClient) -> None:
        client.post("/api/atlas/ingest", json={"path": str(CORPUS)})
        rid = client.post(
            "/api/atlas/ask",
            json={
                "question": "What is the API rate limit?",
                "collection": "default",
                "mode": "simple",
                "verify": True,
            },
        ).json()["run_id"]
        events = events_of(client, rid)
        types = [e["type"] for e in events]
        assert types[0] == "run.start" and types[-1] == "run.done"
        assert "token" in types and "sources" in types
        assert types.index("sources") < types.index("token")  # context first, then the text streams
        streamed = "".join(e["data"]["text"] for e in events if e["type"] == "token")
        result = events[-1]["data"]["result"]
        assert streamed.strip() == result["answer"].strip() or result["answer"].strip() in streamed
        assert result["simulated"] is True and result["mode"] == "simple"
        assert result["citations"] and {"n", "source", "text", "path"} <= set(
            result["citations"][0]
        )
        assert "[" in result["answer"] and result["groundedness"] is not None
        assert result["groundedness"] >= 0.5
        assert result["verification"]["claims"]
        sources = next(e["data"]["sources"] for e in events if e["type"] == "sources")
        assert len(sources) >= 1 and sources[0]["n"] == 1

    def test_agentic_mode_and_no_collection(self, client: TestClient) -> None:
        client.post("/api/atlas/ingest", json={"path": str(CORPUS)})
        rid = client.post(
            "/api/atlas/ask",
            json={
                "question": "How do we handle a Sev1 incident and who is on call?",
                "mode": "agentic",
            },
        ).json()["run_id"]
        detail = finish(client, rid)
        assert detail["status"] == "done" and detail["result"]["mode"] == "agentic"
        assert any(e["type"] == "trace" for e in events_of(client, rid))
        empty = client.post(
            "/api/atlas/ask", json={"question": "anything?", "collection": "nothing-here"}
        ).json()["run_id"]
        res = finish(client, empty)["result"]
        assert res["sources"] == [] and "no documents are indexed" in res["answer"].lower()

    def test_ask_validation(self, client: TestClient) -> None:
        assert (
            client.post("/api/atlas/ask", json={"question": "x", "mode": "weird"}).status_code
            == 422
        )
        assert (
            client.post(
                "/api/atlas/ask", json={"question": "x", "collection": "no spaces"}
            ).status_code
            == 422
        )
        assert client.post("/api/atlas/ask", json={"question": ""}).status_code == 422


class TestArena:
    def test_route(self, client: TestClient) -> None:
        d = client.post(
            "/api/arena/route", json={"prompt": "Write a Python function that merges intervals"}
        ).json()
        assert d["category"] == "coding" and d["model"]
        assert d["candidates"] and {
            "model",
            "utility",
            "quality_term",
            "latency_penalty",
            "cost_penalty",
        } <= set(d["candidates"][0])
        assert (
            d["reasons"]
            and d["profile"] == "balanced"
            and d["source"] in ("registry", "leaderboard")
        )
        math = client.post(
            "/api/arena/route",
            json={"prompt": "Solve for x: 3x + 7 = 22. Show the algebra step by step."},
        ).json()
        assert math["category"] == "math_reasoning"
        assert client.post("/api/arena/route", json={"prompt": ""}).status_code == 422

    def test_run_and_leaderboards(self, client: TestClient) -> None:
        assert client.get("/api/arena/runs").json() == []
        assert "math_reasoning" in client.get("/api/arena/suites").json()
        rid = client.post(
            "/api/arena/run",
            json={
                "suites": ["math_reasoning"],
                "models": ["nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b"],
            },
        ).json()["run_id"]
        detail = finish(client, rid)
        assert detail["status"] == "done", detail["error"]
        events = events_of(client, rid)
        types = [e["type"] for e in events]
        assert types[:2] == ["run.start", "plan"] and "item" in types and types[-1] == "run.done"
        items = [e["data"] for e in events if e["type"] == "item"]
        assert items[-1]["done"] == items[-1]["total"] == len(items)
        result = detail["result"]
        assert result["simulated"] is True and result["status"] == "complete"
        ranks = [s["rank"] for s in result["standings"]]
        assert (
            ranks == [1, 2]
            and result["standings"][0]["overall"] >= result["standings"][1]["overall"]
        )
        (summary,) = client.get("/api/arena/runs").json()
        assert summary["run_id"] == result["run_id"] and summary["suites"] == ["math_reasoning"]
        assert [s["model"] for s in summary["standings"]] == [
            s["model"] for s in result["standings"]
        ]
        # the router now uses the saved leaderboard
        d = client.post(
            "/api/arena/route",
            json={"prompt": "Solve for x: 3x + 7 = 22. Show the algebra step by step."},
        ).json()
        assert d["run_id"] is not None and d["source"] == "leaderboard"

    def test_default_models_and_validation(self, client: TestClient) -> None:
        assert client.post("/api/arena/run", json={"suites": ["nope"]}).status_code == 422
        assert client.post("/api/arena/run", json={"suites": ["/etc/passwd"]}).status_code == 422
        rid = client.post("/api/arena/run", json={"suites": "summarization"}).json()["run_id"]
        plan = events_of(client, rid)[1]["data"]
        assert len(plan["models"]) >= 3 and plan["suites"] == ["summarization"]
        finish(client, rid)


def test_runs_can_overlap(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    with TestClient(app, base_url=HOST) as c:
        ids = [
            c.post("/api/council/run", json={"question": f"Question number {i}?"}).json()["run_id"]
            for i in range(3)
        ]
        assert [finish(c, r)["status"] for r in ids] == ["done"] * 3
        assert len(c.get("/api/runs").json()) == 3


async def test_ask_reindexes_a_collection_embedded_by_another_backend(tmp_path: Path) -> None:
    """Mock-ingested collection + a different embedding space: Studio re-embeds once and retries."""
    from nimlab.atlas.models import StoreError
    from nimlab.studio.atlas_app import start_ask
    from nimlab.studio.state import StudioState

    state = StudioState(make_settings(tmp_path))
    await state.atlas().ingest(CORPUS, "docs")
    atlas = state.atlas()
    real = atlas.ask_stream
    calls = {"ask": 0, "reindex": 0}

    async def flaky(*a: Any, **kw: Any) -> Any:
        calls["ask"] += 1
        if calls["ask"] == 1:
            raise StoreError("query vector has dimension 3; re-embed the collection")
        async for ev in real(*a, **kw):
            yield ev

    real_reindex = atlas.reindex

    async def counting_reindex(name: str) -> int:
        calls["reindex"] += 1
        return await real_reindex(name)

    atlas.ask_stream = flaky
    atlas.reindex = counting_reindex
    run = start_ask(state, "What is the API rate limit?", "docs", "simple", False)
    await run.task  # type: ignore[misc]
    assert run.status == "done" and run.result and run.result["citations"]
    assert calls == {"ask": 2, "reindex": 1}
    await state.shutdown()
