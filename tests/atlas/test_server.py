"""FastAPI app: ingest, JSON query, SSE streaming, collections, health, UI and error mapping."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nimlab.atlas import server
from nimlab.atlas.server import create_app, parse_multipart
from nimlab.core import NIMAuthError, NIMConfig, NIMError, NIMRateLimitError

from .conftest import FakeClient

HEARTBEAT_Q = (
    "How often do robots send heartbeats, and how many missed heartbeats mark a robot as lost?"
)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((name, data))
    return events


def upload(client: TestClient, corpus: Path, collection: str = "default", **params):
    files = [
        ("files", (p.name, p.read_bytes(), "text/markdown")) for p in sorted(corpus.glob("*.md"))
    ]
    return client.post("/ingest", files=files, data={"collection": collection}, params=params)


@pytest.fixture
def web(db_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(db_path, nim_config=NIMConfig(mock=True))) as client:
        yield client


@pytest.fixture
def loaded(web: TestClient, corpus_dir: Path) -> TestClient:
    assert upload(web, corpus_dir).status_code == 200
    return web


# ----------------------------------------------------------------------------- basics


def test_health_reports_mode_and_models(web: TestClient, db_path: Path):
    body = web.get("/health").json()
    assert body["status"] == "ok" and body["mock"] is True and body["db"] == str(db_path)
    assert body["collections"] == 0 and body["models"]["embed"]


def test_the_ui_is_one_self_contained_page(web: TestClient):
    response = web.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    html = response.text
    for marker in (
        "/query/stream",
        "EventSource",
        "prefers-color-scheme",
        "viewport",
        "@media (max-width",
    ):
        assert marker in html
    assert not re.search(r"""(src|href)\s*=\s*["']https?:""", html)  # no CDNs or remote assets
    assert not re.search(r"url\(\s*['\"]?https?:", html) and "<link" not in html
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_collections_start_empty(web: TestClient):
    assert web.get("/collections").json() == {"collections": []}


def test_app_works_without_the_lifespan_context(db_path: Path, corpus_dir: Path):
    client = TestClient(create_app(db_path, nim_config=NIMConfig(mock=True)))  # no `with`
    assert client.get("/health").status_code == 200
    assert upload(client, corpus_dir).status_code == 200
    assert client.post("/query", json={"question": "API rate limit?"}).json()["sources"]


# ----------------------------------------------------------------------------- ingest


def test_ingest_multipart_uploads_and_lists_collections_and_documents(
    web: TestClient, corpus_dir: Path
):
    response = upload(web, corpus_dir, "handbook")
    body = response.json()
    assert response.status_code == 200 and body["collection"] == "handbook"
    assert body["added"] == 5 and body["errors"] == 0 and body["chunks"] > 20
    assert {f["name"] for f in body["files"]} >= {"api_guide.md", "glossary.md"}
    (collection,) = web.get("/collections").json()["collections"]
    assert (
        collection["name"] == "handbook"
        and collection["documents"] == 5
        and collection["dim"] == 1024
    )
    docs = web.get("/documents", params={"collection": "handbook"}).json()["documents"]
    assert {d["name"] for d in docs} == {p.name for p in corpus_dir.glob("*.md")}
    again = upload(web, corpus_dir, "handbook").json()
    assert again["unchanged"] == 5 and again["added"] == 0


def test_ingest_collection_may_be_given_as_a_query_parameter(web: TestClient):
    response = web.post(
        "/ingest?collection=viaquery",
        files=[("files", ("a.md", b"# A\n\nHello there robots.", "text/markdown"))],
    )
    assert response.json()["collection"] == "viaquery"


def test_ingest_accepts_binary_pdf_uploads(web: TestClient, make_pdf):
    pdf = make_pdf([["The charging dock sits at the end of aisle twelve."]])
    response = web.post("/ingest", files=[("files", ("manual.pdf", pdf, "application/pdf"))])
    assert response.status_code == 200 and response.json()["added"] == 1
    answer = web.post("/query", json={"question": "where is the charging dock"}).json()
    assert answer["sources"][0]["page"] == 1 and answer["sources"][0]["source"] == "manual.pdf"


def test_ingest_json_documents(web: TestClient):
    payload = {
        "collection": "notes",
        "documents": [{"name": "n.md", "text": "# N\n\nNote about docks."}],
    }
    response = web.post("/ingest", json=payload)
    assert response.status_code == 200 and response.json()["added"] == 1


def test_ingest_mixed_batch_reports_unsupported_files_as_skipped(web: TestClient):
    files = [
        ("files", ("ok.md", b"# T\n\nGood content about robots.", "text/markdown")),
        ("files", ("pic.png", b"\x89PNG", "image/png")),
    ]
    body = web.post("/ingest", files=files).json()
    status = {f["name"]: f["status"] for f in body["files"]}
    assert status == {"ok.md": "added", "pic.png": "skipped"}


def test_ingest_error_responses(web: TestClient, monkeypatch: pytest.MonkeyPatch):
    only_png = [("files", ("pic.png", b"x", "image/png"))]
    assert web.post("/ingest", files=only_png).status_code == 415
    assert (
        web.post("/ingest", content=b"plain", headers={"content-type": "text/plain"}).status_code
        == 415
    )
    assert web.post("/ingest", json={"nope": 1}).status_code == 400
    assert web.post("/ingest", data={"collection": "x"}, files={}).status_code in (400, 415)
    ok = [("files", ("a.md", b"# A\n\ntext here", "text/markdown"))]
    assert web.post("/ingest", files=ok, data={"collection": "bad name!"}).status_code == 422
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    assert web.post("/ingest", files=ok).status_code == 413


def test_ingest_sanitises_upload_filenames(web: TestClient):
    files = [("files", ("../../etc/evil.md", b"# E\n\nSneaky path content.", "text/markdown"))]
    body = web.post("/ingest", files=files).json()
    assert body["files"][0]["name"] == "evil.md" and body["files"][0]["path"] == "upload://evil.md"


def test_multipart_parser_is_binary_safe_and_handles_quoted_boundaries():
    payload = b"line1\r\n\r\nstill the same part\r\n--not-a-boundary\x00\xff"
    body = (
        b'--"b-1"\r\nContent-Disposition: form-data; name="collection"\r\n\r\ndocs\r\n'
        b'--"b-1"\r\nContent-Disposition: form-data; name="files"; filename="a.bin"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + payload + b'\r\n--"b-1"\r\n'
        b'Content-Disposition: form-data; name="files"; filename=""\r\n\r\n\r\n--"b-1"--\r\n'
    )
    fields, files = parse_multipart(
        'multipart/form-data; boundary="b-1"', body.replace(b'"b-1"', b"b-1")
    )
    assert fields == {"collection": "docs"}
    assert files == [("a.bin", payload)]
    with pytest.raises(ValueError, match="boundary"):
        parse_multipart("multipart/form-data", b"")


# ----------------------------------------------------------------------------- query


def test_query_returns_answer_sources_and_verification(loaded: TestClient):
    response = loaded.post("/query", json={"question": HEARTBEAT_Q, "verify": True, "k": 4})
    body = response.json()
    assert response.status_code == 200
    assert body["sources"][0]["source"] == "architecture.md" and len(body["sources"]) == 4
    assert body["sources"][0]["label"].startswith("architecture.md > ")
    assert body["verification"]["method"] == "lexical" and body["verification"]["claims"]
    assert body["mode"] == "simple" and body["text"] and body["trace"][0]["kind"] == "retrieve"
    assert any("Mock mode" in w for w in body["warnings"])


def test_query_agentic_and_default_options(loaded: TestClient):
    body = loaded.post(
        "/query",
        json={
            "question": "What is needed before on-call? Who is paged after no acknowledgement?",
            "mode": "agentic",
        },
    ).json()
    assert body["mode"] == "agentic" and body["trace"][0]["kind"] == "plan"
    assert body["verification"] is None and body["guard"] == []


def test_query_validation(loaded: TestClient):
    assert loaded.post("/query", json={"question": ""}).status_code == 422
    assert loaded.post("/query", json={"question": "x", "k": 0}).status_code == 422
    assert loaded.post("/query", json={"question": "x", "k": 99}).status_code == 422
    assert loaded.post("/query", json={"question": "x", "mode": "turbo"}).status_code == 422
    assert loaded.post("/query", json={"question": "x", "collection": "../etc"}).status_code == 422
    assert loaded.post("/query", json={}).status_code == 422


def test_query_unknown_collection_is_a_friendly_answer_not_an_error(loaded: TestClient):
    body = loaded.post("/query", json={"question": "anything", "collection": "ghost"}).json()
    assert "No documents are indexed" in body["text"]


def test_delete_collection(loaded: TestClient):
    assert loaded.delete("/collections/default").json() == {"deleted": "default"}
    assert loaded.delete("/collections/default").status_code == 404
    assert loaded.get("/collections").json() == {"collections": []}


# ----------------------------------------------------------------------------- streaming


def test_stream_emits_trace_sources_tokens_and_a_final_event(loaded: TestClient):
    with loaded.stream(
        "GET", "/query/stream", params={"q": HEARTBEAT_Q, "verify": "true"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        text = "".join(response.iter_text())
    events = parse_sse(text)
    names = [name for name, _ in events]
    assert names[-2:] == ["final", "done"] and names.count("final") == 1
    assert (
        names.index("trace") < names.index("sources") < names.index("token") < names.index("final")
    )
    final = events[-2][1]
    assert final["sources"][0]["source"] == "architecture.md" and "citations" in final
    assert final["verification"]["claims"]
    streamed = "".join(data["text"] for name, data in events if name == "token")
    assert streamed.strip() == final["text"]
    sources_event = next(data for name, data in events if name == "sources")
    assert sources_event["sources"][0]["n"] == 1


def test_stream_validation_and_unknown_collection(loaded: TestClient):
    assert loaded.get("/query/stream").status_code == 422
    assert loaded.get("/query/stream", params={"q": "x", "mode": "bogus"}).status_code == 422
    events = parse_sse(loaded.get("/query/stream", params={"q": "hi", "collection": "ghost"}).text)
    assert events[-2][0] == "final" and "No documents" in events[-2][1]["text"]


def test_stream_reports_upstream_failures_as_an_error_event(db_path: Path, corpus_dir: Path):
    client = FakeClient(chat="ok")
    with TestClient(create_app(db_path, client=client)) as web:
        upload(web, corpus_dir)
        client.rerank_error = NIMError("reranker exploded")
        # reranker failures degrade gracefully; a chat failure is fatal for the stream:
        client._chat = lambda model, messages: (_ for _ in ()).throw(NIMError("model exploded"))
        events = parse_sse(web.get("/query/stream", params={"q": HEARTBEAT_Q}).text)
    names = [name for name, _ in events]
    assert names[-2:] == ["error", "done"] and "model exploded" in events[-2][1]["message"]
    assert "final" not in names


# ----------------------------------------------------------------------------- scripted client


@pytest.fixture
def live_web(db_path: Path, corpus_dir: Path):
    def make(client: FakeClient) -> TestClient:
        web = TestClient(create_app(db_path, client=client))
        web.__enter__()
        assert upload(web, corpus_dir).status_code == 200
        return web

    created: list[TestClient] = []

    def factory(client: FakeClient) -> TestClient:
        web = make(client)
        created.append(web)
        return web

    yield factory
    for web in created:
        web.__exit__(None, None, None)


def test_scripted_answer_citations_reach_the_json_response(live_web):
    def chat(model, messages):
        n = int(re.search(r"^\[(\d+)\] architecture\.md", messages[0]["content"], re.M).group(1))
        return f"Heartbeats are sent every 2 seconds [{n}]."

    web = live_web(FakeClient(chat=chat))
    body = web.post("/query", json={"question": HEARTBEAT_Q}).json()
    assert [c["source"] for c in body["citations"]] == ["architecture.md"]
    assert body["citations"][0]["n"] in {s["n"] for s in body["sources"]}
    assert not any("Mock mode" in w for w in body["warnings"])
    assert web.get("/health").json()["mock"] is False


def test_guarded_query_is_blocked_through_the_api(live_web):
    def chat(model, messages):
        if messages[0]["content"].startswith("Task: Check if there is unsafe content"):
            return '{"User Safety": "unsafe", "Safety Categories": "Violence"}'
        return "never reached"

    web = live_web(FakeClient(chat=chat))
    body = web.post("/query", json={"question": "something awful", "guard": True}).json()
    assert (
        body["blocked"] is True and "Violence" in body["text"] and body["guard"][0]["safe"] is False
    )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NIMAuthError("bad key"), 502),
        (NIMError("upstream broke"), 502),
        (NIMRateLimitError("slow down", retry_after=7), 429),
    ],
)
def test_upstream_errors_map_to_http_statuses(live_web, error: Exception, status: int):
    client = FakeClient(chat="ok")
    web = live_web(client)

    def boom(model, messages):
        raise error

    client._chat = boom
    response = web.post("/query", json={"question": HEARTBEAT_Q})
    assert response.status_code == status and "detail" in response.json()
    if status == 429:
        assert response.headers["retry-after"] == "7"


def test_mixing_backends_in_one_collection_is_a_409(db_path: Path, corpus_dir: Path):
    with TestClient(create_app(db_path, nim_config=NIMConfig(mock=True))) as mock_web:
        assert upload(mock_web, corpus_dir).status_code == 200
    with TestClient(create_app(db_path, client=FakeClient())) as live:
        response = upload(live, corpus_dir)
    assert response.status_code == 409 and "reindex" in response.json()["detail"]
