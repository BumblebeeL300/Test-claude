"""FastAPI app: ingest, query (JSON + SSE streaming), collections, health and the web UI."""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nimlab import __version__
from nimlab.core import (
    NIMAuthError,
    NIMClient,
    NIMConfig,
    NIMError,
    NIMRateLimitError,
)

from .config import DEFAULT_COLLECTION, DEFAULT_DB_PATH, AtlasConfig
from .engine import Atlas
from .loaders import SUPPORTED_EXTENSIONS, media_type_for
from .models import Answer, AtlasError, IngestReport, Mode, StoreError

UI_PATH = Path(__file__).with_name("ui") / "index.html"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'"
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    collection: str = DEFAULT_COLLECTION
    mode: Mode = "simple"
    k: int | None = Field(default=None, ge=1, le=20)
    verify: bool = False
    revise: bool = False
    guard: bool = False
    hyde: bool | None = None


# --------------------------------------------------------------------------- multipart


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    """Minimal ``multipart/form-data`` parser: ``(fields, [(filename, bytes)])``.

    Implemented on the standard library so the server needs no extra dependency.
    """
    m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type, re.IGNORECASE)
    if not m:
        raise ValueError("multipart request without a boundary")
    boundary = (m.group(1) or m.group(2)).encode()
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for part in body.split(b"--" + boundary)[1:]:
        if part.startswith(b"--"):
            break
        part = part[2:] if part.startswith(b"\r\n") else part
        head, sep, payload = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        payload = payload[:-2] if payload.endswith(b"\r\n") else payload
        disposition = ""
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line
        name = re.search(r'\bname="([^"]*)"', disposition)
        filename = re.search(r'\bfilename="([^"]*)"', disposition)
        if filename is not None:
            if filename.group(1):
                files.append((filename.group(1), payload))
        elif name:
            fields[name.group(1)] = payload.decode("utf-8", "replace")
    return fields, files


def _safe_name(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name.strip() or "upload"


def _check_collection(name: str) -> str:
    if not _COLLECTION_RE.match(name):
        raise HTTPException(
            422, "collection names use letters, digits, '.', '_' or '-' (max 64 characters)"
        )
    return name


# --------------------------------------------------------------------------- app


class _Runtime:
    """Lazily creates (and later closes) the shared client and :class:`Atlas`."""

    def __init__(
        self,
        db_path: str | Path,
        client: Any | None,
        config: AtlasConfig | None,
        nim_config: NIMConfig | None,
    ) -> None:
        self.db_path = db_path
        self.config = config
        self.nim_config = nim_config
        self.client = client
        self.owns_client = client is None
        self.atlas: Atlas | None = None

    def get(self) -> Atlas:
        if self.atlas is None:
            if self.client is None:
                self.client = NIMClient(self.nim_config)
            self.atlas = Atlas(self.client, self.db_path, config=self.config)
        return self.atlas

    async def close(self) -> None:
        if self.atlas is not None:
            self.atlas.close()
            self.atlas = None
        if self.client is not None and self.owns_client:
            await self.client.aclose()
            self.client = None


def create_app(
    db_path: str | Path | None = None,
    *,
    client: Any | None = None,
    config: AtlasConfig | None = None,
    nim_config: NIMConfig | None = None,
) -> FastAPI:
    """Build the Atlas web app.

    ``db_path`` defaults to ``$NIMLAB_ATLAS_DB`` or ``.nimlab/atlas.db``. Pass ``client`` to share
    an existing NIM client (it is then not closed on shutdown); otherwise one is created from
    ``nim_config`` / the environment, which means mock mode whenever no API key is set.
    """
    path = db_path or os.environ.get("NIMLAB_ATLAS_DB") or DEFAULT_DB_PATH
    runtime = _Runtime(path, client, config, nim_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.get()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Atlas", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime

    def atlas() -> Atlas:
        return runtime.get()

    # -- error mapping
    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(AtlasError)
    async def _atlas_error(_: Request, exc: AtlasError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(NIMRateLimitError)
    async def _rate_limited(_: Request, exc: NIMRateLimitError) -> JSONResponse:
        headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else None
        return JSONResponse({"detail": str(exc)}, status_code=429, headers=headers)

    @app.exception_handler(NIMAuthError)
    async def _auth_error(_: Request, exc: NIMAuthError) -> JSONResponse:
        return JSONResponse({"detail": f"upstream authentication failed: {exc}"}, status_code=502)

    @app.exception_handler(NIMError)
    async def _nim_error(_: Request, exc: NIMError) -> JSONResponse:
        return JSONResponse({"detail": f"upstream model error: {exc}"}, status_code=502)

    # -- routes
    @app.get("/health")
    async def health() -> dict[str, Any]:
        a = atlas()
        return {
            "status": "ok",
            "version": __version__,
            "mock": a.mock,
            "db": a.store.path,
            "collections": len(a.collections()),
            "models": a.stats()["models"],
        }

    @app.get("/collections")
    async def collections() -> dict[str, Any]:
        return {
            "collections": [
                {
                    "name": c.name,
                    "documents": c.documents,
                    "chunks": c.chunks,
                    "embed_model": c.embed_model,
                    "dim": c.dim,
                }
                for c in atlas().collections()
            ]
        }

    @app.get("/documents")
    async def documents(collection: str = DEFAULT_COLLECTION) -> dict[str, Any]:
        docs = atlas().documents(_check_collection(collection))
        return {
            "collection": collection,
            "documents": [
                {
                    "name": d.name,
                    "title": d.title,
                    "type": d.media_type,
                    "chunks": d.n_chunks,
                    "size": d.size,
                    "ingested_at": d.ingested_at,
                }
                for d in docs
            ],
        }

    @app.delete("/collections/{name}")
    async def delete_collection(name: str) -> dict[str, Any]:
        if not atlas().delete_collection(_check_collection(name)):
            raise HTTPException(404, f"unknown collection {name!r}")
        return {"deleted": name}

    @app.post("/ingest", response_model=IngestReport)
    async def ingest(request: Request, collection: str | None = None, force: bool = False) -> IngestReport:
        ctype = request.headers.get("content-type", "")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"upload too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
        body = await request.body()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"upload too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
        files: list[tuple[str, bytes]] = []
        if ctype.lower().startswith("multipart/form-data"):
            try:
                fields, files = parse_multipart(ctype, body)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            collection = collection or fields.get("collection")
        elif "json" in ctype.lower():
            try:
                payload = json.loads(body or b"{}")
                docs = payload["documents"]
                collection = collection or payload.get("collection")
                files = [(str(d["name"]), str(d["text"]).encode("utf-8")) for d in docs]
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(
                    400, 'JSON body must look like {"collection": "...", "documents": '
                    '[{"name": "a.md", "text": "..."}]}'
                ) from exc
        else:
            raise HTTPException(415, "send multipart/form-data (files) or application/json")
        collection = _check_collection(collection or DEFAULT_COLLECTION)
        if not files:
            raise HTTPException(400, "no files in the request")
        accepted = [(_safe_name(n), d) for n, d in files if media_type_for(n)]
        rejected = [_safe_name(n) for n, _ in files if not media_type_for(n)]
        if not accepted:
            raise HTTPException(
                415, f"unsupported file type; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        report = await atlas().ingest_bytes(accepted, collection, force=force)
        for name in rejected:
            from .models import FileReport

            report.files.append(
                FileReport(path=f"upload://{name}", name=name, status="skipped", message="unsupported type")
            )
        return report

    @app.post("/query", response_model=Answer)
    async def query(req: QueryRequest) -> Answer:
        _check_collection(req.collection)
        return await atlas().ask(
            req.question,
            req.collection,
            mode=req.mode,
            k=req.k,
            verify=req.verify,
            revise=req.revise,
            guard=req.guard,
            hyde=req.hyde,
        )

    @app.get("/query/stream")
    async def query_stream(
        q: str = Query(min_length=1, max_length=4000),
        collection: str = DEFAULT_COLLECTION,
        mode: Mode = "simple",
        k: int | None = Query(default=None, ge=1, le=20),
        verify: bool = False,
        revise: bool = False,
        guard: bool = False,
        hyde: bool | None = None,
    ) -> StreamingResponse:
        _check_collection(collection)
        a = atlas()

        async def events() -> AsyncIterator[str]:
            try:
                async for ev in a.ask_stream(
                    q, collection, mode=mode, k=k, verify=verify, revise=revise, guard=guard, hyde=hyde
                ):
                    if ev.type == "token":
                        yield _sse("token", {"text": ev.text})
                    elif ev.type == "trace" and ev.step is not None:
                        yield _sse("trace", ev.step.model_dump(mode="json"))
                    elif ev.type == "sources" and ev.sources is not None:
                        yield _sse("sources", {"sources": [s.model_dump(mode="json") for s in ev.sources]})
                    elif ev.type == "final" and ev.answer is not None:
                        yield _sse("final", ev.answer.model_dump(mode="json"))
            except (AtlasError, NIMError) as exc:
                yield _sse("error", {"message": str(exc)})
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        try:
            html = UI_PATH.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - packaging problem
            raise HTTPException(500, f"UI asset missing: {exc}") from exc
        return HTMLResponse(html, headers={"Content-Security-Policy": CSP, "Cache-Control": "no-cache"})

    return app


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
