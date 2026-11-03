"""The Studio FastAPI app: JSON API under ``/api``, SSE run streams and the static web UI."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from nimlab import __version__
from nimlab.atlas.models import AtlasError
from nimlab.atlas.server import parse_multipart
from nimlab.core import NIMError, default_registry

from . import arena_app, atlas_app, council_app, credentials, forge_app
from .projects import ProjectError
from .security import SecurityMiddleware
from .settings import StudioSettings
from .state import StudioState

__all__ = ["create_app"]

PLACEHOLDER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIM Lab Studio</title>
<style>:root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;place-items:center;
background:#0B0A12;color:#EDEBF7;font:16px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:34rem;padding:2rem}h1{margin:0 0 .5rem}
code{background:#1c1a2b;padding:.1rem .35rem;border-radius:4px}
</style></head><body><main><h1>NIM Lab Studio</h1>
<p>The API is running, but the web UI files (<code>nimlab/studio/web/index.html</code>) are not
installed.</p>
<p>Try <code>/api/health</code>.</p></main></body></html>
"""


# --------------------------------------------------------------------------- request bodies


class KeyBody(BaseModel):
    api_key: str = Field(min_length=1, max_length=512)


class BuildBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    project: str | None = Field(default=None, max_length=64)
    max_iters: int | None = Field(default=None, ge=1, le=20)


class CouncilBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    strategy: str = "moa"
    models: list[str] | None = Field(default=None, max_length=12)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    collection: str | None = None
    mode: Literal["simple", "agentic"] = "simple"
    verify: bool = False


class RouteBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)


class ArenaBody(BaseModel):
    suites: list[str] | str | None = None
    models: list[str] | None = Field(default=None, max_length=12)


# --------------------------------------------------------------------------- app


def create_app(
    settings: StudioSettings | None = None,
    *,
    client: Any | None = None,
    **overrides: Any,
) -> FastAPI:
    """Build the Studio app.

    ``settings`` defaults to :meth:`StudioSettings.from_env`; keyword ``overrides`` replace single
    fields (``token=None`` turns auth off, ``sim_delay=0`` removes simulation pauses). ``client`` is
    an optional pre-built NIM client (tests); otherwise one is created from the environment and the
    stored key, and replaced whenever the key changes.
    """
    cfg = settings or StudioSettings.from_env()
    if overrides:
        cfg = cfg.with_(**overrides)
    state = StudioState(cfg, client=client)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(
        title="NIM Lab Studio",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.studio = state
    app.add_middleware(SecurityMiddleware, settings=cfg)
    api = APIRouter(prefix="/api")

    def bad(status: int, message: str) -> HTTPException:
        return HTTPException(status, message)

    @app.exception_handler(ProjectError)
    async def _project_error(_: Request, exc: ProjectError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(AtlasError)
    async def _atlas_error(_: Request, exc: AtlasError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(NIMError)
    async def _nim_error(_: Request, exc: NIMError) -> JSONResponse:
        return JSONResponse({"detail": f"upstream model error: {exc}"}, status_code=502)

    # ------------------------------------------------------------------ status & key
    @api.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @api.get("/status")
    async def status() -> dict[str, Any]:
        return state.status()

    async def key_response(result: dict[str, Any], warning: str | None = None) -> dict[str, Any]:
        return {
            "ok": True,
            "verified": result["verified"],
            "model_count": result["model_count"],
            "error": result["error"],
            "warning": warning,
            "key": state.key_info(),
        }

    @api.post("/key")
    async def save_key(body: KeyBody) -> dict[str, Any]:
        key = body.api_key.strip()
        try:
            saved = credentials.save_api_key(key)
        except ValueError as exc:
            raise bad(422, str(exc)) from exc
        except OSError as exc:
            raise bad(500, f"could not store the key: {exc}") from exc
        result = await state.verify_key(key)
        state.reset_client()
        warning = saved.warning
        source = credentials.key_source()
        if source in ("env", "dotenv"):
            origin = "the NVIDIA_API_KEY environment variable" if source == "env" else ".env"
            extra = f"A key from {origin} takes precedence over the one you saved."
            warning = f"{warning} {extra}" if warning else extra
        return await key_response(result, warning)

    @api.delete("/key")
    async def delete_key() -> dict[str, Any]:
        credentials.delete_api_key()
        state.reset_client()
        return {"ok": True, "key": state.key_info()}

    @api.post("/key/test")
    async def test_key() -> dict[str, Any]:
        key = state.base_config().api_key
        result = await state.verify_key(key)
        return await key_response(result)

    # ------------------------------------------------------------------ models & usage
    @api.get("/models")
    async def models() -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "roles": sorted(r.value for r in m.roles),
                "cost_tier": m.cost_tier,
                "context_window": m.context_window,
                "notes": m.notes,
            }
            for m in default_registry().all()
        ]

    @api.get("/usage")
    async def usage() -> dict[str, Any]:
        summary = state.usage.summary()
        return {
            **summary,
            "rows": [{"model": m, **stats} for m, stats in summary["models"].items()],
            "runs": dict(state.runs.counts),
        }

    # ------------------------------------------------------------------ runs
    @api.get("/runs")
    async def list_runs(limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
        return [r.summary() for r in state.runs.recent(limit)]

    def get_run(run_id: str) -> Any:
        run = state.runs.get(run_id)
        if run is None:
            raise bad(404, f"unknown run {run_id!r}")
        return run

    @api.get("/runs/{run_id}")
    async def run_detail(run_id: str) -> dict[str, Any]:
        return get_run(run_id).detail()

    @api.get("/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, after: int = Query(0, ge=0)) -> Response:
        run = get_run(run_id)
        last = request.headers.get("last-event-id", "")
        if last.strip().isdigit():
            after = max(after, int(last))
        newest = run.events[-1]["seq"] if run.events else 0
        if run.finished and after >= newest and newest:
            return Response(status_code=204)  # tells EventSource not to reconnect
        return StreamingResponse(
            run.stream(after, cfg.sse_heartbeat_s),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @api.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict[str, Any]:
        run, requested = state.runs.cancel(get_run(run_id).id)
        assert run is not None
        return {"ok": requested, "id": run.id, "status": run.status}

    # ------------------------------------------------------------------ forge & projects
    @api.post("/forge/build")
    async def forge_build(body: BuildBody) -> dict[str, Any]:
        run, project_id = forge_app.start_build(state, body.prompt, body.project, body.max_iters)
        return {"run_id": run.id, "project": project_id}

    @api.get("/projects")
    async def projects() -> list[dict[str, Any]]:
        return state.projects.list()

    @api.get("/projects/{project_id}/tree")
    async def project_tree(project_id: str) -> dict[str, Any]:
        return state.projects.tree(project_id)

    @api.get("/projects/{project_id}/file")
    async def project_file(
        project_id: str, path: str = Query("", max_length=1024)
    ) -> dict[str, Any]:
        return state.projects.read_file(project_id, path)

    @api.get("/projects/{project_id}/diff")
    async def project_diff(project_id: str) -> dict[str, Any]:
        return state.projects.diff(project_id)

    @api.post("/projects/{project_id}/reveal")
    async def project_reveal(project_id: str) -> dict[str, Any]:
        return state.projects.reveal(project_id)

    # ------------------------------------------------------------------ council
    @api.get("/council/strategies")
    async def council_strategies() -> list[dict[str, str]]:
        from nimlab.council.strategies import list_strategies

        return [{"name": s.name, "description": s.description} for s in list_strategies()]

    @api.post("/council/run")
    async def council_run(body: CouncilBody) -> dict[str, Any]:
        try:
            run = council_app.start_council(state, body.question, body.strategy, body.models)
        except ValueError as exc:
            raise bad(422, str(exc)) from exc
        return {"run_id": run.id}

    # ------------------------------------------------------------------ atlas
    def collection_name(name: str | None) -> str:
        try:
            return atlas_app.check_collection(name)
        except ValueError as exc:
            raise bad(422, str(exc)) from exc

    @api.get("/atlas/collections")
    async def atlas_collections() -> list[dict[str, Any]]:
        return atlas_app.collections(state)

    @api.post("/atlas/ingest")
    async def atlas_ingest(
        request: Request, collection: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        ctype = request.headers.get("content-type", "")
        declared = request.headers.get("content-length")
        limit = atlas_app.MAX_UPLOAD_BYTES
        if declared and declared.isdigit() and int(declared) > limit:
            raise bad(413, f"upload too large (limit {limit // 2**20} MB)")
        body = await request.body()
        if len(body) > limit:
            raise bad(413, f"upload too large (limit {limit // 2**20} MB)")
        try:
            if ctype.lower().startswith("multipart/form-data"):
                try:
                    fields, files = parse_multipart(ctype, body)
                except ValueError as exc:
                    raise bad(400, str(exc)) from exc
                if not files:
                    raise bad(400, "no files in the request")
                name = collection_name(collection or fields.get("collection"))
                return await atlas_app.ingest_uploads(
                    state, files, name, force or fields.get("force") in ("1", "true")
                )
            if "json" in ctype.lower():
                try:
                    payload = json.loads(body or b"{}")
                    path = str(payload["path"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise bad(
                        400, 'JSON body must look like {"path": "...", "collection": "..."}'
                    ) from exc
                name = collection_name(collection or payload.get("collection"))
                return await atlas_app.ingest_path(
                    state, path, name, force or bool(payload.get("force"))
                )
        except FileNotFoundError as exc:
            raise bad(404, str(exc)) from exc
        except ValueError as exc:
            raise bad(415, str(exc)) from exc
        raise bad(415, "send multipart/form-data (files) or application/json ({path})")

    @api.post("/atlas/ask")
    async def atlas_ask(body: AskBody) -> dict[str, Any]:
        name = collection_name(body.collection)
        run = atlas_app.start_ask(state, body.question, name, body.mode, body.verify)
        return {"run_id": run.id}

    # ------------------------------------------------------------------ arena
    @api.get("/arena/runs")
    async def arena_runs() -> list[dict[str, Any]]:
        return arena_app.list_runs(state)

    @api.get("/arena/suites")
    async def arena_suites() -> list[str]:
        return arena_app.suite_names()

    @api.post("/arena/route")
    async def arena_route(body: RouteBody) -> dict[str, Any]:
        return await arena_app.route(state, body.prompt)

    @api.post("/arena/run")
    async def arena_run(body: ArenaBody) -> dict[str, Any]:
        try:
            run = arena_app.start_run(state, body.suites, body.models)
        except ValueError as exc:
            raise bad(422, str(exc)) from exc
        return {"run_id": run.id}

    @api.api_route("/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def api_not_found(rest: str) -> JSONResponse:
        return JSONResponse({"detail": f"no such API route: /api/{rest}"}, status_code=404)

    app.include_router(api)

    # ------------------------------------------------------------------ static web UI
    web_dir = cfg.web_dir

    def _index() -> Response:
        index = web_dir / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse(PLACEHOLDER_HTML, headers={"Cache-Control": "no-cache"})

    @app.get("/{path:path}", include_in_schema=False)
    async def web(path: str) -> Response:
        if not path:
            return _index()
        base = web_dir.resolve()
        target = (base / path).resolve()
        if (target == base or base in target.parents) and target.is_file():
            return FileResponse(target, headers={"Cache-Control": "no-cache"})
        if "." in Path(path).name:
            raise HTTPException(404, "not found")
        return _index()  # client-side routes fall back to the app shell

    return app
