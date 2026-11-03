"""Studio routes for Loom: node palette, saved pipelines, lint, export and runs.

Mount ``router`` on the Studio app (``app.include_router(routes_loom.router)``, before the API
catch-all) and add ``APP_INFO`` to the app list. The router also works on a bare FastAPI app
(tests): without ``app.state.studio`` it keeps its own run manager in ``app.state.loom``.

Run events are the engine's (see :mod:`nimlab.loom.engine`) streamed through the usual
``GET /api/runs/{id}/events``; the final ``run.done`` carries ``{outputs, tokens, ...}``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nimlab.loom import export as loom_export
from nimlab.loom import functions as loom_functions
from nimlab.loom import library
from nimlab.loom import linter as loom_lint
from nimlab.loom import nodes as loom_nodes
from nimlab.loom.engine import prepare_inputs, run_pipeline
from nimlab.loom.errors import LoomError
from nimlab.loom.registry import node_types
from nimlab.loom.schema import Pipeline, topo_levels
from nimlab.loom.textops import http_allowlist

from .runs import Run, RunManager

__all__ = ["APP_INFO", "router"]

#: Presentation metadata for the Studio rail / status (``--c-loom`` in the web tokens).
APP_INFO: dict[str, str] = {
    "id": "loom",
    "name": "Loom",
    "tagline": "Patch models into pipelines",
    "color": "#3DFFB4",
    "glyph": "≋",
}

router = APIRouter(prefix="/api/loom")

MAX_PIPELINE_BYTES = 1_000_000


# --------------------------------------------------------------------------- runtime plumbing
class _Standalone:
    """What the routes need when the router is mounted without Studio (tests, embedding)."""

    def __init__(self) -> None:
        self.runs = RunManager(None)
        self._client: Any | None = None

    def client(self) -> Any:
        if self._client is None:
            from nimlab.core import NIMClient, NIMConfig

            self._client = NIMClient(NIMConfig.from_env())
        return self._client

    @property
    def mock(self) -> bool:
        return bool(getattr(self.client(), "mock", False))


class _Runtime:
    """Uniform view over a real ``StudioState`` or the standalone fallback."""

    def __init__(self, request: Request) -> None:
        studio = getattr(request.app.state, "studio", None)
        if studio is None:
            if not hasattr(request.app.state, "loom"):
                request.app.state.loom = _Standalone()
            studio = request.app.state.loom
            self.home: Path | None = None
            self.atlas_db: Path | None = None
        else:
            self.home = getattr(studio.settings, "home", None)
            self.atlas_db = getattr(studio.settings, "atlas_db", None)
        self.state = studio

    @property
    def runs(self) -> RunManager:
        return self.state.runs

    def client(self) -> Any:
        return self.state.client()

    @property
    def mock(self) -> bool:
        return bool(self.state.mock)

    @property
    def pipelines_dir(self) -> Path:
        env = os.environ.get("NIMLAB_LOOM_DIR", "").strip()
        if env or self.home is None:
            return library.user_dir()
        return self.home / "loom"

    @property
    def cache_dir(self) -> Path | None:
        env = os.environ.get("NIMLAB_LOOM_CACHE", "").strip()
        if env:
            return Path(env).expanduser()
        return self.home / "loom" / "cache" if self.home is not None else None


def _bad(status: int, message: str) -> HTTPException:
    return HTTPException(status, message)


def _pipeline_from(body: dict[str, Any]) -> Pipeline:
    data = body.get("pipeline") if isinstance(body.get("pipeline"), dict) else body
    try:
        return Pipeline.from_data(data)
    except LoomError as exc:
        raise _bad(422, str(exc)) from exc


def _check_id(pipeline_id: str) -> str:
    try:
        return library.check_id(pipeline_id)
    except LoomError as exc:
        raise _bad(422, str(exc)) from exc


def _bundled_ids() -> set[str]:
    return {i.id for i in library.list_pipelines(include_user=False)}


# --------------------------------------------------------------------------- node types
@router.get("/node-types")
async def node_types_route() -> list[dict[str, Any]]:
    """Every registered node type with its parameter schema and jacks."""
    out: list[dict[str, Any]] = []
    functions = loom_functions.list_functions()
    allow = http_allowlist()
    for nt in node_types():
        desc = nt.describe()
        if nt.name == "python_fn":
            fn = dict(desc["params"]["properties"]["fn"])
            fn["enum"] = list(functions)
            fn["x-docs"] = functions
            desc["params"] = {
                **desc["params"],
                "properties": {**desc["params"]["properties"], "fn": fn},
            }
        if nt.name == "http_get":
            desc["enabled"] = bool(allow)
            desc["allow"] = allow
        desc["color"] = loom_nodes.CATEGORIES.get(nt.category, "core")
        out.append(desc)
    return out


# --------------------------------------------------------------------------- pipelines
@router.get("/pipelines")
async def list_pipelines_route(request: Request) -> list[dict[str, Any]]:
    rt = _Runtime(request)
    return [i.to_dict() for i in library.list_pipelines(directory=rt.pipelines_dir)]


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline_route(pipeline_id: str, request: Request) -> dict[str, Any]:
    rt = _Runtime(request)
    _check_id(pipeline_id)
    info = library.find_pipeline(pipeline_id, rt.pipelines_dir)
    if info is None:
        raise _bad(404, f"no pipeline {pipeline_id!r}")
    try:
        pipe = Pipeline.load(info.path)
    except LoomError as exc:
        raise _bad(422, str(exc)) from exc
    if any(n.pos is None for n in pipe.nodes):
        library.auto_layout(pipe)
    return {
        "id": info.id,
        "source": info.source,
        "readonly": info.source == "bundled",
        "pipeline": pipe.to_data(),
    }


@router.put("/pipelines/{pipeline_id}")
async def put_pipeline_route(
    pipeline_id: str, body: dict[str, Any], request: Request
) -> dict[str, Any]:
    rt = _Runtime(request)
    _check_id(pipeline_id)
    if pipeline_id in _bundled_ids():
        raise _bad(
            409, f"{pipeline_id!r} is a bundled pipeline and read-only; save it under another id"
        )
    pipe = _pipeline_from(body)
    if len(pipe.to_json()) > MAX_PIPELINE_BYTES:
        raise _bad(413, "pipeline is too large")
    existed = library.find_pipeline(pipeline_id, rt.pipelines_dir) is not None
    try:
        library.save_user_pipeline(pipeline_id, pipe, rt.pipelines_dir)
    except OSError as exc:
        raise _bad(500, f"could not save: {exc}") from exc
    report = loom_lint.lint(pipe)
    return {"id": pipeline_id, "saved": True, "created": not existed, "errors": len(report.errors),
            "warnings": len(report.warnings)}  # fmt: skip


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline_route(pipeline_id: str, request: Request) -> dict[str, Any]:
    rt = _Runtime(request)
    _check_id(pipeline_id)
    if pipeline_id in _bundled_ids():
        raise _bad(403, "bundled pipelines cannot be deleted")
    if not library.delete_user_pipeline(pipeline_id, rt.pipelines_dir):
        raise _bad(404, f"no saved pipeline {pipeline_id!r}")
    return {"ok": True, "id": pipeline_id}


# --------------------------------------------------------------------------- lint & export
@router.post("/lint")
async def lint_route(body: dict[str, Any]) -> dict[str, Any]:
    inner = body.get("pipeline")
    return loom_lint.lint(inner if isinstance(inner, dict) else body).to_dict()


@router.post("/export")
async def export_route(body: dict[str, Any]) -> dict[str, Any]:
    pipe = _pipeline_from(body)
    fmt = str(body.get("format", "mermaid"))
    try:
        if fmt == "mermaid":
            return {"format": fmt, "text": loom_export.to_mermaid(pipe)}
        if fmt == "python":
            return {"format": fmt, "text": loom_export.to_python(pipe)}
    except (LoomError, ValueError) as exc:
        raise _bad(422, str(exc)) from exc
    raise _bad(422, "format must be 'mermaid' or 'python'")


# --------------------------------------------------------------------------- run
def start_loom_run(rt: _Runtime, pipe: Pipeline, inputs: dict[str, Any]) -> Run:
    simulated = rt.mock
    cache_dir = rt.cache_dir
    atlas_db = rt.atlas_db

    async def work(run: Run) -> dict[str, Any]:
        def forward(event: dict[str, Any]) -> None:
            data = dict(event)
            kind = str(data.pop("type"))
            if kind == "run.done":  # the run manager emits the terminal event with the result
                return
            run.emit(kind, data)

        result = await run_pipeline(
            pipe,
            inputs,
            client=rt.client(),
            on_event=forward,
            cache=cache_dir if cache_dir is not None else True,
            stream=True,
            atlas_db=atlas_db,
            validate=False,
        )
        if not result.ok:
            raise RuntimeError(f"{result.failed_node}: {result.error}")
        summary = result.to_dict()
        summary["simulated"] = simulated
        summary["pipeline"] = pipe.name
        return summary

    title = (pipe.title or pipe.name)[:80]
    return rt.runs.start("loom", title, work, simulated=simulated, meta={"pipeline": pipe.name})


@router.post("/run")
async def run_route(body: dict[str, Any], request: Request) -> Any:
    """Start a run. Body: ``{pipeline: {...}}`` or ``{id: "saved-or-bundled"}``, plus ``inputs``."""
    rt = _Runtime(request)
    inputs = body.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise _bad(422, "inputs must be an object")
    if isinstance(body.get("pipeline"), dict):
        pipe = _pipeline_from(body)
    elif isinstance(body.get("id"), str):
        _check_id(body["id"])
        try:
            pipe = library.load_pipeline(body["id"], rt.pipelines_dir)
        except LoomError as exc:
            raise _bad(404, str(exc)) from exc
    else:
        raise _bad(422, "send either 'pipeline' (the document) or 'id' (a saved/bundled name)")
    report = loom_lint.lint(pipe, http_allow=http_allowlist())
    if report.errors:
        return JSONResponse(
            {
                "detail": "pipeline has errors: "
                + "; ".join(i.format() for i in report.errors[:3]),
                "issues": [i.to_dict() for i in report.issues],
            },
            status_code=422,
        )
    try:
        topo_levels(pipe)
        prepare_inputs(pipe, inputs)
    except LoomError as exc:
        raise _bad(422, str(exc)) from exc
    run = start_loom_run(rt, pipe, inputs)
    return {"run_id": run.id}
