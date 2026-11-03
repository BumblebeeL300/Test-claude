"""Studio routes for Distill: recipes, runs, the sample inspector and dataset export.

Mount with ``app.include_router(routes_distill.router)`` *before* the ``/api`` catch-all route.
The router works on a bare FastAPI app too (tests): without ``app.state.studio`` it creates its own
run manager and uses ``./.nimlab/distill/runs`` (override with ``app.state.distill_runs_dir``).

Events on the run stream: ``run.config``, ``seeds``, ``stage.start`` / ``stage.end``,
``sample.generated``, ``sample.scored``, ``sample.dropped``, ``dedup.cluster``, ``funnel``,
``export``, then ``run.done`` (the result summary) or ``run.failed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from nimlab.core import model_scope
from nimlab.distill import Distiller, DistillError, RecipeError, list_recipes
from nimlab.distill.card import render_card
from nimlab.distill.export import FORMATS
from nimlab.distill.models import SAMPLE_STAGES, RunMeta, Sample
from nimlab.distill.report import render_report_html
from nimlab.distill.store import DEFAULT_RUNS_DIR, RunStore, StoreError

from .runs import Run, RunManager
from .scope import clean_overrides, scoped_work

__all__ = ["APP_INFO", "router"]

#: Presentation metadata, same shape as the entries of ``nimlab.studio.state.APPS``.
APP_INFO: dict[str, str] = {
    "id": "distill",
    "name": "Distill",
    "tagline": "Refine model output into clean, deduplicated datasets",
    "color": "#FFC83D",
    "glyph": "⬡",
}

router = APIRouter(prefix="/api/distill")

MAX_N = 2000
SAMPLE_FILTERS = (*SAMPLE_STAGES, "final", "dropped", "all")
#: Simulated runs pause this long per call (scaled by Studio ``sim_delay``) so the UI animates.
SIM_PACE_S = 0.12
SIM_CONCURRENCY = 4


class RunBody(BaseModel):
    recipe: str = Field(min_length=1, max_length=200)
    n: int = Field(default=60, ge=1, le=MAX_N)
    models: list[str] | None = Field(default=None, max_length=8)
    options: dict[str, Any] = Field(default_factory=dict)
    format: str | None = None
    #: Optional ``{role: model}`` overrides for this run (generators, reward model, embedder...).
    model_overrides: dict[str, str] | None = Field(default=None, max_length=16)


# --------------------------------------------------------------------------- state helpers


def _studio(request: Request) -> Any | None:
    return getattr(request.app.state, "studio", None)


def _manager(request: Request) -> RunManager:
    state = _studio(request)
    if state is not None:
        runs: RunManager = state.runs
        return runs
    mgr = getattr(request.app.state, "distill_manager", None)
    if mgr is None:
        mgr = RunManager(None)
        request.app.state.distill_manager = mgr
    manager: RunManager = mgr
    return manager


def _runs_dir(request: Request) -> Path:
    explicit = getattr(request.app.state, "distill_runs_dir", None)
    if explicit is not None:
        return Path(explicit)
    state = _studio(request)
    if state is not None:
        return Path(state.settings.home) / "distill" / "runs"
    return DEFAULT_RUNS_DIR


def _client(request: Request) -> Any | None:
    state = _studio(request)
    return state.client() if state is not None else None


def _sim_pace(request: Request) -> float:
    state = _studio(request)
    return SIM_PACE_S * state.settings.sim_delay if state is not None else 0.0


def _store(request: Request) -> RunStore:
    return RunStore(_runs_dir(request))


def _meta(request: Request, run_id: str) -> RunMeta:
    try:
        return _store(request).load_meta(run_id)
    except StoreError as exc:
        raise HTTPException(404, str(exc)) from exc


# --------------------------------------------------------------------------- views


def _recipe_view(rec: Any) -> dict[str, Any]:
    data: dict[str, Any] = rec.model_dump()
    data["needs_docs"] = rec.kind == "qa_from_docs"
    data["formats"] = [f for f in FORMATS if f != "dpo" or rec.kind == "preference_pairs"]
    data["min_models"] = 2 if rec.kind == "preference_pairs" else 1
    return data


def _run_row(meta: RunMeta) -> dict[str, Any]:
    return {
        "id": meta.id,
        "recipe": meta.recipe.name,
        "kind": meta.recipe.kind,
        "status": meta.status,
        "n": meta.n,
        "models": meta.models,
        "format": meta.format,
        "simulated": meta.simulated,
        "created_at": meta.created_at,
        "finished_at": meta.finished_at,
        "stages_done": meta.stages_done,
        "cost_usd": meta.cost_usd,
        "error": meta.error,
    }


def _sample_view(s: Sample) -> dict[str, Any]:
    return {
        "id": s.id,
        "prompt": s.prompt,
        "response": s.response,
        "rejected": s.rejected,
        "model": s.model,
        "rejected_model": s.rejected_model,
        "temperature": s.temperature,
        "topic": s.topic,
        "subtopic": s.subtopic,
        "source": s.source,
        "context": s.context[:1200],
        "score": s.score,
        "rejected_score": s.rejected_score,
        "score_source": s.score_source,
        "difficulty": s.difficulty,
        "cluster": s.cluster,
        "alive": s.alive,
        "dropped_at": s.dropped_at,
        "reason": s.reason,
        "detail": s.detail,
    }


# --------------------------------------------------------------------------- endpoints


@router.get("/recipes")
def recipes() -> list[dict[str, Any]]:
    """The bundled recipes with their defaults and which export formats fit each."""
    try:
        return [_recipe_view(r) for r in list_recipes().values()]
    except RecipeError as exc:  # pragma: no cover - bundled files are tested
        raise HTTPException(500, str(exc)) from exc


@router.post("/run")
async def start_run(body: RunBody, request: Request) -> dict[str, Any]:
    """Start a run through the Studio run manager; stream it at ``/api/runs/{run_id}/events``."""
    options = dict(body.options)
    if "sim_delay" not in options:
        options["sim_delay"] = _sim_pace(request)
    if "concurrency" not in options:
        options["concurrency"] = SIM_CONCURRENCY
    runs_dir = _runs_dir(request)
    try:
        overrides = clean_overrides(body.model_overrides)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        with model_scope("distill", overrides) as planning:
            probe = Distiller(_client(request), runs_dir=runs_dir)
            rec, opts, panel, fmt = probe.plan(
                body.recipe, n=body.n, models=body.models, fmt=body.format, options=options
            )
            simulated = probe.uses_simulator(opts)
    except (DistillError, RecipeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if rec.kind == "qa_from_docs" and not simulated and not opts.docs:
        raise HTTPException(422, "qa_from_docs needs options.docs (a folder of documents)")
    if not simulated:
        opts = opts.model_copy(update={"concurrency": None, "sim_delay": 0.0})
    client = _client(request)
    recipe_name, n, models = body.recipe, body.n, body.models

    async def work(run: Run) -> dict[str, Any]:
        def sink(type_: str, data: dict[str, Any]) -> None:
            run.emit(type_, data, data.get("stage") if isinstance(data.get("stage"), str) else None)

        distiller = Distiller(client, runs_dir=runs_dir, on_event=sink)
        result = await distiller.run(
            recipe_name, n=n, models=models, fmt=fmt, options=opts, run_id=run.id
        )
        return result.summary()

    run = _manager(request).start(
        "distill",
        f"{rec.name} · n={body.n} · {len(panel)} model{'s' if len(panel) != 1 else ''}",
        scoped_work("distill", overrides, work, planning),
        simulated=simulated,
    )
    return {"run_id": run.id, "simulated": simulated, "format": fmt}


@router.get("/runs")
def list_runs(request: Request, limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    """Runs checkpointed on disk, newest first."""
    return [_run_row(m) for m in _store(request).list_runs()[:limit]]


@router.get("/runs/{run_id}")
def run_detail(run_id: str, request: Request) -> dict[str, Any]:
    """Run metadata plus, when it has progressed, the quality summary (funnel, histogram...)."""
    meta = _meta(request, run_id)
    detail: dict[str, Any] = {**_run_row(meta), "recipe_config": meta.recipe.model_dump()}
    try:
        detail["result"] = Distiller(runs_dir=_runs_dir(request)).load(run_id).summary()
    except (StoreError, DistillError) as exc:  # pragma: no cover - meta was readable a moment ago
        raise HTTPException(404, str(exc)) from exc
    return detail


@router.get("/runs/{run_id}/samples")
def samples(
    run_id: str,
    request: Request,
    stage: str = Query("final", description="a stage name, final, dropped or all"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    reason: str | None = Query(None, max_length=40),
) -> dict[str, Any]:
    """A page of samples: those alive after ``stage``, the dropped ones, or everything."""
    if stage not in SAMPLE_FILTERS:
        raise HTTPException(422, f"stage must be one of {', '.join(SAMPLE_FILTERS)}")
    _meta(request, run_id)
    table = _store(request).load_samples(run_id)
    if stage == "all":
        rows = table
    elif stage == "dropped":
        rows = [s for s in table if not s.alive]
    elif stage == "final":
        rows = [s for s in table if s.alive]
    else:
        rows = [s for s in table if s.alive_after(stage)]
    if reason:
        rows = [s for s in rows if s.reason == reason]
    page = rows[offset : offset + limit]
    return {
        "run_id": run_id,
        "stage": stage,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "items": [_sample_view(s) for s in page],
    }


@router.get("/runs/{run_id}/export")
def export(
    run_id: str,
    request: Request,
    format: str | None = Query(None, description=", ".join(FORMATS)),
) -> FileResponse:
    """Download the kept samples as JSONL in the requested format."""
    meta = _meta(request, run_id)
    fmt = format or meta.format
    if fmt not in FORMATS:
        raise HTTPException(422, f"format must be one of {', '.join(FORMATS)}")
    if "finalize" not in meta.stages_done:
        raise HTTPException(409, "the run has not finished yet; nothing to export")
    try:
        path = Distiller(runs_dir=_runs_dir(request)).export(run_id, fmt)
    except DistillError as exc:
        raise HTTPException(422, str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/x-ndjson",
        filename=f"{meta.recipe.name}-{run_id}.{fmt}.jsonl",
    )


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
def report(run_id: str, request: Request) -> HTMLResponse:
    """The self-contained HTML quality report."""
    meta = _meta(request, run_id)
    result = Distiller(runs_dir=_runs_dir(request)).load(run_id)
    return HTMLResponse(render_report_html(result.report, meta))


@router.get("/runs/{run_id}/card", response_class=PlainTextResponse)
def card(run_id: str, request: Request) -> PlainTextResponse:
    """The dataset card (markdown)."""
    meta = _meta(request, run_id)
    result = Distiller(runs_dir=_runs_dir(request)).load(run_id)
    return PlainTextResponse(
        render_card(meta, result.report, f"{meta.recipe.name}.{meta.format}.jsonl"),
        media_type="text/markdown",
    )
