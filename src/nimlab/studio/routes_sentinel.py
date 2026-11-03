"""Sentinel in Studio: probe suites, blast-test scans, redaction and Shield endpoints.

Mount ``router`` on the Studio app *before* its catch-all ``/api/{rest:path}`` route::

    from .routes_sentinel import APP_INFO, router as sentinel_router
    app.include_router(sentinel_router)      # before app.include_router(api)

Run events (``POST /api/sentinel/scan`` returns ``{"run_id"}``; stream it like any other run):

``plan``          models, suites (with their probe ids) and the total number of probe calls
``probe.start``   a probe was sent to a model
``probe.result``  its verdict: ``outcome`` is ``blocked`` | ``leaked`` | ``refused`` | ``complied``
                  | ``false_refusal``; ``breached`` / ``held`` / ``false_refusal`` are booleans
``scorecard``     one per model when the scan has finished (grade, score, per-category rates)
``run.done``      the terminal event, ``result`` holds all scorecards and the saved run id
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nimlab.core import NIMError, default_registry
from nimlab.sentinel import (
    Action,
    Redactor,
    Runner,
    ScanSettings,
    ShieldConfig,
    SimulatedClient,
    SuiteError,
    list_suites,
    resolve_suites,
)
from nimlab.sentinel.llm import default_models
from nimlab.sentinel.models import Probe, ProbeResult, ProbeSuite
from nimlab.sentinel.pii import PII_KINDS
from nimlab.sentinel.report import render_html
from nimlab.sentinel.shield import Shield
from nimlab.sentinel.store import RunStore

from .runs import Run
from .settings import StudioSettings
from .state import StudioState

__all__ = ["APP_INFO", "router", "start_scan"]

#: Presentation metadata for the app rail / switchboard (same keys as ``StudioState`` APPS).
APP_INFO: dict[str, str] = {
    "id": "sentinel",
    "name": "Sentinel",
    "tagline": "Blast-test your models for leaks, injections and jailbreaks",
    "color": "#FF4D4D",
    "glyph": "⬢",
}

MAX_LISTED_RUNS = 20
MAX_TEXT = 100_000
#: Base per-call pause of simulated scans (scaled by ``StudioSettings.sim_delay``).
SIM_CALL_DELAY_S = 0.12

router = APIRouter(prefix="/api/sentinel")


# --------------------------------------------------------------------------- request bodies


class ScanBody(BaseModel):
    models: list[str] | None = Field(default=None, max_length=12)
    suites: list[str] | str | None = None
    shield: bool = False
    limit: int | None = Field(default=None, ge=1, le=50)


class RedactBody(BaseModel):
    text: str = Field(max_length=MAX_TEXT)
    action: Literal["redact", "restore"] = "redact"
    style: Literal["token", "label", "mask"] = "token"
    kinds: list[str] | None = None
    #: With ``action="restore"``: the ``token -> original`` mapping returned by a redact call.
    mapping: dict[str, str] | None = None


class ShieldBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    system: str | None = Field(default=None, max_length=8000)
    model: str | None = Field(default=None, max_length=200)
    canaries: list[str] = Field(default_factory=list, max_length=20)
    #: Per-check action overrides, e.g. ``{"input_injection": "redact"}``.
    actions: dict[str, Action] | None = None
    #: ``False`` runs only the input checks (no model call).
    call_model: bool = True


# --------------------------------------------------------------------------- plumbing


def get_state(request: Request) -> StudioState:
    """The Studio state of the app this router is mounted on (a private one on a bare app)."""
    app_state = request.app.state
    state = getattr(app_state, "studio", None)
    if state is None:
        state = getattr(app_state, "sentinel_state", None)
        if state is None:
            state = StudioState(StudioSettings.from_env())
            app_state.sentinel_state = state
    assert isinstance(state, StudioState)
    return state


def runs_dir(state: StudioState) -> Path:
    return Path(state.settings.home) / "sentinel" / "runs"


def _client_for(state: StudioState) -> tuple[Any, bool]:
    """``(client, owned)``: the simulator in mock mode (own instance), else the shared client."""
    if state.mock:
        client = SimulatedClient(
            config=state.base_config().model_copy(update={"mock": True}),
            delay_s=SIM_CALL_DELAY_S * state.settings.sim_delay,
        )
        client.usage = state.usage
        return client, True
    return state.client(), False


def _suite_names() -> list[str]:
    return list(list_suites())


def _resolve(spec: list[str] | str | None, limit: int | None) -> list[ProbeSuite]:
    try:
        return resolve_suites("all" if spec in (None, "", []) else spec, limit=limit)
    except SuiteError as exc:
        raise ValueError(str(exc)) from exc


def _result_event(r: ProbeResult, done: int, total: int) -> dict[str, Any]:
    return {
        "done": done,
        "total": total,
        "model": r.model,
        "suite": r.suite,
        "category": r.category,
        "probe": r.probe_id,
        "title": r.title,
        "outcome": r.outcome,
        "breached": r.breached,
        "false_refusal": r.false_refusal,
        "held": r.error is None and not r.breached and not r.false_refusal,
        "unsafe": r.unsafe,
        "shielded": r.shielded,
        "error": r.error,
        "snippet": " ".join(r.response.split())[:160],
    }


def start_scan(
    state: StudioState,
    models: list[str] | None,
    suites: list[str] | str | None,
    *,
    shield: bool = False,
    limit: int | None = None,
) -> Run:
    """Register a Sentinel scan as a Studio run (raises ``ValueError`` for unknown suites)."""
    resolved = _resolve(suites, limit)
    chosen = [m.strip() for m in (models or []) if m and m.strip()] or default_models(
        default_registry()
    )
    simulated = state.mock
    store = RunStore(runs_dir(state))

    async def work(run: Run) -> dict[str, Any]:
        client, owned = _client_for(state)
        try:
            runner = Runner(
                client,
                settings=ScanSettings(concurrency=6, shield=ShieldConfig() if shield else None),
                store=store,
            )
            total = len(chosen) * sum(len(s.probes) for s in resolved)
            run.emit(
                "plan",
                {
                    "models": chosen,
                    "suites": [
                        {
                            "name": s.name,
                            "category": s.category,
                            "title": s.title,
                            "probes": [{"id": p.id, "title": p.title} for p in s.probes],
                        }
                        for s in resolved
                    ],
                    "total": total,
                    "simulated": simulated,
                    "shield": shield,
                },
            )

            def on_start(model: str, suite: ProbeSuite, probe: Probe) -> None:
                run.emit(
                    "probe.start",
                    {
                        "model": model,
                        "suite": suite.name,
                        "category": suite.category,
                        "probe": probe.id,
                        "title": probe.title,
                    },
                )

            def on_result(r: ProbeResult, done: int, total_: int) -> None:
                run.emit("probe.result", _result_event(r, done, total_))

            result = await runner.run(chosen, resolved, on_start=on_start, on_result=on_result)
            cards = [c.model_dump(mode="json") for c in result.scorecards]
            for card in cards:
                run.emit("scorecard", card)
            return {
                "sentinel_run_id": result.run_id,
                "status": result.status,
                "simulated": simulated,
                "shield": shield,
                "models": result.models,
                "suites": result.suites,
                "scorecards": cards,
            }
        finally:
            if owned and hasattr(client, "aclose"):
                await client.aclose()

    return state.runs.start(
        "sentinel", f"{len(chosen)} models, {len(resolved)} suites", work, simulated=simulated
    )


# --------------------------------------------------------------------------- endpoints


@router.get("/suites")
async def sentinel_suites() -> list[dict[str, Any]]:
    out = []
    for s in list_suites().values():
        out.append(
            {
                "name": s.name,
                "category": s.category,
                "title": s.title,
                "description": s.description,
                "n_probes": len(s.probes),
                "benign": s.category == "over_refusal",
                "probes": [{"id": p.id, "title": p.title, "expect": p.expect} for p in s.probes],
            }
        )
    return out


@router.post("/scan")
async def sentinel_scan(body: ScanBody, request: Request) -> dict[str, Any]:
    state = get_state(request)
    try:
        run = start_scan(state, body.models, body.suites, shield=body.shield, limit=body.limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"run_id": run.id}


@router.get("/runs")
async def sentinel_runs(request: Request) -> list[dict[str, Any]]:
    """Saved scans, newest first, each with its scorecards."""
    store = RunStore(runs_dir(get_state(request)))
    out: list[dict[str, Any]] = []
    for run_id in reversed(store.list_ids()[-MAX_LISTED_RUNS:]):
        try:
            run = store.load(run_id)
        except (OSError, ValueError):
            continue
        out.append(
            {
                "run_id": run.run_id,
                "created_at": run.created_at,
                "status": run.status,
                "simulated": run.simulated,
                "shield": run.shield,
                "models": run.models,
                "suites": run.suites,
                "n_results": len(run.results),
                "scorecards": [c.model_dump(mode="json") for c in run.scorecards],
            }
        )
    return out


def _load(request: Request, run_id: str) -> Any:
    try:
        return RunStore(runs_dir(get_state(request))).load(run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"unknown Sentinel run {run_id!r}") from exc


@router.get("/runs/{run_id}")
async def sentinel_run(run_id: str, request: Request) -> dict[str, Any]:
    return _load(request, run_id).model_dump(mode="json")  # type: ignore[no-any-return]


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
async def sentinel_report(run_id: str, request: Request) -> HTMLResponse:
    return HTMLResponse(render_html(_load(request, run_id)))


@router.post("/redact")
async def sentinel_redact(body: RedactBody) -> dict[str, Any]:
    if body.action == "restore":
        if not body.mapping:
            raise HTTPException(422, "restore needs the token mapping returned by a redact call")
        return {"text": Redactor().restore(body.text, body.mapping)}
    unknown = set(body.kinds or []) - set(PII_KINDS)
    if unknown:
        raise HTTPException(422, f"unknown PII kind(s): {', '.join(sorted(unknown))}")
    redactor = Redactor(style=body.style, kinds=body.kinds)
    res = redactor.redact(body.text)
    return {
        "text": res.text,
        "style": body.style,
        "reversible": body.style == "token",
        "counts": res.counts,
        "mapping": res.mapping,
        "matches": [
            {"kind": m.kind, "start": m.start, "end": m.end, "text": m.text, "token": t}
            for m, t in zip(res.matches, res.tokens, strict=True)
        ],
    }


@router.post("/shield")
async def sentinel_shield(body: ShieldBody, request: Request) -> dict[str, Any]:
    state = get_state(request)
    valid = set(ShieldConfig.model_fields) & {
        "input_injection",
        "input_pii",
        "input_guard",
        "output_guard",
        "output_leak",
        "output_pii",
    }
    overrides = body.actions or {}
    bad = set(overrides) - valid
    if bad:
        raise HTTPException(422, f"unknown check(s): {', '.join(sorted(bad))}")
    cfg = ShieldConfig(**{**{k: v for k, v in overrides.items()}, "canaries": body.canaries})
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": body.system}] if body.system else []
    )
    messages.append({"role": "user", "content": body.prompt})
    model = body.model or default_models(default_registry())[0]
    client, owned = _client_for(state)
    try:
        async with Shield(client, cfg) as sh:
            if not body.call_model:
                sent, checks, _ = await sh.check_input(messages)
                blocked = any(c.applied == "blocked" for c in checks)
                return {
                    "model": model,
                    "simulated": state.mock,
                    "blocked": blocked,
                    "content": "",
                    "sent": sent[-1]["content"],
                    "checks": [c.model_dump(mode="json") for c in checks],
                    "annotations": [],
                    "redactions": 0,
                }
            res = await sh.chat(model, messages, max_tokens=300)
    except NIMError as exc:
        raise HTTPException(502, f"upstream model error: {exc}") from exc
    finally:
        if owned and hasattr(client, "aclose"):
            await client.aclose()
    return {
        "model": model,
        "simulated": state.mock,
        "blocked": res.blocked,
        "blocked_by": res.blocked_by,
        "content": res.content,
        "checks": [c.model_dump(mode="json") for c in res.checks],
        "annotations": res.annotations,
        "redactions": res.redactions,
    }
