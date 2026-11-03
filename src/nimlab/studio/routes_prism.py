"""Prism in Studio: ``POST /api/prism/run`` (multipart images + mode) and ``GET /api/prism/modes``.

Include ``router`` in the Studio app *before* the catch-all ``/api/{rest:path}`` router. A run emits
``run.start``, then per stage ``stage.start`` / ``stage.end``, ``token`` (streamed stage text),
one ``result`` per image and finally ``run.done`` whose ``result`` is::

    {"mode", "batch", "simulated", "items": [PrismResult | {"name", "error"}]}
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from nimlab.atlas.server import parse_multipart
from nimlab.core import Role, default_registry
from nimlab.prism import (
    ImageError,
    Prism,
    PrismError,
    PrismImage,
    UnknownModeError,
    get_mode,
    have_pillow,
    list_modes,
    schema_names,
)
from nimlab.prism.schemas import BUILTIN_SCHEMAS

from .runs import Run
from .state import StudioState

__all__ = ["APP_INFO", "MAX_IMAGES", "MAX_UPLOAD_BYTES", "router"]

APP_INFO: dict[str, str] = {
    "id": "prism",
    "name": "Prism",
    "tagline": "Put an image on the light table: describe, read, extract, rebuild",
    "color": "#B07CFF",
    "glyph": "△",
}

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_IMAGES = 12
#: Form fields that are not mode options.
RESERVED = frozenset({"mode", "model", "files"})
SIM_DELAY_SCALE = 1.0

router = APIRouter(prefix="/api/prism")


def _state(request: Request) -> StudioState:
    state: StudioState = request.app.state.studio
    return state


def _bad(status: int, message: str) -> HTTPException:
    return HTTPException(status, message)


@router.get("/modes")
async def modes(request: Request) -> dict[str, Any]:
    """Modes with their options and image counts, the built-in extract schemas and the model."""
    state = _state(request)
    return {
        "modes": [m.info() for m in list_modes()],
        "schemas": [
            {"name": n, "description": BUILTIN_SCHEMAS[n].description} for n in schema_names()
        ],
        "model": default_registry().pick(Role.VISION),
        "vision_models": [m.id for m in default_registry().by_role(Role.VISION)],
        "pillow": have_pillow(),
        "simulated": state.mock,
        "max_images": MAX_IMAGES,
    }


async def _read_upload(request: Request) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    ctype = request.headers.get("content-type", "")
    if not ctype.lower().startswith("multipart/form-data"):
        raise _bad(415, "send multipart/form-data: a 'mode' field and one or more image files")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise _bad(413, f"upload too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise _bad(413, f"upload too large (limit {MAX_UPLOAD_BYTES // 2**20} MB)")
    try:
        return parse_multipart(ctype, body)
    except ValueError as exc:
        raise _bad(400, str(exc)) from exc


@router.post("/run")
async def run_prism(request: Request) -> dict[str, Any]:
    state = _state(request)
    fields, files = await _read_upload(request)
    if not files:
        raise _bad(400, "no image in the request")
    if len(files) > MAX_IMAGES:
        raise _bad(422, f"at most {MAX_IMAGES} images per run")
    try:
        mode = get_mode(fields.get("mode", ""))
    except UnknownModeError as exc:
        raise _bad(422, str(exc)) from exc
    options: dict[str, Any] = {k: v for k, v in fields.items() if k not in RESERVED}
    model = (fields.get("model") or "").strip() or None

    schema = str(options.get("schema") or "").strip()
    if schema and schema.lower().replace("-", "_") not in BUILTIN_SCHEMAS:
        if not schema.startswith("{"):
            raise _bad(422, f"schema must be one of {', '.join(schema_names())} or inline JSON")
        if len(schema) > 50_000:
            raise _bad(422, "schema is too large (limit 50 KB)")

    sizing = Prism(max_side=1568)
    try:
        images: list[PrismImage] = [
            await asyncio.to_thread(sizing.load, data, name=name) for name, data in files
        ]
        mode.validate(
            len(images) if mode.max_images > 1 else 1, options
        )  # fail now, as a 422, not later inside the run
    except (ImageError, PrismError) as exc:
        raise _bad(422, str(exc)) from exc

    client = state.client()
    simulated = bool(getattr(client, "mock", False))
    delay = state.settings.sim_delay * SIM_DELAY_SCALE
    single = len(images) == mode.max_images

    async def work(run: Run) -> dict[str, Any]:
        prism = Prism(
            client, model=model, sim_delay=delay
        )  # the shared client is not ours to close

        def observer(type_: str, data: dict[str, Any]) -> None:
            run.emit(type_, data, data.get("stage"))

        if single:
            result = await prism.run(
                mode.name,
                images if mode.max_images > 1 else images[0],
                observer=observer,
                **options,
            )
            items: list[dict[str, Any]] = [result.model_dump(mode="json")]
        else:
            batch = await prism.batch(images, mode.name, observer=observer, **options)
            items = [
                b.result.model_dump(mode="json") if b.result else {"name": b.name, "error": b.error}
                for b in batch
            ]
        sim = any(i.get("simulated") for i in items) or simulated
        return {"mode": mode.name, "batch": not single, "simulated": sim, "items": items}

    title = f"{mode.name}: {images[0].name}" + (f" +{len(images) - 1}" if len(images) > 1 else "")
    run = state.runs.start("prism", title[:80], work, simulated=simulated)
    return {"run_id": run.id}
