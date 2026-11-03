"""Model selection API: live role table, picker options, global and per-app overrides.

``GET  /api/models/roles``          every role's effective model + source, stale-default notice
``GET  /api/models/options``        picker options for one role (``?role=chat&all=1``)
``PUT  /api/models/overrides``      global per-role overrides (``null`` / ``"auto"`` clears)
``PUT  /api/models/app/{app}``      per-app overrides (win over the global ones)
``DELETE /api/models/app/{app}``    reset an app to Auto
``POST /api/models/refresh``        refetch the live catalog now
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from nimlab.core import Role, default_registry

from .scope import APP_ROLES

__all__ = ["models_payload", "router"]

router = APIRouter(prefix="/api/models")

SIM_NOTE = (
    "Simulation mode: showing the curated catalog. The live model list appears once a key is added."
)


class OverridesBody(BaseModel):
    overrides: dict[str, str | None] = Field(default_factory=dict, max_length=16)


def _state(request: Request) -> Any:
    return request.app.state.studio


def _validate(overrides: dict[str, str | None]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, value in overrides.items():
        try:
            role = Role(str(key).strip().lower())
        except ValueError:
            valid = ", ".join(r.value for r in Role)
            raise HTTPException(422, f"unknown role {key!r}; valid roles: {valid}") from None
        if value is not None:
            value = value.strip()
            if len(value) > 200 or any(ch.isspace() for ch in value):
                raise HTTPException(422, f"invalid model id for role {role.value!r}")
        out[role.value] = value or None
    return out


def models_payload(state: Any, *, fetch: bool = False) -> dict[str, Any]:
    """Everything the model pickers need in one object."""
    reg = default_registry()
    snap = reg.snapshot(fetch=fetch)
    mock = bool(state.mock)
    studio = reg.studio_overrides()
    roles = []
    for r in Role:
        info = reg.describe(r)
        layers = reg.override_layers(r)
        roles.append(
            {
                **info,
                "override": studio.get(r),
                "pinned_by": next((k for k in ("env", "toml") if k in layers), None),
            }
        )
    apps: dict[str, Any] = {}
    saved = reg.app_overrides()
    for app, slots in APP_ROLES.items():
        entries = []
        for slot in slots:
            r = Role(slot["role"])
            info = reg.describe(r, app=app)
            entries.append(
                {
                    **slot,
                    "model": info["model"],
                    "source": info["source"],
                    "override": saved.get(app, {}).get(r),
                }
            )
        apps[app] = {"roles": entries, "overridden": bool(saved.get(app))}
    return {
        "live": snap is not None,
        "mock": mock,
        "note": SIM_NOTE if snap is None else "",
        "catalog": (
            {
                "source": snap.source,
                "count": len(snap.ids),
                "age_s": int(snap.age_s()),
                "fetched_at": snap.fetched_at,
            }
            if snap is not None
            else None
        ),
        "stale": reg.stale_defaults(),
        "unavailable": reg.unavailable(),
        "substitutions": reg.substitutions(),
        "roles": roles,
        "apps": apps,
    }


@router.get("/roles")
async def roles(request: Request) -> dict[str, Any]:
    return models_payload(_state(request))


@router.get("/options")
async def options(
    request: Request, role: str = Query(...), all: bool = Query(False)
) -> dict[str, Any]:
    try:
        r = Role(role.strip().lower())
    except ValueError:
        raise HTTPException(422, f"unknown role {role!r}") from None
    reg = default_registry()
    snap = reg.snapshot(fetch=False)
    return {
        "role": r.value,
        "live": snap is not None,
        "note": "" if snap is not None else SIM_NOTE,
        "all": all,
        "models": reg.models_for_role(r, include_all=all),
    }


@router.put("/overrides")
async def set_overrides(body: OverridesBody, request: Request) -> dict[str, Any]:
    default_registry().set_studio_overrides(_validate(body.overrides))
    return models_payload(_state(request))


@router.put("/app/{app}")
async def set_app_overrides(app: str, body: OverridesBody, request: Request) -> dict[str, Any]:
    if app not in APP_ROLES:
        raise HTTPException(404, f"unknown app {app!r}")
    default_registry().set_app_overrides(app, _validate(body.overrides))
    return models_payload(_state(request))


@router.delete("/app/{app}")
async def reset_app_overrides(app: str, request: Request) -> dict[str, Any]:
    if app not in APP_ROLES:
        raise HTTPException(404, f"unknown app {app!r}")
    reg = default_registry()
    current = reg.app_overrides(app).get(app, {})
    reg.set_app_overrides(app, {r: None for r in current})
    return models_payload(_state(request))


@router.post("/refresh")
async def refresh(request: Request) -> dict[str, Any]:
    state = _state(request)
    reg = default_registry()
    await asyncio.to_thread(reg.refresh_catalog, config=state.base_config())
    return models_payload(state)
