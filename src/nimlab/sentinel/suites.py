"""Loading and resolving probe suites (bundled JSON files plus user-provided ones)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import CATEGORIES, ProbeSuite

__all__ = [
    "BUNDLED_DIR",
    "SuiteError",
    "list_suites",
    "load_suite",
    "resolve_suites",
]

BUNDLED_DIR = Path(__file__).parent / "suites"


class SuiteError(ValueError):
    """A suite file is missing or invalid."""


def load_suite(path: str | Path) -> ProbeSuite:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SuiteError(f"suite file not found: {p}") from exc
    except (OSError, ValueError) as exc:
        raise SuiteError(f"cannot read suite {p}: {exc}") from exc
    try:
        return ProbeSuite.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(x) for x in first["loc"]) or "suite"
        raise SuiteError(f"invalid suite {p.name}: {where}: {first['msg']}") from exc


def list_suites(extra_dirs: list[str | Path] | None = None) -> dict[str, ProbeSuite]:
    """Bundled suites, then any ``*.json`` in ``extra_dirs`` (later ones override by name)."""
    found: dict[str, ProbeSuite] = {}
    for directory in [BUNDLED_DIR, *(Path(d) for d in extra_dirs or [])]:
        if not directory.is_dir():
            raise SuiteError(f"suite directory not found: {directory}")
        for path in sorted(directory.glob("*.json")):
            suite = load_suite(path)
            found[suite.name] = suite
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return dict(sorted(found.items(), key=lambda kv: (order[kv[1].category], kv[0])))


def resolve_suites(
    spec: str | list[str] | None = "all",
    *,
    extra_dirs: list[str | Path] | None = None,
    limit: int | None = None,
) -> list[ProbeSuite]:
    """``"all"``, a comma list or list of suite names / category names / JSON file paths.

    ``limit`` keeps only the first N probes of every suite (quick smoke runs).
    """
    available = list_suites(extra_dirs)
    names: list[str]
    if spec is None or spec == "all" or spec == ["all"]:
        names = ["all"]
    elif isinstance(spec, str):
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = [s.strip() for s in spec if s.strip()]
    if not names:
        names = ["all"]
    out: list[ProbeSuite] = []
    for name in names:
        if name == "all":
            out += list(available.values())
        elif name in available:
            out.append(available[name])
        elif name.endswith(".json") or Path(name).is_file():
            out.append(load_suite(name))
        else:
            raise SuiteError(f"unknown suite {name!r}; available: {', '.join(available)}")
    unique: dict[str, ProbeSuite] = {}
    for s in out:
        unique[s.name] = s
    result = list(unique.values())
    if limit is not None and limit > 0:
        result = [s.model_copy(update={"probes": s.probes[:limit]}) for s in result]
    return result
