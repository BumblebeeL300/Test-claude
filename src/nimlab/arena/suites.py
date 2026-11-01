"""Eval suite schema (pydantic) and loading of bundled / user-provided suite files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .graders import available_graders, grader_class

__all__ = [
    "BUNDLED_DIR",
    "CATEGORIES",
    "EvalItem",
    "GraderSpec",
    "Suite",
    "SuiteError",
    "list_suites",
    "load_suite",
    "load_suites",
    "resolve_suites",
]

BUNDLED_DIR = Path(__file__).parent / "suites"

#: Task categories shared by suites and the router. ``general_chat`` has no suite of its own.
CATEGORIES: tuple[str, ...] = (
    "math_reasoning",
    "coding",
    "factual_qa",
    "instruction_following",
    "json_compliance",
    "summarization",
    "general_chat",
)

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SuiteError(ValueError):
    """A suite file is missing or invalid."""


class GraderSpec(BaseModel):
    """``{"type": "numeric", "expected": 42}``: ``type`` picks the grader, the rest are params."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" not in data and "type" in data:
            rest = {k: v for k, v in data.items() if k != "type"}
            return {"type": data["type"], "params": rest}
        return data

    @model_validator(mode="after")
    def _check(self) -> GraderSpec:
        try:
            cls = grader_class(self.type)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        cls.validate_params(self.params)  # raises pydantic ValidationError (a ValueError)
        return self

    def flat(self) -> dict[str, Any]:
        return {"type": self.type, **self.params}


class EvalItem(BaseModel):
    """One prompt with its grader. ``reference`` is a gold answer (used by judges / simulation)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str = Field(min_length=1)
    system: str | None = None
    grader: GraderSpec | None = None
    reference: str | None = None
    tags: list[str] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=1)
    weight: float = Field(default=1.0, gt=0)

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError("item id may only contain letters, digits, '_', '-' and '.'")
        return v


class Suite(BaseModel):
    """A named collection of items sharing a category, defaults and (usually) a grader."""

    model_config = ConfigDict(extra="forbid")

    name: str
    title: str = ""
    description: str = ""
    category: str = "general_chat"
    version: str = "1"
    system: str | None = None
    grader: GraderSpec | None = None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)
    open_ended: bool = False
    battle_rubric: str | None = None
    items: list[EvalItem] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _SLUG.match(v):
            raise ValueError("suite name may only contain letters, digits, '_', '-' and '.'")
        return v

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v!r}; use one of {', '.join(CATEGORIES)}")
        return v

    @model_validator(mode="after")
    def _items_valid(self) -> Suite:
        seen: set[str] = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError(f"duplicate item id {item.id!r}")
            seen.add(item.id)
            if item.grader is None and self.grader is None:
                raise ValueError(f"item {item.id!r} has no grader and the suite sets no default")
        return self

    # ------------------------------------------------------------------ helpers
    def grader_for(self, item: EvalItem) -> GraderSpec:
        spec = item.grader or self.grader
        assert spec is not None  # guaranteed by validation
        return spec

    def system_for(self, item: EvalItem) -> str | None:
        return item.system if item.system is not None else self.system

    def max_tokens_for(self, item: EvalItem) -> int:
        return item.max_tokens or self.max_tokens

    def item_hash(self, item: EvalItem) -> str:
        """Fingerprint of everything that determines an item's result (for safe resume)."""
        blob = json.dumps(
            {
                "p": item.prompt,
                "s": self.system_for(item),
                "g": self.grader_for(item).model_dump(),
                "t": self.temperature,
                "m": self.max_tokens_for(item),
            },
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def limited(self, n: int | None) -> Suite:
        """A copy restricted to the first ``n`` items (``None`` = all)."""
        if n is None or n >= len(self.items):
            return self
        return self.model_copy(update={"items": self.items[: max(1, n)]})

    def grader_types(self) -> list[str]:
        return sorted({self.grader_for(i).type for i in self.items})


# ---------------------------------------------------------------------------- loading


def load_suite(path: str | Path) -> Suite:
    """Load and validate one suite JSON file."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SuiteError(f"suite file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise SuiteError(f"{p}: invalid JSON: {exc}") from exc
    try:
        return Suite.model_validate(raw)
    except ValueError as exc:
        raise SuiteError(f"{p}: {exc}") from exc


def load_suites(directory: str | Path | None = None) -> dict[str, Suite]:
    """All ``*.json`` suites in ``directory`` (default: the bundled ones), keyed by name."""
    root = Path(directory) if directory is not None else BUNDLED_DIR
    if not root.is_dir():
        raise SuiteError(f"suite directory not found: {root}")
    suites: dict[str, Suite] = {}
    for path in sorted(root.glob("*.json")):
        suite = load_suite(path)
        if suite.name in suites:
            raise SuiteError(f"{path}: duplicate suite name {suite.name!r}")
        suites[suite.name] = suite
    return suites


def list_suites(extra_dirs: list[str | Path] | None = None) -> dict[str, Suite]:
    """Bundled suites plus those from ``extra_dirs`` (later directories override earlier ones)."""
    suites = load_suites()
    for d in extra_dirs or []:
        suites.update(load_suites(d))
    return suites


def resolve_suites(
    spec: str | list[str] | None,
    *,
    extra_dirs: list[str | Path] | None = None,
    limit: int | None = None,
) -> list[Suite]:
    """Turn ``"all"`` / ``"a,b"`` / paths to suite files into loaded suites.

    Each comma-separated token is a bundled (or ``extra_dirs``) suite name, a path to a ``.json``
    suite file, or ``all``.
    """
    available = list_suites(extra_dirs)
    tokens: list[str]
    if spec is None:
        tokens = ["all"]
    elif isinstance(spec, str):
        tokens = [t.strip() for t in spec.split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in spec if str(t).strip()]
    out: dict[str, Suite] = {}
    for tok in tokens or ["all"]:
        if tok.lower() == "all":
            out.update(available)
        elif tok in available:
            out[tok] = available[tok]
        elif tok.endswith(".json") or Path(tok).is_file():
            suite = load_suite(tok)
            out[suite.name] = suite
        else:
            known = ", ".join(sorted(available))
            raise SuiteError(f"unknown suite {tok!r}; available: {known}")
    return [s.limited(limit) for s in out.values()]


def describe_graders() -> dict[str, str]:  # used by `nimlab arena suites --graders`
    return {name: grader_class(name).description for name in available_graders()}
