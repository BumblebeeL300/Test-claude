"""A small dependency-free JSON Schema validator and the ``json_schema`` grader.

Supports the subset of draft 2020-12 that evaluation schemas need: ``type``, ``enum``,
``const``, ``properties`` / ``required`` / ``additionalProperties``, ``items`` / ``prefixItems``,
size and range keywords, ``pattern``, ``format`` (date, date-time, email), the combinators
``allOf`` / ``anyOf`` / ``oneOf`` / ``not`` and local ``$ref`` (``#/$defs/...``).
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nimlab.core.text import JSONExtractionError, extract_json, strip_think

from .base import GradeContext, Grader, GradeResult, register_grader

__all__ = ["validate_json_schema"]

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _check_format(value: str, fmt: str) -> bool:
    try:
        if fmt == "date":
            return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)) and bool(date.fromisoformat(value))
        if fmt == "date-time":
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        if fmt == "email":
            return bool(_EMAIL.match(value))
    except ValueError:
        return False
    return True  # unknown formats are annotations only


def _resolve(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in [p for p in ref[1:].split("/") if p]:
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _validate(inst: Any, schema: Any, root: dict[str, Any], path: str, errs: list[str]) -> None:
    if schema is True or schema == {}:
        return
    if schema is False:
        errs.append(f"{path}: no value is allowed here")
        return
    if "$ref" in schema:
        _validate(inst, _resolve(schema["$ref"], root), root, path, errs)

    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(inst, n) for n in names):
            errs.append(f"{path}: expected {'|'.join(names)}, got {type(inst).__name__}")
            return  # further keywords would only produce noise
    if "enum" in schema and not any(_canon(inst) == _canon(v) for v in schema["enum"]):
        errs.append(f"{path}: {inst!r} is not one of {schema['enum']!r}")
    if "const" in schema and _canon(inst) != _canon(schema["const"]):
        errs.append(f"{path}: expected constant {schema['const']!r}, got {inst!r}")

    if isinstance(inst, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in inst:
                errs.append(f"{path}: missing required property {name!r}")
        for name, sub in props.items():
            if name in inst:
                _validate(inst[name], sub, root, f"{path}.{name}", errs)
        extra = schema.get("additionalProperties", True)
        for name in inst:
            if name in props:
                continue
            if extra is False:
                errs.append(f"{path}: unexpected property {name!r}")
            elif isinstance(extra, dict):
                _validate(inst[name], extra, root, f"{path}.{name}", errs)
        if "minProperties" in schema and len(inst) < schema["minProperties"]:
            errs.append(f"{path}: fewer than {schema['minProperties']} properties")
        if "maxProperties" in schema and len(inst) > schema["maxProperties"]:
            errs.append(f"{path}: more than {schema['maxProperties']} properties")

    if isinstance(inst, list):
        prefix = schema.get("prefixItems", [])
        for i, sub in enumerate(prefix[: len(inst)]):
            _validate(inst[i], sub, root, f"{path}[{i}]", errs)
        items = schema.get("items")
        if isinstance(items, (dict, bool)):
            for i in range(len(prefix), len(inst)):
                _validate(inst[i], items, root, f"{path}[{i}]", errs)
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: has {len(inst)} items, need at least {schema['minItems']}")
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: has {len(inst)} items, need at most {schema['maxItems']}")
        if schema.get("uniqueItems") and len({_canon(v) for v in inst}) != len(inst):
            errs.append(f"{path}: items are not unique")

    if isinstance(inst, str):
        if "minLength" in schema and len(inst) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(inst) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            errs.append(f"{path}: {inst!r} does not match pattern {schema['pattern']!r}")
        if "format" in schema and not _check_format(inst, schema["format"]):
            errs.append(f"{path}: {inst!r} is not a valid {schema['format']}")

    if isinstance(inst, (int, float)) and not isinstance(inst, bool):
        if "minimum" in schema and inst < schema["minimum"]:
            errs.append(f"{path}: {inst} is below minimum {schema['minimum']}")
        if "maximum" in schema and inst > schema["maximum"]:
            errs.append(f"{path}: {inst} is above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and inst <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {inst} must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and inst >= schema["exclusiveMaximum"]:
            errs.append(f"{path}: {inst} must be < {schema['exclusiveMaximum']}")
        if "multipleOf" in schema:
            q = inst / schema["multipleOf"]
            if not math.isclose(q, round(q), abs_tol=1e-9):
                errs.append(f"{path}: {inst} is not a multiple of {schema['multipleOf']}")

    for sub in schema.get("allOf", []):
        _validate(inst, sub, root, path, errs)
    if "anyOf" in schema:
        if not any(_sub_ok(inst, s, root, path) for s in schema["anyOf"]):
            errs.append(f"{path}: does not match any allowed alternative (anyOf)")
    if "oneOf" in schema:
        n = sum(1 for s in schema["oneOf"] if _sub_ok(inst, s, root, path))
        if n != 1:
            errs.append(f"{path}: matches {n} alternatives, need exactly one (oneOf)")
    if "not" in schema and _sub_ok(inst, schema["not"], root, path):
        errs.append(f"{path}: must not match the 'not' schema")


def _sub_ok(inst: Any, schema: Any, root: dict[str, Any], path: str) -> bool:
    sub: list[str] = []
    _validate(inst, schema, root, path, sub)
    return not sub


def validate_json_schema(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable violations (empty when ``instance`` is valid)."""
    errors: list[str] = []
    _validate(instance, schema, schema, "$", errors)
    return errors


def _check_schema_shape(schema: Any, where: str = "$") -> None:
    if isinstance(schema, bool) or schema is None:
        return
    if not isinstance(schema, dict):
        raise ValueError(f"{where}: schema must be an object")
    t = schema.get("type")
    for name in [t] if isinstance(t, str) else (t or []):
        if name not in _TYPES:
            raise ValueError(f"{where}: unknown type {name!r}")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise ValueError(f"{where}: invalid pattern: {exc}") from exc
    for key, sub in schema.get("properties", {}).items():
        _check_schema_shape(sub, f"{where}.{key}")
    if isinstance(schema.get("items"), dict):
        _check_schema_shape(schema["items"], f"{where}[]")
    for key in ("allOf", "anyOf", "oneOf"):
        for i, sub in enumerate(schema.get(key, [])):
            _check_schema_shape(sub, f"{where}.{key}[{i}]")


class JsonSchemaParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # `schema` is the natural key in suite files but shadows a BaseModel attribute, hence the alias
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = False

    def model_post_init(self, context: Any, /) -> None:
        _check_schema_shape(self.schema_)


def parse_model_json(text: str, *, strict: bool) -> Any:
    """Parse JSON from model output. ``strict`` forbids prose around a single JSON value."""
    content = strip_think(text)[0].strip()
    if strict:
        body = content
        fenced = re.fullmatch(r"```(?:json|JSON)?\s*\n(.*?)\n?```", body, re.DOTALL)
        if fenced:
            body = fenced.group(1).strip()
        return json.loads(body)
    return extract_json(content)


@register_grader
class JsonSchemaGrader(Grader):
    """Output must contain JSON that validates against the item's schema (score 1 or 0)."""

    name = "json_schema"
    description = "Output parses as JSON and validates against a JSON Schema."
    Params = JsonSchemaParams

    async def run(self, params: JsonSchemaParams, ctx: GradeContext) -> GradeResult:
        try:
            value = parse_model_json(ctx.output, strict=params.strict)
        except (ValueError, JSONExtractionError) as exc:
            return GradeResult(score=0.0, passed=False, detail=f"no valid JSON: {exc}")
        errors = validate_json_schema(value, params.schema_)
        if errors:
            shown = "; ".join(errors[:4]) + (f" (+{len(errors) - 4} more)" if len(errors) > 4 else "")
            return GradeResult(
                score=0.0, passed=False, detail=f"schema violations: {shown}",
                extracted=json.dumps(value)[:200],
            )
        return GradeResult(
            score=1.0, passed=True, detail="valid against schema", extracted=json.dumps(value)[:200]
        )
