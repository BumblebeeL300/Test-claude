"""A tiny JSON-Schema subset validator (no dependency): enough for node params and ``json`` nodes.

Supported keywords: ``type`` (string or list), ``enum``, ``const``, ``properties``, ``required``,
``additionalProperties: false``, ``items``, ``minItems``/``maxItems``, ``minLength``/``maxLength``,
``minimum``/``maximum``, ``pattern``, ``anyOf``. Unknown keywords are ignored.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pydantic import BaseModel

__all__ = ["JsonSchemaModel", "schema_model", "validate"]

_NAMES: dict[type, str] = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    list: "array",
    dict: "object",
}


def _name(value: Any) -> str:
    return "null" if value is None else _NAMES.get(type(value), type(value).__name__)


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _check_scalar(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and len(pattern) < 300:
            try:
                if not re.search(pattern, value):
                    errors.append(f"{path}: does not match /{pattern}/")
            except re.error:
                pass
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    return errors


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable problems (empty when ``value`` conforms)."""
    if not isinstance(schema, dict):
        return []
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if not any(not validate(value, s, path) for s in any_of if isinstance(s, dict)):
            return [f"{path}: does not match any allowed shape"]
        return []
    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, str(t)) for t in options):
            return [f"{path}: expected {' or '.join(map(str, options))}, got {_name(value)}"]
    errors = _check_scalar(value, schema, path)
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: allows at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        for key, sub in props.items():
            if key in value and isinstance(sub, dict):
                errors.extend(validate(value[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                errors.append(f"{path}: unexpected field(s) {', '.join(extra)}")
    return errors


class JsonSchemaModel(BaseModel):
    """Adapter so ``NIMClient.chat_json`` can be driven by an inline JSON Schema."""

    data: Any = None
    loom_schema: ClassVar[dict[str, Any]] = {}

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return cls.loom_schema

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        problems = validate(obj, cls.loom_schema)
        if problems:
            raise ValueError("; ".join(problems[:6]))
        return cls(data=obj)


def schema_model(schema: dict[str, Any], name: str = "LoomSchema") -> type[JsonSchemaModel]:
    """A model class whose JSON Schema is ``schema`` and whose ``.data`` holds the parsed value."""
    return type(name, (JsonSchemaModel,), {"loom_schema": dict(schema)})
