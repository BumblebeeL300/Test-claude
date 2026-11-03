"""JSON Schemas for ``extract``: built-ins, resolution of user schemas and a small validator."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import PrismError

__all__ = [
    "BUILTIN_SCHEMAS",
    "BuiltinSchema",
    "resolve_schema",
    "schema_names",
    "validate_instance",
]


@dataclass(frozen=True)
class BuiltinSchema:
    name: str
    description: str
    schema: dict[str, Any]
    sample: Callable[[int], dict[str, Any]]  # deterministic example used by simulation


def _money() -> dict[str, Any]:
    return {"type": "number"}


_RECEIPT: dict[str, Any] = {
    "title": "Receipt",
    "type": "object",
    "properties": {
        "merchant": {"type": "string", "description": "Store or business name"},
        "address": {"type": "string"},
        "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
        "currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": _money(),
                    "total": _money(),
                },
                "required": ["description", "total"],
            },
        },
        "subtotal": _money(),
        "tax": _money(),
        "tip": _money(),
        "total": _money(),
        "payment_method": {"type": "string"},
    },
    "required": ["merchant", "date", "items", "total"],
}

_INVOICE: dict[str, Any] = {
    "title": "Invoice",
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "issue_date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
        "due_date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
        "vendor": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name"],
        },
        "bill_to": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "address": {"type": "string"}},
            "required": ["name"],
        },
        "currency": {"type": "string"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": _money(),
                    "amount": _money(),
                },
                "required": ["description", "amount"],
            },
        },
        "subtotal": _money(),
        "tax": _money(),
        "total_due": _money(),
    },
    "required": ["invoice_number", "issue_date", "vendor", "line_items", "total_due"],
}

_BUSINESS_CARD: dict[str, Any] = {
    "title": "BusinessCard",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "job_title": {"type": "string"},
        "company": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "website": {"type": "string"},
        "address": {"type": "string"},
        "social": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name"],
}

_CHART_DATA: dict[str, Any] = {
    "title": "ChartData",
    "type": "object",
    "properties": {
        "chart_type": {
            "type": "string",
            "enum": ["bar", "line", "pie", "scatter", "area", "other"],
        },
        "title": {"type": "string"},
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}, "y": {"type": "number"}},
                            "required": ["x", "y"],
                        },
                    },
                },
                "required": ["name", "points"],
            },
        },
    },
    "required": ["chart_type", "series"],
}

_MERCHANTS = (
    ("Blue Heron Cafe", "214 Harbor St, Portland, ME 04101"),
    ("Corner Market", "9 Elm Avenue, Austin, TX 78701"),
    ("Juniper Hardware", "1180 Mill Road, Boulder, CO 80302"),
    ("Nori Kitchen", "52 Canal Street, Seattle, WA 98101"),
)
_ITEMS = (
    ("Flat white", 4.5), ("Almond croissant", 3.75), ("Sparkling water", 2.25),
    ("Sourdough loaf", 6.0), ("Cold brew", 5.0), ("Breakfast burrito", 9.5),
    ("Garden salad", 8.25), ("Oat cookie", 2.5),
)  # fmt: skip
_PEOPLE = (
    ("Maya Okafor", "Head of Design", "Lumen Labs"),
    ("Jonas Weber", "Staff Engineer", "Northwind Systems"),
    ("Priya Raman", "Product Manager", "Fathom Analytics"),
    ("Leo Marchetti", "Founder", "Marchetti & Co."),
)
_SERIES = (
    ("Revenue", ("Q1", "Q2", "Q3", "Q4")),
    ("Active users", ("Jan", "Feb", "Mar", "Apr", "May")),
    ("Signups", ("Mon", "Tue", "Wed", "Thu", "Fri")),
)


def _pick(options: tuple[Any, ...], seed: int, salt: int = 0) -> Any:
    return options[(seed >> salt) % len(options)]


def _receipt_sample(seed: int) -> dict[str, Any]:
    merchant, address = _pick(_MERCHANTS, seed)
    items: list[dict[str, Any]] = []
    for k in range(3 + seed % 2):
        desc, price = _ITEMS[(seed // 3 + k * 3) % len(_ITEMS)]
        qty = 1 + (seed >> (k + 2)) % 2
        items.append(
            {
                "description": desc,
                "quantity": qty,
                "unit_price": price,
                "total": round(price * qty, 2),
            }
        )
    subtotal = round(sum(i["total"] for i in items), 2)
    tax = round(subtotal * 0.0825, 2)
    return {
        "merchant": merchant,
        "address": address,
        "date": f"2024-{1 + seed % 12:02d}-{1 + (seed >> 4) % 28:02d}",
        "currency": "USD",
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "tip": 0.0,
        "total": round(subtotal + tax, 2),
        "payment_method": _pick(("Visa ****4242", "Cash", "Mastercard ****1881"), seed, 5),
    }


def _invoice_sample(seed: int) -> dict[str, Any]:
    merchant, address = _pick(_MERCHANTS, seed, 2)
    lines: list[dict[str, Any]] = []
    for k, (desc, rate) in enumerate((("Design sprint", 1800.0), ("Prototype build", 2400.0))):
        qty = 1 + (seed >> k) % 3
        lines.append(
            {"description": desc, "quantity": qty, "unit_price": rate, "amount": rate * qty}
        )
    subtotal = sum(line["amount"] for line in lines)
    tax = round(subtotal * 0.07, 2)
    return {
        "invoice_number": f"INV-{2000 + seed % 900}",
        "issue_date": f"2024-{1 + seed % 12:02d}-03",
        "due_date": f"2024-{1 + seed % 12:02d}-28",
        "vendor": {"name": merchant, "address": address, "email": "billing@example.com"},
        "bill_to": {"name": "Acme Corp", "address": "400 Market St, San Francisco, CA"},
        "currency": "USD",
        "line_items": lines,
        "subtotal": subtotal,
        "tax": tax,
        "total_due": round(subtotal + tax, 2),
    }


def _card_sample(seed: int) -> dict[str, Any]:
    name, title, company = _pick(_PEOPLE, seed)
    slug = name.split()[0].lower()
    return {
        "name": name,
        "job_title": title,
        "company": company,
        "email": f"{slug}@{company.split()[0].lower().strip(',&.')}.example",
        "phone": f"+1 555 01{seed % 10}{(seed >> 3) % 10} {1000 + seed % 9000}",
        "website": f"https://{company.split()[0].lower().strip(',&.')}.example",
        "address": _pick(_MERCHANTS, seed, 1)[1],
        "social": [f"@{slug}"],
    }


def chart_series(seed: int) -> tuple[str, list[str], list[tuple[str, list[float]]], str]:
    """``(title, x labels, [(series, values)], chart_type)`` shared by simulation of two modes."""
    name, xs = _pick(_SERIES, seed)
    base = 20 + seed % 40
    first = [float(base + ((seed >> (i + 1)) % 25) + i * 6) for i in range(len(xs))]
    second = [round(v * (0.55 + ((seed >> i) % 20) / 100), 1) for i, v in enumerate(first)]
    kind = _pick(("bar", "line", "area"), seed, 3)
    return f"{name} by period", list(xs), [(name, first), ("Previous", second)], kind


def _chart_sample(seed: int) -> dict[str, Any]:
    title, xs, series, kind = chart_series(seed)
    return {
        "chart_type": kind,
        "title": title,
        "x_label": "Period",
        "y_label": series[0][0],
        "series": [
            {"name": n, "points": [{"x": x, "y": y} for x, y in zip(xs, ys, strict=True)]}
            for n, ys in series
        ],
    }


BUILTIN_SCHEMAS: dict[str, BuiltinSchema] = {
    s.name: s
    for s in (
        BuiltinSchema(
            "receipt", "Merchant, date, line items, tax and total", _RECEIPT, _receipt_sample
        ),
        BuiltinSchema(
            "invoice", "Vendor, parties, line items and amount due", _INVOICE, _invoice_sample
        ),
        BuiltinSchema(
            "business_card",
            "Name, title, company and contact details",
            _BUSINESS_CARD,
            _card_sample,
        ),
        BuiltinSchema(
            "chart_data", "Chart type, labels and data series", _CHART_DATA, _chart_sample
        ),
    )
}  # fmt: skip


def schema_names() -> list[str]:
    return list(BUILTIN_SCHEMAS)


def _key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_schema(spec: str | Path | dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """``(name, schema)`` from a built-in name, a path to a ``.json`` file, JSON text or a dict."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        spec = "receipt"
    if isinstance(spec, dict):
        schema, name = spec, str(spec.get("title") or "custom")
    else:
        text = str(spec).strip()
        if _key(text) in BUILTIN_SCHEMAS:
            built = BUILTIN_SCHEMAS[_key(text)]
            return built.name, built.schema
        if text.startswith("{"):
            try:
                schema = json.loads(text)
            except json.JSONDecodeError as exc:
                raise PrismError(f"schema is not valid JSON: {exc}") from exc
            name = str(schema.get("title") or "custom") if isinstance(schema, dict) else "custom"
        else:
            path = Path(text).expanduser()
            if not path.is_file():
                raise PrismError(
                    f"unknown schema {text!r}: use one of {', '.join(schema_names())}, "
                    "a path to a JSON Schema file, or inline JSON"
                )
            try:
                schema = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PrismError(f"could not read schema file {path}: {exc}") from exc
            name = str(schema.get("title") or path.stem) if isinstance(schema, dict) else path.stem
    if not isinstance(schema, dict):
        raise PrismError("a schema must be a JSON object")
    if schema.get("type", "object") != "object":
        raise PrismError('the schema root must describe an object (use {"type": "object", ...})')
    return name, schema


# --------------------------------------------------------------------------- validation

_TYPES: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer())
    ),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    node: Any = root
    for part in ref.lstrip("#").strip("/").split("/"):
        if part:
            node = (
                node.get(part.replace("~1", "/").replace("~0", "~"))
                if isinstance(node, dict)
                else None
            )
    return node if isinstance(node, dict) else {}


def validate_instance(
    value: Any, schema: dict[str, Any], *, root: dict[str, Any] | None = None, path: str = "$"
) -> list[str]:
    """Errors (empty when valid) for the commonly used JSON Schema keywords.

    Covers ``type``, ``required``, ``properties``, ``items``, ``enum``, ``const``, ``minimum`` /
    ``maximum``, ``minLength`` / ``maxLength``, ``minItems`` / ``maxItems``, ``anyOf`` / ``oneOf``
    / ``allOf`` and local ``$ref``. Unknown keywords are ignored.
    """
    root = root or schema
    if "$ref" in schema:
        schema = {
            **_resolve_ref(root, str(schema["$ref"])),
            **{k: v for k, v in schema.items() if k != "$ref"},
        }
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        wanted = types if isinstance(types, list) else [types]
        if not any(_TYPES.get(t, lambda _v: True)(value) for t in wanted):
            return [f"{path}: expected {' or '.join(map(str, wanted))}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    for key in ("anyOf", "oneOf"):
        if key in schema and not any(
            not validate_instance(value, sub, root=root, path=path) for sub in schema[key]
        ):
            errors.append(f"{path}: matches none of the {key} options")
    for sub in schema.get("allOf", []):
        errors += validate_instance(value, sub, root=root, path=path)
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        for prop, sub in (schema.get("properties") or {}).items():
            if prop in value:
                errors += validate_instance(value[prop], sub, root=root, path=f"{path}.{prop}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors += validate_instance(item, schema["items"], root=root, path=f"{path}[{i}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            errors.append(f"{path}: does not match pattern {pattern!r}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    return errors
