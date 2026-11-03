"""Prism: a vision workbench on NVIDIA NIM vision-language models.

    from nimlab.prism import Prism

    async with Prism() as prism:
        print((await prism.describe("photo.jpg")).text)
        receipt = await prism.extract("receipt.png", schema="receipt")
        page = await prism.ui2code("screenshot.png", refine=2)

Modes: describe, ocr, extract, ui2code, compare, chart, ask (see :func:`list_modes`). Runs offline
in mock mode with deterministic, clearly flagged simulated output.
"""

from __future__ import annotations

from .errors import ImageError, PrismError, UnknownModeError
from .images import PrismImage, dominant_colors, have_pillow, list_images, load_image
from .modes import (
    Mode,
    Option,
    check_html,
    extract_html,
    get_mode,
    list_modes,
    mode_names,
    register_mode,
)
from .prism import BatchItem, Prism, PrismResult, write_jsonl
from .schemas import BUILTIN_SCHEMAS, resolve_schema, schema_names, validate_instance

__all__ = [
    "BUILTIN_SCHEMAS",
    "BatchItem",
    "ImageError",
    "Mode",
    "Option",
    "Prism",
    "PrismError",
    "PrismImage",
    "PrismResult",
    "UnknownModeError",
    "check_html",
    "dominant_colors",
    "extract_html",
    "get_mode",
    "have_pillow",
    "list_images",
    "list_modes",
    "load_image",
    "mode_names",
    "register_mode",
    "resolve_schema",
    "schema_names",
    "validate_instance",
    "write_jsonl",
]
