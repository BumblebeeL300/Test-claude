"""Helpers shared by the Distill tests (importable because conftest puts this folder on sys.path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nimlab.distill import DistillResult, Distiller

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "distill"
DOCS = EXAMPLES / "docs"


async def run_recipe(distiller: Distiller, recipe: str, n: int = 80, **kw: Any) -> DistillResult:
    if recipe == "qa_from_docs":
        kw.setdefault("docs", DOCS)
    return await distiller.run(recipe, n=n, **kw)
