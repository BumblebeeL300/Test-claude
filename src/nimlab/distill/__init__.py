"""Distill: a synthetic dataset foundry built on NVIDIA NIM models.

>>> from nimlab.distill import Distiller
>>> result = await Distiller(mock=True).run("instruction_diversity", n=200)  # doctest: +SKIP
>>> len(result.kept), result.report.funnel[-1].out  # doctest: +SKIP

Recipes describe the dataset; the pipeline seeds topics, generates with several models, scores with
the reward model (judge fallback), filters, deduplicates, decontaminates and exports JSONL plus a
dataset card and an HTML quality report. Runs are checkpointed and resumable.
"""

from __future__ import annotations

from .distiller import Distiller, DistillError, DistillResult
from .export import FORMATS, Record, read_dataset
from .models import RunMeta, RunOptions, Sample, Seed
from .quality import QualityReport
from .recipes import Recipe, RecipeError, get_recipe, list_recipes
from .scoring import parse_reward
from .store import RunStore

__all__ = [
    "FORMATS",
    "DistillError",
    "DistillResult",
    "Distiller",
    "QualityReport",
    "Recipe",
    "RecipeError",
    "Record",
    "RunMeta",
    "RunOptions",
    "RunStore",
    "Sample",
    "Seed",
    "get_recipe",
    "list_recipes",
    "parse_reward",
    "read_dataset",
]
