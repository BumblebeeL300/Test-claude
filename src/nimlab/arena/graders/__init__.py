"""Pluggable graders. Importing this package registers all built-in grader types."""

from __future__ import annotations

from . import basic, checks, code, jsonschema, judge  # noqa: F401  (register built-ins)
from .base import (
    GradeContext,
    Grader,
    GraderError,
    GradeResult,
    available_graders,
    get_grader,
    grader_class,
    register_grader,
)
from .jsonschema import validate_json_schema
from .judge import JudgeVerdict
from .sandbox import SandboxResult, extract_code, run_python_tests

__all__ = [
    "GradeContext",
    "GradeResult",
    "Grader",
    "GraderError",
    "JudgeVerdict",
    "SandboxResult",
    "available_graders",
    "extract_code",
    "get_grader",
    "grader_class",
    "register_grader",
    "run_python_tests",
    "validate_json_schema",
]
