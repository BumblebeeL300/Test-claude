"""Forge: an autonomous, test-driven coding agent built on NVIDIA NIM.

>>> from nimlab.forge import Forge                       # doctest: +SKIP
>>> report = await Forge(client).run("Write fizzbuzz", "./work")   # doctest: +SKIP
"""

from __future__ import annotations

from .events import EventBus, ForgeEvent
from .forge import Forge
from .models import (
    ForgeConfig,
    ForgeReport,
    Issue,
    ModelAssignment,
    PhaseTiming,
    Plan,
    Review,
    Step,
)
from .patching import PatchError, apply_patch_text
from .pytest_parser import PytestReport, parse_pytest_output
from .sandbox import (
    CommandNotAllowedError,
    PathEscapeError,
    Sandbox,
    SandboxError,
    SandboxLimits,
)
from .snapshot import Snapshot

__all__ = [
    "CommandNotAllowedError",
    "EventBus",
    "Forge",
    "ForgeConfig",
    "ForgeEvent",
    "ForgeReport",
    "Issue",
    "ModelAssignment",
    "PatchError",
    "PathEscapeError",
    "PhaseTiming",
    "Plan",
    "PytestReport",
    "Review",
    "Sandbox",
    "SandboxError",
    "SandboxLimits",
    "Snapshot",
    "Step",
    "apply_patch_text",
    "parse_pytest_output",
]
