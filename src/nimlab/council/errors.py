"""Council-specific exceptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .records import Trace


class CouncilError(Exception):
    """Base class for council failures. ``trace`` holds whatever was recorded before the failure."""

    def __init__(self, message: str, *, trace: Trace | None = None) -> None:
        super().__init__(message)
        self.trace = trace


class UnknownStrategyError(CouncilError, KeyError):
    """Requested strategy name is not registered."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""


class NotEnoughModelsError(CouncilError):
    """The strategy needs more distinct models than were configured."""


class AllModelsFailedError(CouncilError):
    """Every model in a stage failed, so the run cannot continue."""
