"""Prism-specific exceptions."""

from __future__ import annotations


class PrismError(Exception):
    """Base class for Prism failures (bad input, unusable image, bad schema)."""


class ImageError(PrismError):
    """An image could not be read, is of an unsupported type, or is too large."""


class UnknownModeError(PrismError, KeyError):
    """Requested mode name is not registered."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""
