"""Small word-statistics helpers."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9]+")


def count_words(text: str) -> dict[str, int]:
    """Return how often each (lower-cased) word occurs in ``text``."""
    counts: dict[str, int] = {}
    for word in _WORD.findall(text.lower()):
        counts[word] = counts.get(word, 0) + 1
    return counts
