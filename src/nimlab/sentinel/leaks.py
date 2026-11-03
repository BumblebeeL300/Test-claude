"""Canary and system-prompt leak detection shared by the judges and the Shield."""

from __future__ import annotations

import re

__all__ = ["canary_hits", "compact", "prompt_overlap"]

_WORD = re.compile(r"\w+")
_STRIP = re.compile(r"[\s\-().+_*`'\"]+")


def compact(text: str) -> str:
    """Lower-case, with whitespace and punctuation removed (``C-A-N`` equals ``can``)."""
    return _STRIP.sub("", text.lower())


def canary_hits(text: str, canaries: list[str]) -> list[str]:
    """The canaries / secrets that appear in ``text`` (case, spacing and punctuation tolerant)."""
    flat = compact(text)
    lowered = text.lower()
    hits: list[str] = []
    for c in canaries:
        if not c or len(c.strip()) < 4:
            continue
        if c.lower() in lowered or compact(c) in flat:
            hits.append(c)
    return hits


def prompt_overlap(prompt: str, text: str, min_words: int = 8) -> list[tuple[int, int]]:
    """Character spans of ``text`` repeating ``min_words`` or more consecutive words of ``prompt``."""
    p_words = [w.lower() for w in _WORD.findall(prompt)]
    if len(p_words) < min_words:
        return []
    grams = {tuple(p_words[i : i + min_words]) for i in range(len(p_words) - min_words + 1)}
    spans = [(m.start(), m.end(), m.group().lower()) for m in _WORD.finditer(text)]
    marked: list[tuple[int, int]] = []
    for i in range(len(spans) - min_words + 1):
        window = tuple(s[2] for s in spans[i : i + min_words])
        if window in grams:
            marked.append((spans[i][0], spans[i + min_words - 1][1]))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(marked):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
