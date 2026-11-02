"""Text helpers: reasoning-block stripping, robust JSON extraction, chunking, cosine."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

__all__ = [
    "JSONExtractionError",
    "ThinkStreamFilter",
    "chunk_text",
    "cosine",
    "extract_json",
    "strip_think",
]

_OPEN = "<think>"
_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_think(text: str) -> tuple[str, str | None]:
    """Split ``<think>...</think>`` reasoning from the visible answer.

    Returns ``(content, reasoning)``. Handles several blocks, an unterminated ``<think>`` (output
    cut off by ``max_tokens``: everything after it is reasoning) and a lone ``</think>`` with no
    opener (chat templates that pre-fill ``<think>``: everything before it is reasoning).
    ``reasoning`` is ``None`` when there is none. Text without think tags is returned unchanged.
    """
    if not text:
        return "", None
    parts: list[str] = []
    found = False

    def _collect(m: re.Match[str]) -> str:
        parts.append(m.group(1).strip())
        return ""

    content = _THINK_RE.sub(_collect, text)
    if content != text:
        found = True
    opener = _OPEN_RE.search(content)
    if opener:
        parts.append(content[opener.end() :].strip())
        content = content[: opener.start()]
        found = True
    else:
        closer = _CLOSE_RE.search(content)
        if closer:
            parts.append(content[: closer.start()].strip())
            content = content[closer.end() :]
            found = True
    if not found:
        return text, None
    reasoning = "\n\n".join(p for p in parts if p) or None
    return content.strip(), reasoning


def _partial_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of ``tag``."""
    low = buf.lower()
    for n in range(min(len(tag) - 1, len(buf)), 0, -1):
        if low.endswith(tag[:n]):
            return n
    return 0


class ThinkStreamFilter:
    """Incrementally removes ``<think>...</think>`` blocks from a stream of text chunks.

    Tags split across chunk boundaries are handled. Call :meth:`feed` per chunk and
    :meth:`flush` at the end of the stream.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._after_close = False

    def _emit(self, piece: str) -> str:
        if self._after_close:
            piece = piece.lstrip()
            if piece:
                self._after_close = False
        return piece

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            low = self._buf.lower()
            if not self._in_think:
                i = low.find(_OPEN)
                if i >= 0:
                    out.append(self._emit(self._buf[:i]))
                    self._buf = self._buf[i + len(_OPEN) :]
                    self._in_think = True
                    continue
                keep = _partial_suffix_len(self._buf, _OPEN)
                cut = len(self._buf) - keep
                out.append(self._emit(self._buf[:cut]))
                self._buf = self._buf[cut:]
                break
            j = low.find(_CLOSE)
            if j >= 0:
                self._buf = self._buf[j + len(_CLOSE) :]
                self._in_think = False
                self._after_close = True
                continue
            keep = _partial_suffix_len(self._buf, _CLOSE)
            self._buf = self._buf[len(self._buf) - keep :]
            break
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._in_think else self._emit(self._buf)
        self._buf = ""
        self._in_think = False
        return rest


class JSONExtractionError(ValueError):
    """No JSON value could be located in the text."""


_FENCE_RE = re.compile(r"```(?:json|JSON|javascript|js)?\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", s)
        if repaired != s:
            return json.loads(repaired)
        raise


def _scan(s: str) -> Any:
    """First JSON object/array embedded in ``s`` (raw_decode at each ``{`` / ``[``)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            # one cheap repair attempt: trailing commas inside this candidate
            tail = s[i:]
            fixed = _TRAILING_COMMA_RE.sub(r"\1", tail)
            if fixed == tail:
                continue
            try:
                value, _ = decoder.raw_decode(fixed)
            except json.JSONDecodeError:
                continue
        if isinstance(value, (dict, list)):
            return value
    raise JSONExtractionError("no JSON object or array found in text")


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model output.

    Strips ``<think>`` blocks, then tries (in order): the whole text, fenced code blocks, and the
    first JSON object/array embedded in surrounding prose. Tolerates trailing commas.
    Raises :class:`JSONExtractionError` (a ``ValueError``) when nothing parses.
    """
    content, _ = strip_think(text)
    content = content.strip()
    if not content:
        raise JSONExtractionError("empty text")
    try:
        return _loads(content)
    except json.JSONDecodeError:
        pass
    for block in _FENCE_RE.findall(content):
        block = block.strip()
        try:
            return _loads(block)
        except json.JSONDecodeError:
            try:
                return _scan(block)
            except JSONExtractionError:
                continue
    return _scan(content)


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _hard_split(s: str, max_chars: int) -> list[str]:
    """Split an over-long sentence on whitespace, falling back to fixed slices."""
    out: list[str] = []
    cur = ""
    for word in s.split():
        while len(word) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:max_chars])
            word = word[max_chars:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur = f"{cur} {word}"
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _units(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Break text into (unit, separator-before) pieces, each <= max_chars."""
    units: list[tuple[str, str]] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for para in paragraphs:
        first = True
        pieces = [para] if len(para) <= max_chars else _SENT_RE.split(para)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            subs = [piece] if len(piece) <= max_chars else _hard_split(piece, max_chars)
            for sub in subs:
                units.append((sub, "\n\n" if first and units else " "))
                first = False
    return units


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Tail of ``chunk`` of at most ``overlap`` chars, starting on a word boundary."""
    if overlap <= 0 or not chunk:
        return ""
    if len(chunk) <= overlap:
        return chunk
    tail = chunk[-overlap:]
    if not chunk[-overlap - 1].isspace():
        m = re.search(r"\s", tail)
        tail = tail[m.end() :] if m else ""
    return tail.strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most ``max_chars`` characters.

    Paragraphs are kept together where possible; over-long paragraphs are split on sentence
    boundaries, then on words. Each chunk after the first starts with up to ``overlap`` characters
    from the end of the previous chunk (snapped to a word boundary) for retrieval context.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    units = _units(text, max_chars)
    if not units:
        return []
    chunks: list[str] = []
    cur = ""
    for unit, sep in units:
        if not cur:
            cur = unit
            continue
        if len(cur) + len(sep) + len(unit) <= max_chars:
            cur = f"{cur}{sep}{unit}"
            continue
        chunks.append(cur)
        tail = _overlap_tail(cur, overlap)
        # make room for the carried overlap; drop it if the unit would not fit alongside
        fits = bool(tail) and len(tail) + 1 + len(unit) <= max_chars
        cur = f"{tail} {unit}" if fits else unit
    if cur:
        chunks.append(cur)
    return chunks


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is the zero vector."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.shape != vb.shape:
        raise ValueError(f"dimension mismatch: {va.shape} vs {vb.shape}")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    val = float(np.dot(va, vb) / (na * nb))
    return max(-1.0, min(1.0, val)) if math.isfinite(val) else 0.0
