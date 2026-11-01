"""Structure-aware chunking.

Sections (already split on headings / PDF pages by the loaders) are cut into paragraph blocks that
are packed into chunks of at most ``max_chars``. Fenced code blocks, lists and tables stay intact
when they fit; over-long blocks fall back to lines, sentences and finally words. Consecutive chunks
of one section share ``overlap`` characters, and tiny chunks are merged with a neighbour (re-adding
the neighbour's heading line so no context is lost).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import Chunk, LoadedDocument, Section

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")


@dataclass
class _Piece:
    text: str
    path: tuple[str, ...]
    page: int | None
    section_start: bool


# --------------------------------------------------------------------------- blocks


def split_blocks(text: str) -> list[str]:
    """Split text into paragraph-like blocks; each fenced code block is one block."""
    blocks: list[str] = []
    cur: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if cur:
            block = "\n".join(cur).strip("\n")
            if block.strip():
                blocks.append(block)
            cur.clear()

    for line in text.split("\n"):
        if fence is not None:
            cur.append(line)
            close = _FENCE_CLOSE_RE.match(line)
            if close and close.group(1)[0] == fence[0] and len(close.group(1)) >= len(fence):
                fence = None
                flush()
            continue
        opening = _FENCE_RE.match(line)
        if opening:
            flush()
            fence = opening.group(1)
            cur.append(line)
            continue
        if not line.strip():
            flush()
            continue
        cur.append(line)
    flush()
    return blocks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split on whitespace into pieces <= max_chars; unbreakable words are sliced."""
    out: list[str] = []
    cur = ""
    for word in text.split():
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


def _pack(pieces: Iterable[str], max_chars: int, sep: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
        elif len(cur) + len(sep) + len(piece) <= max_chars:
            cur = f"{cur}{sep}{piece}"
        else:
            out.append(cur)
            cur = piece
    if cur:
        out.append(cur)
    return out


def _split_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    pieces: list[str] = []
    for sentence in _SENT_RE.split(line):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split(sentence, max_chars))
    return _pack(pieces, max_chars, " ")


def _split_code(block: str, max_chars: int) -> list[str]:
    """Split an over-long fenced block by lines, re-wrapping every part in the fence."""
    lines = block.split("\n")
    opener = lines[0]
    closer_match = _FENCE_CLOSE_RE.match(lines[-1]) if len(lines) > 1 else None
    fence_chars = _FENCE_RE.match(opener)
    closer = lines[-1].strip() if closer_match else (fence_chars.group(1) if fence_chars else "```")
    body = lines[1:-1] if closer_match else lines[1:]
    capacity = max(max_chars - len(opener) - len(closer) - 2, 40)
    pieces: list[str] = []
    for line in body:
        pieces.extend(
            [line[i : i + capacity] for i in range(0, len(line), capacity)] if line else [""]
        )
    parts = _pack(pieces, capacity, "\n")
    return [f"{opener}\n{part}\n{closer}" for part in parts]


def split_long(block: str, max_chars: int) -> list[str]:
    """Break a block larger than ``max_chars`` into parts that each fit."""
    if len(block) <= max_chars:
        return [block]
    if _FENCE_RE.match(block):
        return _split_code(block, max_chars)
    lines = block.split("\n")
    if len(lines) > 1:
        pieces: list[str] = []
        for line in lines:
            pieces.extend(_split_line(line, max_chars))
        return _pack(pieces, max_chars, "\n")
    return _split_line(block, max_chars)


# --------------------------------------------------------------------------- chunks


def _overlap_tail(text: str, overlap: int) -> str:
    """Last ``overlap`` chars of ``text`` snapped forward to a word boundary."""
    if overlap <= 0 or len(text) <= overlap or "```" in text[-overlap:]:
        return ""
    tail = text[-overlap:]
    if not text[-overlap - 1].isspace():
        m = re.search(r"\s", tail)
        tail = tail[m.end() :] if m else ""
    return tail.strip()


def _chunk_section(section: Section, max_chars: int, overlap: int) -> list[_Piece]:
    blocks: list[str] = []
    for block in split_blocks(section.text):
        blocks.extend(split_long(block, max_chars))
    texts: list[str] = []
    cur = ""
    for block in blocks:
        if not cur:
            cur = block
        elif len(cur) + 2 + len(block) <= max_chars:
            cur = f"{cur}\n\n{block}"
        else:
            texts.append(cur)
            tail = _overlap_tail(cur, overlap)
            cur = (
                f"{tail}\n\n{block}" if tail and len(tail) + 2 + len(block) <= max_chars else block
            )
    if cur:
        texts.append(cur)
    return [
        _Piece(t, section.heading_path, section.page, section_start=(i == 0))
        for i, t in enumerate(texts)
    ]


def _common_prefix(a: Sequence[str], b: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return tuple(out)


def _joined(a: _Piece, b: _Piece) -> _Piece:
    """Merge ``b`` into ``a``, restating headings that the shared breadcrumb no longer covers."""
    prefix = _common_prefix(a.path, b.path)
    head, addition = a.text, b.text
    if a.path and prefix != a.path:
        head = f"{'#' * len(a.path)} {a.path[-1]}\n\n{head}"
    if b.path and b.section_start and b.path != a.path:
        addition = f"{'#' * len(b.path)} {b.path[-1]}\n\n{addition}"
    return _Piece(f"{head}\n\n{addition}", prefix, a.page, a.section_start)


def _merge_small(pieces: list[_Piece], max_chars: int, min_chars: int) -> list[_Piece]:
    def fits(a: _Piece, b: _Piece) -> bool:
        return a.page == b.page and len(_joined(a, b).text) <= max_chars

    out: list[_Piece] = []
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        while len(cur.text) < min_chars and i + 1 < len(pieces) and fits(cur, pieces[i + 1]):
            cur = _joined(cur, pieces[i + 1])
            i += 1
        if len(cur.text) < min_chars and out and fits(out[-1], cur):
            out[-1] = _joined(out[-1], cur)
        else:
            out.append(cur)
        i += 1
    return out


def chunk_sections(
    sections: Iterable[Section],
    *,
    max_chars: int = 1200,
    overlap: int = 150,
    min_chars: int = 200,
) -> list[Chunk]:
    """Chunk sections into :class:`~nimlab.atlas.models.Chunk` objects (path and page kept)."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    pieces: list[_Piece] = []
    for section in sections:
        if section.text.strip():
            pieces.extend(_chunk_section(section, max_chars, overlap))
    if min_chars > 0:
        pieces = _merge_small(pieces, max_chars, min_chars)
    return [
        Chunk(
            text=re.sub(r"\n{3,}", "\n\n", p.text).strip(),
            heading_path=p.path,
            page=p.page,
            ord=i,
            section_start=p.section_start,
        )
        for i, p in enumerate(pieces)
    ]


def chunk_document(
    doc: LoadedDocument, *, max_chars: int = 1200, overlap: int = 150, min_chars: int = 200
) -> list[Chunk]:
    return chunk_sections(doc.sections, max_chars=max_chars, overlap=overlap, min_chars=min_chars)
