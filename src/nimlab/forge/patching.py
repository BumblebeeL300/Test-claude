"""Search/replace edit blocks (the ``apply_patch`` format).

A patch is one or more blocks::

    <<<<<<< SEARCH
    def add(a, b):
        return a - b
    =======
    def add(a, b):
        return a + b
    >>>>>>> REPLACE

Matching is line based and must be *unique*. Three tiers are tried in order and the first tier
with any match decides: exact lines, lines compared without trailing whitespace, and lines compared
without any indentation (the replacement is then re-indented to the file). Edits are atomic: if any
block fails, nothing is applied. CRLF files keep their line endings.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "PATCH_FORMAT_HELP",
    "EditBlock",
    "PatchError",
    "PatchResult",
    "apply_edit_blocks",
    "apply_patch_text",
    "parse_edit_blocks",
]

PATCH_FORMAT_HELP = (
    "Use one or more blocks:\n"
    "<<<<<<< SEARCH\n<exact existing lines>\n=======\n<replacement lines>\n>>>>>>> REPLACE"
)

_HEAD = re.compile(r"^\s*<{5,9}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^\s*={5,9}\s*$")
_TAIL = re.compile(r"^\s*>{5,9}\s*REPLACE\s*$")


class PatchError(ValueError):
    """The patch is malformed or cannot be applied."""


@dataclass(frozen=True)
class EditBlock:
    """One search/replace pair, each as a list of lines (no trailing newlines)."""

    search: tuple[str, ...]
    replace: tuple[str, ...]


@dataclass
class PatchResult:
    """Result of applying edit blocks to a text."""

    text: str
    applied: int
    tiers: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0


def parse_edit_blocks(patch: str) -> list[EditBlock]:
    """Parse SEARCH/REPLACE blocks. Text outside blocks (fences, file names) is ignored."""
    if not isinstance(patch, str) or not patch.strip():
        raise PatchError(f"empty patch. {PATCH_FORMAT_HELP}")
    blocks: list[EditBlock] = []
    state = "outside"
    search: list[str] = []
    replace: list[str] = []
    for raw in patch.replace("\r\n", "\n").split("\n"):
        if state == "outside":
            if _HEAD.match(raw):
                state, search, replace = "search", [], []
        elif state == "search":
            if _DIVIDER.match(raw):
                state = "replace"
            elif _HEAD.match(raw):
                raise PatchError("found a second SEARCH marker before '=======' divider")
            else:
                search.append(raw)
        elif _TAIL.match(raw):
            blocks.append(EditBlock(tuple(search), tuple(replace)))
            state = "outside"
        elif _HEAD.match(raw) or _DIVIDER.match(raw):
            raise PatchError(
                "unexpected marker inside REPLACE section (missing '>>>>>>> REPLACE'?)"
            )
        else:
            replace.append(raw)
    if state == "search":
        raise PatchError(f"block is missing the '=======' divider. {PATCH_FORMAT_HELP}")
    if state == "replace":
        raise PatchError(f"block is missing the closing '>>>>>>> REPLACE'. {PATCH_FORMAT_HELP}")
    if not blocks:
        raise PatchError(f"no SEARCH/REPLACE blocks found. {PATCH_FORMAT_HELP}")
    return blocks


def _find(lines: Sequence[str], needle: Sequence[str], key: Callable[[str], str]) -> list[int]:
    n = len(needle)
    if n == 0 or n > len(lines):
        return []
    want = [key(x) for x in needle]
    have = [key(x) for x in lines]
    return [i for i in range(len(lines) - n + 1) if have[i : i + n] == want]


def _strip_blank_edges(lines: Sequence[str]) -> list[str]:
    out = list(lines)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(replace: Sequence[str], file_first: str, search_first: str) -> list[str]:
    """Shift ``replace`` so the search block's indentation lines up with the file's."""
    f_ind, s_ind = _indent(file_first), _indent(search_first)
    if f_ind == s_ind:
        return list(replace)
    if len(f_ind) > len(s_ind) and f_ind.endswith(s_ind):
        extra = f_ind[: len(f_ind) - len(s_ind)]
        return [extra + line if line.strip() else line for line in replace]
    if len(s_ind) > len(f_ind) and s_ind.endswith(f_ind):
        drop = s_ind[: len(s_ind) - len(f_ind)]
        return [line[len(drop) :] if line.startswith(drop) else line for line in replace]
    return list(replace)


def _closest_hint(lines: Sequence[str], search: Sequence[str]) -> str:
    """Describe the region of the file that most resembles ``search`` (for error messages)."""
    needle = _strip_blank_edges(search)
    if not needle or not lines:
        return ""
    n = len(needle)
    want = "\n".join(x.strip() for x in needle)
    first = needle[0].strip()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == first]
    if not starts:
        close = difflib.get_close_matches(first, [ln.strip() for ln in lines], n=3, cutoff=0.6)
        starts = [i for i, ln in enumerate(lines) if ln.strip() in close]
    if not starts:
        return ""
    best_ratio, best = 0.0, starts[0]
    for i in starts[:200]:
        window = "\n".join(x.strip() for x in lines[i : i + n])
        ratio = difflib.SequenceMatcher(None, window, want).ratio()
        if ratio > best_ratio:
            best_ratio, best = ratio, i
    if best_ratio < 0.5:
        return ""
    end = min(len(lines), best + max(n, 1) + 1)
    shown = "\n".join(f"{i + 1:>4}| {lines[i]}" for i in range(best, min(end, best + 14)))
    return f"\nClosest region in the file (similarity {best_ratio:.0%}):\n{shown}"


def apply_edit_blocks(original: str, blocks: Sequence[EditBlock]) -> PatchResult:
    """Apply ``blocks`` to ``original`` atomically.

    Raises:
        PatchError: if any block fails to match exactly once, or is otherwise invalid.
    """
    crlf = "\r\n" in original and original.count("\r\n") >= original.count("\n") / 2
    text = original.replace("\r\n", "\n")
    trailing_nl = text.endswith("\n")
    lines = text.split("\n")
    if trailing_nl:
        lines.pop()
    if text == "":
        lines = []
    tiers: list[str] = []
    total = len(blocks)

    for number, block in enumerate(blocks, start=1):
        label = f"block {number}/{total}"
        search = list(block.search)
        replace = list(block.replace)
        if not any(x.strip() for x in search):
            if any(x.strip() for x in lines):
                raise PatchError(
                    f"{label}: empty SEARCH is only allowed for new or empty files; "
                    "use write_file to replace a whole file, or quote the lines to change"
                )
            lines = list(replace)
            tiers.append("create")
            continue

        matches: list[int] = []
        tier = ""
        used_search, used_replace = search, replace
        for tier_name, key in (
            ("exact", lambda s: s),
            ("rstrip", lambda s: s.rstrip()),
        ):
            matches = _find(lines, search, key)
            if matches:
                tier = tier_name
                break
        if not matches:
            trimmed = _strip_blank_edges(search)
            if trimmed:
                matches = _find(lines, trimmed, lambda s: s.strip())
                if matches:
                    tier = "indent-insensitive"
                    used_search = trimmed
                    used_replace = _strip_blank_edges(replace)
        if not matches:
            raise PatchError(
                f"{label}: SEARCH text not found. It must match the file's lines exactly "
                f"(whitespace included).{_closest_hint(lines, search)}"
            )
        if len(matches) > 1:
            where = ", ".join(str(i + 1) for i in matches[:6])
            raise PatchError(
                f"{label}: SEARCH text is ambiguous ({len(matches)} matches, starting at lines "
                f"{where}). Include more surrounding lines so it matches once."
            )
        start = matches[0]
        if tier == "indent-insensitive":
            used_replace = _reindent(used_replace, lines[start], used_search[0])
        lines[start : start + len(used_search)] = used_replace
        tiers.append(tier)

    result = "\n".join(lines)
    if lines and (trailing_nl or text == ""):
        result += "\n"
    if crlf:
        result = result.replace("\n", "\r\n")

    old_lines = original.replace("\r\n", "\n").splitlines()
    new_lines = result.replace("\r\n", "\n").splitlines()
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return PatchResult(text=result, applied=total, tiers=tiers, added=added, removed=removed)


def apply_patch_text(original: str, patch: str) -> PatchResult:
    """Parse ``patch`` and apply it to ``original`` (convenience wrapper)."""
    return apply_edit_blocks(original, parse_edit_blocks(patch))
