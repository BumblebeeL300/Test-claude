"""PII detection and reversible redaction (emails, phones, cards, IBANs, IPs, API keys).

Detection is regex-based with checksum validation where one exists (Luhn for payment cards,
mod-97 for IBANs), which keeps false positives low. :class:`Redactor` replaces matches with stable
tokens such as ``[EMAIL_1]`` and keeps the token -> original mapping in memory so the text can be
restored later; the same value always gets the same token.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "PII_KINDS",
    "PIIMatch",
    "RedactionResult",
    "Redactor",
    "find_pii",
    "iban_valid",
    "luhn_valid",
    "redact",
    "restore",
]

#: Detection order = priority when matches overlap (an API key beats a phone-looking digit run).
PII_KINDS: tuple[str, ...] = ("api_key", "email", "iban", "credit_card", "ip", "phone")
Style = Literal["token", "label", "mask"]


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    start: int
    end: int
    text: str


# ------------------------------------------------------------------------------ validators


def luhn_valid(number: str) -> bool:
    """Luhn checksum of a digit string (spaces and dashes are ignored)."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 12 or any(c not in "0123456789 -" for c in number):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(value: str) -> bool:
    """ISO 13616 mod-97 check (length and country-code shape included)."""
    s = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    moved = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in moved)
    return int(numeric) % 97 == 1


# ------------------------------------------------------------------------------ patterns

_API_KEY = re.compile(
    r"\b(?:nvapi-[A-Za-z0-9_-]{20,}"
    r"|sk-(?:proj-|live-|ant-[a-z0-9]+-)?[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AIza[0-9A-Za-z_-]{35})(?![A-Za-z0-9_-])"
)
_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_CARD = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?![\d]|\.\d)"
)
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_PHONE = re.compile(
    r"(?<![\w@.+-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]\d{2,4}){1,4}(?![\w@-])"
    r"|(?<![\w@.])\+\d{8,15}(?!\w)"
)


def _digits(text: str) -> int:
    return sum(c.isdigit() for c in text)


def _candidates(text: str, kinds: set[str]) -> list[PIIMatch]:
    out: list[PIIMatch] = []
    if "api_key" in kinds:
        out += [PIIMatch("api_key", m.start(), m.end(), m.group()) for m in _API_KEY.finditer(text)]
    if "email" in kinds:
        out += [PIIMatch("email", m.start(), m.end(), m.group()) for m in _EMAIL.finditer(text)]
    if "iban" in kinds:
        for m in _IBAN.finditer(text):
            if iban_valid(m.group()):
                out.append(PIIMatch("iban", m.start(), m.end(), m.group()))
    if "credit_card" in kinds:
        for m in _CARD.finditer(text):
            if 13 <= _digits(m.group()) <= 19 and luhn_valid(m.group()):
                out.append(PIIMatch("credit_card", m.start(), m.end(), m.group()))
    if "ip" in kinds:
        out += [PIIMatch("ip", m.start(), m.end(), m.group()) for m in _IPV4.finditer(text)]
        for m in _IPV6.finditer(text):
            if m.group().count(":") >= 2:
                try:
                    ipaddress.IPv6Address(m.group())
                except ValueError:
                    continue
                out.append(PIIMatch("ip", m.start(), m.end(), m.group()))
    if "phone" in kinds:
        for m in _PHONE.finditer(text):
            n = _digits(m.group())
            plus = m.group().lstrip().startswith("+")
            if (9 <= n <= 15 or (plus and 8 <= n <= 15)) and not re.fullmatch(
                r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", m.group()
            ):
                out.append(PIIMatch("phone", m.start(), m.end(), m.group()))
    return out


def find_pii(text: str, kinds: list[str] | tuple[str, ...] | None = None) -> list[PIIMatch]:
    """Non-overlapping PII matches in document order (higher-priority kinds win overlaps)."""
    wanted = set(kinds) if kinds is not None else set(PII_KINDS)
    unknown = wanted - set(PII_KINDS)
    if unknown:
        raise ValueError(f"unknown PII kind(s): {', '.join(sorted(unknown))}")
    rank = {k: i for i, k in enumerate(PII_KINDS)}
    ordered = sorted(_candidates(text, wanted), key=lambda m: (rank[m.kind], -(m.end - m.start)))
    taken: list[PIIMatch] = []
    for cand in ordered:
        if all(cand.end <= t.start or cand.start >= t.end for t in taken):
            taken.append(cand)
    return sorted(taken, key=lambda m: m.start)


# ------------------------------------------------------------------------------ redaction


@dataclass
class RedactionResult:
    text: str
    matches: list[PIIMatch] = field(default_factory=list)
    #: ``token -> original`` for this call (only for the reversible ``token`` style).
    mapping: dict[str, str] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.matches:
            out[m.kind] = out.get(m.kind, 0) + 1
        return out


class Redactor:
    """Replaces PII with tokens and remembers the mapping so :meth:`restore` can undo it.

    ``style="token"`` gives ``[EMAIL_1]`` (reversible), ``"label"`` gives ``[EMAIL]`` and
    ``"mask"`` gives a block of ``█`` of the same length; the last two are not reversible.
    """

    def __init__(
        self, *, style: Style = "token", kinds: list[str] | tuple[str, ...] | None = None
    ) -> None:
        if style not in ("token", "label", "mask"):
            raise ValueError("style must be 'token', 'label' or 'mask'")
        self.style: Style = style
        self.kinds = tuple(kinds) if kinds is not None else None
        #: ``token -> original``, kept in memory for the lifetime of this redactor.
        self.mapping: dict[str, str] = {}
        self._by_value: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    def _token(self, m: PIIMatch) -> str:
        label = m.kind.upper()
        if self.style == "label":
            return f"[{label}]"
        if self.style == "mask":
            return "█" * len(m.text)
        key = (m.kind, m.text)
        if key not in self._by_value:
            self._counters[m.kind] = self._counters.get(m.kind, 0) + 1
            token = f"[{label}_{self._counters[m.kind]}]"
            self._by_value[key] = token
            self.mapping[token] = m.text
        return self._by_value[key]

    def redact(self, text: str) -> RedactionResult:
        matches = find_pii(text, self.kinds)
        parts: list[str] = []
        pos = 0
        used: dict[str, str] = {}
        tokens: list[str] = []
        for m in matches:
            token = self._token(m)
            tokens.append(token)
            if self.style == "token":
                used[token] = m.text
            parts += [text[pos : m.start], token]
            pos = m.end
        parts.append(text[pos:])
        return RedactionResult("".join(parts), matches, used, tokens)

    def restore(self, text: str, mapping: dict[str, str] | None = None) -> str:
        """Put the original values back (uses this redactor's mapping unless one is given)."""
        table = self.mapping if mapping is None else mapping
        if not table:
            return text
        pattern = re.compile("|".join(re.escape(t) for t in sorted(table, key=len, reverse=True)))
        return pattern.sub(lambda m: table[m.group()], text)

    def clear(self) -> None:
        self.mapping.clear()
        self._by_value.clear()
        self._counters.clear()


def redact(
    text: str, *, style: Style = "token", kinds: list[str] | tuple[str, ...] | None = None
) -> RedactionResult:
    """One-shot redaction (see :class:`Redactor` for session-long, restorable redaction)."""
    return Redactor(style=style, kinds=kinds).redact(text)


def restore(text: str, mapping: dict[str, str]) -> str:
    """Undo a ``token``-style redaction with the mapping it returned."""
    return Redactor().restore(text, mapping)
