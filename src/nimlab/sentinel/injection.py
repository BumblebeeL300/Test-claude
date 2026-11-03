# ruff: noqa: E501  (regex rule table)
"""Heuristic prompt-injection detection for text that is about to reach a model.

The rules look for the phrasing attacks use to talk to the *model* instead of the reader:
"ignore previous instructions", fake system tags, notes addressed to AI assistants, requests to
reveal hidden prompts, or to send data to a URL. Each rule has a weight; the final score combines
them as a noisy-or, so one strong signal or several weak ones cross the default threshold of 0.5.
Heuristics miss novel phrasing and flag some security write-ups: treat the score as one signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["RULES", "InjectionHit", "InjectionScan", "expand_span", "injection_scan"]

_I = re.IGNORECASE | re.DOTALL

#: ``(rule name, weight, pattern)``
RULES: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "override-instructions",
        0.9,
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(?:previous|prior|above|"
            r"earlier|preceding|all|any|your|the|these)\b[^.\n]{0,30}\b(?:instructions?|prompts?|"
            r"rules?|directions?|guidelines?|polic(?:y|ies)|tasks?|questions?|requests?|quer(?:y|ies))\b",
            _I,
        ),
    ),
    (
        "new-instructions",
        0.6,
        re.compile(r"\bnew\s+(?:system\s+)?(?:instructions?|rules?|prompt|task)\s*[:\-]", _I),
    ),
    (
        "reveal-prompt",
        0.8,
        re.compile(
            r"\b(?:reveal|print|show|repeat|output|leak|display|echo)\b[^.\n]{0,40}\b(?:system prompt|"
            r"hidden (?:prompt|instructions?)|your (?:initial |original )?(?:instructions|prompt|rules)|"
            r"initial prompt|everything above)\b",
            _I,
        ),
    ),
    (
        "persona-switch",
        0.5,
        re.compile(
            r"\byou\s+are\s+now\b|\bpretend\s+(?:to\s+be|you\s+are)\b|"
            r"\bact\s+as\s+(?:if|an?\s+(?:unrestricted|unfiltered)|dan)\b|\bfrom now on,? you\b",
            _I,
        ),
    ),
    (
        "jailbreak-mode",
        0.7,
        re.compile(
            r"\b(?:developer|debug|jailbreak|god|admin)\s+mode\b|\bDAN\b|\bdo anything now\b|"
            r"\bsafety (?:policy|filters?) (?:is |are |has been )?(?:disabled|suspended|off)\b",
            _I,
        ),
    ),
    (
        "fake-role-tag",
        0.6,
        re.compile(
            r"</?(?:system|assistant|instructions?)>|\[\s*(?:system|inst)\s*\]|<\|im_start\|>|"
            r"^\s*#{1,3}\s*(?:system|instructions?)\b|^\s*system\s*:",
            _I | re.MULTILINE,
        ),
    ),
    (
        "system-label",
        0.5,
        re.compile(r"(?-i:\bSYSTEM\s*:)"),
    ),
    (
        "conceal-from-user",
        0.7,
        re.compile(
            r"\b(?:do not|don't|never)\s+(?:tell|inform|mention|reveal|show)\b[^.\n]{0,40}\b(?:user|human|reader)\b",
            _I,
        ),
    ),
    (
        "addressed-to-ai",
        0.7,
        re.compile(
            r"\b(?:note|message|instruction|p\.?s\.?)s?\s+(?:to|for)\s+(?:the\s+)?(?:ai|assistant|llm|"
            r"language model|model|screener|bot|chatbot)\b|"
            r"\b(?:ai|llm|language model)\s+(?:assistants?|models?|agents?)\s+(?:reading|processing|summari[sz]ing)\b|"
            r"\bhidden instruction for\b|\b(?:ai|assistant|chatbot)\s*:\s*(?:do not|don't|skip|ignore|reply|write|print)\b|"
            r"\byour\s+new\s+(?:task|job|goal)\s+is\b",
            _I,
        ),
    ),
    (
        "html-comment-directive",
        0.6,
        re.compile(
            r"<!--[^>]{0,400}?\b(?:ignore|instruction|assistant|ai\b|model|reply with|respond with)[^>]{0,400}?-->",
            _I,
        ),
    ),
    (
        "exfiltration",
        0.6,
        re.compile(
            r"\b(?:send|post|forward|upload|exfiltrate|transmit)\b[^.\n]{0,60}\bto\b[^.\n]{0,20}https?://|"
            r"!\[[^\]]*\]\(https?://[^)\s]*[?&](?:q|data|secret|d)=",
            _I,
        ),
    ),
    (
        "output-only-token",
        0.5,
        re.compile(
            r"\b(?:reply|respond|answer|output|print|write)\s+(?:with\s+)?(?:only|just|exactly)\b[^.\n<>]{0,60}",
            _I,
        ),
    ),
]


@dataclass(frozen=True)
class InjectionHit:
    rule: str
    weight: float
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class InjectionScan:
    score: float
    hits: list[InjectionHit]

    def flagged(self, threshold: float = 0.5) -> bool:
        return self.score >= threshold

    @property
    def rules(self) -> list[str]:
        out: list[str] = []
        for h in self.hits:
            if h.rule not in out:
                out.append(h.rule)
        return out


def injection_scan(text: str) -> InjectionScan:
    """Score ``text`` between 0 (clean) and 1 (almost certainly an injection)."""
    hits: list[InjectionHit] = []
    for name, weight, pattern in RULES:
        for m in pattern.finditer(text):
            hits.append(InjectionHit(name, weight, m.start(), m.end(), m.group()[:120]))
    best: dict[str, float] = {}
    for h in hits:
        best[h.rule] = max(best.get(h.rule, 0.0), h.weight)
    remaining = 1.0
    for w in best.values():
        remaining *= 1.0 - w
    return InjectionScan(round(1.0 - remaining, 4), sorted(hits, key=lambda h: h.start))


def expand_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a hit to its whole sentence / line / HTML comment so redaction leaves clean text."""
    for m in re.finditer(r"<!--.*?-->", text, re.DOTALL):
        if m.start() <= start and end <= m.end():
            return m.start(), m.end()
    left = max(text.rfind(c, 0, start) for c in ".!?\n>") + 1
    stops = [i for i in (text.find(c, end) for c in "\n<") if i != -1]
    right = min(stops) if stops else len(text)
    for c in ".!?":
        i = text.find(c, end, right)
        if i != -1:
            right = min(right, i + 1)
    return left, max(right, end)
