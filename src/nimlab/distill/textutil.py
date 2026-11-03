"""Small text helpers shared by the filters, dedup, decontamination and the simulator."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

__all__ = [
    "STOPWORDS",
    "content_words",
    "distinct_n",
    "exact_key",
    "looks_english",
    "ngrams",
    "normalize",
    "repeat_ratio",
    "sentences",
    "word_tokens",
]

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "he",
        "she",
        "his",
        "her",
        "i",
        "if",
        "then",
        "than",
        "so",
        "not",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "about",
        "into",
        "over",
        "under",
        "between",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "when",
        "where",
        "there",
        "here",
        "also",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "other",
        "such",
        "only",
        "just",
        "very",
    ]
)


def word_tokens(text: str) -> list[str]:
    """Lower-cased word tokens (letters and digits, apostrophes kept inside words)."""
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def content_words(text: str) -> list[str]:
    """Word tokens without stop words and one-letter tokens."""
    return [w for w in word_tokens(text) if w not in STOPWORDS and len(w) > 1]


def normalize(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form used for exact duplicate keys."""
    return " ".join(word_tokens(text))


def exact_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(" ".join(text.split()))]
    return [s for s in parts if s]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: unique n-grams / all n-grams over a corpus (1.0 = no repetition at all)."""
    total = 0
    seen: set[tuple[str, ...]] = set()
    for text in texts:
        grams = ngrams(word_tokens(text), n)
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else 0.0


def repeat_ratio(text: str, n: int = 3) -> float:
    """Share of word n-grams that repeat an earlier n-gram of the same text (0 = none)."""
    grams = ngrams(word_tokens(text), n)
    if len(grams) < 4:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values())
    return repeated / len(grams)


# Words that are very common in English prose and rare elsewhere (Spanish/French/German "a", "de",
# "la" ... are deliberately absent).
_EN_MARKERS = frozenset(
    [
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "are",
        "was",
        "were",
        "that",
        "this",
        "with",
        "for",
        "it",
        "as",
        "on",
        "be",
        "by",
        "at",
        "from",
        "or",
        "an",
        "not",
        "you",
        "we",
        "they",
        "have",
        "has",
        "can",
        "will",
        "your",
        "their",
        "which",
        "what",
        "when",
        "how",
        "why",
    ]
)


def looks_english(text: str) -> bool:
    """Cheap language heuristic: mostly Latin letters and a normal share of English function words.

    Short texts (under 8 words) only fail on non-Latin scripts. This is a heuristic, not language
    identification: it is meant to drop obvious non-English or garbled generations.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    foreign = sum(1 for c in letters if ord(c) > 0x24F)
    if foreign / len(letters) > 0.2:
        return False
    words = word_tokens(text)
    if len(words) < 8:
        return True
    hits = sum(1 for w in words if w in _EN_MARKERS)
    return hits / len(words) >= 0.12
