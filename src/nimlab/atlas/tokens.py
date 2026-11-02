"""A small, dependency-free tokenizer shared by BM25 and the lexical fallbacks."""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# fmt: off
STOPWORDS: frozenset[str] = frozenset((
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "as", "at",
    "by", "from", "but", "not", "no", "do", "does", "did", "can", "could", "should", "would",
    "will", "may", "might", "have", "has", "had", "i", "you", "he", "she", "we", "they", "them",
    "his", "her", "our", "your", "their", "my", "me", "what", "which", "who", "whom", "how",
    "why", "when", "where", "if", "then", "than", "so", "such", "into", "about", "over",
    "under", "there", "here", "also",
))
# fmt: on


def stem(token: str) -> str:
    """Very light suffix stripping so that ``tokens``/``token`` and ``rotating``/``rotate`` meet."""
    n = len(token)
    if n <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and n > 4:
        token = token[:-3] + "y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif token.endswith(("ss", "us", "is")):
        return token
    elif token.endswith("s"):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str, *, stopwords: bool = True, stemming: bool = True) -> list[str]:
    """Lower-case alphanumeric tokens, optionally without stop words and lightly stemmed."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if stopwords and raw in STOPWORDS:
            continue
        out.append(stem(raw) if stemming else raw)
    return out


def token_set(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(tokenize(t))
    return out
