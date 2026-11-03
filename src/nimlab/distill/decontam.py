"""Decontamination: drop training samples that overlap a held-out eval set (word n-grams)."""

from __future__ import annotations

from collections.abc import Iterable

from .textutil import ngrams, normalize, word_tokens

__all__ = ["EvalIndex"]


class EvalIndex:
    """An index of the n-grams (and whole normalised texts) of an eval set."""

    def __init__(self, texts: Iterable[str], n: int = 8) -> None:
        self.n = n
        self._grams: set[tuple[str, ...]] = set()
        self._whole: set[str] = set()
        count = 0
        for text in texts:
            tokens = word_tokens(text)
            if not tokens:
                continue
            count += 1
            self._whole.add(normalize(text))
            self._grams.update(ngrams(tokens, n))
        self.size = count

    def overlap(self, text: str) -> float:
        """Share (0-1) of ``text``'s n-grams found in the eval set (1.0 for an exact whole match).

        Texts shorter than ``n`` words have no n-grams; they only match when they are, as a whole,
        an eval text.
        """
        if normalize(text) in self._whole and normalize(text):
            return 1.0
        grams = ngrams(word_tokens(text), self.n)
        if not grams:
            return 0.0
        return sum(1 for g in grams if g in self._grams) / len(grams)

    def contaminated(self, text: str, min_overlap: float) -> bool:
        return self.size > 0 and self.overlap(text) >= min_overlap
