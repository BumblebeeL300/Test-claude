# Task: extend the `wordstats` module

The workspace already contains `wordstats.py` with `count_words` and passing tests. Extend it without
breaking existing behaviour:

1. Add `top_words(text: str, n: int = 3) -> list[tuple[str, int]]` returning the `n` most common words
   (case-insensitive) ordered by count descending, then alphabetically.
2. Add `longest_word(text: str) -> str | None` (first longest on ties, `None` for text without words).
3. Make `count_words` treat apostrophes inside words as part of the word (`don't` is one word).

Add tests for every new behaviour next to the existing ones in `test_wordstats.py`.
