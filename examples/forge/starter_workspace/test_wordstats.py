from wordstats import count_words


def test_counts_case_insensitively():
    assert count_words("The the THE cat") == {"the": 3, "cat": 1}


def test_ignores_punctuation():
    assert count_words("Hello, world! Hello?") == {"hello": 2, "world": 1}


def test_empty_text():
    assert count_words("") == {}
