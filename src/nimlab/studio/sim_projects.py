"""Small, real Python projects used by the simulated Forge build.

Each template has a correct implementation, one *planted bug* (a snippet swapped for a wrong one)
and a test-suite in which exactly one test catches that bug. The simulated coder first writes the
buggy code, Forge's own verify step runs real pytest and sees one failure, the coder applies a
SEARCH/REPLACE patch, and the suite goes green.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ProjectTemplate", "pick_template"]


@dataclass(frozen=True)
class ProjectTemplate:
    key: str
    module: str
    goal: str
    steps: tuple[tuple[str, tuple[str, ...]], ...]
    source: str  # the correct module
    good: str  # snippet in ``source`` ...
    bad: str  # ... replaced by this wrong version in the first draft
    tests: str
    failing_test: str
    readme: str
    risks: tuple[str, ...] = field(default_factory=tuple)
    review_note: str = ""

    @property
    def buggy_source(self) -> str:
        assert self.source.count(self.good) == 1, self.key
        return self.source.replace(self.good, self.bad)

    @property
    def patch(self) -> str:
        return f"<<<<<<< SEARCH\n{self.bad}=======\n{self.good}>>>>>>> REPLACE\n"

    @property
    def test_module(self) -> str:
        return f"test_{self.module}.py"

    @property
    def source_file(self) -> str:
        return f"{self.module}.py"


# --------------------------------------------------------------------------- fizzbuzz
_FIZZBUZZ = ProjectTemplate(
    key="fizzbuzz",
    module="fizzbuzz",
    goal="A fizzbuzz(n) function and a fizzbuzz_range helper, covered by pytest tests",
    steps=(
        (
            "Define the FizzBuzz rules and edge cases (multiples of 15, invalid input)",
            ("README.md",),
        ),
        ("Implement fizzbuzz() and fizzbuzz_range()", ("fizzbuzz.py",)),
        (
            "Write pytest tests for plain numbers, 3, 5, 15, ranges and errors",
            ("test_fizzbuzz.py",),
        ),
    ),
    source='''\
"""FizzBuzz: the classic, with input validation."""

from __future__ import annotations


def fizzbuzz(n: int) -> str:
    """Return the FizzBuzz word for ``n`` (or ``str(n)``)."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)


def fizzbuzz_range(stop: int) -> list[str]:
    """FizzBuzz for 1..stop inclusive."""
    return [fizzbuzz(i) for i in range(1, stop + 1)]
''',
    good="""\
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
""",
    bad="""\
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 15 == 0:
        return "FizzBuzz"
""",
    tests="""\
import pytest

from fizzbuzz import fizzbuzz, fizzbuzz_range


def test_plain_number():
    assert fizzbuzz(1) == "1"
    assert fizzbuzz(7) == "7"


def test_fizz():
    assert fizzbuzz(9) == "Fizz"


def test_buzz():
    assert fizzbuzz(10) == "Buzz"


def test_fizzbuzz_for_multiples_of_15():
    assert fizzbuzz(15) == "FizzBuzz"


def test_range_has_expected_length_and_ends():
    out = fizzbuzz_range(5)
    assert len(out) == 5
    assert out[0] == "1" and out[-1] == "Buzz"


def test_rejects_non_positive():
    with pytest.raises(ValueError):
        fizzbuzz(0)
""",
    failing_test="test_fizzbuzz.py::test_fizzbuzz_for_multiples_of_15",
    readme="# FizzBuzz\n\nTiny, fully tested FizzBuzz.\n\n```python\nfrom fizzbuzz import fizzbuzz_range\nprint(fizzbuzz_range(15))\n```\n\nRun the tests with `pytest`.\n",
    risks=("multiples of 15 must be checked before 3 and 5",),
    review_note="Rules are checked in the right order and invalid input is rejected.",
)

# --------------------------------------------------------------------------- todo
_TODO = ProjectTemplate(
    key="todo",
    module="todo",
    goal="An in-memory TodoList with add, complete, remove and summary operations",
    steps=(
        ("Sketch the TodoList API and error behaviour", ("README.md",)),
        ("Implement TodoList (add, complete, remove, pending, summary)", ("todo.py",)),
        ("Write tests including the unknown-id error paths", ("test_todo.py",)),
    ),
    source='''\
"""A tiny todo list."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    id: int
    title: str
    done: bool = False


class TodoList:
    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    def add(self, title: str) -> Task:
        if not title.strip():
            raise ValueError("title must not be empty")
        task = Task(self._next_id, title.strip())
        self._tasks[task.id] = task
        self._next_id += 1
        return task

    def complete(self, task_id: int) -> Task:
        task = self._tasks[task_id]
        task.done = True
        return task

    def remove(self, task_id: int) -> None:
        if task_id not in self._tasks:
            raise KeyError(task_id)
        del self._tasks[task_id]

    def pending(self) -> list[Task]:
        return [t for t in self._tasks.values() if not t.done]

    def summary(self) -> str:
        done = sum(t.done for t in self._tasks.values())
        return f"{done}/{len(self._tasks)} done"
''',
    good="""\
        if task_id not in self._tasks:
            raise KeyError(task_id)
        del self._tasks[task_id]
""",
    bad="""\
        self._tasks.pop(task_id, None)
""",
    tests="""\
import pytest

from todo import TodoList


def test_add_assigns_increasing_ids():
    todos = TodoList()
    assert [todos.add("a").id, todos.add("b").id] == [1, 2]


def test_add_rejects_blank_titles():
    with pytest.raises(ValueError):
        TodoList().add("   ")


def test_complete_and_pending():
    todos = TodoList()
    a, b = todos.add("write"), todos.add("test")
    todos.complete(a.id)
    assert [t.title for t in todos.pending()] == ["test"]
    assert b.done is False


def test_summary():
    todos = TodoList()
    todos.complete(todos.add("one").id)
    todos.add("two")
    assert todos.summary() == "1/2 done"


def test_complete_unknown_raises():
    with pytest.raises(KeyError):
        TodoList().complete(42)


def test_remove_unknown_id_raises():
    with pytest.raises(KeyError):
        TodoList().remove(7)
""",
    failing_test="test_todo.py::test_remove_unknown_id_raises",
    readme="# Todo\n\nAn in-memory todo list.\n\n```python\nfrom todo import TodoList\n\ntodos = TodoList()\ntodos.add('ship it')\nprint(todos.summary())\n```\n",
    risks=("removing or completing an unknown id must fail loudly",),
    review_note="Unknown ids raise KeyError consistently; titles are validated.",
)

# --------------------------------------------------------------------------- slugify
_SLUG = ProjectTemplate(
    key="slugify",
    module="slugify",
    goal="slugify(text): turn titles into clean, URL-safe slugs",
    steps=(
        ("List the slug rules: lower-case, ASCII, single hyphens, trimmed", ("README.md",)),
        ("Implement slugify() with accent folding and hyphen collapsing", ("slugify.py",)),
        (
            "Write tests for accents, punctuation, repeated separators and empty input",
            ("test_slugify.py",),
        ),
    ),
    source='''\
"""Turn arbitrary text into URL-friendly slugs."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str, max_length: int = 80) -> str:
    """Lower-case ASCII slug with single hyphens, e.g. ``"Hello, World!" -> "hello-world"``."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:max_length].rstrip("-")
''',
    good="""\
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
""",
    bad="""\
    slug = re.sub(r"[^a-z0-9]", "-", folded.lower()).strip("-")
""",
    tests="""\
from slugify import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_strips_punctuation_and_edges():
    assert slugify("  Hello!  ") == "hello"


def test_folds_accents():
    assert slugify("Crème brûlée") == "creme-brulee"


def test_collapses_repeated_separators():
    assert slugify("a   b___c") == "a-b-c"


def test_empty_and_symbols_only():
    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_max_length_does_not_leave_trailing_hyphen():
    assert slugify("one two three", max_length=4) == "one"
""",
    failing_test="test_slugify.py::test_collapses_repeated_separators",
    readme="# slugify\n\n```python\nfrom slugify import slugify\n\nslugify('Crème brûlée, please!')  # 'creme-brulee-please'\n```\n",
    risks=("runs of separators must collapse to one hyphen",),
    review_note="Accent folding and hyphen handling look correct; max_length trims cleanly.",
)

# --------------------------------------------------------------------------- lru cache
_LRU = ProjectTemplate(
    key="lru",
    module="lru_cache",
    goal="An LRUCache class with O(1) get/put and least-recently-used eviction",
    steps=(
        ("Define the eviction contract: reads count as use", ("README.md",)),
        ("Implement LRUCache on top of OrderedDict", ("lru_cache.py",)),
        ("Write tests for capacity, eviction order and recency updates", ("test_lru_cache.py",)),
    ),
    source='''\
"""A small least-recently-used cache."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data
''',
    good="""\
        self._data.move_to_end(key)
        return self._data[key]
""",
    bad="""\
        return self._data[key]
""",
    tests="""\
import pytest

from lru_cache import LRUCache


def test_put_and_get():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing", 0) == 0


def test_evicts_oldest_when_full():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert "a" not in cache and len(cache) == 2


def test_overwrite_does_not_grow():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("a", 2)
    assert len(cache) == 1 and cache.get("a") == 2


def test_get_refreshes_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert "a" in cache and "b" not in cache


def test_rejects_bad_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)
""",
    failing_test="test_lru_cache.py::test_get_refreshes_recency",
    readme="# LRU cache\n\n```python\nfrom lru_cache import LRUCache\n\ncache = LRUCache(2)\ncache.put('a', 1)\n```\n",
    risks=("a read must count as a use, otherwise hot keys get evicted",),
    review_note="Eviction order is correct and reads refresh recency.",
)

# --------------------------------------------------------------------------- word stats
_WORDS = ProjectTemplate(
    key="wordstats",
    module="wordstats",
    goal="Word-frequency helpers: word_counts, top_words and average_word_length",
    steps=(
        ("Decide tokenisation: letters and apostrophes, case-insensitive", ("README.md",)),
        ("Implement word_counts(), top_words() and average_word_length()", ("wordstats.py",)),
        ("Write tests for casing, ties, punctuation and empty text", ("test_wordstats.py",)),
    ),
    source='''\
"""Word statistics for plain text."""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z']+")


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def word_counts(text: str) -> Counter[str]:
    return Counter(words(text))


def top_words(text: str, n: int = 3) -> list[tuple[str, int]]:
    """Most common words; ties are broken alphabetically."""
    counts = word_counts(text)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def average_word_length(text: str) -> float:
    found = words(text)
    return sum(map(len, found)) / len(found) if found else 0.0
''',
    good="""\
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
""",
    bad="""\
    return counts.most_common(n)
""",
    tests="""\
from wordstats import average_word_length, top_words, word_counts


def test_counts_ignore_case_and_punctuation():
    assert word_counts("The cat. THE dog!") == {"the": 2, "cat": 1, "dog": 1}


def test_top_words_orders_by_count():
    assert top_words("b a b c b a", 2) == [("b", 3), ("a", 2)]


def test_top_words_breaks_ties_alphabetically():
    assert top_words("pear apple fig", 2) == [("apple", 1), ("fig", 1)]


def test_average_word_length():
    assert average_word_length("aa bbbb") == 3.0


def test_empty_text():
    assert word_counts("") == {}
    assert average_word_length("") == 0.0
""",
    failing_test="test_wordstats.py::test_top_words_breaks_ties_alphabetically",
    readme="# wordstats\n\n```python\nfrom wordstats import top_words\n\ntop_words('to be or not to be', 2)\n```\n",
    risks=("equal counts need a deterministic order",),
    review_note="Deterministic tie-breaking and sensible tokenisation.",
)

# --------------------------------------------------------------------------- calculator
_CALC = ProjectTemplate(
    key="calculator",
    module="calculator",
    goal="A safe arithmetic expression evaluator (no eval) with clear errors",
    steps=(
        ("Define the grammar: + - * / ** and parentheses, numbers only", ("README.md",)),
        ("Implement evaluate() on top of the ast module", ("calculator.py",)),
        (
            "Write tests for precedence, parentheses, errors and unsafe input",
            ("test_calculator.py",),
        ),
    ),
    source='''\
"""A safe calculator: parses with ``ast`` and only allows arithmetic."""

from __future__ import annotations

import ast
import operator as op

_BINARY = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ValueError("division by zero") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def evaluate(expression: str) -> float:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from None
    return _eval(tree.body)
''',
    good="""\
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ValueError("division by zero") from None
""",
    bad="""\
        return _BINARY[type(node.op)](left, right)
""",
    tests="""\
import pytest

from calculator import evaluate


def test_precedence():
    assert evaluate("2 + 3 * 4") == 14


def test_parentheses_and_unary():
    assert evaluate("-(2 + 3) * 2") == -10


def test_power_and_division():
    assert evaluate("2 ** 3 / 4") == 2.0


def test_division_by_zero_is_a_value_error():
    with pytest.raises(ValueError, match="division by zero"):
        evaluate("1 / 0")


def test_rejects_names_and_calls():
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('echo hi')")


def test_rejects_garbage():
    with pytest.raises(ValueError):
        evaluate("2 +")
""",
    failing_test="test_calculator.py::test_division_by_zero_is_a_value_error",
    readme="# calculator\n\n```python\nfrom calculator import evaluate\n\nevaluate('2 + 3 * 4')  # 14\n```\n",
    risks=("never use eval()", "division by zero must be a clean ValueError"),
    review_note="Uses ast, so there is no code-execution path; errors are consistent.",
)

# --------------------------------------------------------------------------- generic
_TEXTKIT = ProjectTemplate(
    key="textkit",
    module="textkit",
    goal="A small text toolkit: title_case, truncate and is_palindrome",
    steps=(
        ("Agree on the helper semantics and edge cases", ("README.md",)),
        ("Implement title_case(), truncate() and is_palindrome()", ("textkit.py",)),
        ("Write tests for each helper, including punctuation and limits", ("test_textkit.py",)),
    ),
    source='''\
"""Handy text helpers."""

from __future__ import annotations

import re

_SMALL = {"a", "an", "and", "of", "the", "to", "in", "on"}


def title_case(text: str) -> str:
    words = text.split()
    out = [w.capitalize() if i == 0 or w.lower() not in _SMALL else w.lower() for i, w in enumerate(words)]
    return " ".join(out)


def truncate(text: str, limit: int, ellipsis: str = "...") -> str:
    if limit < len(ellipsis):
        raise ValueError("limit is shorter than the ellipsis")
    if len(text) <= limit:
        return text
    return text[: limit - len(ellipsis)].rstrip() + ellipsis


def is_palindrome(text: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9]", "", text.lower())
    return cleaned == cleaned[::-1]
''',
    good="""\
    cleaned = re.sub(r"[^a-z0-9]", "", text.lower())
""",
    bad="""\
    cleaned = text.lower()
""",
    tests="""\
import pytest

from textkit import is_palindrome, title_case, truncate


def test_title_case_keeps_small_words_lower():
    assert title_case("the lord of the rings") == "The Lord of the Rings"


def test_truncate_short_text_is_unchanged():
    assert truncate("hello", 10) == "hello"


def test_truncate_adds_ellipsis():
    assert truncate("hello wonderful world", 12) == "hello won..."


def test_truncate_rejects_tiny_limit():
    with pytest.raises(ValueError):
        truncate("hello", 2)


def test_is_palindrome_ignores_punctuation_and_case():
    assert is_palindrome("A man, a plan, a canal: Panama")


def test_is_palindrome_false_case():
    assert not is_palindrome("hello")
""",
    failing_test="test_textkit.py::test_is_palindrome_ignores_punctuation_and_case",
    readme="# textkit\n\nSmall text helpers: `title_case`, `truncate`, `is_palindrome`.\n",
    risks=("palindromes must ignore case, spaces and punctuation",),
    review_note="Helpers are small, pure and well covered.",
)

_RULES: tuple[tuple[str, ProjectTemplate], ...] = (
    (r"fizz", _FIZZBUZZ),
    (r"todo|to-do|task|checklist|reminder", _TODO),
    (r"slug|permalink|url", _SLUG),
    (r"lru|cache|memoi", _LRU),
    (r"word|frequen|count|wc\b|text stat", _WORDS),
    (r"calc|arithmetic|expression|math", _CALC),
)


def pick_template(prompt: str) -> ProjectTemplate:
    """The template whose keywords appear in ``prompt`` (a generic text toolkit otherwise)."""
    low = prompt.lower()
    for pattern, template in _RULES:
        if re.search(pattern, low):
            return template
    return _TEXTKIT
