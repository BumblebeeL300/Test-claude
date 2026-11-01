"""Shared fixtures for the Forge test-suite (everything runs offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The suite runs with --import-mode=importlib, which does not put test directories on sys.path.
sys.path.insert(0, str(Path(__file__).parent))

from forge_fakes import FakeClient  # noqa: E402

from nimlab.forge import Sandbox  # noqa: E402

FIZZBUZZ_BUGGY = '''\
def fizzbuzz(n: int) -> str:
    """Return the FizzBuzz word for ``n``."""
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 15 == 0:
        return "FizzBuzz"
    return str(n)
'''

FIZZBUZZ_TESTS = '''\
import pytest

from fizzbuzz import fizzbuzz


def test_plain_number():
    assert fizzbuzz(1) == "1"


def test_fizz():
    assert fizzbuzz(9) == "Fizz"


def test_buzz():
    assert fizzbuzz(10) == "Buzz"


def test_fizzbuzz_15():
    assert fizzbuzz(15) == "FizzBuzz"


@pytest.mark.parametrize("n", [30, 45, 60])
def test_multiples_of_15(n):
    assert fizzbuzz(n) == "FizzBuzz"
'''

FIZZBUZZ_PATCH = '''\
<<<<<<< SEARCH
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 15 == 0:
        return "FizzBuzz"
=======
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
>>>>>>> REPLACE
'''

PLAN_JSON = {
    "goal": "Implement fizzbuzz(n) with a pytest suite",
    "steps": [
        {"id": 1, "description": "Write tests for plain numbers, 3, 5 and 15", "files": ["test_fizzbuzz.py"]},
        {"id": 2, "description": "Implement fizzbuzz", "files": ["fizzbuzz.py"]},
    ],
    "test_strategy": "pytest with parametrized multiples of 15",
    "risks": ["multiples of 15 must be checked before 3 and 5"],
}

APPROVE_JSON = {"verdict": "approve", "summary": "Looks correct and well tested.", "issues": []}


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """An empty workspace directory."""
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def sandbox(ws: Path) -> Sandbox:
    return Sandbox(ws)


@pytest.fixture
def fizzbuzz_sources() -> dict[str, str]:
    return {"buggy": FIZZBUZZ_BUGGY, "tests": FIZZBUZZ_TESTS, "patch": FIZZBUZZ_PATCH}


@pytest.fixture
def fake_client_cls() -> type[FakeClient]:
    return FakeClient
