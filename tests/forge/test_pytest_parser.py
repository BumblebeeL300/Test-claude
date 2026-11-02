"""The pytest output parser: counts, failures, collection errors and the condensed LLM report."""

from __future__ import annotations

from pathlib import Path

from nimlab.forge import Sandbox, parse_pytest_output
from nimlab.forge.pytest_parser import PytestReport
from nimlab.forge.tools import WorkspaceTools

SHORT_FAILURES = """\
..F..FFF                                                                 [100%]
=================================== FAILURES ===================================
________________________________ test_fizzbuzz _________________________________
test_fizzbuzz.py:11: in test_fizzbuzz
    assert fizzbuzz(15) == "FizzBuzz"
E   AssertionError: assert 'Fizz' == 'FizzBuzz'
E
E     - FizzBuzz
E     + Fizz
___________________________ test_param[30-FizzBuzz] ____________________________
test_fizzbuzz.py:15: in test_param
    assert fizzbuzz(n) == exp
E   AssertionError: assert 'Fizz' == 'FizzBuzz'
E
E     - FizzBuzz
E     + Fizz
__________________________________ test_error __________________________________
test_fizzbuzz.py:19: in test_error
    d["x"]
E   KeyError: 'x'
_____________________________ TestK.test_in_class ______________________________
test_fizzbuzz.py:23: in test_in_class
    assert [1,2,3] == [1,2,4]
E   assert [1, 2, 3] == [1, 2, 4]
E
E     At index 2 diff: 3 != 4
E     Use -v to get more diff
=========================== short test summary info ============================
FAILED test_fizzbuzz.py::test_fizzbuzz - AssertionError: assert 'Fizz' == 'Fi...
FAILED test_fizzbuzz.py::test_param[30-FizzBuzz] - AssertionError: assert 'Fi...
FAILED test_fizzbuzz.py::test_error - KeyError: 'x'
FAILED test_fizzbuzz.py::TestK::test_in_class - assert [1, 2, 3] == [1, 2, 4]
4 failed, 4 passed in 0.02s
"""

LONG_NESTED = """\
============================= test session starts ==============================
collected 1 item

test_calc.py F                                                           [100%]

=================================== FAILURES ===================================
__________________________________ test_zero ___________________________________

    def test_zero():
>       assert div(1, 0) == 0
               ^^^^^^^^^

test_calc.py:8:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

a = 1, b = 0

    def div(a, b):
>       return helper(a, b)
               ^^^^^^^^^^^^

calc.py:2:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

a = 1, b = 0

    def helper(a, b):
>       return a / b
               ^^^^^
E       ZeroDivisionError: division by zero

calc.py:5: ZeroDivisionError
=========================== short test summary info ============================
FAILED test_calc.py::test_zero - ZeroDivisionError: division by zero
============================== 1 failed in 0.02s ===============================
"""

MIXED = """\
.FsxFE                                                                   [100%]
==================================== ERRORS ====================================
_____________________ ERROR at setup of test_fixture_error _____________________
test_calc.py:24: in boom
    raise RuntimeError("fixture exploded")
E   RuntimeError: fixture exploded
=================================== FAILURES ===================================
__________________________________ test_zero ___________________________________
test_calc.py:8: in test_zero
    assert div(1, 0) == 0
           ^^^^^^^^^
calc.py:2: in div
    return helper(a, b)
           ^^^^^^^^^^^^
calc.py:5: in helper
    return a / b
           ^^^^^
E   ZeroDivisionError: division by zero
_________________________________ test_prints __________________________________
test_calc.py:20: in test_prints
    assert 1 == 2
E   assert 1 == 2
----------------------------- Captured stdout call -----------------------------
debug output here
=========================== short test summary info ============================
FAILED test_calc.py::test_zero - ZeroDivisionError: division by zero
FAILED test_calc.py::test_prints - assert 1 == 2
ERROR test_calc.py::test_fixture_error - RuntimeError: fixture exploded
2 failed, 1 passed, 1 skipped, 1 xfailed, 1 error in 0.04s
"""

COLLECTION_ERRORS = """\
==================================== ERRORS ====================================
_______________________ ERROR collecting test_broken.py ________________________
ImportError while importing test module '/work/proj/test_broken.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
test_broken.py:1: in <module>
    import nonexistent_mod
E   ModuleNotFoundError: No module named 'nonexistent_mod'
_______________________ ERROR collecting test_syntax.py ________________________
/usr/local/lib/python3.13/site-packages/_pytest/python.py:528: in importtestmodule
    mod = import_path(
E     File "/work/proj/test_syntax.py", line 1
E       def test_x(:
E                  ^
E   SyntaxError: invalid syntax
=========================== short test summary info ============================
ERROR test_broken.py
ERROR test_syntax.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors in 0.17s
"""

ALL_PASS = """\
.....                                                                    [100%]
5 passed in 0.01s
"""


class TestCounts:
    def test_all_passing(self) -> None:
        rep = parse_pytest_output(ALL_PASS, exit_code=0)
        assert rep.ok and rep.passed == 5 and rep.failed == 0 and rep.total == 5
        assert rep.duration_s == 0.01
        assert rep.failures == []
        assert rep.condensed() == "pytest: 5 passed (exit code 0)"

    def test_failures_counts(self) -> None:
        rep = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        assert (rep.passed, rep.failed, rep.errors) == (4, 4, 0)
        assert not rep.ok and rep.exit_code == 1

    def test_mixed_outcomes_with_banner_style_footer(self) -> None:
        rep = parse_pytest_output(MIXED, exit_code=1)
        assert (rep.passed, rep.failed, rep.errors, rep.skipped, rep.xfailed) == (1, 2, 1, 1, 1)

    def test_equals_decorated_final_line(self) -> None:
        rep = parse_pytest_output(LONG_NESTED, exit_code=1)
        assert rep.failed == 1 and rep.duration_s == 0.02

    def test_warnings_and_deselected_counts(self) -> None:
        out = "..\n=== 2 passed, 3 deselected, 4 warnings in 0.50s ===\n"
        rep = parse_pytest_output(out, exit_code=0)
        assert (rep.passed, rep.deselected, rep.warnings, rep.duration_s) == (2, 3, 4, 0.5)

    def test_no_tests_ran(self) -> None:
        rep = parse_pytest_output("\nno tests ran in 0.00s\n", exit_code=5)
        assert rep.no_tests and not rep.ok and rep.signature == "NO_TESTS"
        assert "No tests were found" in rep.condensed()

    def test_exit_five_alone_means_no_tests(self) -> None:
        assert parse_pytest_output("", exit_code=5).no_tests

    def test_exit_zero_with_only_skips_is_not_ok(self) -> None:
        rep = parse_pytest_output("s\n1 skipped in 0.01s\n", exit_code=0)
        assert rep.skipped == 1 and not rep.ok

    def test_timeout(self) -> None:
        rep = parse_pytest_output("tests/test_slow.py ..", exit_code=None, timed_out=True)
        assert rep.timed_out and not rep.ok and rep.signature == "TIMEOUT"
        assert "time limit" in rep.condensed()

    def test_usage_error_marks_crash(self) -> None:
        rep = parse_pytest_output(
            "\nno tests ran in 0.00s\n", exit_code=4, stderr="ERROR: file or directory not found: x"
        )
        assert rep.crashed and not rep.ok and not rep.no_tests
        assert rep.signature.startswith("CRASH:")
        assert "file or directory not found: x" in rep.output_tail
        text = rep.condensed()
        assert (
            "could not run" in text and "not found: x" in text and "Create test files" not in text
        )

    def test_empty_output_is_handled(self) -> None:
        rep = parse_pytest_output("", exit_code=1)
        assert not rep.ok and rep.failures == []
        assert rep.condensed().startswith("pytest:")


class TestFailureDetails:
    def test_node_ids_come_from_summary(self) -> None:
        rep = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        assert [f.nodeid for f in rep.failures] == [
            "test_fizzbuzz.py::test_fizzbuzz",
            "test_fizzbuzz.py::test_param[30-FizzBuzz]",
            "test_fizzbuzz.py::test_error",
            "test_fizzbuzz.py::TestK::test_in_class",
        ]

    def test_assertion_messages_and_error_types(self) -> None:
        rep = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        first = rep.failures[0]
        assert first.error_type == "AssertionError"
        assert first.message == "AssertionError: assert 'Fizz' == 'FizzBuzz'"
        assert "- FizzBuzz" in first.detail and "+ Fizz" in first.detail
        assert rep.failures[2].error_type == "KeyError"
        assert rep.failures[3].error_type == "AssertionError"  # bare "assert a == b"

    def test_source_line_and_location(self) -> None:
        rep = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        frame = rep.failures[0].frames[0]
        assert (frame.file, frame.line, frame.func) == ("test_fizzbuzz.py", 11, "test_fizzbuzz")
        assert frame.source == 'assert fizzbuzz(15) == "FizzBuzz"'

    def test_nested_frames_short_style(self) -> None:
        rep = parse_pytest_output(MIXED, exit_code=1)
        zero = next(f for f in rep.failures if f.nodeid.endswith("test_zero"))
        assert [(f.file, f.line) for f in zero.frames] == [
            ("test_calc.py", 8),
            ("calc.py", 2),
            ("calc.py", 5),
        ]
        assert zero.frames[-1].source == "return a / b"

    def test_long_style_traceback(self) -> None:
        rep = parse_pytest_output(LONG_NESTED, exit_code=1)
        (zero,) = rep.failures
        assert zero.nodeid == "test_calc.py::test_zero"
        assert zero.error_type == "ZeroDivisionError"
        assert [f.line for f in zero.frames] == [8, 2, 5]
        assert zero.frames[0].source == "assert div(1, 0) == 0"
        assert zero.frames[-1].source == "return a / b"

    def test_setup_error_and_captured_output(self) -> None:
        rep = parse_pytest_output(MIXED, exit_code=1)
        err = next(f for f in rep.failures if f.kind == "error")
        assert err.nodeid == "test_calc.py::test_fixture_error"
        assert err.error_type == "RuntimeError"
        prints = next(f for f in rep.failures if f.nodeid.endswith("test_prints"))
        assert prints.captured == ["debug output here"]

    def test_collection_errors(self) -> None:
        rep = parse_pytest_output(COLLECTION_ERRORS, exit_code=2, root="/work/proj")
        assert rep.errors == 2 and not rep.ok
        assert [f.kind for f in rep.failures] == ["collection", "collection"]
        broken, syntax = rep.failures
        assert broken.nodeid == "test_broken.py"
        assert broken.error_type == "ModuleNotFoundError"
        assert "nonexistent_mod" in broken.message
        assert syntax.error_type == "SyntaxError" or "SyntaxError" in " ".join(syntax.detail)
        assert "/work/proj" not in " ".join(syntax.detail)  # absolute workspace paths stripped

    def test_framework_frames_are_dropped(self) -> None:
        rep = parse_pytest_output(COLLECTION_ERRORS, exit_code=2, root="/work/proj")
        files = [fr.file for f in rep.failures for fr in f.frames]
        assert all("site-packages" not in f and "/lib/python" not in f for f in files)

    def test_workspace_root_is_made_relative(self) -> None:
        out = SHORT_FAILURES.replace("test_fizzbuzz.py:11:", "/abs/ws/test_fizzbuzz.py:11:")
        rep = parse_pytest_output(out, exit_code=1, root="/abs/ws")
        assert rep.failures[0].frames[0].file == "test_fizzbuzz.py"

    def test_summary_only_output_still_yields_failures(self) -> None:
        out = (
            "=== short test summary info ===\n"
            "FAILED tests/test_a.py::test_x - ValueError: bad\n"
            "ERROR tests/test_b.py::test_y - OSError: nope\n"
            "=== 1 failed, 1 error in 0.1s ===\n"
        )
        rep = parse_pytest_output(out, exit_code=1)
        assert [(f.nodeid, f.kind, f.error_type) for f in rep.failures] == [
            ("tests/test_a.py::test_x", "failure", "ValueError"),
            ("tests/test_b.py::test_y", "error", "OSError"),
        ]


class TestCondensedReport:
    def test_shape_and_content(self) -> None:
        text = parse_pytest_output(SHORT_FAILURES, exit_code=1).condensed()
        assert text.splitlines()[0] == "pytest: 4 passed, 4 failed (exit code 1)"
        assert "FAIL test_fizzbuzz.py::test_fizzbuzz" in text
        assert "at test_fizzbuzz.py:11 in test_fizzbuzz" in text
        assert 'assert fizzbuzz(15) == "FizzBuzz"' in text
        assert "E AssertionError: assert 'Fizz' == 'FizzBuzz'" in text

    def test_nested_frames_show_first_and_innermost_only(self) -> None:
        text = parse_pytest_output(MIXED, exit_code=1).condensed()
        assert "at test_calc.py:8 in test_zero" in text
        assert "at calc.py:5 in helper" in text
        assert "at calc.py:2 " not in text  # the middle frame is trimmed
        assert "captured output:" in text and "debug output here" in text

    def test_limits_failures_and_length(self) -> None:
        body = "".join(
            f"_____ test_{i} _____\nt.py:{i}: in test_{i}\n"
            f"    assert {i} == 0\nE   assert {i} == 0\n"
            for i in range(1, 31)
        )
        summary = "".join(f"FAILED t.py::test_{i} - assert {i} == 0\n" for i in range(1, 31))
        out = (
            f"=== FAILURES ===\n{body}=== short test summary info ===\n{summary}30 failed in 1.0s\n"
        )
        rep = parse_pytest_output(out, exit_code=1)
        assert len(rep.failures) == 30
        text = rep.condensed(max_failures=5)
        assert text.count("FAIL t.py::") == 5
        assert "and 25 more failing test(s)" in text
        assert len(rep.condensed(max_chars=300)) <= 300

    def test_long_messages_are_trimmed(self) -> None:
        out = (
            "=== FAILURES ===\n_____ test_a _____\nt.py:1: in test_a\n    f()\n"
            f"E   ValueError: {'x' * 1000}\n=== short test summary info ===\n"
            "FAILED t.py::test_a - ValueError\n1 failed in 0.1s\n"
        )
        rep = parse_pytest_output(out, exit_code=1)
        assert len(rep.failures[0].message) <= 220


class TestSignature:
    def test_same_failures_same_signature_despite_timing(self) -> None:
        a = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        b = parse_pytest_output(SHORT_FAILURES.replace("0.02s", "9.99s"), exit_code=1)
        assert a.signature == b.signature

    def test_different_failure_sets_differ(self) -> None:
        a = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        b = parse_pytest_output(MIXED, exit_code=1)
        assert a.signature != b.signature

    def test_different_message_for_same_test_differs(self) -> None:
        a = parse_pytest_output(SHORT_FAILURES, exit_code=1)
        b = parse_pytest_output(
            SHORT_FAILURES.replace("assert 'Fizz' ==", "assert 'Buzz' =="), exit_code=1
        )
        assert a.signature != b.signature

    def test_memory_addresses_and_paths_do_not_change_signature(self) -> None:
        tmpl = (
            "=== FAILURES ===\n_____ test_a _____\nt.py:1: in test_a\n    f()\n"
            "E   AssertionError: <Foo object at {addr}> in {path}\n"
            "=== short test summary info ===\n"
            "FAILED t.py::test_a - AssertionError\n1 failed in 0.1s\n"
        )
        a = parse_pytest_output(tmpl.format(addr="0x7f01", path="/tmp/aaa/bbb/ccc.py"), exit_code=1)
        b = parse_pytest_output(tmpl.format(addr="0x7f99", path="/tmp/xxx/yyy/zzz.py"), exit_code=1)
        assert a.signature == b.signature


class TestAgainstRealPytest:
    """End-to-end: run the real pytest through the sandbox and parse what comes out."""

    async def test_real_run_failing_then_passing(self, ws: Path) -> None:
        (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (ws / "test_calc.py").write_text(
            "from calc import add\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n\n"
            "def test_zero():\n    assert add(0, 0) == 0\n"
        )
        tools = WorkspaceTools(Sandbox(ws))
        bad = await tools.run_tests()
        assert not bad.ok and (bad.passed, bad.failed) == (1, 1)
        assert bad.failures[0].nodeid == "test_calc.py::test_add"
        assert "assert -1 == 5" in bad.failures[0].message
        assert str(ws) not in bad.condensed()

        (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        good = await tools.run_tests()
        assert good.ok and good.passed == 2

    async def test_real_run_collection_error(self, ws: Path) -> None:
        (ws / "test_x.py").write_text("import not_a_module_xyz\n\ndef test_x():\n    pass\n")
        rep = await WorkspaceTools(Sandbox(ws)).run_tests()
        assert not rep.ok
        assert rep.failures and rep.failures[0].kind == "collection"
        assert "not_a_module_xyz" in rep.failures[0].message

    async def test_real_run_without_tests(self, ws: Path) -> None:
        (ws / "mod.py").write_text("x = 1\n")
        rep = await WorkspaceTools(Sandbox(ws)).run_tests()
        assert rep.no_tests and not rep.ok

    async def test_real_run_timeout(self, ws: Path) -> None:
        (ws / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n")
        sb = Sandbox(ws)
        proc = await sb.run_pytest(timeout=1.5)
        rep = parse_pytest_output(proc.stdout, exit_code=proc.returncode, timed_out=proc.timed_out)
        assert rep.timed_out and not rep.ok


def test_report_roundtrips_through_json() -> None:
    rep = parse_pytest_output(MIXED, exit_code=1)
    again = PytestReport.model_validate_json(rep.model_dump_json())
    assert again.signature == rep.signature and again.failures[0].frames == rep.failures[0].frames
