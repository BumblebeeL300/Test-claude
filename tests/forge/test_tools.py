"""WorkspaceTools: the coder's tool surface, and its tolerance for bad calls."""

# ruff: noqa: ASYNC240

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nimlab.forge import EventBus, Sandbox, SandboxLimits
from nimlab.forge.tools import TOOL_NAMES, TOOL_SPECS, WorkspaceTools, clip


@pytest.fixture
def tools(sandbox: Sandbox) -> WorkspaceTools:
    return WorkspaceTools(sandbox)


def seed(ws: Path, name: str, text: str | bytes) -> Path:
    path = ws / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return path


class TestSchemas:
    def test_all_tools_have_valid_openai_schemas(self) -> None:
        schemas = WorkspaceTools.schemas()
        assert [s["function"]["name"] for s in schemas] == list(TOOL_NAMES)
        assert set(TOOL_NAMES) == {
            "list_dir", "read_file", "write_file", "apply_patch",
            "run_tests", "run_python", "search_code",
        }  # fmt: skip
        for s in schemas:
            assert s["type"] == "function"
            fn = s["function"]
            assert fn["description"]
            params = fn["parameters"]
            assert params["type"] == "object"
            assert set(params["required"]) <= set(params["properties"])
            for prop in params["properties"].values():
                assert prop["type"] in {"string", "integer", "boolean", "array"}
            json.dumps(s)  # serialisable

    def test_list_dir_is_first_for_the_mock_backend(self) -> None:
        # The offline mock calls the first tool with placeholder args; keep that one harmless.
        assert TOOL_SPECS[0].name == "list_dir"
        assert TOOL_SPECS[0].required == ()


class TestListDir:
    async def test_lists_dirs_first_and_hides_noise(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "b.py", "x")
        seed(ws, "a.py", "xx")
        seed(ws, "pkg/m.py", "")
        seed(ws, "__pycache__/c.pyc", "")
        seed(ws, ".git/HEAD", "")
        out = await tools.call("list_dir", {})
        assert out.ok
        lines = out.output.splitlines()
        assert lines[1] == "pkg/"
        assert "a.py (2 B)" in out.output and "b.py (1 B)" in out.output
        assert "__pycache__" not in out.output and ".git" not in out.output

    async def test_recursive_and_subpath(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "pkg/sub/deep.py", "")
        flat = await tools.call("list_dir", {"path": "pkg"})
        assert "sub/" in flat.output and "deep.py" not in flat.output
        deep = await tools.call("list_dir", {"path": ".", "recursive": True})
        assert "deep.py" in deep.output

    async def test_errors(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "f.txt", "x")
        assert not (await tools.call("list_dir", {"path": "missing"})).ok
        assert "not a directory" in (await tools.call("list_dir", {"path": "f.txt"})).output
        assert not (await tools.call("list_dir", {"path": "../"})).ok

    async def test_empty_directory(self, tools: WorkspaceTools) -> None:
        out = await tools.call("list_dir", {})
        assert out.ok and "(empty)" in out.output


class TestReadFile:
    async def test_full_and_ranges(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "n.txt", "\n".join(f"line{i}" for i in range(1, 11)) + "\n")
        full = await tools.call("read_file", {"path": "n.txt"})
        assert full.ok and "(10 lines)" in full.output and "line10" in full.output
        part = await tools.call("read_file", {"path": "n.txt", "start_line": 3, "end_line": 4})
        assert "lines 3-4 of 10" in part.output
        assert part.output.splitlines()[1:] == ["line3", "line4"]
        tail = await tools.call("read_file", {"path": "n.txt", "start_line": 9})
        assert tail.output.splitlines()[1:] == ["line9", "line10"]

    async def test_range_beyond_file(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "n.txt", "a\nb\n")
        out = await tools.call("read_file", {"path": "n.txt", "start_line": 50})
        assert not out.ok and "only 2 lines" in out.output

    async def test_failures(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "blob.bin", b"\x00\x01")
        assert "not found" in (await tools.call("read_file", {"path": "ghost.py"})).output
        assert "binary" in (await tools.call("read_file", {"path": "blob.bin"})).output
        assert not (await tools.call("read_file", {"path": "/etc/passwd"})).ok
        assert not (await tools.call("read_file", {"path": "../secret"})).ok
        assert not (await tools.call("read_file", {"path": ".git/config"})).ok

    async def test_huge_files_point_to_line_ranges(self, ws: Path) -> None:
        tools = WorkspaceTools(Sandbox(ws, SandboxLimits(max_read_bytes=100)))
        seed(ws, "big.txt", "x" * 1000)
        out = await tools.call("read_file", {"path": "big.txt"})
        assert not out.ok and "line range" in out.output

    @pytest.mark.skipif(os.name != "posix", reason="symlinks")
    async def test_symlink_escape_refused(
        self, tools: WorkspaceTools, ws: Path, tmp_path: Path
    ) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        (ws / "innocent.txt").symlink_to(secret)
        out = await tools.call("read_file", {"path": "innocent.txt"})
        assert not out.ok and "classified" not in out.output


class TestWriteFile:
    async def test_create_overwrite_unchanged(self, tools: WorkspaceTools, ws: Path) -> None:
        created = await tools.call("write_file", {"path": "pkg/a.py", "content": "x = 1\n"})
        assert created.ok and created.output.startswith("Created pkg/a.py (1 lines)")
        assert created.changed == ["pkg/a.py"]
        over = await tools.call("write_file", {"path": "pkg/a.py", "content": "x = 2\n"})
        assert over.output.startswith("Overwrote pkg/a.py")
        same = await tools.call("write_file", {"path": "pkg/a.py", "content": "x = 2\n"})
        assert "unchanged" in same.output and same.changed == []
        assert (ws / "pkg" / "a.py").read_text() == "x = 2\n"
        assert tools.changed_files == {"pkg/a.py"}

    async def test_python_syntax_errors_are_flagged_but_written(
        self, tools: WorkspaceTools, ws: Path
    ) -> None:
        out = await tools.call("write_file", {"path": "bad.py", "content": "def f(:\n    pass\n"})
        assert out.ok and "SyntaxError" in out.output and "line 1" in out.output
        assert (ws / "bad.py").exists()
        ok = await tools.call("write_file", {"path": "notes.txt", "content": "def f(:"})
        assert "SyntaxError" not in ok.output

    @pytest.mark.parametrize(
        "path", ["../evil.py", "/tmp/evil.py", "a/../../evil.py", ".forge/x", ".git/x"]
    )
    async def test_escapes_and_reserved_paths_refused(
        self, tools: WorkspaceTools, ws: Path, path: str
    ) -> None:
        out = await tools.call("write_file", {"path": path, "content": "x"})
        assert not out.ok and out.output.startswith("Error")
        assert out.changed == []
        assert not (ws.parent / "evil.py").exists() and not Path("/tmp/evil.py").exists()

    async def test_read_only_sandbox(self, ws: Path) -> None:
        tools = WorkspaceTools(Sandbox(ws, read_only=True))
        out = await tools.call("write_file", {"path": "a.py", "content": "x"})
        assert not out.ok and "read-only" in out.output
        assert not (ws / "a.py").exists()

    async def test_oversized_content_refused(self, ws: Path) -> None:
        tools = WorkspaceTools(Sandbox(ws, SandboxLimits(max_write_bytes=10)))
        out = await tools.call("write_file", {"path": "a.txt", "content": "x" * 11})
        assert not out.ok and "limit" in out.output

    async def test_structured_content_is_serialised_as_json(
        self, tools: WorkspaceTools, ws: Path
    ) -> None:
        out = await tools.call("write_file", {"path": "data.json", "content": {"a": [1, 2]}})
        assert out.ok
        assert json.loads((ws / "data.json").read_text()) == {"a": [1, 2]}

    async def test_empty_content_creates_empty_file(self, tools: WorkspaceTools, ws: Path) -> None:
        out = await tools.call("write_file", {"path": "pkg/__init__.py", "content": ""})
        assert out.ok and (ws / "pkg" / "__init__.py").read_text() == ""

    async def test_numbers_are_coerced_to_text(self, tools: WorkspaceTools, ws: Path) -> None:
        out = await tools.call("write_file", {"path": "n.txt", "content": 12345})
        assert out.ok and (ws / "n.txt").read_text() == "12345"


class TestApplyPatch:
    PATCH = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"

    async def test_applies_and_reports_stats(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "m.py", "x = 1\ny = 2\n")
        out = await tools.call("apply_patch", {"path": "m.py", "patch": self.PATCH})
        assert out.ok and "Applied 1 edit block(s) to m.py (+1/-1 lines)" in out.output
        assert out.changed == ["m.py"]
        assert (ws / "m.py").read_text() == "x = 2\ny = 2\n"

    async def test_failure_leaves_file_untouched_and_explains(
        self, tools: WorkspaceTools, ws: Path
    ) -> None:
        seed(ws, "m.py", "x = 1\ny = 2\n")
        patch = self.PATCH + "<<<<<<< SEARCH\nnot there\n=======\nz\n>>>>>>> REPLACE\n"
        out = await tools.call("apply_patch", {"path": "m.py", "patch": patch})
        assert not out.ok and "patch not applied" in out.output and "block 2/2" in out.output
        assert (ws / "m.py").read_text() == "x = 1\ny = 2\n"
        assert tools.changed_files == set()

    async def test_malformed_patch(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "m.py", "x = 1\n")
        out = await tools.call("apply_patch", {"path": "m.py", "patch": "make it better please"})
        assert not out.ok and "SEARCH" in out.output

    async def test_missing_file_can_only_be_created_with_empty_search(
        self, tools: WorkspaceTools, ws: Path
    ) -> None:
        bad = await tools.call("apply_patch", {"path": "new.py", "patch": self.PATCH})
        assert not bad.ok
        create = "<<<<<<< SEARCH\n=======\nprint('hi')\n>>>>>>> REPLACE\n"
        good = await tools.call("apply_patch", {"path": "new.py", "patch": create})
        assert good.ok and (ws / "new.py").read_text() == "print('hi')\n"

    async def test_syntax_warning_after_patch(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "m.py", "x = 1\n")
        patch = "<<<<<<< SEARCH\nx = 1\n=======\nx = (\n>>>>>>> REPLACE\n"
        out = await tools.call("apply_patch", {"path": "m.py", "patch": patch})
        assert out.ok and "SyntaxError" in out.output

    async def test_path_escape_and_directories(self, tools: WorkspaceTools, ws: Path) -> None:
        (ws / "d").mkdir()
        assert not (await tools.call("apply_patch", {"path": "../x.py", "patch": self.PATCH})).ok
        out = await tools.call("apply_patch", {"path": "d", "patch": self.PATCH})
        assert not out.ok and "not a file" in out.output


class TestRunTools:
    async def test_run_tests_pass_and_fail(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "test_a.py", "def test_a():\n    assert 1 + 1 == 3\n")
        bad = await tools.call("run_tests", {})
        assert (
            not bad.ok
            and "FAIL test_a.py::test_a" in bad.output
            and "assert (1 + 1) == 3" in bad.output
        )
        assert tools.last_pytest is not None and not tools.last_pytest.ok
        seed(ws, "test_a.py", "def test_a():\n    assert 1 + 1 == 2\n")
        good = await tools.call("run_tests", {})
        assert good.ok and "1 passed" in good.output

    async def test_run_tests_with_path_and_keyword(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(
            ws,
            "tests/test_one.py",
            "def test_alpha():\n    pass\n\ndef test_beta():\n    assert False\n",
        )
        out = await tools.call("run_tests", {"path": "tests/test_one.py", "keyword": "alpha"})
        assert out.ok and "1 passed" in out.output
        escape = await tools.call("run_tests", {"path": "../"})
        assert not escape.ok and escape.output.startswith("Error")

    async def test_run_tests_rejects_option_smuggling_via_keyword(
        self, tools: WorkspaceTools, ws: Path
    ) -> None:
        seed(ws, "test_a.py", "def test_a():\n    pass\n")
        out = await tools.call("run_tests", {"keyword": "-p evil"})
        # "-p evil" is a -k expression value, not a flag; pytest just selects nothing
        assert "evil" not in out.output.lower() or not out.ok

    async def test_run_python(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "hello.py", "import sys\nprint('hi', sys.argv[1:])\n")
        out = await tools.call("run_python", {"script": "hello.py", "args": ["a", "b"]})
        assert out.ok and "exit code 0" in out.output and "hi ['a', 'b']" in out.output

    async def test_run_python_failures(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "boom.py", "raise SystemExit(7)\n")
        assert "exit code 7" in (await tools.call("run_python", {"script": "boom.py"})).output
        missing = await tools.call("run_python", {"script": "nope.py"})
        assert not missing.ok and "not allowed" in missing.output
        shell = await tools.call("run_python", {"script": "-c"})
        assert not shell.ok and "not allowed" in shell.output
        assert not (await tools.call("run_python", {"script": "../x.py"})).ok

    async def test_run_python_timeout(self, ws: Path) -> None:
        tools = WorkspaceTools(Sandbox(ws, SandboxLimits(command_timeout_s=1)))
        seed(ws, "hang.py", "while True:\n    pass\n")
        out = await tools.call("run_python", {"script": "hang.py"})
        assert not out.ok and "timed out after 1s" in out.output

    async def test_args_as_json_string_or_plain_string(
        self, tools: WorkspaceTools, ws: Path
    ) -> None:
        seed(ws, "echo.py", "import sys\nprint(sys.argv[1:])\n")
        a = await tools.call("run_python", {"script": "echo.py", "args": '["x", "y"]'})
        b = await tools.call("run_python", {"script": "echo.py", "args": "p q"})
        assert "['x', 'y']" in a.output and "['p', 'q']" in b.output


class TestSearchCode:
    @pytest.fixture(autouse=True)
    def _files(self, ws: Path) -> None:
        seed(ws, "a.py", "def alpha():\n    return 'Needle'\n")
        seed(ws, "pkg/b.py", "# needle in a haystack\nvalue = 3\n")
        seed(ws, "notes.md", "needle docs\n")
        seed(ws, "node_modules/x.js", "needle\n")
        seed(ws, ".forge/hidden.txt", "needle\n")
        seed(ws, "blob.bin", b"\x00needle")

    async def test_regex_search_with_locations(self, tools: WorkspaceTools) -> None:
        out = await tools.call("search_code", {"pattern": r"def \w+\(\)"})
        assert out.ok and out.output.strip() == "a.py:1: def alpha():"

    async def test_case_insensitive_and_glob(self, tools: WorkspaceTools) -> None:
        sensitive = await tools.call("search_code", {"pattern": "needle", "glob": "*.py"})
        assert "pkg/b.py:1" in sensitive.output and "a.py" not in sensitive.output
        insensitive = await tools.call(
            "search_code", {"pattern": "needle", "glob": "*.py", "ignore_case": True}
        )
        assert "a.py:2" in insensitive.output and "pkg/b.py:1" in insensitive.output

    async def test_skips_noise_and_binaries(self, tools: WorkspaceTools) -> None:
        out = await tools.call("search_code", {"pattern": "needle", "ignore_case": True})
        assert "node_modules" not in out.output and ".forge" not in out.output
        assert "blob.bin" not in out.output and "notes.md" in out.output

    async def test_subpath_no_match_and_invalid_regex(self, tools: WorkspaceTools) -> None:
        sub = await tools.call("search_code", {"pattern": "needle", "path": "pkg"})
        assert sub.output.strip() == "pkg/b.py:1: # needle in a haystack"
        none = await tools.call("search_code", {"pattern": "zzz_absent"})
        assert none.ok and "No matches" in none.output
        bad = await tools.call("search_code", {"pattern": "(unclosed"})
        assert bad.ok and "searched literally" in bad.output
        assert not (await tools.call("search_code", {"pattern": "x", "path": "nope"})).ok
        assert not (await tools.call("search_code", {"pattern": "x", "path": "../"})).ok

    async def test_match_limit(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "many.txt", "hit\n" * 500)
        out = await tools.call("search_code", {"pattern": "hit"})
        assert "stopped at 60 matches" in out.output and out.output.count("many.txt:") == 60


class TestRobustness:
    async def test_unknown_tool(self, tools: WorkspaceTools) -> None:
        out = await tools.call("rm_rf", {"path": "/"})
        assert not out.ok and "unknown tool 'rm_rf'" in out.output and "write_file" in out.output

    @pytest.mark.parametrize("name", ["", "run_shell", "bash", "LIST_DIR", "__class__"])
    async def test_odd_tool_names(self, tools: WorkspaceTools, name: str) -> None:
        out = await tools.call(name, {})
        assert not out.ok and "unknown tool" in out.output

    async def test_missing_required_arguments(self, tools: WorkspaceTools) -> None:
        out = await tools.call("write_file", {"path": "a.py"})
        assert not out.ok and "missing required argument(s): content" in out.output
        assert "write_file(path: string, content: string)" in out.output
        both = await tools.call("apply_patch", {})
        assert "path, patch" in both.output

    async def test_wrong_types(self, tools: WorkspaceTools) -> None:
        assert "must be an integer" in (await tools.call("run_tests", {"maxfail": "many"})).output
        assert "must be a boolean" in (await tools.call("list_dir", {"recursive": "maybe"})).output
        assert (
            "must be an integer"
            in (await tools.call("read_file", {"path": "a", "start_line": True})).output
        )
        assert (
            "list of strings"
            in (await tools.call("run_python", {"script": "a.py", "args": 5})).output
        )

    async def test_lenient_coercions(self, tools: WorkspaceTools, ws: Path) -> None:
        seed(ws, "a.txt", "l1\nl2\nl3\n")
        out = await tools.call("read_file", {"path": "a.txt", "start_line": "2", "end_line": 2.0})
        assert out.ok and out.output.splitlines()[1:] == ["l2"]
        assert (await tools.call("list_dir", {"recursive": "true"})).ok

    @pytest.mark.parametrize(
        "arguments", [None, "", "  ", "{}", [], 42, "not json", "[1, 2]", {"path": None}]
    )
    async def test_garbage_arguments_never_raise(
        self, tools: WorkspaceTools, arguments: object
    ) -> None:
        for name in TOOL_NAMES:
            out = await tools.call(name, arguments)
            assert isinstance(out.output, str) and out.output
            assert out.ok is False or name in {"list_dir", "run_tests"}

    async def test_json_string_arguments_are_parsed(self, tools: WorkspaceTools, ws: Path) -> None:
        out = await tools.call("write_file", json.dumps({"path": "j.txt", "content": "hi"}))
        assert out.ok and (ws / "j.txt").read_text() == "hi"

    async def test_extra_arguments_are_ignored(self, tools: WorkspaceTools) -> None:
        assert (await tools.call("list_dir", {"path": ".", "color": "blue", "verbose": 1})).ok

    async def test_placeholder_arguments_like_the_mock_backend_sends(
        self, tools: WorkspaceTools
    ) -> None:
        out = await tools.call("list_dir", {"path": "Path about tests and code", "recursive": True})
        assert not out.ok and "no such directory" in out.output

    async def test_internal_failures_become_error_outputs(
        self, tools: WorkspaceTools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(**_: object) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(tools, "_t_list_dir", boom)
        out = await tools.call("list_dir", {})
        assert not out.ok and "internal tool failure: RuntimeError: kaboom" in out.output

    async def test_long_output_is_clipped(self, ws: Path) -> None:
        tools = WorkspaceTools(Sandbox(ws), max_output_chars=500)
        seed(ws, "long.txt", "line\n" * 2000)
        out = await tools.call("read_file", {"path": "long.txt"})
        assert len(out.output) < 700 and "chars omitted" in out.output

    async def test_events_are_emitted(self, sandbox: Sandbox) -> None:
        bus = EventBus()
        tools = WorkspaceTools(sandbox, bus, phase="implement")
        await tools.call("list_dir", {})
        await tools.call("nope", {})
        kinds = [(e.kind, e.phase) for e in bus.history]
        assert ("tool_call", "implement") in kinds and ("tool_result", "implement") in kinds
        results = [e for e in bus.history if e.kind == "tool_result"]
        assert [e.data["ok"] for e in results] == [True, False]

    def test_clip_keeps_head_and_tail(self) -> None:
        text = "A" * 100 + "B" * 100
        out = clip(text, 50)
        assert out.startswith("A") and out.endswith("B") and "omitted" in out
        assert clip("short", 50) == "short"
