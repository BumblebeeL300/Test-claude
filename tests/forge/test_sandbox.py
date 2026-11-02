"""Sandbox: path confinement, env scrubbing, timeouts, output caps and the command allowlist."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from nimlab.forge import (
    CommandNotAllowedError,
    PathEscapeError,
    Sandbox,
    SandboxError,
    SandboxLimits,
)
from nimlab.forge.sandbox import FileTooLargeError, ReservedPathError, SandboxReadOnlyError

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process semantics")


def write(ws: Path, name: str, text: str) -> Path:
    path = ws / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- path confinement


class TestPathConfinement:
    def test_relative_paths_resolve_inside(self, sandbox: Sandbox, ws: Path) -> None:
        assert sandbox.resolve("a.py") == ws.resolve() / "a.py"
        assert sandbox.resolve("pkg/sub/../mod.py") == ws.resolve() / "pkg" / "mod.py"
        assert sandbox.resolve("") == ws.resolve()
        assert sandbox.resolve(".") == ws.resolve()
        assert sandbox.resolve("  spaced.py  ") == ws.resolve() / "spaced.py"

    @pytest.mark.parametrize(
        "bad",
        ["../x", "..", "a/../../x", "pkg/../../../etc/passwd", "./../outside", "a/b/../../../c"],
    )
    def test_traversal_rejected(self, sandbox: Sandbox, bad: str) -> None:
        with pytest.raises(PathEscapeError):
            sandbox.resolve(bad)

    @pytest.mark.parametrize("bad", ["/etc/passwd", "/", "/tmp", "/root/.ssh/id_rsa"])
    def test_absolute_outside_rejected(self, sandbox: Sandbox, bad: str) -> None:
        with pytest.raises(PathEscapeError):
            sandbox.resolve(bad)

    def test_absolute_inside_is_accepted(self, sandbox: Sandbox, ws: Path) -> None:
        assert sandbox.resolve(str(ws / "ok.py")) == ws.resolve() / "ok.py"

    def test_sibling_directory_with_same_prefix_is_outside(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        evil = tmp_path / "proj-evil"
        root.mkdir()
        evil.mkdir()
        with pytest.raises(PathEscapeError):
            Sandbox(root).resolve(str(evil / "x.py"))

    def test_null_byte_rejected(self, sandbox: Sandbox) -> None:
        with pytest.raises(PathEscapeError):
            sandbox.resolve("a\x00b.py")

    @posix_only
    def test_symlink_to_outside_file_rejected(
        self, sandbox: Sandbox, ws: Path, tmp_path: Path
    ) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret")
        (ws / "link.txt").symlink_to(secret)
        with pytest.raises(PathEscapeError):
            sandbox.resolve("link.txt")
        with pytest.raises(PathEscapeError):
            sandbox.read_text("link.txt")

    @posix_only
    def test_symlinked_directory_escape_rejected(
        self, sandbox: Sandbox, ws: Path, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "data.txt").write_text("x")
        (ws / "portal").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscapeError):
            sandbox.resolve("portal/data.txt")
        with pytest.raises(PathEscapeError):
            sandbox.write_text("portal/new.txt", "pwned")
        assert not (outside / "new.txt").exists()

    @posix_only
    def test_dangling_symlink_to_outside_rejected_for_write(
        self, sandbox: Sandbox, ws: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "does-not-exist-yet.txt"
        (ws / "dangling").symlink_to(target)
        with pytest.raises(PathEscapeError):
            sandbox.write_text("dangling", "payload")
        assert not target.exists()

    @posix_only
    def test_symlink_staying_inside_is_fine(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "real.py", "x = 1\n")
        (ws / "alias.py").symlink_to(ws / "real.py")
        assert sandbox.read_text("alias.py") == "x = 1\n"

    @pytest.mark.parametrize(
        "reserved", [".forge/report.json", ".git/config", ".forge", ".git/hooks/x"]
    )
    def test_reserved_dirs_blocked(self, sandbox: Sandbox, reserved: str) -> None:
        with pytest.raises(ReservedPathError):
            sandbox.resolve(reserved)

    def test_write_creates_parents_and_roundtrips(self, sandbox: Sandbox, ws: Path) -> None:
        path = sandbox.write_text("pkg/deep/mod.py", "VALUE = 1\n")
        assert path == ws.resolve() / "pkg" / "deep" / "mod.py"
        assert sandbox.read_text("pkg/deep/mod.py") == "VALUE = 1\n"

    def test_write_to_directory_refused(self, sandbox: Sandbox, ws: Path) -> None:
        (ws / "dir").mkdir()
        with pytest.raises(SandboxError):
            sandbox.write_text("dir", "x")
        with pytest.raises(SandboxError):
            sandbox.write_text(".", "x")

    def test_read_only_sandbox_refuses_writes(self, ws: Path) -> None:
        sb = Sandbox(ws, read_only=True)
        with pytest.raises(SandboxReadOnlyError):
            sb.write_text("a.py", "x")
        assert not (ws / "a.py").exists()
        write(ws, "b.py", "ok")
        assert sb.read_text("b.py") == "ok"

    def test_size_caps(self, ws: Path) -> None:
        sb = Sandbox(ws, SandboxLimits(max_write_bytes=100, max_read_bytes=50))
        with pytest.raises(FileTooLargeError):
            sb.write_text("big.txt", "x" * 101)
        write(ws, "large.txt", "y" * 200)
        with pytest.raises(FileTooLargeError):
            sb.read_text("large.txt")

    def test_binary_files_not_read_as_text(self, sandbox: Sandbox, ws: Path) -> None:
        (ws / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        with pytest.raises(SandboxError, match="binary"):
            sandbox.read_text("blob.bin")

    def test_iter_files_skips_ignored_and_symlinks(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "a.py", "")
        write(ws, "pkg/b.py", "")
        write(ws, "__pycache__/a.cpython-313.pyc", "")
        write(ws, ".git/config", "")
        write(ws, ".forge/forge_report.json", "")
        write(ws, "node_modules/x/index.js", "")
        found = {sandbox.relpath(p) for p in sandbox.iter_files()}
        assert found == {"a.py", "pkg/b.py"}

    def test_nonexistent_workspace_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SandboxError):
            Sandbox(tmp_path / "nope")


# --------------------------------------------------------------------------- environment


class TestEnvironmentScrubbing:
    SECRETS = {
        "NVIDIA_API_KEY": "nvapi-secret",
        "NIM_API_KEY": "nim-secret",
        "OPENAI_API_KEY": "sk-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "ghp_secret",
        "HTTP_PROXY": "http://proxy:3128",
        "HTTPS_PROXY": "http://proxy:3128",
        "ALL_PROXY": "socks5://proxy:1080",
        "http_proxy": "http://proxy:3128",
        "https_proxy": "http://proxy:3128",
        "NO_PROXY": "localhost",
        "PYTHONPATH": "/somewhere",
        "PYTEST_ADDOPTS": "--evil",
    }

    def test_scrubbed_env_has_no_secrets(
        self, sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key, value in self.SECRETS.items():
            monkeypatch.setenv(key, value)
        env = sandbox.scrubbed_env()
        for key in self.SECRETS:
            assert key not in env
        assert "PATH" in env
        assert env["HOME"].endswith(".forge/home") or env["HOME"].endswith(".forge\\home")
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"

    async def test_child_process_sees_scrubbed_env(
        self, sandbox: Sandbox, ws: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key, value in self.SECRETS.items():
            monkeypatch.setenv(key, value)
        write(ws, "dump_env.py", "import json, os\nprint(json.dumps(dict(os.environ)))\n")
        res = await sandbox.run_python("dump_env.py")
        assert res.ok, res.output
        seen = json.loads(res.stdout)
        for key in self.SECRETS:
            assert key not in seen
        assert not any("secret" in v.lower() for v in seen.values())
        assert "nvapi" not in res.stdout

    async def test_child_runs_in_workspace_cwd(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "where.py", "import os\nprint(os.getcwd())\n")
        res = await sandbox.run_python("where.py")
        assert Path(res.stdout.strip()).resolve() == ws.resolve()

    async def test_stdin_is_closed(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "ask.py", "input('say something: ')\n")
        start = time.monotonic()
        res = await sandbox.run_python("ask.py", timeout=20)
        assert time.monotonic() - start < 15
        assert not res.ok and not res.timed_out
        assert "EOFError" in res.stderr


# --------------------------------------------------------------------------- execution limits


class TestExecutionLimits:
    async def test_timeout_kills_the_process(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "spin.py", "import time\nwhile True:\n    time.sleep(0.05)\n")
        start = time.monotonic()
        res = await sandbox.run_python("spin.py", timeout=1.0)
        assert res.timed_out is True
        assert res.returncode is None
        assert res.ok is False
        assert time.monotonic() - start < 10

    @posix_only
    async def test_timeout_kills_grandchildren(self, sandbox: Sandbox, ws: Path) -> None:
        write(
            ws,
            "spawn.py",
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "open('child.pid', 'w').write(str(p.pid))\n"
            "time.sleep(60)\n",
        )
        res = await sandbox.run_python("spawn.py", timeout=1.5)
        assert res.timed_out
        pid = int((ws / "child.pid").read_text())
        deadline = time.monotonic() + 5
        alive = True
        while time.monotonic() < deadline and alive:
            try:
                os.kill(pid, 0)
                # A zombie still answers kill(0); check /proc state when available.
                state = Path(f"/proc/{pid}/stat")
                alive = state.exists() and state.read_text().split()[2] != "Z"
            except ProcessLookupError:
                alive = False
            if alive:
                time.sleep(0.1)
        assert not alive, "grandchild survived the timeout"

    async def test_output_is_capped_keeping_head_and_tail(self, ws: Path) -> None:
        sb = Sandbox(ws, SandboxLimits(max_output_bytes=2000))
        write(
            ws,
            "flood.py",
            "import sys\nprint('HEAD-MARKER')\nprint('x' * 1_000_000)\nprint('TAIL-MARKER')\n"
            "sys.stderr.write('E' * 500_000 + '\\nERR-TAIL\\n')\n",
        )
        res = await sb.run_python("flood.py")
        assert res.truncated
        assert len(res.stdout) < 2600 and len(res.stderr) < 2600
        assert "HEAD-MARKER" in res.stdout and "TAIL-MARKER" in res.stdout
        assert "truncated" in res.stdout
        assert "ERR-TAIL" in res.stderr

    @posix_only
    async def test_child_file_size_limit(self, ws: Path) -> None:
        sb = Sandbox(ws, SandboxLimits(max_child_file_bytes=100_000))
        write(ws, "fill.py", "open('big.bin', 'wb').write(b'0' * 5_000_000)\n")
        res = await sb.run_python("fill.py")
        assert not res.ok
        assert (ws / "big.bin").stat().st_size <= 100_000

    async def test_exit_code_and_stderr_reported(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "fail.py", "import sys\nprint('out')\nsys.stderr.write('err\\n')\nsys.exit(3)\n")
        res = await sandbox.run_python("fail.py")
        assert res.returncode == 3 and not res.ok
        assert res.stdout.strip() == "out" and res.stderr.strip() == "err"
        assert "[stderr]" in res.output

    async def test_invalid_utf8_output_does_not_crash(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "bytes.py", "import sys\nsys.stdout.buffer.write(b'ok \\xff\\xfe bytes\\n')\n")
        res = await sandbox.run_python("bytes.py")
        assert res.ok and "ok" in res.stdout

    async def test_script_arguments_are_passed(self, sandbox: Sandbox, ws: Path) -> None:
        write(ws, "argv.py", "import sys\nprint(sys.argv[1:])\n")
        res = await sandbox.run_python("argv.py", ["a b", "--flag; rm -rf /"])
        assert res.stdout.strip() == "['a b', '--flag; rm -rf /']"  # no shell involved


# --------------------------------------------------------------------------- allowlist


class TestCommandAllowlist:
    @pytest.mark.parametrize(
        "argv",
        [
            ["bash", "-c", "echo hi"],
            ["sh", "-c", "id"],
            ["ls"],
            ["rm", "-rf", "/"],
            ["curl", "http://example.com"],
            ["git", "status"],
            ["pip", "install", "x"],
            [sys.executable, "script.py"],
            ["/usr/bin/python3", "script.py"],
            ["python3.13", "script.py"],
            ["./python", "script.py"],
            [" python", "script.py"],
            [],
        ],
    )
    async def test_non_allowlisted_commands_refused(
        self, sandbox: Sandbox, argv: list[str]
    ) -> None:
        with pytest.raises(CommandNotAllowedError):
            await sandbox.run(argv)

    @pytest.mark.parametrize(
        "args",
        [
            ["-c", "print(1)"],
            ["-m", "http.server"],
            ["-m", "pip", "list"],
            ["-"],
            [],
            ["../escape.py"],
            ["/etc/passwd"],
            ["notes.txt"],
            ["missing.py"],
        ],
    )
    async def test_python_only_runs_workspace_scripts(
        self, sandbox: Sandbox, ws: Path, args: list[str]
    ) -> None:
        write(ws, "notes.txt", "hello")
        with pytest.raises((CommandNotAllowedError, PathEscapeError)):
            await sandbox.run(["python", *args])

    @pytest.mark.parametrize(
        "args",
        [
            ["-p", "evil_plugin"],
            ["-c", "other.ini"],
            ["--rootdir=/"],
            ["--rootdir", "/"],
            ["-o", "addopts=--foo"],
            ["--basetemp=/etc"],
            ["--pyargs", "os"],
            ["--pdb"],
            ["--junitxml=/tmp/x.xml"],
            ["--confcutdir=/"],
            ["-k"],
            ["--maxfail"],
        ],
    )
    async def test_pytest_flags_are_allowlisted(self, sandbox: Sandbox, args: list[str]) -> None:
        with pytest.raises(CommandNotAllowedError):
            await sandbox.run(["pytest", *args])

    @pytest.mark.parametrize("target", ["../x", "/etc", "/tmp/test_x.py", "a/../../b.py::test"])
    async def test_pytest_targets_must_be_inside(self, sandbox: Sandbox, target: str) -> None:
        with pytest.raises(PathEscapeError):
            await sandbox.run(["pytest", target])

    def test_allowed_pytest_argv_is_built_safely(self, sandbox: Sandbox, ws: Path) -> None:
        argv = sandbox.build_argv(
            ["pytest", "-x", "-q", "-k", "foo and not bar", "--maxfail=2", "sub/test_a.py::test_x"]
        )
        assert argv[:3] == [sys.executable, "-m", "pytest"]
        assert "no:cacheprovider" in argv
        assert f"--rootdir={sandbox.root}" in argv
        assert "-k" in argv and "foo and not bar" in argv
        assert argv[-1] == "sub/test_a.py::test_x"

    def test_python_dash_m_pytest_is_routed_through_pytest_validation(
        self, sandbox: Sandbox
    ) -> None:
        argv = sandbox.build_argv(["python", "-m", "pytest", "-x"])
        assert argv[1:3] == ["-m", "pytest"]
        with pytest.raises(CommandNotAllowedError):
            sandbox.build_argv(["python", "-m", "pytest", "-p", "evil"])

    def test_restricting_allowed_commands(self, ws: Path) -> None:
        sb = Sandbox(ws, allowed_commands=("pytest",))
        write(ws, "a.py", "")
        with pytest.raises(CommandNotAllowedError):
            sb.build_argv(["python", "a.py"])
        assert sb.build_argv(["pytest"])
        with pytest.raises(SandboxError):
            Sandbox(ws, allowed_commands=("bash",))

    async def test_pytest_runs_and_ignores_parent_project_config(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        root = parent / "child"
        root.mkdir(parents=True)
        # A hostile/odd parent config must not leak into the workspace run.
        (parent / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\naddopts = ['--this-flag-does-not-exist']\n"
        )
        write(root, "test_ok.py", "def test_ok():\n    assert True\n")
        res = await Sandbox(root).run_pytest()
        assert res.ok, res.output
        assert "1 passed" in res.stdout

    async def test_workspace_pytest_config_is_respected(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        root.mkdir()
        (root / "pytest.ini").write_text("[pytest]\npython_files = check_*.py\n")
        write(root, "check_it.py", "def test_it():\n    assert True\n")
        write(root, "test_ignored.py", "def test_bad():\n    assert False\n")
        res = await Sandbox(root).run_pytest()
        assert res.ok, res.output


@posix_only
async def test_cancelling_a_run_kills_the_child(sandbox: Sandbox, ws: Path) -> None:
    import asyncio

    write(
        ws,
        "sleeper.py",
        "import os, time\nopen('sleeper.pid', 'w').write(str(os.getpid()))\ntime.sleep(60)\n",
    )
    task = asyncio.ensure_future(sandbox.run_python("sleeper.py", timeout=60))
    pid_file = ws / "sleeper.pid"
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    deadline = time.monotonic() + 5
    alive = True
    while alive and time.monotonic() < deadline:
        stat = Path(f"/proc/{pid}/stat")
        alive = stat.exists() and stat.read_text().split()[2] != "Z"
        if alive:
            await asyncio.sleep(0.05)
    assert not alive, "child survived cancellation"


@pytest.mark.parametrize("dirname", ["plain", "dir with spaces"])
async def test_default_pytest_config_supports_conftest_and_src_layouts(
    tmp_path: Path, dirname: str
) -> None:
    root = tmp_path / dirname
    write(root, "conftest.py", "import pytest\n\n@pytest.fixture\ndef magic():\n    return 42\n")
    write(
        root, "tests/conftest.py", "import pytest\n\n@pytest.fixture\ndef inner():\n    return 1\n"
    )
    write(
        root,
        "tests/test_fixtures.py",
        "def test_f(magic, inner):\n    assert (magic, inner) == (42, 1)\n",
    )
    write(root, "src/pkgx.py", "V = 1\n")
    write(root, "tests/test_src.py", "from pkgx import V\n\ndef test_src():\n    assert V == 1\n")
    write(root, "toplevel.py", "T = 2\n")
    write(
        root, "tests/test_top.py", "from toplevel import T\n\ndef test_top():\n    assert T == 2\n"
    )
    res = await Sandbox(root).run_pytest()
    assert res.ok, res.output
    assert "3 passed" in res.stdout
