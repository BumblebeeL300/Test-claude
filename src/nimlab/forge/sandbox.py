"""Workspace sandbox: path confinement and constrained subprocess execution.

Everything the agent can touch goes through :class:`Sandbox`:

* **Paths** are resolved (``..`` and symlinks followed) and must stay inside the workspace root.
  Absolute paths are accepted only if they already point inside the root.
* **Commands** are limited to an allowlist (``python <script.py>``, ``pytest``); there is no shell,
  no ``python -c``, and pytest flags are checked against an allowlist as well.
* **Subprocesses** get a hard timeout (the whole process group is killed), ``cwd=<workspace>``, a
  scrubbed environment (nothing from the parent except a few harmless variables, so no API keys and
  no proxy settings), POSIX resource limits, and head+tail capped output.

This is a *guard rail*, not a security boundary: code that Forge writes and runs executes with
the privileges of the current user and may use the network. Run Forge in a container or VM when
working with untrusted tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ALLOWED_COMMANDS",
    "IGNORED_DIR_NAMES",
    "RESERVED_DIR_NAMES",
    "CommandNotAllowedError",
    "FileTooLargeError",
    "PathEscapeError",
    "ProcessResult",
    "ReservedPathError",
    "Sandbox",
    "SandboxError",
    "SandboxLimits",
    "SandboxReadOnlyError",
    "looks_binary",
]

#: Directories the agent tools may never read or write (Forge's own state and VCS metadata).
RESERVED_DIR_NAMES = frozenset({".forge", ".git"})
#: Directories skipped when walking the workspace (snapshots, listings, searches).
IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".forge",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
        ".eggs",
    }
)
IGNORED_FILE_SUFFIXES = (".pyc", ".pyo")
IGNORED_FILE_NAMES = frozenset({".DS_Store"})

#: Logical commands the sandbox will run. Everything else is refused.
ALLOWED_COMMANDS = ("python", "python3", "pytest")

# Environment variables that are copied from the parent process. Everything else is dropped, which
# is deliberately stricter than a deny-list: API keys, proxy settings, PYTHONPATH, PYTEST_ADDOPTS...
_ENV_PASSTHROUGH = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
)

# pytest options we accept. Value: does the flag take an argument?
_PYTEST_FLAGS: dict[str, bool] = {
    "-x": False,
    "--exitfirst": False,
    "-q": False,
    "-qq": False,
    "--quiet": False,
    "-v": False,
    "-vv": False,
    "-vvv": False,
    "--verbose": False,
    "-s": False,
    "-l": False,
    "--showlocals": False,
    "--lf": False,
    "--last-failed": False,
    "--ff": False,
    "--collect-only": False,
    "--co": False,
    "--disable-warnings": False,
    "--no-header": False,
    "-k": True,
    "-m": True,
    "--maxfail": True,
    "--tb": True,
    "-r": True,
    "--durations": True,
}


class SandboxError(Exception):
    """Base class for every sandbox refusal."""


class PathEscapeError(SandboxError):
    """A path resolves outside the workspace root."""


class ReservedPathError(SandboxError):
    """A path points into a reserved directory (``.forge``, ``.git``)."""


class CommandNotAllowedError(SandboxError):
    """A command or argument is not on the allowlist."""


class FileTooLargeError(SandboxError):
    """A file exceeds the configured size cap."""


class SandboxReadOnlyError(SandboxError):
    """A write was attempted on a read-only (dry-run) sandbox."""


@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits applied by :class:`Sandbox`."""

    command_timeout_s: float = 90.0
    max_output_bytes: int = 48_000  # kept per stream (head + tail)
    max_read_bytes: int = 400_000
    max_write_bytes: int = 500_000
    max_child_file_bytes: int = 64 * 1024 * 1024  # RLIMIT_FSIZE for child processes


@dataclass
class ProcessResult:
    """Outcome of one sandboxed subprocess."""

    argv: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    duration_s: float = 0.0
    timeout_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """stdout followed by stderr (labelled), as one string."""
        parts = [self.stdout.rstrip("\n")] if self.stdout.strip() else []
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip("\n"))
        return "\n".join(parts)


def looks_binary(data: bytes) -> bool:
    """Heuristic: a NUL byte in the first 8 KiB means "not text"."""
    return b"\x00" in data[:8192]


class _CappedBuffer:
    """Keeps the first ``cap/2`` and last ``cap/2`` bytes of a stream; counts what it drops."""

    def __init__(self, cap: int) -> None:
        self.head_cap = max(cap // 2, 1)
        self.tail_cap = max(cap - self.head_cap, 1)
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> None:
        if len(self.head) < self.head_cap:
            take = self.head_cap - len(self.head)
            self.head += chunk[:take]
            chunk = chunk[take:]
        if chunk:
            self.tail += chunk
            overflow = len(self.tail) - self.tail_cap
            if overflow > 0:
                del self.tail[:overflow]
                self.dropped += overflow

    def render(self) -> tuple[str, bool]:
        head = self.head.decode("utf-8", errors="replace")
        tail = self.tail.decode("utf-8", errors="replace")
        if self.dropped:
            return f"{head}\n... [{self.dropped} bytes of output truncated] ...\n{tail}", True
        return head + tail, False


async def _drain(stream: asyncio.StreamReader | None, cap: int) -> _CappedBuffer:
    buf = _CappedBuffer(cap)
    if stream is None:
        return buf
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return buf
        buf.feed(chunk)


def _child_limits(cpu_seconds: int, max_file_bytes: int) -> Callable[[], None] | None:
    """``preexec_fn`` applying POSIX resource limits to the child (None where unsupported)."""
    if os.name != "posix":  # pragma: no cover - windows
        return None

    def apply() -> None:  # pragma: no cover - runs in the forked child
        import resource

        for res, soft, hard in (
            (resource.RLIMIT_CORE, 0, 0),
            (resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 5),
            (resource.RLIMIT_FSIZE, max_file_bytes, max_file_bytes),
        ):
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(res, (soft, hard))

    return apply


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - windows
            proc.kill()


def _is_pytest_config(root: Path) -> bool:
    """True when the workspace carries its own pytest configuration."""
    if (root / "pytest.ini").is_file():
        return True
    probes = (
        ("pyproject.toml", "[tool.pytest"),
        ("tox.ini", "[pytest]"),
        ("setup.cfg", "[tool:pytest]"),
    )
    for name, marker in probes:
        path = root / name
        try:
            if path.is_file() and marker in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


class Sandbox:
    """A confined view of one workspace directory.

    Args:
        root: Workspace directory (must exist).
        limits: Resource limits.
        read_only: Refuse every write (used by ``--dry-run``).
        allowed_commands: Subset of :data:`ALLOWED_COMMANDS` that may run.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        limits: SandboxLimits | None = None,
        *,
        read_only: bool = False,
        allowed_commands: Sequence[str] = ALLOWED_COMMANDS,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SandboxError(f"workspace is not a directory: {self.root}")
        self.limits = limits or SandboxLimits()
        self.read_only = read_only
        unknown = set(allowed_commands) - set(ALLOWED_COMMANDS)
        if unknown:
            raise SandboxError(f"commands cannot be allowed (no handler): {sorted(unknown)}")
        self.allowed_commands = tuple(allowed_commands)

    # ------------------------------------------------------------------ paths

    def resolve(self, path: str | os.PathLike[str], *, write: bool = False) -> Path:
        """Resolve ``path`` against the root and ensure it stays inside.

        Raises:
            PathEscapeError: traversal, an absolute path outside the root, or a symlink escape.
            ReservedPathError: the path is inside ``.forge`` / ``.git``.
            SandboxReadOnlyError: ``write=True`` on a read-only sandbox.
        """
        raw = os.fspath(path)
        if not isinstance(raw, str) or "\x00" in raw:
            raise PathEscapeError("invalid path")
        raw = raw.strip()
        if write and self.read_only:
            raise SandboxReadOnlyError("workspace is read-only (dry-run): writes are disabled")
        candidate = Path(raw) if raw else Path(".")
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
            rel = resolved.relative_to(self.root)
        except (ValueError, RuntimeError, OSError) as exc:
            raise PathEscapeError(f"path escapes the workspace: {raw!r}") from exc
        if rel.parts and rel.parts[0] in RESERVED_DIR_NAMES:
            raise ReservedPathError(f"path is reserved: {rel.parts[0]}/")
        return resolved

    def relpath(self, path: str | os.PathLike[str]) -> str:
        """Workspace-relative POSIX path (``.`` for the root)."""
        try:
            rel = Path(path).resolve(strict=False).relative_to(self.root)
        except ValueError:
            return str(path)
        return rel.as_posix() or "."

    # ------------------------------------------------------------------ file access

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        """Read a UTF-8 text file (invalid bytes replaced); refuses binaries and huge files."""
        target = self.resolve(path)
        if not target.is_file():
            raise SandboxError(f"not a file: {self.relpath(target)}")
        cap = max_bytes if max_bytes is not None else self.limits.max_read_bytes
        size = target.stat().st_size
        with target.open("rb") as fh:
            data = fh.read(cap + 1)
        if looks_binary(data):
            raise SandboxError(f"binary file, not readable as text: {self.relpath(target)}")
        if len(data) > cap:
            raise FileTooLargeError(
                f"{self.relpath(target)} is {size} bytes (limit {cap}); read a line range instead"
            )
        return data.decode("utf-8", errors="replace")

    def write_text(self, path: str, content: str) -> Path:
        """Create or overwrite a text file (parents are created). Returns the resolved path."""
        target = self.resolve(path, write=True)
        if target == self.root or target.is_dir():
            raise SandboxError(f"is a directory: {self.relpath(target)}")
        data = content.encode("utf-8")
        if len(data) > self.limits.max_write_bytes:
            raise FileTooLargeError(
                f"content is {len(data)} bytes (limit {self.limits.max_write_bytes})"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir: a racing symlink must not redirect the write.
        self.resolve(target.relative_to(self.root).as_posix(), write=True)
        target.write_bytes(data)
        return target

    def iter_files(self, start: str = ".") -> Iterator[Path]:
        """Yield regular, non-ignored files below ``start`` (symlinks are not followed)."""
        base = self.resolve(start)
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in IGNORED_DIR_NAMES and not Path(dirpath, d).is_symlink()
            )
            for name in sorted(filenames):
                if name in IGNORED_FILE_NAMES or name.endswith(IGNORED_FILE_SUFFIXES):
                    continue
                full = Path(dirpath, name)
                if full.is_symlink() or not full.is_file():
                    continue
                yield full

    # ------------------------------------------------------------------ environment

    def forge_dir(self) -> Path:
        """``<root>/.forge`` (created on demand); Forge's private state directory."""
        path = self.root / ".forge"
        path.mkdir(exist_ok=True)
        return path

    def scrubbed_env(self) -> dict[str, str]:
        """The environment passed to every child: a tiny allowlist plus Python hygiene flags."""
        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        env.setdefault("PATH", os.defpath)
        home = self.root / ".forge" / "home"
        env.update(
            {
                "HOME": str(home),
                "USERPROFILE": str(home),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "NO_COLOR": "1",
                "PYTHONHASHSEED": "0",
                "CI": "1",
            }
        )
        return env

    # ------------------------------------------------------------------ commands

    def _check_name(self, name: str) -> None:
        if name not in ALLOWED_COMMANDS or name not in self.allowed_commands:
            raise CommandNotAllowedError(
                f"command {name!r} is not allowed (allowed: {', '.join(self.allowed_commands)})"
            )

    def _pytest_argv(self, args: Sequence[str]) -> list[str]:
        """Validate pytest arguments and build the real argv."""
        checked: list[str] = []
        it = iter(args)
        for arg in it:
            if not isinstance(arg, str) or "\x00" in arg:
                raise CommandNotAllowedError("invalid pytest argument")
            if arg.startswith("-"):
                flag, eq, value = arg.partition("=")
                if flag.startswith("-r") and len(flag) > 2 and flag[2:].isalpha() and not eq:
                    checked.append(arg)  # e.g. -rA, -rfE
                    continue
                if flag not in _PYTEST_FLAGS:
                    raise CommandNotAllowedError(f"pytest option {flag!r} is not allowed")
                if _PYTEST_FLAGS[flag]:
                    if not eq:
                        value = next(it, "")
                        if not value or value.startswith("-"):
                            raise CommandNotAllowedError(f"pytest option {flag!r} needs a value")
                    if "\x00" in value:
                        raise CommandNotAllowedError("invalid pytest argument")
                    checked.append(f"{flag}={value}" if flag.startswith("--") else flag)
                    if not flag.startswith("--"):
                        checked.append(value)
                elif eq:
                    raise CommandNotAllowedError(f"pytest option {flag!r} takes no value")
                else:
                    checked.append(flag)
            else:
                # Positional: a path (optionally ``::node``) that must live inside the workspace.
                path_part = arg.split("::", 1)[0]
                target = self.resolve(path_part)
                node = arg[len(path_part) :]
                checked.append(self.relpath(target) + node)
        base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--rootdir={self.root}"]
        if not _is_pytest_config(self.root):
            ini = self._write_default_pytest_ini()
            # The ini lives in .forge/, so conftest discovery must be anchored at the root.
            base += ["-c", str(ini), f"--confcutdir={self.root}"]
        return base + checked

    def _write_default_pytest_ini(self) -> Path:
        ini = self.root / ".forge" / "pytest.ini"
        if not self.read_only:
            ini.parent.mkdir(exist_ok=True)
            # pythonpath entries are relative to the ini file, so spell out absolute paths.
            paths = " ".join(shlex.quote(str(p)) for p in (self.root, self.root / "src"))
            text = f"[pytest]\npythonpath = {paths}\naddopts =\n"
            if not ini.exists() or ini.read_text(encoding="utf-8") != text:
                ini.write_text(text, encoding="utf-8")
        return ini

    def _python_argv(self, args: Sequence[str]) -> list[str]:
        if not args:
            raise CommandNotAllowedError(
                "python needs a script path (interactive/stdin is refused)"
            )
        first = args[0]
        if first == "-m" and len(args) >= 2 and args[1] == "pytest":
            return self._pytest_argv(list(args[2:]))
        if first.startswith("-"):
            raise CommandNotAllowedError(
                f"python option {first!r} is not allowed (only `python <script.py>`)"
            )
        script = self.resolve(first)
        if script.suffix != ".py":
            raise CommandNotAllowedError("only .py scripts may be run")
        if not script.is_file():
            raise CommandNotAllowedError(f"script not found: {self.relpath(script)}")
        if any("\x00" in a for a in args[1:]):
            raise CommandNotAllowedError("invalid script argument")
        return [sys.executable, "-B", str(script), *args[1:]]

    def build_argv(self, argv: Sequence[str]) -> list[str]:
        """Validate a logical command (``["pytest", "-x"]``) and return the real argv."""
        if not argv:
            raise CommandNotAllowedError("empty command")
        name = argv[0]
        if os.sep in name or "/" in name or name != name.strip():
            raise CommandNotAllowedError(f"command must be a bare name, got {name!r}")
        self._check_name(name)
        if name == "pytest":
            return self._pytest_argv(list(argv[1:]))
        return self._python_argv(list(argv[1:]))

    async def run(self, argv: Sequence[str], *, timeout: float | None = None) -> ProcessResult:
        """Run an allowlisted command in the workspace and return its (capped) output.

        ``argv[0]`` is a logical name (``python``, ``python3`` or ``pytest``), never a path. The
        call never raises for non-zero exit codes or timeouts; those are reported in the result.
        """
        real = self.build_argv(argv)
        limit = float(timeout if timeout is not None else self.limits.command_timeout_s)
        self.forge_dir()
        (self.root / ".forge" / "home").mkdir(exist_ok=True)
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *real,
            cwd=str(self.root),
            env=self.scrubbed_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
            preexec_fn=_child_limits(int(limit) + 5, self.limits.max_child_file_bytes),
        )
        cap = self.limits.max_output_bytes
        out_task = asyncio.ensure_future(_drain(proc.stdout, cap))
        err_task = asyncio.ensure_future(_drain(proc.stderr, cap))
        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=limit)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_group(proc)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        except BaseException:
            _kill_group(proc)
            out_task.cancel()
            err_task.cancel()
            raise
        # Pipes close once the (killed) group exits; guard against grandchildren holding them.
        done, pending = await asyncio.wait({out_task, err_task}, timeout=2.0)
        for task in pending:
            task.cancel()
        out_buf = out_task.result() if out_task in done else _CappedBuffer(cap)
        err_buf = err_task.result() if err_task in done else _CappedBuffer(cap)
        stdout, out_trunc = out_buf.render()
        stderr, err_trunc = err_buf.render()
        return ProcessResult(
            argv=real,
            returncode=None if timed_out else proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=out_trunc or err_trunc,
            duration_s=round(time.monotonic() - start, 3),
            timeout_s=limit,
        )

    async def run_pytest(
        self,
        targets: Sequence[str] = (),
        *,
        keyword: str = "",
        maxfail: int = 0,
        timeout: float | None = None,
    ) -> ProcessResult:
        """Run pytest with the flags Forge's parser expects (``-q --tb=short -rfE``)."""
        args = ["-q", "--tb=short", "-rfE", "--no-header", "--disable-warnings"]
        if keyword:
            args += ["-k", keyword]
        if maxfail and maxfail > 0:
            args.append(f"--maxfail={int(maxfail)}")
        args += list(targets)
        return await self.run(["pytest", *args], timeout=timeout)

    async def run_python(
        self, script: str, args: Sequence[str] = (), *, timeout: float | None = None
    ) -> ProcessResult:
        """Run ``python <script> [args...]`` for a script inside the workspace."""
        return await self.run(["python", script, *args], timeout=timeout)
