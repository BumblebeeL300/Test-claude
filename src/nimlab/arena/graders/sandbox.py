"""Best-effort sandbox for executing model-written Python against hidden unit tests.

Isolation layers (this is *not* a security boundary against a determined attacker; run Arena in a
container if you grade untrusted code at scale):

* a separate interpreter in isolated mode (``-I -B``): no user site, no ``PYTHON*`` env vars;
* a **scrubbed environment**: only ``PATH``, ``LANG``, ``HOME``/``TMPDIR`` (a throwaway directory),
  so API keys and other secrets of the parent process are invisible;
* a fresh temp directory as cwd, deleted afterwards;
* POSIX resource limits (CPU seconds, address space, file size, open files, no core dumps);
* sockets are disabled inside the interpreter (no network);
* a per-test alarm plus a wall-clock timeout; on timeout the whole process group is killed;
* stdout / stderr go to size-capped files, and results travel over a nonce-tagged line so the
  solution cannot forge a verdict by printing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["SandboxResult", "extract_code", "run_python_tests", "scrubbed_env"]

OUTPUT_CAP = 20_000

_HARNESS = r'''
import json, sys, signal, traceback
NONCE = {nonce!r}
TEST_TIMEOUT = {test_timeout!r}
try:
    import resource
    def _lim(which, soft, hard=None):
        try:
            resource.setrlimit(which, (soft, soft if hard is None else hard))
        except (ValueError, OSError):
            pass
    _lim(resource.RLIMIT_CPU, {cpu_s!r})
    _lim(resource.RLIMIT_AS, {mem_bytes!r})
    _lim(resource.RLIMIT_FSIZE, 1 << 20)
    _lim(resource.RLIMIT_CORE, 0)
    _lim(resource.RLIMIT_NOFILE, 64)
except ImportError:
    pass

import socket
def _no_net(*a, **k):
    raise OSError("network access is disabled in the sandbox")
socket.socket = _no_net
socket.create_connection = _no_net
socket.getaddrinfo = _no_net

class _Timeout(BaseException):
    pass

def _on_alarm(signum, frame):
    raise _Timeout()

_HAS_ALARM = hasattr(signal, "setitimer")
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _on_alarm)

def _guarded(fn):
    if _HAS_ALARM:
        signal.setitimer(signal.ITIMER_REAL, TEST_TIMEOUT)
    try:
        fn()
    finally:
        if _HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)

def _emit(payload):
    out = sys.__stdout__
    out.write("\n" + NONCE + json.dumps(payload) + "\n")
    out.flush()

code = open("solution.py", encoding="utf-8").read()
tests = json.load(open("tests.json", encoding="utf-8"))
ns = {{"__name__": "solution"}}
results = []
try:
    _guarded(lambda: exec(compile(code, "solution.py", "exec"), ns))
except _Timeout:
    _emit({{"load_error": "timeout while loading the solution", "results": []}})
    sys.exit(0)
except BaseException as exc:
    _emit({{"load_error": "".join(traceback.format_exception_only(type(exc), exc)).strip(), "results": []}})
    sys.exit(0)

for i, src in enumerate(tests):
    scope = dict(ns)
    try:
        _guarded(lambda: exec(compile(src, "<hidden test %d>" % i, "exec"), scope))
        results.append({{"ok": True}})
    except _Timeout:
        results.append({{"ok": False, "error": "timeout"}})
    except AssertionError as exc:
        results.append({{"ok": False, "error": "assertion failed " + str(exc)[:200]}})
    except BaseException as exc:
        results.append({{"ok": False, "error": "".join(traceback.format_exception_only(type(exc), exc)).strip()[:300]}})
_emit({{"load_error": None, "results": results}})
'''


@dataclass
class SandboxResult:
    """Outcome of one sandboxed execution."""

    passed: int = 0
    total: int = 0
    timed_out: bool = False
    load_error: str | None = None
    crashed: bool = False
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total and not self.timed_out

    @property
    def fraction(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        if self.timed_out:
            return "timed out: the process was killed at the wall-clock limit"
        if self.load_error:
            return f"solution failed to load: {self.load_error}"
        if self.crashed:
            return f"interpreter crashed (rc={self.returncode}): {self.stderr.strip()[-160:]}"
        failed = [d.get("error", "failed") for d in self.details if not d.get("ok")]
        base = f"{self.passed}/{self.total} hidden tests passed"
        return base + (f" | first failure: {failed[0]}" if failed else "")


def scrubbed_env(home: str) -> dict[str, str]:
    """The only environment variables a sandboxed child ever sees."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": home,
        "TMPDIR": home,
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str, entrypoint: str | None = None) -> str:
    """Pull Python source out of model output (fenced block preferred, else the raw text)."""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    py = [b for lang, b in blocks if lang in ("python", "py", "python3", "")]
    if entrypoint:
        pat = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(entrypoint)}\b", re.MULTILINE)
        for body in reversed(py):
            if pat.search(body):
                return body
    if py:
        return py[-1]
    if blocks:
        return blocks[-1][1]
    open_fence = re.search(r"```[A-Za-z0-9_+-]*\n(.*)$", text, re.DOTALL)  # unterminated fence
    return open_fence.group(1) if open_fence else text


async def _kill_group(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - non-POSIX
            proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5)


def _read_capped(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(OUTPUT_CAP).decode("utf-8", errors="replace")
    except OSError:
        return ""


async def run_python_tests(
    code: str,
    tests: list[str],
    *,
    timeout_s: float = 10.0,
    test_timeout_s: float = 2.0,
    memory_mb: int = 512,
) -> SandboxResult:
    """Execute ``code`` and then each snippet of ``tests`` (``assert`` statements) in isolation.

    Every test runs in its own namespace copy and passes if it raises nothing. ``timeout_s`` is the
    wall-clock cap for the whole child process; ``test_timeout_s`` is a per-test alarm so a single
    infinite loop costs only that test.
    """
    if not tests:
        raise ValueError("at least one test is required")
    nonce = f"__NIMLAB_RESULT_{secrets.token_hex(8)}__"
    workdir = Path(tempfile.mkdtemp(prefix="nimlab-sandbox-"))
    result = SandboxResult(total=len(tests))
    try:
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        (workdir / "harness.py").write_text(
            _HARNESS.format(
                nonce=nonce,
                test_timeout=float(test_timeout_s),
                cpu_s=int(timeout_s) + 2,
                mem_bytes=memory_mb * 1024 * 1024,
            ),
            encoding="utf-8",
        )
        out_path, err_path = workdir / "stdout.txt", workdir / "stderr.txt"
        with out_path.open("wb") as out, err_path.open("wb") as err:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-B",
                str(workdir / "harness.py"),
                cwd=workdir,
                env=scrubbed_env(str(workdir)),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                result.timed_out = True
                await _kill_group(proc)
            except BaseException:
                await _kill_group(proc)
                raise
        result.returncode = proc.returncode
        stdout, result.stderr = _read_capped(out_path), _read_capped(err_path)
        payload = None
        for line in reversed(stdout.splitlines()):
            if line.startswith(nonce):
                with contextlib.suppress(ValueError):
                    payload = json.loads(line[len(nonce) :])
                break
        result.stdout = "\n".join(ln for ln in stdout.splitlines() if not ln.startswith(nonce))
        if payload is not None:
            result.load_error = payload.get("load_error")
            result.details = payload.get("results", [])
            result.passed = sum(1 for d in result.details if d.get("ok"))
        elif not result.timed_out:
            result.crashed = True
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
