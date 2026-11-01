"""Parse pytest terminal output into structured results and a condensed failure report.

Forge runs pytest as ``-q --tb=short -rfE`` but the parser also copes with the default (long)
traceback style, collection errors, ``-x`` runs, timeouts and "no tests ran". The goal is a report
small enough to hand back to a coding model: failing test ids, the assertion messages, and only the
interesting traceback frames.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["FailureInfo", "Frame", "PytestReport", "parse_pytest_output"]

_SECTION = re.compile(r"^=+ (?P<title>.+?) =+$")
_CASE_HEADER = re.compile(r"^_+ (?P<name>.+?) _+$")
_CAPTURED = re.compile(r"^-+ Captured (?P<what>.+?) -+$")
_FRAME = re.compile(r"^(?P<file>[^\s:][^:\n]*?\.py):(?P<line>\d+): ?(?P<what>.*)$")
_SUMMARY_ITEM = re.compile(r"^(?P<kind>FAILED|ERROR) (?P<nodeid>\S+?)(?: - (?P<msg>.*))?$")
_COUNT = re.compile(
    r"(?P<n>\d+) (?P<what>passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|rerun)"
)
_FINAL = re.compile(r"\bin (?P<secs>\d+(?:\.\d+)?)s\b")
_ERROR_TYPE = re.compile(
    r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failed))\b"
)
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_ABS_PATH = re.compile(r"(?:/[\w.\-]+){3,}")

_NO_TESTS_EXIT = 5

CaseKind = Literal["failure", "error", "collection"]


class Frame(BaseModel):
    """One traceback frame."""

    file: str
    line: int
    func: str = ""
    source: str = ""


class FailureInfo(BaseModel):
    """One failing (or erroring) test or collection error."""

    nodeid: str
    kind: CaseKind = "failure"
    error_type: str = ""
    message: str = ""  # first meaningful ``E`` line, trimmed
    detail: list[str] = Field(default_factory=list)  # the ``E`` lines (trimmed)
    frames: list[Frame] = Field(default_factory=list)
    captured: list[str] = Field(default_factory=list)  # captured stdout/stderr, trimmed

    @property
    def key(self) -> str:
        """Stable identity used to detect repeated failures."""
        msg = _ABS_PATH.sub("<path>", _HEX.sub("0x?", self.message))
        return f"{self.nodeid}|{self.error_type}|{msg[:200]}"


class PytestReport(BaseModel):
    """Parsed outcome of one pytest run."""

    exit_code: int | None = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    warnings: int = 0
    deselected: int = 0
    duration_s: float = 0.0
    timed_out: bool = False
    no_tests: bool = False
    crashed: bool = False  # pytest itself did not run properly (usage/internal error)
    failures: list[FailureInfo] = Field(default_factory=list)
    output_tail: str = ""  # last lines of raw output, kept for diagnostics

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped + self.xfailed + self.xpassed

    @property
    def ok(self) -> bool:
        """All tests passed and at least one test ran."""
        return self.exit_code == 0 and not self.timed_out and self.passed > 0

    @property
    def summary_line(self) -> str:
        if self.timed_out:
            return "pytest timed out"
        if self.crashed:
            return "could not run"
        if self.no_tests:
            return "no tests collected"
        parts = [
            f"{n} {label}"
            for n, label in (
                (self.passed, "passed"),
                (self.failed, "failed"),
                (self.errors, "error(s)"),
                (self.skipped, "skipped"),
                (self.xfailed, "xfailed"),
            )
            if n
        ]
        return ", ".join(parts) or f"exit code {self.exit_code}"

    @property
    def signature(self) -> str:
        """Fingerprint of *which* things fail (and how), independent of timing and counts."""
        if self.timed_out:
            return "TIMEOUT"
        if self.crashed:
            last = "\n".join(self.output_tail.splitlines()[-3:])
            return "CRASH:" + hashlib.sha256(last.encode("utf-8")).hexdigest()[:12]
        if self.no_tests:
            return "NO_TESTS"
        if not self.failures:
            return f"EXIT:{self.exit_code}:{self.failed}:{self.errors}"
        keys = sorted(f.key for f in self.failures)
        return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]

    def condensed(self, max_chars: int = 3500, max_failures: int = 8, max_detail: int = 8) -> str:
        """A report for an LLM: counts, then per-failure id, location, source line, messages."""
        head = f"pytest: {self.summary_line}"
        if self.exit_code is not None:
            head += f" (exit code {self.exit_code})"
        if self.ok:
            return head
        if self.timed_out:
            tail = "\n".join(self.output_tail.splitlines()[-15:])
            note = "The test run exceeded the time limit; look for infinite loops."
            return f"{head}\n{note}\n{tail}".strip()
        if self.crashed and not self.failures:
            lines = [ln for ln in self.output_tail.splitlines() if _meaningful(ln)]
            return f"{head}\nPytest failed before running any test (check paths and options):\n" + (
                "\n".join(lines[-12:])
            )
        if self.no_tests and not self.failures:
            return (
                f"{head}\nNo tests were found. Create test files named test_*.py containing "
                "functions named test_*."
            )
        chunks = [head]
        shown = self.failures[:max_failures]
        for fail in shown:
            chunks.append(_render_failure(fail, max_detail))
        if len(self.failures) > len(shown):
            chunks.append(f"... and {len(self.failures) - len(shown)} more failing test(s)")
        if not self.failures:
            tail = "\n".join(self.output_tail.splitlines()[-25:])
            chunks.append(tail)
        text = "\n".join(chunks)
        if len(text) > max_chars:
            text = text[: max_chars - 40].rstrip() + "\n... [report truncated]"
        return text


_NOISE = ("PytestDeprecationWarning", "warnings.warn(", "The event loop scope", "Future versions")


def _meaningful(line: str) -> bool:
    return bool(line.strip()) and not any(n in line for n in _NOISE)


def _render_failure(fail: FailureInfo, max_detail: int) -> str:
    label = {"failure": "FAIL", "error": "ERROR", "collection": "COLLECTION ERROR"}[fail.kind]
    lines = [f"{label} {fail.nodeid}"]
    if fail.frames:
        first, last = fail.frames[0], fail.frames[-1]
        frames = [first] if first is last else [first, last]
        for fr in frames:
            where = f"  at {fr.file}:{fr.line}" + (f" in {fr.func}" if fr.func else "")
            lines.append(where + (f"  >  {fr.source}" if fr.source else ""))
    detail = fail.detail[:max_detail]
    for ln in detail:
        lines.append(f"  E {ln}")
    if len(fail.detail) > len(detail):
        lines.append(f"  E ... ({len(fail.detail) - len(detail)} more lines)")
    if fail.captured:
        lines.append("  captured output:")
        lines.extend(f"    {ln}" for ln in fail.captured[:5])
    return "\n".join(lines)


def _trim(text: str, limit: int = 220) -> str:
    text = text.rstrip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalise_name(name: str) -> str:
    """``TestK.test_in_class[1.5]`` -> ``TestK::test_in_class[1.5]`` (only dots outside ``[]``)."""
    base, bracket, param = name.partition("[")
    return base.replace(".", "::") + bracket + param


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"_preamble": []}
    current = "_preamble"
    for line in lines:
        match = _SECTION.match(line)
        if match:
            current = match.group("title").lower()
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return sections


def _case_header(line: str) -> str | None:
    """Name from a ``____ test_name ____`` header (None for ``_ _ _ _`` frame separators)."""
    match = _CASE_HEADER.match(line)
    if not match:
        return None
    name = match.group("name")
    return name if name.strip("_ ") else None


_NOISE_FRAME = ("site-packages", "/lib/python", "_pytest", "<frozen", "pluggy")


def _parse_case(name: str, body: list[str], kind: CaseKind) -> FailureInfo:
    info = FailureInfo(nodeid=name, kind=kind)
    long_style = any(ln.startswith(">") for ln in body)
    capture_mode: str | None = None
    current: Frame | None = None
    pending_source = ""
    for raw in body:
        line = raw.rstrip()
        cap = _CAPTURED.match(line)
        if cap:
            capture_mode = cap.group("what")
            continue
        if capture_mode is not None:
            if line.strip() and len(info.captured) < 8:
                info.captured.append(_trim(line, 160))
            continue
        if line.startswith("E") and (len(line) == 1 or line[1] in " \t"):
            msg = line[1:].strip()
            if msg:
                info.detail.append(_trim(msg))
            continue
        frame = _FRAME.match(line)
        if frame:
            what = frame.group("what").strip()
            func = what[3:].strip() if what.startswith("in ") else ""
            current = Frame(
                file=frame.group("file"),
                line=int(frame.group("line")),
                func=func,
                source=pending_source,
            )
            pending_source = ""
            if not any(tag in current.file for tag in _NOISE_FRAME):
                info.frames.append(current)
            continue
        if line.startswith(">"):
            pending_source = _trim(line.lstrip("> ").strip(), 160)
        elif (
            not long_style
            and current is not None
            and not current.source
            and line.startswith("    ")
            and line.strip()
        ):
            current.source = _trim(line.strip(), 160)
    if info.detail:
        info.message = info.detail[0]
        match = _ERROR_TYPE.match(info.message)
        if match:
            info.error_type = match.group("type")
        elif info.message.startswith("assert"):
            info.error_type = "AssertionError"
    return info


def parse_pytest_output(
    stdout: str,
    *,
    exit_code: int | None = None,
    stderr: str = "",
    timed_out: bool = False,
    root: str | None = None,
) -> PytestReport:
    """Parse pytest's terminal output (``stdout``) into a :class:`PytestReport`.

    ``root`` (the workspace path) is stripped from paths so reports use relative file names.
    """
    if root:
        prefix = root.rstrip("/") + "/"
        stdout = stdout.replace(prefix, "")
        stderr = stderr.replace(prefix, "")
    lines = stdout.replace("\r\n", "\n").split("\n")
    report = PytestReport(exit_code=exit_code, timed_out=timed_out)
    if exit_code in (3, 4):  # usage/internal error: the explanation is on stderr
        raw_tail = (stdout.strip() + "\n" + stderr.strip()).strip()
    else:
        raw_tail = stdout.strip() or stderr.strip()
    report.output_tail = "\n".join(raw_tail.splitlines()[-40:])

    # Final counts line, searched from the bottom: "4 failed, 4 passed in 0.02s", "no tests ran".
    for line in reversed(lines):
        text = line.strip().strip("=").strip()
        if not text:
            continue
        if text.startswith("no tests ran") or (_FINAL.search(text) and _COUNT.search(text)):
            if text.startswith("no tests ran"):
                report.no_tests = True
            for m in _COUNT.finditer(text):
                n, what = int(m.group("n")), m.group("what")
                if what == "passed":
                    report.passed = n
                elif what == "failed":
                    report.failed = n
                elif what.startswith("error"):
                    report.errors = n
                elif what == "skipped":
                    report.skipped = n
                elif what == "xfailed":
                    report.xfailed = n
                elif what == "xpassed":
                    report.xpassed = n
                elif what.startswith("warning"):
                    report.warnings = n
                elif what == "deselected":
                    report.deselected = n
            dur = _FINAL.search(text)
            if dur:
                report.duration_s = float(dur.group("secs"))
            break
        if _FINAL.search(text):  # e.g. "8 deselected in 0.01s" handled above; this is a fallback
            break

    if exit_code == _NO_TESTS_EXIT and report.total == 0:
        report.no_tests = True
    if exit_code in (3, 4) or (exit_code not in (0, 1, 2, 5, None) and report.total == 0):
        report.crashed = True
        report.no_tests = False

    sections = _split_sections(lines)

    # Summary lines give canonical node ids.
    summary_items: list[tuple[str, str, str]] = []
    for line in sections.get("short test summary info", []):
        m = _SUMMARY_ITEM.match(line.strip())
        if m:
            summary_items.append((m.group("kind"), m.group("nodeid"), m.group("msg") or ""))
    nodeids = [nid for _, nid, _ in summary_items]

    def canonical(name: str) -> str:
        if name.startswith("ERROR collecting "):
            return name[len("ERROR collecting ") :].strip()
        for prefix in ("ERROR at setup of ", "ERROR at teardown of "):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        wanted = _normalise_name(name)
        for nid in nodeids:
            if nid == wanted or nid.endswith("::" + wanted):
                return nid
        return name

    cases: dict[str, FailureInfo] = {}
    for title in ("failures", "errors"):
        body = sections.get(title, [])
        starts = [(i, _case_header(ln)) for i, ln in enumerate(body)]
        starts = [(i, name) for i, name in starts if name]
        for idx, (i, name) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(body)
            is_collection = name.startswith("ERROR collecting ")
            kind: CaseKind = (
                "collection" if is_collection else ("error" if title == "errors" else "failure")
            )
            nodeid = canonical(name)
            info = _parse_case(nodeid, body[i + 1 : end], kind)
            cases[nodeid] = info

    for kind_word, nodeid, msg in summary_items:
        if nodeid not in cases:
            kind = "failure" if kind_word == "FAILED" else "error"
            info = FailureInfo(nodeid=nodeid, kind=kind)
            if msg:
                info.message = _trim(msg)
                info.detail = [info.message]
                m = _ERROR_TYPE.match(msg)
                if m:
                    info.error_type = m.group("type")
            cases[nodeid] = info
        elif msg and not cases[nodeid].message:
            cases[nodeid].message = _trim(msg)

    report.failures = list(cases.values())
    if report.failures and report.failed == 0 and report.errors == 0 and exit_code not in (0, None):
        report.failed = sum(1 for f in report.failures if f.kind == "failure")
        report.errors = sum(1 for f in report.failures if f.kind != "failure")
    return report
