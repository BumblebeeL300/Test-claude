"""Check for and apply updates (git checkouts), with clear messages for every other install."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nimlab import __version__

from .frozen import is_frozen

__all__ = ["UpdateStatus", "apply_update", "check_for_updates", "find_repo"]

UPSTREAM_BRANCH = "claude/nimlab"
REPO_URL = "https://github.com/BumblebeeL300/Test-claude"
GIT_TIMEOUT_S = 25.0
PIP_TIMEOUT_S = 600.0

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _run(cmd: list[str], cwd: Path | None, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@dataclass
class UpdateStatus:
    version: str = __version__
    kind: str = "unknown"  # "git" | "frozen-app" | "pip" | "no-git"
    # up-to-date | behind | ahead | diverged | offline | error | unsupported
    state: str = "unknown"
    message: str = ""
    branch: str | None = None
    upstream: str | None = None
    behind: int = 0
    ahead: int = 0
    dirty: bool = False
    commits: list[str] = field(default_factory=list)
    can_update: bool = False
    repo: str | None = None
    instructions: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def find_repo(start: Path | None = None) -> Path | None:
    """The git checkout NIM Lab was installed from (editable install), if any."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists() and (parent / "pyproject.toml").is_file():
            return parent
    return None


def _git(
    repo: Path, args: list[str], run: Runner, timeout: float = GIT_TIMEOUT_S
) -> tuple[int, str]:
    proc = run(["git", *args], repo, timeout)
    return proc.returncode, (proc.stdout or proc.stderr or "").strip()


def check_for_updates(
    *,
    repo: Path | None = None,
    fetch: bool = True,
    frozen: bool | None = None,
    run: Runner = _run,
) -> UpdateStatus:
    """Compare this install with the upstream branch. Never raises."""
    status = UpdateStatus()
    if is_frozen() if frozen is None else frozen:
        status.kind, status.state = "frozen-app", "unsupported"
        status.message = (
            f"NIM Lab Studio {__version__} is a packaged app. To update, pull the latest "
            "source and "
            "rebuild it with packaging/macos/build_app.sh (or download a newer build)."
        )
        status.instructions = f"git pull && packaging/macos/build_app.sh   ({REPO_URL})"
        return status
    root = repo if repo is not None else find_repo()
    if root is None or shutil.which("git") is None:
        status.kind, status.state = ("pip" if root is None else "no-git"), "unsupported"
        status.message = (
            f"NIM Lab {__version__} was not installed from a git checkout"
            + ("" if root is None else " and git is not available")
            + f". Update with `{sys.executable} -m pip install -U nimlab` or from {REPO_URL}."
        )
        return status
    status.kind, status.repo = "git", str(root)
    try:
        code, branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"], run)
        if code != 0:
            status.state, status.message = "error", f"git could not read the repository: {branch}"
            return status
        status.branch = branch
        code, out = _git(root, ["status", "--porcelain", "--untracked-files=no"], run)
        status.dirty = bool(out.strip()) if code == 0 else False
        if fetch:
            code, out = _git(root, ["fetch", "--quiet"], run)
            if code != 0:
                status.state = "offline"
                status.message = (
                    "Could not reach the remote (offline, or no credentials): "
                    f"{out.splitlines()[0] if out else 'git fetch failed'}"
                )
                return status
        code, upstream = _git(
            root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], run
        )
        if code != 0:
            upstream = f"origin/{UPSTREAM_BRANCH}"
            code, _ = _git(root, ["rev-parse", "--verify", "--quiet", upstream], run)
            if code != 0:
                status.state = "error"
                status.message = f"No upstream branch found (looked for @{{u}} and {upstream})."
                return status
        status.upstream = upstream
        code, counts = _git(
            root, ["rev-list", "--left-right", "--count", f"HEAD...{upstream}"], run
        )
        if code != 0:
            status.state, status.message = (
                "error",
                f"git could not compare with {upstream}: {counts}",
            )
            return status
        ahead_s, behind_s = counts.split()[:2]
        status.ahead, status.behind = int(ahead_s), int(behind_s)
        if status.behind:
            _, log = _git(root, ["log", "--format=%h %s", "-n", "10", f"HEAD..{upstream}"], run)
            status.commits = [ln for ln in log.splitlines() if ln.strip()]
    except subprocess.TimeoutExpired:
        status.state, status.message = "offline", "git timed out (offline or the remote is slow)."
        return status
    except OSError as exc:
        status.state, status.message = "error", f"could not run git: {exc}"
        return status
    if status.behind and status.ahead:
        status.state = "diverged"
        status.message = (
            f"{status.behind} new commit(s) upstream, but you have {status.ahead} local commit(s): "
            "a fast-forward update is not possible. Merge or rebase manually."
        )
    elif status.behind:
        status.state = "behind"
        n = status.behind
        status.can_update = not status.dirty
        status.message = f"{n} new commit{'s' if n != 1 else ''} available on {upstream}."
        if status.dirty:
            status.message += " Commit or stash your local changes first."
    elif status.ahead:
        status.state = "ahead"
        status.message = f"Up to date ({status.ahead} local commit(s) not pushed)."
    else:
        status.state = "up-to-date"
        status.message = f"NIM Lab {__version__} is up to date ({upstream})."
    return status


def apply_update(
    *, repo: Path | None = None, run: Runner = _run, python: str | None = None
) -> dict[str, Any]:
    """``git pull --ff-only`` then ``pip install -e .[desktop]``: ``ok``, ``steps``, ``message``."""
    status = check_for_updates(repo=repo, fetch=True, run=run)
    steps: list[dict[str, Any]] = []
    if status.kind != "git":
        return {"ok": False, "steps": steps, "message": status.message, "status": status.as_dict()}
    if status.state != "behind":
        return {
            "ok": status.state in ("up-to-date", "ahead"),
            "steps": steps,
            "message": status.message,
            "status": status.as_dict(),
        }
    if status.dirty:
        return {"ok": False, "steps": steps, "message": status.message, "status": status.as_dict()}
    root = Path(status.repo or "")
    py = python or sys.executable
    for label, cmd, timeout in (
        ("git pull --ff-only", ["git", "pull", "--ff-only"], GIT_TIMEOUT_S * 4),
        ("pip install", [py, "-m", "pip", "install", "-e", ".[desktop]"], PIP_TIMEOUT_S),
    ):
        try:
            proc = run(cmd, root, timeout)
        except (subprocess.TimeoutExpired, OSError) as exc:
            steps.append({"step": label, "ok": False, "output": str(exc)})
            return {"ok": False, "steps": steps, "message": f"{label} failed: {exc}"}
        output = (proc.stdout or "") + (proc.stderr or "")
        steps.append({"step": label, "ok": proc.returncode == 0, "output": output.strip()[-1500:]})
        if proc.returncode != 0:
            return {
                "ok": False,
                "steps": steps,
                "message": f"{label} failed (exit {proc.returncode}); nothing else was changed.",
            }
    return {
        "ok": True,
        "steps": steps,
        "restart": True,
        "message": f"Updated from {__version__}. Restart NIM Lab to use the new version.",
    }
