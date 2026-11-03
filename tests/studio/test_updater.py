"""Update checks against real (local, tmp) git repositories; no network."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nimlab.studio import updater

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")

ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def git(cwd: Path, *args: str) -> str:
    import os

    out = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **ENV},
    )
    return out.stdout.strip()


def commit(repo: Path, name: str, text: str = "x") -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


@pytest.fixture(autouse=True)
def _git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(origin bare repo, the user's checkout, another developer's checkout)."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "claude/nimlab", str(origin))
    seed = tmp_path / "seed"
    git(tmp_path, "clone", "-q", str(origin), str(seed))
    git(seed, "checkout", "-q", "-b", "claude/nimlab")
    (seed / "pyproject.toml").write_text("[project]\nname='x'\n")
    git(seed, "add", ".")
    git(seed, "commit", "-q", "-m", "init")
    git(seed, "push", "-q", "-u", "origin", "claude/nimlab")
    mine = tmp_path / "mine"
    git(tmp_path, "clone", "-q", str(origin), str(mine))
    return origin, mine, seed


class Recorder:
    """Runs git for real; fakes pip so nothing is installed."""

    def __init__(self, pip_rc: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.pip_rc = pip_rc

    def __call__(self, cmd: list[str], cwd: Path | None, timeout: float) -> Any:
        self.calls.append(cmd)
        if "pip" in cmd:
            return subprocess.CompletedProcess(cmd, self.pip_rc, "Successfully installed", "")
        return updater._run(cmd, cwd, timeout)


def test_up_to_date(repos: tuple[Path, Path, Path]) -> None:
    _, mine, _ = repos
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.state == "up-to-date" and st.behind == 0 and not st.can_update
    assert st.kind == "git" and st.branch == "claude/nimlab"
    assert st.upstream == "origin/claude/nimlab"


def test_new_commits_are_counted_and_listed(repos: tuple[Path, Path, Path]) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt")
    commit(seed, "b.txt")
    git(seed, "push", "-q")
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.state == "behind" and st.behind == 2 and st.can_update
    assert "2 new commits" in st.message
    assert [c.split(" ", 1)[1] for c in st.commits] == ["add b.txt", "add a.txt"]


def test_one_commit_message_is_singular(repos: tuple[Path, Path, Path]) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt")
    git(seed, "push", "-q")
    assert "1 new commit available" in updater.check_for_updates(repo=mine, frozen=False).message


def test_dirty_tree_blocks_the_update(repos: tuple[Path, Path, Path]) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt")
    git(seed, "push", "-q")
    (mine / "pyproject.toml").write_text("changed locally")
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.state == "behind" and st.dirty and not st.can_update
    assert "Commit or stash" in st.message
    rec = Recorder()
    result = updater.apply_update(repo=mine, run=rec)
    assert not result["ok"] and not any(c[:2] == ["git", "pull"] for c in rec.calls)
    assert not (mine / "a.txt").exists()


def test_diverged_history_is_reported(repos: tuple[Path, Path, Path]) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt")
    git(seed, "push", "-q")
    commit(mine, "local.txt")
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.state == "diverged" and not st.can_update and "fast-forward" in st.message


def test_local_commits_only_is_ahead(repos: tuple[Path, Path, Path]) -> None:
    _, mine, _ = repos
    commit(mine, "local.txt")
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.state == "ahead" and st.ahead == 1 and not st.can_update


def test_offline_is_graceful(repos: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, mine, _ = repos
    git(mine, "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"))
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.state == "offline" and "Could not reach the remote" in st.message
    assert not st.can_update


def test_check_without_fetch_does_not_touch_the_network(
    repos: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt")
    git(seed, "push", "-q")
    st = updater.check_for_updates(repo=mine, fetch=False, frozen=False)
    assert st.state == "up-to-date"  # the new commit is only visible after a fetch


def test_apply_pulls_fast_forward_then_reinstalls(repos: tuple[Path, Path, Path]) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt", "new feature")
    git(seed, "push", "-q")
    rec = Recorder()
    result = updater.apply_update(repo=mine, run=rec, python="/venv/bin/python")
    assert result["ok"] and result["restart"] and "Restart" in result["message"]
    assert (mine / "a.txt").read_text() == "new feature"
    assert [c[:3] for c in rec.calls if "pull" in c or "pip" in c] == [
        ["git", "pull", "--ff-only"],
        ["/venv/bin/python", "-m", "pip"],
    ]
    pip_cmd = next(c for c in rec.calls if "pip" in c)
    assert pip_cmd[-2:] == ["-e", ".[desktop]"]
    assert [s["ok"] for s in result["steps"]] == [True, True]


def test_failed_reinstall_is_reported(repos: tuple[Path, Path, Path]) -> None:
    _, mine, seed = repos
    commit(seed, "a.txt")
    git(seed, "push", "-q")
    result = updater.apply_update(repo=mine, run=Recorder(pip_rc=1))
    assert not result["ok"] and "pip install failed" in result["message"]


def test_apply_when_already_current_does_nothing(repos: tuple[Path, Path, Path]) -> None:
    _, mine, _ = repos
    rec = Recorder()
    result = updater.apply_update(repo=mine, run=rec)
    assert result["ok"] and not result["steps"]
    assert not any("pull" in c for c in rec.calls)


def test_frozen_app_gets_rebuild_instructions() -> None:
    st = updater.check_for_updates(frozen=True)
    assert st.kind == "frozen-app" and st.state == "unsupported"
    assert "build_app.sh" in st.message and not st.can_update


def test_no_checkout_means_pip_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(updater, "find_repo", lambda start=None: None)
    st = updater.check_for_updates(frozen=False)
    assert st.kind == "pip" and "pip install -U nimlab" in st.message


def test_missing_git_binary_is_handled(
    repos: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, mine, _ = repos
    monkeypatch.setattr(updater.shutil, "which", lambda name: None)
    st = updater.check_for_updates(repo=mine, frozen=False)
    assert st.kind == "no-git" and "git is not available" in st.message


def test_find_repo_walks_up_to_the_checkout(repos: tuple[Path, Path, Path]) -> None:
    _, mine, _ = repos
    nested = mine / "src" / "deep"
    nested.mkdir(parents=True)
    assert updater.find_repo(nested / "f.py") == mine.resolve()
