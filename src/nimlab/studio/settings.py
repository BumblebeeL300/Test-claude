"""Studio settings: where data lives, the auth token and a few test knobs."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_ALLOWED_HOSTS", "StudioSettings", "new_token"]

#: Host names a request may use. Anything else (DNS rebinding) is refused with 403.
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]", "::1")


def new_token() -> str:
    """A fresh URL-safe auth token (256 bits)."""
    return secrets.token_urlsafe(32)


def _path_env(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


@dataclass
class StudioSettings:
    """Everything configurable about a Studio instance.

    Folder layout (``home`` defaults to ``~/NIMLab`` or ``$NIMLAB_STUDIO_HOME``)::

        ~/NIMLab/Projects/<slug>/     Forge projects         ($NIMLAB_PROJECTS_DIR)
        ~/NIMLab/atlas.db             Atlas knowledge base   ($NIMLAB_ATLAS_DB)
        ~/NIMLab/arena/runs/*.json    Arena runs
        ~/NIMLab/studio_history.json  run count + recent runs
    """

    home: Path = field(default_factory=lambda: Path.home() / "NIMLab")
    projects_dir: Path | None = None
    atlas_db: Path | None = None
    arena_runs_dir: Path | None = None
    history_path: Path | None = None
    web_dir: Path = field(default_factory=lambda: Path(__file__).parent / "web")
    #: Auth token required on ``/api/*`` (except ``/api/health``). ``None`` disables the check.
    token: str | None = field(default_factory=new_token)
    check_host: bool = True
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    #: Multiplier for the artificial pauses of simulated runs (0 = no pauses, used by tests).
    sim_delay: float = 1.0
    #: Report ``needs-key`` for apps while no API key is set (default: apps simulate instead).
    require_key: bool = False
    sse_heartbeat_s: float = 15.0
    max_history: int = 100
    max_memory_runs: int = 200

    def __post_init__(self) -> None:
        self.home = Path(self.home).expanduser()
        if self.projects_dir is None:
            self.projects_dir = _path_env("NIMLAB_PROJECTS_DIR") or self.home / "Projects"
        if self.atlas_db is None:
            self.atlas_db = _path_env("NIMLAB_ATLAS_DB") or self.home / "atlas.db"
        if self.arena_runs_dir is None:
            self.arena_runs_dir = self.home / "arena" / "runs"
        if self.history_path is None:
            self.history_path = self.home / "studio_history.json"
        for name in ("projects_dir", "atlas_db", "arena_runs_dir", "history_path"):
            setattr(self, name, Path(getattr(self, name)).expanduser())

    @classmethod
    def from_env(cls, **overrides: Any) -> StudioSettings:
        """Settings from the environment (``NIMLAB_STUDIO_HOME`` etc.); keyword overrides win."""
        home = (
            overrides.pop("home", None) or _path_env("NIMLAB_STUDIO_HOME") or Path.home() / "NIMLab"
        )
        return cls(home=Path(home), **overrides)

    def with_(self, **changes: Any) -> StudioSettings:
        """A copy with ``changes``; moving ``home`` moves every path derived from it too."""
        if "home" in changes:
            for name in ("projects_dir", "atlas_db", "arena_runs_dir", "history_path"):
                changes.setdefault(name, None)
        return replace(self, **changes)
