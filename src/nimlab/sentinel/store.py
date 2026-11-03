"""On-disk run store: one JSON file per scan under ``.nimlab/sentinel/runs/``."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import SentinelRun

__all__ = ["RunStore", "default_runs_dir", "new_run_id", "utcnow"]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_runs_dir() -> Path:
    """``$NIMLAB_HOME/sentinel/runs`` (default ``./.nimlab/sentinel/runs``)."""
    return Path(os.environ.get("NIMLAB_HOME", ".nimlab")) / "sentinel" / "runs"


def new_run_id() -> str:
    return utcnow().strftime("%Y%m%dT%H%M%S") + "-" + os.urandom(2).hex()


class RunStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_runs_dir()

    def path_for(self, run_id: str) -> Path:
        if not _SAFE_ID.match(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.root / f"{run_id}.json"

    def save(self, run: SentinelRun) -> Path:
        run.updated_at = utcnow().isoformat(timespec="seconds")
        path = self.path_for(run.run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".run-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(run.model_dump_json(indent=1))
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return path

    def list_ids(self) -> list[str]:
        """Run ids, oldest first (ids start with a UTC timestamp)."""
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json") if _SAFE_ID.match(p.stem))

    def load(self, ref: str | Path = "latest") -> SentinelRun:
        """Load by run id, file path, or ``latest``."""
        text = str(ref)
        if text == "latest":
            ids = self.list_ids()
            if not ids:
                raise FileNotFoundError(f"no Sentinel runs in {self.root}")
            path = self.path_for(ids[-1])
        elif _SAFE_ID.match(text) and self.path_for(text).is_file():
            path = self.path_for(text)
        elif Path(text).is_file():
            path = Path(text)
        else:
            raise FileNotFoundError(f"no such Sentinel run: {text} (looked in {self.root})")
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
            return SentinelRun.model_validate(data)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read run {path}: {exc}") from exc
