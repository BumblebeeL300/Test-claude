"""Persistent result types (pydantic) and the on-disk run store."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "BattleRecord",
    "ItemResult",
    "RunResult",
    "RunStore",
    "SlotVerdict",
    "SuiteInfo",
    "default_runs_dir",
    "utcnow",
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_runs_dir() -> Path:
    """``$NIMLAB_HOME/arena/runs`` (default ``./.nimlab/arena/runs``)."""
    home = os.environ.get("NIMLAB_HOME", ".nimlab")
    return Path(home) / "arena" / "runs"


class ItemResult(BaseModel):
    """One model answering one item: raw output, grade and performance measurements."""

    model: str
    suite: str
    item_id: str
    item_hash: str = ""
    weight: float = 1.0
    grader: str = ""
    output: str = ""
    score: float = 0.0
    passed: bool = False
    detail: str = ""
    error: str | None = None
    attempts: int = 1
    latency_s: float = 0.0
    ttft_s: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_per_s: float | None = None
    usage_estimated: bool = True
    finished_at: str = ""

    @property
    def scored(self) -> bool:
        """Errored items (API or grader failures) say nothing about the model and are skipped."""
        return self.error is None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model, self.suite, self.item_id)


class SuiteInfo(BaseModel):
    name: str
    title: str = ""
    category: str = "general_chat"
    open_ended: bool = False
    n_items: int = 0


class SlotVerdict(BaseModel):
    """One judge call. ``first`` / ``second`` say which model sat in slot A."""

    slot_a: str
    slot_b: str
    pick: Literal["A", "B", "tie"]
    winner: str | None = None  # model id, None for a tie
    rationale: str = ""


class BattleRecord(BaseModel):
    """A position-swapped pairwise comparison of two models on one item.

    ``score_a`` is the mean of the two verdicts' scores for ``model_a`` (1 win, 0.5 tie, 0 loss),
    so it is 1.0 / 0.75 / 0.5 / 0.25 / 0.0; disagreement between the two orders (position bias)
    cancels to 0.5.
    """

    suite: str
    item_id: str
    model_a: str
    model_b: str
    score_a: float = Field(ge=0.0, le=1.0)
    verdicts: list[SlotVerdict] = Field(default_factory=list)
    judge_model: str = ""

    @property
    def consistent(self) -> bool:
        winners = {v.winner for v in self.verdicts}
        return len(winners) <= 1

    @property
    def winner(self) -> str | None:
        if self.score_a > 0.5:
            return self.model_a
        if self.score_a < 0.5:
            return self.model_b
        return None


class RunResult(BaseModel):
    """Everything produced by one ``Arena.run`` / ``Arena.battle`` invocation."""

    run_id: str
    created_at: str
    updated_at: str = ""
    status: Literal["running", "partial", "complete"] = "running"
    models: list[str] = Field(default_factory=list)
    suites: dict[str, SuiteInfo] = Field(default_factory=dict)
    judge_model: str | None = None
    mock: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    items: list[ItemResult] = Field(default_factory=list)
    battles: list[BattleRecord] = Field(default_factory=list)

    @classmethod
    def new(cls, **kw: Any) -> RunResult:
        now = utcnow()
        return cls(
            run_id=kw.pop("run_id", None) or now.strftime("%Y%m%dT%H%M%SZ"),
            created_at=now.isoformat(timespec="seconds"),
            **kw,
        )

    def index(self) -> dict[tuple[str, str, str], ItemResult]:
        return {r.key: r for r in self.items}

    def for_model(self, model: str, suite: str | None = None) -> list[ItemResult]:
        return [r for r in self.items if r.model == model and (suite is None or r.suite == suite)]

    def merge_item(self, result: ItemResult) -> None:
        """Insert ``result``, replacing a previous result with the same key."""
        for i, old in enumerate(self.items):
            if old.key == result.key:
                self.items[i] = result
                return
        self.items.append(result)
        if result.model not in self.models:
            self.models.append(result.model)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RunStore:
    """JSON files under ``<root>`` named ``<run_id>.json`` (run ids are UTC timestamps)."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_runs_dir()

    # ------------------------------------------------------------------ paths
    def path_for(self, run_id: str) -> Path:
        if not _SAFE_ID.match(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.root / f"{run_id}.json"

    def unique_run_id(self) -> str:
        base = utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id, n = base, 1
        while self.path_for(run_id).exists():
            n += 1
            run_id = f"{base}-{n}"
        return run_id

    def new_run(self, **kw: Any) -> RunResult:
        return RunResult.new(run_id=self.unique_run_id(), **kw)

    # ------------------------------------------------------------------ io
    def save(self, run: RunResult) -> Path:
        """Atomically write ``run`` (temp file + rename) so a crash never leaves a torn file."""
        self.root.mkdir(parents=True, exist_ok=True)
        run.updated_at = utcnow().isoformat(timespec="seconds")
        path = self.path_for(run.run_id)
        payload = run.model_dump_json(indent=1)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{run.run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return path

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json") if _SAFE_ID.match(p.stem))

    def load(self, ref: str | Path) -> RunResult:
        """Load by run id, path to a run file, or ``"latest"``."""
        ref_s = str(ref)
        if ref_s == "latest":
            ids = self.list_ids()
            if not ids:
                raise FileNotFoundError(f"no runs found in {self.root}")
            path = self.path_for(ids[-1])
        elif ref_s.endswith(".json") or os.sep in ref_s or Path(ref_s).is_file():
            path = Path(ref_s)
        else:
            path = self.path_for(ref_s)
        if not path.is_file():
            raise FileNotFoundError(f"run not found: {path}")
        try:
            return RunResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise ValueError(f"{path} is not a valid Arena run file: {exc}") from exc

    def latest(self) -> RunResult | None:
        ids = self.list_ids()
        return self.load(ids[-1]) if ids else None

    def merged(self, max_runs: int = 20) -> RunResult | None:
        """The newest result for every (model, suite, item) across the last ``max_runs`` runs.

        This is what the router uses as "latest leaderboard data": a model benchmarked last week
        on one suite and yesterday on another contributes both.
        """
        ids = self.list_ids()[-max_runs:]
        if not ids:
            return None
        runs = [self.load(i) for i in ids]
        newest = runs[-1]
        out = RunResult(
            run_id=f"merged-{newest.run_id}",
            created_at=newest.created_at,
            status="complete",
            judge_model=newest.judge_model,
            mock=all(r.mock for r in runs),
        )
        seen: dict[tuple[str, str, str], ItemResult] = {}
        for run in runs:  # oldest first, so newer results overwrite
            out.suites.update(run.suites)
            for r in run.items:
                seen[r.key] = r
            out.battles.extend(run.battles)
        out.items = [r for r in seen.values()]
        out.models = sorted({r.model for r in out.items})
        return out
