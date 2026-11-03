"""On-disk run state: ``.nimlab/distill/runs/<id>/``.

::

    run.json         RunMeta: recipe, models, options, stages done, usage, status
    seeds.json       output of the seed stage
    samples.jsonl    every sample and its fate (rewritten atomically at checkpoints)
    eval_texts.json  copy of the decontamination eval set (so resume needs no original file)
    vectors.npz      embeddings computed for dedup and the diversity metric
    quality.json     the quality report
    export/          dataset files, dataset card and HTML report
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import RunMeta, Sample, Seed, utc_now

__all__ = ["DEFAULT_RUNS_DIR", "RunStore", "StoreError", "new_run_id"]

DEFAULT_RUNS_DIR = Path(".nimlab/distill/runs")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StoreError(RuntimeError):
    """A run directory is missing or unreadable."""


def new_run_id(recipe: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{recipe}-{uuid.uuid4().hex[:4]}"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class RunStore:
    """Read and write the files of runs under ``root``."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_RUNS_DIR

    # ------------------------------------------------------------------ paths
    def path(self, run_id: str) -> Path:
        if not _ID_RE.match(run_id):
            raise StoreError(f"invalid run id {run_id!r}")
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.path(run_id) / "run.json").is_file()

    def export_dir(self, run_id: str) -> Path:
        return self.path(run_id) / "export"

    # ------------------------------------------------------------------ meta
    def save_meta(self, meta: RunMeta) -> None:
        meta.updated_at = utc_now()
        _atomic_write(
            self.path(meta.id) / "run.json", meta.model_dump_json(indent=2).encode("utf-8")
        )

    def load_meta(self, run_id: str) -> RunMeta:
        path = self.path(run_id) / "run.json"
        try:
            return RunMeta.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise StoreError(f"no such run: {run_id} (looked in {self.root})") from exc
        except ValueError as exc:
            raise StoreError(f"run {run_id} has an unreadable run.json: {exc}") from exc

    def list_runs(self) -> list[RunMeta]:
        """All readable runs, newest first."""
        metas: list[RunMeta] = []
        if self.root.is_dir():
            for child in self.root.iterdir():
                if (child / "run.json").is_file():
                    with contextlib.suppress(StoreError):
                        metas.append(self.load_meta(child.name))
        return sorted(metas, key=lambda m: (m.created_at, m.id), reverse=True)

    # ------------------------------------------------------------------ seeds / samples
    def save_seeds(self, run_id: str, seeds: list[Seed]) -> None:
        payload = json.dumps([s.model_dump() for s in seeds], ensure_ascii=False)
        _atomic_write(self.path(run_id) / "seeds.json", payload.encode("utf-8"))

    def load_seeds(self, run_id: str) -> list[Seed]:
        path = self.path(run_id) / "seeds.json"
        if not path.is_file():
            return []
        return [Seed.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]

    def save_samples(self, run_id: str, samples: list[Sample]) -> None:
        lines = "\n".join(s.model_dump_json() for s in samples)
        _atomic_write(self.path(run_id) / "samples.jsonl", (lines + "\n" if lines else "").encode())

    def load_samples(self, run_id: str) -> list[Sample]:
        path = self.path(run_id) / "samples.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Sample.model_validate_json(line))
        return out

    # ------------------------------------------------------------------ side files
    def save_eval_texts(self, run_id: str, texts: list[str]) -> None:
        _atomic_write(
            self.path(run_id) / "eval_texts.json", json.dumps(texts, ensure_ascii=False).encode()
        )

    def load_eval_texts(self, run_id: str) -> list[str]:
        path = self.path(run_id) / "eval_texts.json"
        return [str(x) for x in json.loads(path.read_text("utf-8"))] if path.is_file() else []

    def save_vectors(self, run_id: str, vectors: dict[str, list[float]]) -> None:
        path = self.path(run_id) / "vectors.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(vectors)
        matrix = np.array([vectors[i] for i in ids], dtype=np.float32).reshape(len(ids), -1)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(ids), matrix=matrix)
        os.replace(tmp, path)

    def load_vectors(self, run_id: str) -> dict[str, list[float]]:
        path = self.path(run_id) / "vectors.npz"
        if not path.is_file():
            return {}
        with np.load(path, allow_pickle=False) as data:
            ids = [str(i) for i in data["ids"]]
            return {i: row.tolist() for i, row in zip(ids, data["matrix"], strict=True)}

    def save_json(self, run_id: str, name: str, payload: Any) -> None:
        _atomic_write(
            self.path(run_id) / name,
            json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        )

    def load_json(self, run_id: str, name: str) -> Any | None:
        path = self.path(run_id) / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
