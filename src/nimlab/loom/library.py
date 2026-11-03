"""Finding pipelines: bundled examples, the user's folder, and scaffolding new ones."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .errors import PipelineError
from .registry import get_node_type, ports_for
from .schema import Pipeline, Position, topo_levels

__all__ = [
    "PipelineInfo",
    "auto_layout",
    "bundled_dir",
    "delete_user_pipeline",
    "find_pipeline",
    "list_pipelines",
    "load_pipeline",
    "new_pipeline",
    "save_user_pipeline",
    "slugify",
    "user_dir",
]

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
NODE_W, ROW_H, HEAD_H, LCD_H = 232, 24, 34, 70
COL_GAP, ROW_GAP = 64, 30


@dataclass(frozen=True)
class PipelineInfo:
    id: str
    name: str
    title: str
    description: str
    source: str  # bundled | user
    nodes: int
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "nodes": self.nodes,
            "readonly": self.source == "bundled",
        }


def bundled_dir() -> Path:
    return Path(str(resources.files("nimlab.loom") / "pipelines"))


def user_dir() -> Path:
    """``$NIMLAB_LOOM_DIR``, else ``$NIMLAB_STUDIO_HOME/loom``, else ``~/NIMLab/loom``."""
    env = os.environ.get("NIMLAB_LOOM_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    home = os.environ.get("NIMLAB_STUDIO_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / "NIMLab"
    return base / "loom"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:64].strip("-")
    return slug or "pipeline"


def check_id(pipeline_id: str) -> str:
    if not _SLUG.match(pipeline_id):
        raise PipelineError(
            f"invalid pipeline id {pipeline_id!r}: use lowercase letters, digits, '-' or '_'"
        )
    return pipeline_id


def _scan(directory: Path, source: str) -> list[PipelineInfo]:
    out: list[PipelineInfo] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        if not _SLUG.match(path.stem):
            continue
        try:
            p = Pipeline.load(path)
        except PipelineError:
            continue
        out.append(
            PipelineInfo(
                path.stem, p.name, p.title or p.name, p.description, source, len(p.nodes), path
            )
        )
    return out


def list_pipelines(
    *, include_user: bool = True, directory: Path | None = None
) -> list[PipelineInfo]:
    """Bundled pipelines first, then saved ones from ``directory`` (default :func:`user_dir`)."""
    items = _scan(bundled_dir(), "bundled")
    if include_user:
        items += _scan(directory or user_dir(), "user")
    return items


def find_pipeline(ref: str, directory: Path | None = None) -> PipelineInfo | None:
    user = {i.id: i for i in _scan(directory or user_dir(), "user")}
    if ref in user:
        return user[ref]
    for info in _scan(bundled_dir(), "bundled"):
        if info.id == ref or info.name == ref:
            return info
    return None


def load_pipeline(ref: str | Path, directory: Path | None = None) -> Pipeline:
    """Load by file path, bundled name or saved id."""
    path = Path(ref)
    if path.suffix == ".json" or path.exists():
        return Pipeline.load(path)
    info = find_pipeline(str(ref), directory)
    if info is None:
        names = ", ".join(i.id for i in list_pipelines(directory=directory))
        raise PipelineError(f"no pipeline file or bundled name {str(ref)!r} (available: {names})")
    return Pipeline.load(info.path)


def save_user_pipeline(pipeline_id: str, pipeline: Pipeline, directory: Path | None = None) -> Path:
    check_id(pipeline_id)
    return pipeline.save((directory or user_dir()) / f"{pipeline_id}.json")


def delete_user_pipeline(pipeline_id: str, directory: Path | None = None) -> bool:
    check_id(pipeline_id)
    path = (directory or user_dir()) / f"{pipeline_id}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True


def node_height(node_type: str, params: dict[str, Any]) -> int:
    ins, outs = ports_for(get_node_type(node_type), params)
    return HEAD_H + max(len(ins), len(outs), 1) * ROW_H + LCD_H + 8


def auto_layout(pipeline: Pipeline, *, force: bool = False) -> Pipeline:
    """Give every node without a position a place on a left-to-right layered grid."""
    levels = topo_levels(pipeline)
    heights = {n.id: node_height(n.type, n.params) for n in pipeline.nodes}
    columns = []
    for level in levels:
        columns.append((level, sum(heights[i] for i in level) + ROW_GAP * (len(level) - 1)))
    tallest = max((h for _, h in columns), default=0)
    by_id = {n.id: n for n in pipeline.nodes}
    for col, (level, total) in enumerate(columns):
        y = 40 + (tallest - total) / 2
        for nid in level:
            node = by_id[nid]
            if force or node.pos is None:
                node.pos = Position(x=40 + col * (NODE_W + COL_GAP), y=round(y))
            y += heights[nid] + ROW_GAP
    return pipeline


def new_pipeline(name: str) -> Pipeline:
    """A starter pipeline: input -> prompt -> output."""
    p = Pipeline.from_data(
        {
            "name": slugify(name),
            "title": name.strip() or "New pipeline",
            "description": "Describe what this pipeline does.",
            "nodes": [
                {
                    "id": "question",
                    "type": "input",
                    "params": {"type": "string", "default": "What is a patch bay?"},
                },
                {
                    "id": "answer",
                    "type": "prompt",
                    "params": {
                        "template": "Answer in two sentences.\n\nQuestion: {{question}}",
                        "role": "chat",
                    },
                },
                {"id": "result", "type": "output", "params": {"name": "answer"}},
            ],
            "edges": [
                {"from": "question.value", "to": "answer.question"},
                {"from": "answer.text", "to": "result.value"},
            ],
        }
    )
    return auto_layout(p)
