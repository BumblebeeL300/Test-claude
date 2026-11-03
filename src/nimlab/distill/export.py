"""Dataset export (``openai-chat``, ``alpaca``, ``sharegpt``, ``dpo``) and reading datasets back."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Sample

__all__ = [
    "FORMATS",
    "ExportError",
    "Record",
    "detect_format",
    "from_json",
    "read_dataset",
    "read_texts",
    "sample_meta",
    "sample_to_record",
    "to_json",
    "write_jsonl",
]

FORMATS = ("openai-chat", "alpaca", "sharegpt", "dpo")


class ExportError(ValueError):
    """The samples cannot be written in the requested format."""


@dataclass
class Record:
    """A format-independent training example."""

    prompt: str
    response: str
    rejected: str | None = None
    system: str | None = None
    context: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def sample_meta(s: Sample) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "id": s.id,
        "model": s.model,
        "temperature": s.temperature,
        "topic": s.topic,
        "score": None if s.score is None else round(s.score, 4),
        "score_source": s.score_source,
    }
    if s.source:
        meta["source"] = s.source
    if s.difficulty is not None:
        meta["difficulty"] = round(s.difficulty, 4)
    if s.rejected is not None:
        meta["rejected_model"] = s.rejected_model
        meta["rejected_score"] = None if s.rejected_score is None else round(s.rejected_score, 4)
    return meta


def sample_to_record(s: Sample, system: str | None = None) -> Record:
    return Record(
        prompt=s.prompt,
        response=s.response,
        rejected=s.rejected,
        system=system,
        context=s.context,
        meta=sample_meta(s),
    )


# --------------------------------------------------------------------------- writing


def to_json(rec: Record, fmt: str, *, include_meta: bool = False) -> dict[str, Any]:
    """One record in the given format. ``include_meta`` adds a ``metadata`` object."""
    out: dict[str, Any]
    if fmt == "openai-chat":
        messages = []
        if rec.system:
            messages.append({"role": "system", "content": rec.system})
        messages.append({"role": "user", "content": rec.prompt})
        messages.append({"role": "assistant", "content": rec.response})
        out = {"messages": messages}
    elif fmt == "alpaca":
        out = {"instruction": rec.prompt, "input": rec.context, "output": rec.response}
        if rec.system:
            out["system"] = rec.system
    elif fmt == "sharegpt":
        turns = []
        if rec.system:
            turns.append({"from": "system", "value": rec.system})
        turns.append({"from": "human", "value": rec.prompt})
        turns.append({"from": "gpt", "value": rec.response})
        out = {"conversations": turns}
    elif fmt == "dpo":
        if rec.rejected is None:
            raise ExportError("the dpo format needs a rejected response; use a preference recipe")
        out = {"prompt": rec.prompt, "chosen": rec.response, "rejected": rec.rejected}
    else:
        raise ExportError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    if include_meta:
        out["metadata"] = rec.meta
    return out


def write_jsonl(
    records: Iterable[Record], path: str | Path, fmt: str, *, include_meta: bool = False
) -> int:
    """Write records as JSON lines; returns how many were written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(to_json(rec, fmt, include_meta=include_meta), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


# --------------------------------------------------------------------------- reading


def detect_format(obj: dict[str, Any]) -> str | None:
    if "messages" in obj:
        return "openai-chat"
    if "conversations" in obj:
        return "sharegpt"
    if "chosen" in obj and "rejected" in obj:
        return "dpo"
    if "instruction" in obj and "output" in obj:
        return "alpaca"
    return None


def from_json(obj: dict[str, Any]) -> Record:
    """Parse one record of any supported format back into a :class:`Record`."""
    fmt = detect_format(obj)
    raw_meta = obj.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    if fmt == "openai-chat":
        system = prompt = response = None
        for m in obj["messages"]:
            role, content = m.get("role"), m.get("content", "")
            if role == "system" and system is None:
                system = content
            elif role == "user" and prompt is None:
                prompt = content
            elif role == "assistant":
                response = content
        return Record(prompt or "", response or "", system=system, meta=meta)
    if fmt == "sharegpt":
        turns = obj["conversations"]
        sysv = next((t["value"] for t in turns if t.get("from") == "system"), None)
        human = next((t["value"] for t in turns if t.get("from") == "human"), "")
        gpt = next((t["value"] for t in reversed(turns) if t.get("from") == "gpt"), "")
        return Record(human, gpt, system=sysv, meta=meta)
    if fmt == "dpo":
        return Record(obj.get("prompt", ""), obj["chosen"], rejected=obj["rejected"], meta=meta)
    if fmt == "alpaca":
        return Record(
            obj["instruction"],
            obj["output"],
            system=obj.get("system"),
            context=obj.get("input", "") or "",
            meta=meta,
        )
    raise ExportError(
        "unrecognised record; expected messages, conversations, instruction/output or "
        "prompt/chosen/rejected"
    )


def read_dataset(path: str | Path) -> list[Record]:
    """Read a JSONL (or JSON array) dataset in any supported format."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"cannot read {p}: {exc}") from exc
    rows: list[Any]
    if p.suffix == ".json":
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else [loaded]
    else:
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ExportError(f"{p}:{i}: not valid JSON ({exc})") from exc
    return [from_json(r) for r in rows if isinstance(r, dict)]


_TEXT_KEYS = ("prompt", "question", "instruction", "query", "text", "input")


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        if detect_format(row) is not None:
            try:
                return from_json(row).prompt
            except (ExportError, KeyError):
                return ""
        for key in _TEXT_KEYS:
            if isinstance(row.get(key), str) and row[key].strip():
                return str(row[key])
    return ""


def read_texts(path: str | Path) -> list[str]:
    """Texts from an eval-set file: JSONL/JSON of strings or objects, ``.csv`` or plain lines.

    Objects may be dataset records (their prompt is used) or carry ``question``/``prompt``/
    ``instruction``/``text`` fields.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExportError(f"cannot read {p}: {exc}") from exc
    rows: list[Any]
    if p.suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
    elif p.suffix == ".json":
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else [loaded]
    elif p.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = [line for line in text.splitlines() if line.strip()]
    return [t for t in (_row_text(r) for r in rows) if t.strip()]
