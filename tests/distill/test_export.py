"""Every export format round-trips through ``write_jsonl`` / ``read_dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nimlab.distill.export import (
    FORMATS,
    ExportError,
    Record,
    detect_format,
    from_json,
    read_dataset,
    read_texts,
    sample_to_record,
    to_json,
    write_jsonl,
)
from nimlab.distill.models import Sample

RECORDS = [
    Record("What is 2+2?", "It is 4.", rejected="5", context="arithmetic"),
    Record("Say éè 你好", "Unicode ✓ and\nnewlines", rejected="nope"),
]


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip(tmp_path: Path, fmt: str) -> None:
    path = tmp_path / f"data.{fmt}.jsonl"
    assert write_jsonl(RECORDS, path, fmt) == 2
    back = read_dataset(path)
    assert [(r.prompt, r.response) for r in back] == [(r.prompt, r.response) for r in RECORDS]
    if fmt == "dpo":
        assert [r.rejected for r in back] == ["5", "nope"]
    if fmt == "alpaca":
        assert back[0].context == "arithmetic"
    assert detect_format(json.loads(path.read_text().splitlines()[0])) == fmt


def test_exact_shapes() -> None:
    rec = Record("p", "r", rejected="x", system="be brief", context="ctx")
    assert to_json(rec, "openai-chat") == {
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "r"},
        ]
    }
    assert to_json(rec, "alpaca") == {
        "instruction": "p",
        "input": "ctx",
        "output": "r",
        "system": "be brief",
    }
    assert to_json(rec, "sharegpt")["conversations"][-1] == {"from": "gpt", "value": "r"}
    assert to_json(rec, "dpo") == {"prompt": "p", "chosen": "r", "rejected": "x"}


def test_system_prompt_and_metadata_round_trip(tmp_path: Path) -> None:
    rec = Record("p", "r", system="sys", meta={"id": "s1", "score": 1.5})
    for fmt in ("openai-chat", "sharegpt"):
        path = tmp_path / f"{fmt}.jsonl"
        write_jsonl([rec], path, fmt, include_meta=True)
        (back,) = read_dataset(path)
        assert back.system == "sys" and back.meta == {"id": "s1", "score": 1.5}
        write_jsonl([rec], path, fmt)
        assert "metadata" not in path.read_text()


def test_dpo_needs_rejected_and_unknown_format_fails() -> None:
    with pytest.raises(ExportError, match="rejected"):
        to_json(Record("p", "r"), "dpo")
    with pytest.raises(ExportError, match="unknown format"):
        to_json(Record("p", "r"), "csv")
    with pytest.raises(ExportError, match="unrecognised"):
        from_json({"foo": 1})


def test_sample_to_record_carries_scores_and_pair() -> None:
    s = Sample(
        id="s1",
        seed_idx=1,
        prompt="p",
        response="good",
        rejected="bad",
        model="a",
        rejected_model="b",
        score=2.0,
        rejected_score=-1.0,
        score_source="reward",
        difficulty=0.5,
        context="c",
    )
    rec = sample_to_record(s, "sys")
    assert (rec.prompt, rec.response, rec.rejected, rec.system, rec.context) == (
        "p",
        "good",
        "bad",
        "sys",
        "c",
    )
    assert rec.meta["rejected_model"] == "b" and rec.meta["difficulty"] == 0.5


def test_read_dataset_errors_and_json_arrays(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="cannot read"):
        read_dataset(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": []}\nnot json\n')
    with pytest.raises(ExportError, match=r"bad\.jsonl:2"):
        read_dataset(bad)
    arr = tmp_path / "rows.json"
    arr.write_text(json.dumps([to_json(RECORDS[0], "alpaca")]))
    assert read_dataset(arr)[0].prompt == "What is 2+2?"


def test_read_texts_accepts_many_eval_formats(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("first question\n\nsecond question\n")
    assert read_texts(tmp_path / "a.txt") == ["first question", "second question"]
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"question": "q1"})
        + "\n"
        + json.dumps("plain string")
        + "\n"
        + json.dumps(to_json(RECORDS[0], "openai-chat"))
        + "\n"
    )
    assert read_texts(tmp_path / "b.jsonl") == ["q1", "plain string", "What is 2+2?"]
    (tmp_path / "c.json").write_text(json.dumps([{"prompt": "p1"}, {"instruction": "i1"}, {}]))
    assert read_texts(tmp_path / "c.json") == ["p1", "i1"]
    (tmp_path / "d.csv").write_text("id,text\n1,hello there\n")
    assert read_texts(tmp_path / "d.csv") == ["hello there"]
    with pytest.raises(ExportError):
        read_texts(tmp_path / "none.txt")
