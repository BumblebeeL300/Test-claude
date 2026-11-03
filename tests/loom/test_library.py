from __future__ import annotations

from pathlib import Path

import pytest

from nimlab.loom import Pipeline, PipelineError, list_pipelines, load_pipeline, new_pipeline
from nimlab.loom.library import (
    auto_layout,
    bundled_dir,
    delete_user_pipeline,
    find_pipeline,
    save_user_pipeline,
    slugify,
    user_dir,
)

from .conftest import inp, make, out, prompt


def test_user_dir_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NIMLAB_LOOM_DIR", raising=False)
    monkeypatch.setenv("NIMLAB_STUDIO_HOME", str(tmp_path / "home"))
    assert user_dir() == tmp_path / "home" / "loom"
    monkeypatch.setenv("NIMLAB_LOOM_DIR", str(tmp_path / "custom"))
    assert user_dir() == tmp_path / "custom"


def test_bundled_ids_and_loading_by_name_or_path(tmp_path: Path) -> None:
    ids = [i.id for i in list_pipelines(include_user=False)]
    assert ids == sorted(ids) and len(ids) >= 4
    by_name = load_pipeline("support-triage")
    by_path = load_pipeline(bundled_dir() / "support-triage.json")
    assert by_name.to_data() == by_path.to_data()
    with pytest.raises(PipelineError, match="no pipeline file or bundled name 'nope'"):
        load_pipeline("nope")


def test_save_list_shadow_and_delete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NIMLAB_LOOM_DIR", str(tmp_path / "saved"))
    pipe = new_pipeline("Mine")
    path = save_user_pipeline("mine", pipe)
    assert path == tmp_path / "saved" / "mine.json"
    infos = {i.id: i for i in list_pipelines()}
    assert infos["mine"].source == "user" and infos["mine"].to_dict()["readonly"] is False
    assert load_pipeline("mine").name == "mine"
    assert find_pipeline("mine") is not None and find_pipeline("missing") is None
    assert delete_user_pipeline("mine") is True and delete_user_pipeline("mine") is False
    with pytest.raises(PipelineError, match="invalid pipeline id"):
        save_user_pipeline("../evil", pipe)


def test_broken_files_are_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "bad.json").write_text("{nope")
    (folder / "Not A Slug.json").write_text("{}")
    monkeypatch.setenv("NIMLAB_LOOM_DIR", str(folder))
    assert [i.id for i in list_pipelines() if i.source == "user"] == []


def test_slugify() -> None:
    assert slugify("  My Cool Pipeline! ") == "my-cool-pipeline"
    assert slugify("???") == "pipeline"
    assert len(slugify("x" * 200)) <= 64


def test_auto_layout_places_columns_without_overlap() -> None:
    pipe = make(
        [inp(), prompt("a", "{{q}}"), prompt("b", "{{q}}"), prompt("c", "{{a}}{{b}}"), out()],
        [
            ("q.value", "a.q"),
            ("q.value", "b.q"),
            ("a.text", "c.a"),
            ("b.text", "c.b"),
            ("c.text", "out.value"),
        ],
    )
    auto_layout(pipe)
    pos = {n.id: (n.pos.x, n.pos.y) for n in pipe.nodes if n.pos}
    assert len(pos) == 5
    assert pos["q"][0] < pos["a"][0] == pos["b"][0] < pos["c"][0] < pos["out"][0]
    assert abs(pos["a"][1] - pos["b"][1]) >= 100  # same column, stacked
    again = Pipeline.from_data(pipe.to_data())
    pos_before = pos["a"]
    auto_layout(again)  # existing positions are kept
    assert (again.node("a").pos.x, again.node("a").pos.y) == pos_before


def test_scaffold_is_valid_and_roundtrips(tmp_path: Path) -> None:
    pipe = new_pipeline("Brand New")
    assert pipe.name == "brand-new" and pipe.title == "Brand New"
    saved = pipe.save(tmp_path / "p.json")
    assert Pipeline.load(saved).to_data() == pipe.to_data()
