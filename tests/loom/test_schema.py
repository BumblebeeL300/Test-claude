from __future__ import annotations

import json

import pytest

from nimlab.loom import CycleError, Pipeline, PipelineError, SCHEMA_VERSION
from nimlab.loom.schema import find_cycle, topo_levels

from .conftest import inp, make, out, prompt


def test_roundtrip_json(tmp_path) -> None:
    pipe = make([inp(), prompt("a", "Say {{q}}"), out()], [("q.value", "a.q"), ("a.text", "out.value")])
    path = pipe.save(tmp_path / "p.json")
    again = Pipeline.load(path)
    assert again.to_data() == pipe.to_data()
    data = json.loads(path.read_text())
    assert data["version"] == SCHEMA_VERSION
    assert data["edges"][0] == {"from": "q.value", "to": "a.q"}  # wire format uses from/to


def test_defaults_and_positions() -> None:
    pipe = Pipeline.from_data(
        {"name": "x", "nodes": [{"id": "a", "type": "output", "pos": {"x": 3, "y": 4}}]}
    )
    node = pipe.node("a")
    assert node.retries == 0 and node.timeout is None and node.cache is None
    assert node.pos is not None and (node.pos.x, node.pos.y) == (3, 4)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"name": "x", "version": 99, "nodes": []}, "unsupported pipeline schema version"),
        ({"name": "x", "nodes": [{"id": "a", "type": "input"}, {"id": "a", "type": "output"}]}, "duplicate node id"),
        ({"name": "x", "nodes": [{"id": "1bad", "type": "input"}]}, "invalid node id"),
        ({"name": "x", "nodes": [{"id": "a", "type": "output"}], "edges": [{"from": "a.x", "to": "ghost.value"}]}, "unknown node"),
        ({"name": "x", "nodes": [], "edges": [{"from": "nope", "to": "a.b"}]}, "expected 'node.port'"),
        ({"name": "x", "nodes": [{"id": "a", "type": "input", "bogus": 1}]}, "bogus"),
        ({"name": "x", "nodes": [{"id": "a", "type": "input", "retries": -1}]}, "retries"),
        ({"name": "", "nodes": []}, "name"),
    ],
)
def test_schema_validation_errors(data, message) -> None:
    with pytest.raises(PipelineError, match=message):
        Pipeline.from_data(data)


def test_invalid_json_text() -> None:
    with pytest.raises(PipelineError, match="invalid JSON"):
        Pipeline.from_json("{nope")


def test_missing_file(tmp_path) -> None:
    with pytest.raises(PipelineError, match="cannot read"):
        Pipeline.load(tmp_path / "missing.json")


def test_cycle_detection_names_the_path() -> None:
    pipe = make(
        [prompt("a", "{{x}}"), prompt("b", "{{x}}"), prompt("c", "{{x}}")],
        [("a.text", "b.x"), ("b.text", "c.x"), ("c.text", "a.x")],
    )
    assert find_cycle(pipe) == ["a", "b", "c", "a"]
    with pytest.raises(CycleError, match=r"a -> b -> c -> a") as info:
        pipe.topological_order()
    assert info.value.cycle == ["a", "b", "c", "a"]


def test_self_loop_is_a_cycle() -> None:
    pipe = make([prompt("a", "{{x}}")], [("a.text", "a.x")])
    with pytest.raises(CycleError, match="a -> a"):
        topo_levels(pipe)


def test_topological_levels_group_parallel_nodes() -> None:
    pipe = make(
        [inp(), prompt("a", "{{q}}"), prompt("b", "{{q}}"), prompt("c", "{{a}} {{b}}"), out()],
        [("q.value", "a.q"), ("q.value", "b.q"), ("a.text", "c.a"), ("b.text", "c.b"), ("c.text", "out.value")],
    )
    levels = topo_levels(pipe)
    assert levels == [["q"], ["a", "b"], ["c"], ["out"]]
