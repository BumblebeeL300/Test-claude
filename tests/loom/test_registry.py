from __future__ import annotations

import pytest

from nimlab.loom import (
    NodeType,
    PipelineError,
    Port,
    VarSource,
    lint,
    list_functions,
    node_types,
    register_function,
    register_node_type,
    run_pipeline,
)
from nimlab.loom.context import NodeContext, NodeResult
from nimlab.loom.registry import unregister_node_type

from .conftest import inp, make, out


async def _shout(ctx: NodeContext) -> NodeResult:
    text = str(ctx.inputs.get("text", ""))
    return NodeResult({"text": text.upper() + ctx.params.get("bang", "!")})


@pytest.fixture
def shout_type():
    nt = register_node_type(
        NodeType(
            "shout",
            "Shout",
            "data",
            "Uppercase the text.",
            _shout,
            params={"bang": {"type": "string", "default": "!"}},
            inputs=(Port("text", "text"),),
            outputs=(Port("text", "text"),),
        )
    )
    yield nt
    unregister_node_type("shout")


def test_all_documented_node_types_are_registered() -> None:
    names = {t.name for t in node_types()}
    assert {
        "input", "prompt", "json", "embed", "rerank", "branch", "map", "merge", "join",
        "template", "regex", "http_get", "python_fn", "council", "atlas_ask", "output",
    } <= names  # fmt: skip


async def test_custom_node_types_plug_in(shout_type, fake) -> None:
    pipe = make(
        [inp(default="hi"), {"id": "s", "type": "shout", "params": {"bang": "?!"}}, out()],
        [("q.value", "s.text"), ("s.text", "out.value")],
    )
    assert lint(pipe).ok
    assert (await run_pipeline(pipe, client=fake)).outputs["out"] == "HI?!"
    bad = make(
        [inp(), {"id": "s", "type": "shout", "params": {"bang": 3}}, out()],
        [("q.value", "s.text"), ("s.text", "out.value")],
    )
    assert any("expected string" in i.message for i in lint(bad).issues)


def test_registering_twice_needs_replace(shout_type) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_node_type(shout_type)
    register_node_type(shout_type, replace=True)


def test_unknown_type_error_lists_known_types() -> None:
    from nimlab.loom.registry import get_node_type

    with pytest.raises(PipelineError, match=r"unknown node type 'warp'.*prompt"):
        get_node_type("warp")


def test_input_vars_define_jacks() -> None:
    from nimlab.loom.registry import get_node_type, ports_for

    ins, outs = ports_for(
        get_node_type("prompt"), {"template": "{{a}} {{b.c}} {{a}}", "system": "{{s}}"}
    )
    assert [p.name for p in ins] == ["a", "b", "s"] and outs[0].name == "text"
    ins, _ = ports_for(get_node_type("branch"), {"condition": "value['x'] > limit and 'y' in tags"})
    assert [p.name for p in ins] == ["value", "limit", "tags"]
    ins, outs = ports_for(get_node_type("input"), {"type": "number"})
    assert ins == [] and outs[0].type == "number"
    ins, _ = ports_for(
        get_node_type("map"),
        {"step": {"type": "prompt", "params": {"template": "{{item}} {{index}} {{ctx}}"}}},
    )
    assert [p.name for p in ins] == ["items", "ctx"]
    assert VarSource("x").kind == "template"


def test_register_function_rules() -> None:
    register_function("tmp_fn_a", lambda value, **_: value, doc="identity", replace=True)
    assert list_functions()["tmp_fn_a"] == "identity"
    with pytest.raises(ValueError, match="already registered"):
        register_function("tmp_fn_a", lambda value, **_: value)
    with pytest.raises(ValueError, match="invalid function name"):
        register_function("not a name", lambda value, **_: value)
    assert {"flatten", "pluck", "flatten_field", "numbered", "count"} <= set(list_functions())


def test_builtin_functions_behave() -> None:
    from nimlab.loom.functions import get_function

    call = lambda name, value, **kw: get_function(name)(value, **kw)  # noqa: E731
    assert call("flatten", [[1, 2], [3], 4]) == [1, 2, 3, 4]
    assert call("pluck", [{"a": 1}, {"a": 2}], field="a") == [1, 2]
    assert call("flatten_field", [{"q": [1, 2]}, {"q": [3]}], field="q") == [1, 2, 3]
    assert call("get_field", {"a": {"b": [10, 20]}}, field="a.b.1") == 20
    assert call("unique", ["a", "b", "a"]) == ["a", "b"]
    assert call("top_k", [1, 2, 3], k=2) == [1, 2]
    assert call("numbered", ["x", "y"]) == "1. x\n2. y"
    assert call("sentences", "One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert call("word_count", "a b  c") == 3
    assert call("truncate", "abcdef", chars=3) == "abc…"
    assert call("from_json", '{"a": 1}') == {"a": 1}
