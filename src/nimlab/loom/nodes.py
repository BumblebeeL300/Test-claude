"""The built-in node types. Importing this module registers them."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
from typing import Any

from nimlab.core import chunk_text, cosine, default_registry

from . import expr as _expr
from . import template as _tpl
from .context import NodeContext, NodeResult, estimate_tokens
from .errors import LoomError
from .functions import get_function, list_functions
from .jsonschema_lite import schema_model
from .registry import (
    NodeType,
    Port,
    VarSource,
    apply_defaults,
    get_node_type,
    ports_for,
    register_node_type,
)
from .schema import Node
from .textops import check_http_url, dedupe_indices, http_allowlist, regex_apply
from .datatypes import coerce

__all__ = ["CATEGORIES", "MAP_STEP_FORBIDDEN", "http_allowlist"]

#: palette order and colour hint (``--c-*`` token or a literal colour) per category
CATEGORIES: dict[str, str] = {
    "io": "core",
    "model": "council",
    "logic": "arena",
    "data": "forge",
    "apps": "atlas",
    "net": "loom",
}
MAP_STEP_FORBIDDEN = frozenset({"map", "branch", "merge", "join", "input", "output"})

MAX_HTTP_BYTES = 1_000_000
ROLES = ["chat", "reasoning", "coder", "vision", "fast"]
INPUT_TYPES = {"string": "text", "number": "number", "boolean": "bool", "json": "json", "list": "list"}


# --------------------------------------------------------------------------- helpers
def _s(title: str, default: Any = None, **kw: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "string", "title": title}
    if default is not None:
        spec["default"] = default
    spec.update(kw)
    return spec


def _n(title: str, default: Any = None, integer: bool = False, **kw: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "integer" if integer else "number", "title": title}
    if default is not None:
        spec["default"] = default
    spec.update(kw)
    return spec


def _b(title: str, default: bool = False, **kw: Any) -> dict[str, Any]:
    return {"type": "boolean", "title": title, "default": default, **kw}


def _model_params(role: str, roles: list[str] | None = None) -> dict[str, dict[str, Any]]:
    return {
        "model": _s(
            "Model", "", format="model",
            description="NIM model id. Leave empty to use the role's default.",
        ),
        "role": _s(
            "Role", role, enum=roles or ROLES,
            description="Model role used when no model id is set.",
        ),
    }  # fmt: skip


def resolve_model(params: dict[str, Any], default_role: str) -> str:
    model = str(params.get("model") or "").strip()
    return model or default_registry().pick(str(params.get("role") or default_role))


def _as_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [_tpl.stringify(v) for v in value]
    return [_tpl.stringify(value)]


def _messages(ctx: NodeContext) -> tuple[list[dict[str, Any]], int]:
    scope = ctx.scope()
    user = _tpl.render(str(ctx.params.get("template") or ""), scope)
    system = _tpl.render(str(ctx.params.get("system") or ""), scope).strip()
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs, estimate_tokens(user + system)


# --------------------------------------------------------------------------- io
def _input_ports(params: dict[str, Any]) -> tuple[list[Port], list[Port]]:
    kind = INPUT_TYPES.get(str(params.get("type", "string")), "text")
    return [], [Port("value", kind)]


async def _run_input(ctx: NodeContext) -> NodeResult:
    name = str(ctx.params.get("name") or ctx.node.id)
    if name in ctx.env.input_values:
        value = ctx.env.input_values[name]
    else:
        value = ctx.params.get("default")
    return NodeResult({"value": value})


async def _run_output(ctx: NodeContext) -> NodeResult:
    return NodeResult({"value": ctx.inputs.get("value")})


# --------------------------------------------------------------------------- model nodes
async def _run_prompt(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    model = resolve_model(p, "chat")
    messages, prompt_tokens = _messages(ctx)
    temperature = float(p.get("temperature", 0.2))
    max_tokens = int(p.get("max_tokens", 1024))
    client = ctx.client
    if ctx.streaming and hasattr(client, "stream_chat"):
        parts: list[str] = []
        async for piece in client.stream_chat(
            model, messages, temperature=temperature, max_tokens=max_tokens
        ):
            parts.append(piece)
            await ctx.token(piece)
        ctx.flush_tokens()
        text = "".join(parts)
        tokens = prompt_tokens + estimate_tokens(text)
    else:
        result = await client.chat(
            model, messages, temperature=temperature, max_tokens=max_tokens
        )
        text = result.content
        tokens = int(getattr(result.usage, "total_tokens", 0) or 0)
    return NodeResult({"text": text.strip()}, tokens, model)


async def _run_json(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    model = resolve_model(p, "chat")
    messages, prompt_tokens = _messages(ctx)
    schema = p.get("schema") or {"type": "object"}
    holder = await ctx.client.chat_json(
        model,
        messages,
        schema_model(schema),
        retries=int(p.get("repair_attempts", 2)),
        temperature=float(p.get("temperature", 0.0)),
        max_tokens=int(p.get("max_tokens", 1024)),
    )
    data = holder.data
    tokens = prompt_tokens + estimate_tokens(json.dumps(data, default=str))
    return NodeResult({"data": data}, tokens, model)


async def _run_embed(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    model = resolve_model(p, "embed")
    texts = _as_texts(ctx.inputs.get("texts"))
    vectors = await ctx.client.embed(model, texts, input_type=str(p.get("input_type", "passage")))
    threshold = float(p.get("dedupe_threshold") or 0)
    unique = list(texts)
    if threshold > 0:
        unique = [texts[i] for i in dedupe_indices(vectors, threshold, cosine)]
    tokens = sum(estimate_tokens(t) for t in texts)
    return NodeResult(
        {"vectors": vectors, "unique": unique, "count": len(unique)}, tokens, model
    )


async def _run_rerank(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    model = resolve_model(p, "rerank")
    passages = _as_texts(ctx.inputs.get("passages"))
    query = _tpl.stringify(ctx.inputs.get("query"))
    top_n = int(p["top_n"]) if p.get("top_n") else None
    ranked = await ctx.client.rerank(model, query, passages, top_n=top_n)
    tokens = estimate_tokens(query) + sum(estimate_tokens(t) for t in passages)
    return NodeResult(
        {"ranked": [r.text for r in ranked], "scores": [round(r.score, 4) for r in ranked]},
        tokens,
        model,
    )


# --------------------------------------------------------------------------- logic
async def _run_branch(ctx: NodeContext) -> NodeResult:
    scope = ctx.scope()
    try:
        taken = bool(_expr.evaluate(str(ctx.params.get("condition") or "value"), scope))
    except LoomError as exc:
        raise LoomError(f"condition failed: {exc}") from exc
    value = ctx.inputs.get("value")
    return NodeResult({"then" if taken else "else": value})


async def _run_merge(ctx: NodeContext) -> NodeResult:
    live = ctx.inputs.get("in") or []
    return NodeResult({"out": live[0] if live else None})


async def _run_join(ctx: NodeContext) -> NodeResult:
    live = ctx.inputs.get("in") or []
    items: list[Any] = []
    for value in live:
        items.extend(value if isinstance(value, list) else [value])
    sep = str(ctx.params.get("separator", "\n\n")).encode().decode("unicode_escape")
    return NodeResult({"text": sep.join(_tpl.stringify(v) for v in items)})


async def _run_template(ctx: NodeContext) -> NodeResult:
    return NodeResult({"text": _tpl.render(str(ctx.params.get("template") or ""), ctx.scope())})


async def _run_regex(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    out = regex_apply(
        _tpl.stringify(ctx.inputs.get("text")),
        str(p.get("pattern") or ""),
        str(p.get("mode", "search")),
        int(p.get("group", 0)),
        str(p.get("flags") or ""),
        str(p.get("replacement", "")),
    )
    return NodeResult(out)


async def _run_chunk(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    text = _tpl.stringify(ctx.inputs.get("text"))
    chunks = chunk_text(text, int(p.get("max_chars", 1200)), int(p.get("overlap", 100)))
    return NodeResult({"chunks": chunks, "count": len(chunks)})


async def _run_python_fn(ctx: NodeContext) -> NodeResult:
    name = str(ctx.params.get("fn") or "")
    fn = get_function(name)
    if fn is None:
        raise LoomError(f"function {name!r} is not registered (known: {', '.join(list_functions())})")
    args = ctx.params.get("args") or {}
    result = fn(ctx.inputs.get("value"), **args)
    if inspect.isawaitable(result):
        result = await result
    try:
        json.dumps(result, default=str)
    except (TypeError, ValueError) as exc:
        raise LoomError(f"function {name!r} returned a non-JSON value: {exc}") from exc
    return NodeResult({"result": result})


# --------------------------------------------------------------------------- map
def _step_type_and_params(params: dict[str, Any]) -> tuple[NodeType, dict[str, Any]]:
    step = params.get("step") or {}
    st = get_node_type(str(step.get("type") or "prompt"))
    if st.name in MAP_STEP_FORBIDDEN:
        raise LoomError(f"a map step cannot be a {st.name!r} node")
    return st, apply_defaults(st, dict(step.get("params") or {}))


def _map_ports(params: dict[str, Any]) -> tuple[list[Port], list[Port]]:
    ins = [Port("items", "list")]
    step = params.get("step") or {}
    try:
        st = get_node_type(str(step.get("type") or "prompt"))
    except LoomError:
        st = None
    if st is not None and st.name not in MAP_STEP_FORBIDDEN:
        sins, _ = ports_for(st, dict(step.get("params") or {}))
        item_port = next((p.name for p in st.inputs if p.required), None)
        for p in sins:
            if p.name in ("item", "index") or p.name == item_port:
                continue
            ins.append(Port(p.name, p.type, p.required))
    return ins, [Port("results", "list"), Port("count", "number")]


async def _run_map(ctx: NodeContext) -> NodeResult:
    p = ctx.params
    st, step_params = _step_type_and_params(p)
    items = coerce(ctx.inputs.get("items"), "list") or []
    if not isinstance(items, list):
        raise LoomError("map needs a list on its 'items' input")
    limit = int(p.get("max_items", 200))
    if len(items) > limit:
        raise LoomError(f"map got {len(items)} items; the limit is {limit} (raise max_items)")
    concurrency = max(1, int(p.get("concurrency", 4)))
    on_error = str(p.get("on_error", "fail"))
    _, outs = ports_for(st, step_params)
    primary = outs[0].name if outs else "text"
    item_port = next((q.name for q in st.inputs if q.required), None)
    shared = {k: v for k, v in ctx.inputs.items() if k != "items"}
    sem = asyncio.Semaphore(concurrency)
    done = 0
    tokens = 0
    model: str | None = None
    ctx.env.event("node.progress", node=ctx.node.id, done=0, total=len(items))

    async def one(i: int, item: Any) -> Any:
        nonlocal done, tokens, model
        async with sem:
            inputs = {**shared, "index": i}
            inputs["item"] = item
            if item_port:
                inputs[item_port] = item
            sub_node = Node(id=ctx.node.id, type=st.name, params=step_params, defaults=ctx.node.defaults)
            sub = NodeContext(sub_node, step_params, inputs, ctx.env, quiet=True)
            try:
                res = await st.run(sub)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if on_error == "fail":
                    raise LoomError(f"item {i}: {exc}") from exc
                value: Any = ... if on_error == "skip" else None
            else:
                tokens += res.tokens
                model = model or res.model
                value = res.outputs.get(primary)
            done += 1
            ctx.env.event("node.progress", node=ctx.node.id, done=done, total=len(items))
            return value

    results = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    kept = [r for r in results if r is not ...]
    return NodeResult({"results": kept, "count": len(kept)}, tokens, model)


# --------------------------------------------------------------------------- net & apps
async def _run_http_get(ctx: NodeContext) -> NodeResult:
    import httpx

    allow = http_allowlist(ctx.env.http_allow)
    url = _tpl.render(str(ctx.params.get("url") or ""), ctx.scope()).strip()
    check_http_url(url, allow)
    timeout = float(ctx.params.get("timeout", 15))
    max_bytes = min(int(ctx.params.get("max_bytes", 200_000)), MAX_HTTP_BYTES)
    owned = ctx.env.http_client is None
    client = ctx.env.http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    try:
        async with client.stream("GET", url, timeout=timeout) as resp:
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) >= max_bytes:
                    break
            status = resp.status_code
    finally:
        if owned:
            await client.aclose()
    text = bytes(body[:max_bytes]).decode("utf-8", errors="replace")
    return NodeResult({"body": text, "status": status})


async def _run_council(ctx: NodeContext) -> NodeResult:
    from nimlab.council import Council

    p = ctx.params
    question = _tpl.stringify(ctx.inputs.get("question")).strip()
    overrides: dict[str, Any] = {}
    for key in ("rounds", "layers", "n"):
        if p.get(key):
            overrides[key] = int(p[key])
    models = [str(m) for m in (p.get("models") or []) if str(m).strip()]
    result = await Council(ctx.client).run(
        question, strategy=str(p.get("strategy", "moa")), models=models or None, **overrides
    )
    return NodeResult(
        {
            "answer": result.answer,
            "agreement": round(result.agreement, 4) if result.agreement is not None else 0,
            "rationale": result.rationale or "",
        },
        int(result.trace.total_tokens),
        None,
    )


async def _run_atlas_ask(ctx: NodeContext) -> NodeResult:
    from nimlab.atlas import Atlas
    from nimlab.atlas.config import DEFAULT_DB_PATH

    p = ctx.params
    db = p.get("db") or ctx.env.atlas_db or os.environ.get("NIMLAB_ATLAS_DB") or DEFAULT_DB_PATH
    atlas = Atlas(ctx.client, db)
    try:
        answer = await atlas.ask(
            _tpl.stringify(ctx.inputs.get("question")).strip(),
            str(p.get("collection") or "default"),
            mode=p.get("mode", "simple"),
            verify=bool(p.get("verify", False)),
        )
    finally:
        with contextlib.suppress(Exception):
            atlas.close()
    cites = [c.model_dump(mode="json") for c in answer.citations]
    tokens = int(sum(int(v.get("total_tokens", 0)) for v in answer.usage.values() if isinstance(v, dict)))
    return NodeResult({"answer": answer.text, "citations": cites}, tokens, answer.model or None)


# --------------------------------------------------------------------------- registration
_PROMPT_PARAMS: dict[str, dict[str, Any]] = {
    "template": _s(
        "Prompt", "", format="textarea",
        description="User message. Use {{variable}} to add an input jack per variable.",
    ),
    "system": _s("System message", "", format="textarea", description="Optional; supports {{variables}}."),
    **_model_params("chat"),
    "temperature": _n("Temperature", 0.2, minimum=0, maximum=2),
    "max_tokens": _n("Max tokens", 1024, integer=True, minimum=16, maximum=32768),
}  # fmt: skip


def _register_all() -> None:
    T = register_node_type
    T(NodeType(
        "input", "Input", "io", "A named, typed pipeline parameter.", _run_input,
        params={
            "name": _s("Name", "", description="Input name (defaults to the node id)."),
            "type": _s("Type", "string", enum=list(INPUT_TYPES)),
            "default": {"title": "Default value", "description": "Used when the run gives no value."},
            "description": _s("Description", ""),
        },
        ports_hook=_input_ports, glyph="⏵",
    ))  # fmt: skip
    T(NodeType(
        "output", "Output", "io", "Collects a result under a name.", _run_output,
        params={"name": _s("Name", "", description="Output name (defaults to the node id).")},
        inputs=(Port("value", "any"),), glyph="⏹",
    ))  # fmt: skip
    T(NodeType(
        "prompt", "Prompt", "model", "Fill a template and ask a model.", _run_prompt,
        params=_PROMPT_PARAMS, required=("template",),
        input_vars=(VarSource("template"), VarSource("system")),
        outputs=(Port("text", "text"),), cacheable=True, glyph="✎",
    ))  # fmt: skip
    T(NodeType(
        "json", "JSON", "model", "Ask a model for JSON that matches a schema (chat_json).", _run_json,
        params={
            **{k: v for k, v in _PROMPT_PARAMS.items() if k != "temperature"},
            "temperature": _n("Temperature", 0.0, minimum=0, maximum=2),
            "schema": {
                "type": "object", "title": "JSON Schema", "format": "json",
                "default": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "description": "Inline JSON Schema the reply must satisfy.",
            },
            "repair_attempts": _n("Repair attempts", 2, integer=True, minimum=0, maximum=5),
        },
        required=("template",), input_vars=(VarSource("template"), VarSource("system")),
        outputs=(Port("data", "json"),), cacheable=True, glyph="{}",
    ))  # fmt: skip
    T(NodeType(
        "embed", "Embed", "model", "Embed texts; optionally drop near-duplicates.", _run_embed,
        params={
            **_model_params("embed", ["embed"]),
            "input_type": _s("Input type", "passage", enum=["passage", "query"]),
            "dedupe_threshold": _n(
                "Dedupe threshold", 0, minimum=0, maximum=1,
                description="Cosine similarity at or above which a text counts as a duplicate. 0 = off.",
            ),
        },
        inputs=(Port("texts", "list"),),
        outputs=(Port("vectors", "vectors"), Port("unique", "list"), Port("count", "number")),
        cacheable=True, glyph="∷",
    ))  # fmt: skip
    T(NodeType(
        "rerank", "Rerank", "model", "Order passages by relevance to a query.", _run_rerank,
        params={**_model_params("rerank", ["rerank"]), "top_n": _n("Top N", 0, integer=True, minimum=0)},
        inputs=(Port("query", "text"), Port("passages", "list")),
        outputs=(Port("ranked", "list"), Port("scores", "list")), cacheable=True, glyph="⇅",
    ))  # fmt: skip
    T(NodeType(
        "branch", "Branch", "logic",
        "Send the value down 'then' or 'else' depending on a safe expression.", _run_branch,
        params={
            "condition": _s(
                "Condition", "value", format="expr",
                description="Safe expression, e.g. value['category'] == 'billing'. "
                "No attribute access or calls.",
            )
        },
        required=("condition",), inputs=(Port("value", "any"),),
        input_vars=(VarSource("condition", "expr"),), input_exclude=("value",),
        outputs=(Port("then", "any"), Port("else", "any")), glyph="⑂",
    ))  # fmt: skip
    T(NodeType(
        "map", "Map", "logic", "Run one step for every item of a list, a few at a time.", _run_map,
        params={
            "step": {
                "type": "object", "title": "Step", "format": "step",
                "default": {"type": "prompt", "params": {"template": "{{item}}"}},
                "description": "The node run for each item. {{item}} and {{index}} are available.",
            },
            "concurrency": _n("Concurrency", 4, integer=True, minimum=1, maximum=32),
            "max_items": _n("Max items", 200, integer=True, minimum=1, maximum=2000),
            "on_error": _s("On item error", "fail", enum=["fail", "skip", "null"]),
        },
        ports_hook=_map_ports, cacheable=True, glyph="⇶",
    ))  # fmt: skip
    T(NodeType(
        "merge", "Merge", "logic", "Take the first live value (joins the arms of a branch).",
        _run_merge, inputs=(Port("in", "any", required=False, multi=True),),
        outputs=(Port("out", "any"),), any_input=True, glyph="⑃",
    ))  # fmt: skip
    T(NodeType(
        "join", "Join", "logic", "Concatenate every live input into one text.", _run_join,
        params={"separator": _s("Separator", "\\n\\n")},
        inputs=(Port("in", "any", required=False, multi=True),),
        outputs=(Port("text", "text"),), any_input=True, glyph="⊕",
    ))  # fmt: skip
    T(NodeType(
        "template", "Template", "data", "Pure string formatting with {{variables}} and filters.",
        _run_template, params={"template": _s("Template", "", format="textarea")},
        required=("template",), input_vars=(VarSource("template"),),
        outputs=(Port("text", "text"),), glyph="❝",
    ))  # fmt: skip
    T(NodeType(
        "regex", "Regex", "data", "Search, find all, split or substitute with a regular expression.",
        _run_regex,
        params={
            "pattern": _s("Pattern", "", format="code"),
            "mode": _s("Mode", "search", enum=["search", "findall", "split", "sub"]),
            "group": _n("Group", 0, integer=True, minimum=0, maximum=20),
            "flags": _s("Flags", "", description="Any of i (ignore case), m (multiline), s (dotall)."),
            "replacement": _s("Replacement", "", description="For mode 'sub'."),
        },
        required=("pattern",), inputs=(Port("text", "text"),),
        outputs=(Port("match", "text"), Port("matches", "list"), Port("found", "bool")), glyph=".*",
    ))  # fmt: skip
    T(NodeType(
        "chunk", "Chunk", "data", "Split long text into overlapping chunks.", _run_chunk,
        params={
            "max_chars": _n("Max characters", 1200, integer=True, minimum=50, maximum=20000),
            "overlap": _n("Overlap", 100, integer=True, minimum=0, maximum=2000),
        },
        inputs=(Port("text", "text"),),
        outputs=(Port("chunks", "list"), Port("count", "number")), glyph="▤",
    ))  # fmt: skip
    T(NodeType(
        "python_fn", "Python fn", "data", "Call a registered function (never arbitrary code).",
        _run_python_fn,
        params={
            "fn": _s("Function", "flatten", format="function", description="A registered function name."),
            "args": {"type": "object", "title": "Arguments", "format": "json", "default": {}},
        },
        required=("fn",), inputs=(Port("value", "any", required=False),),
        outputs=(Port("result", "any"),), glyph="ƒ",
    ))  # fmt: skip
    T(NodeType(
        "http_get", "HTTP GET", "net", "Fetch a URL. Disabled unless the host is allowlisted.",
        _run_http_get,
        params={
            "url": _s("URL", "", description="Supports {{variables}}."),
            "timeout": _n("Timeout (s)", 15, minimum=1, maximum=120),
            "max_bytes": _n("Max bytes", 200000, integer=True, minimum=1, maximum=MAX_HTTP_BYTES),
        },
        required=("url",), input_vars=(VarSource("url"),),
        outputs=(Port("body", "text"), Port("status", "number")), glyph="⇣",
    ))  # fmt: skip
    T(NodeType(
        "council", "Council", "apps", "Ask the multi-model Council (nimlab.council).", _run_council,
        params={
            "strategy": _s("Strategy", "moa", enum=["moa", "debate", "best-of-n", "critic"]),
            "models": {"type": "array", "title": "Models", "items": {"type": "string"}, "default": [], "format": "models"},
            "rounds": _n("Rounds", 0, integer=True, minimum=0, maximum=6, description="0 = strategy default."),
            "layers": _n("Layers", 0, integer=True, minimum=0, maximum=6, description="0 = strategy default."),
            "n": _n("Candidates (n)", 0, integer=True, minimum=0, maximum=16, description="0 = strategy default."),
        },
        inputs=(Port("question", "text"),),
        outputs=(Port("answer", "text"), Port("agreement", "number"), Port("rationale", "text")),
        cacheable=True, glyph="◎",
    ))  # fmt: skip
    T(NodeType(
        "atlas_ask", "Atlas ask", "apps", "Ask an Atlas collection (cited RAG).", _run_atlas_ask,
        params={
            "collection": _s("Collection", "default"),
            "mode": _s("Mode", "simple", enum=["simple", "agentic"]),
            "verify": _b("Verify claims"),
            "db": _s("Database path", "", description="Defaults to the Atlas database."),
        },
        inputs=(Port("question", "text"),),
        outputs=(Port("answer", "text"), Port("citations", "list")), cacheable=True, glyph="◈",
    ))  # fmt: skip


_register_all()
