"""Export a pipeline as a mermaid diagram or as a standalone Python script.

The script depends only on ``nimlab`` (``nimlab.core`` for every model call). The pure helpers it
shares with the engine (templates, the safe expression evaluator, type coercion, regex and URL
policy) are copied into it verbatim, so it behaves like ``loom run``. Each dependency level runs
concurrently; branches skip the untaken side exactly as the engine does.
"""

# ruff: noqa: E501  (code generator: long f-string templates are clearer unsplit)
from __future__ import annotations

import inspect
import json
from typing import Any

from . import datatypes as _datatypes
from . import expr as _expr
from . import template as _template
from . import textops as _textops
from .registry import get_node_type, ports_for
from .schema import Pipeline, topo_levels

__all__ = ["to_mermaid", "to_python"]

_MERMAID_STYLE = {
    "io": ("#2b2540", "#8B7BFF"),
    "model": ("#3a1a2e", "#FF3D9A"),
    "logic": ("#3a2616", "#FF8A3D"),
    "data": ("#26361a", "#C6FF3D"),
    "apps": ("#12313a", "#22E1FF"),
    "net": ("#12372c", "#3DFFB4"),
}


# --------------------------------------------------------------------------- mermaid
def _q(text: str) -> str:
    return text.replace('"', "#quot;").replace("<", "&lt;").replace(">", "&gt;")


def to_mermaid(pipeline: Pipeline) -> str:
    """A ``flowchart LR`` diagram: shapes by node kind, cables labelled with their jacks."""
    lines = [f"%% {pipeline.title or pipeline.name}", "flowchart LR"]
    cats: dict[str, str] = {}
    outs_count: dict[str, int] = {}
    ins_count: dict[str, int] = {}
    for n in pipeline.nodes:
        nt = get_node_type(n.type)
        ins, outs = ports_for(nt, n.params)
        outs_count[n.id], ins_count[n.id] = len(outs), len(ins)
        title = _q(n.label or n.id)
        text = f'"{title}<br/><i>{nt.name}</i>"'
        nid = f"n_{n.id}".replace("-", "_")
        shape = {
            "input": f"([{text}])",
            "output": f"([{text}])",
            "branch": "{" + text + "}",
            "map": f"[[{text}]]",
            "merge": f"(({text}))",
            "join": f"(({text}))",
        }.get(n.type, f"[{text}]")
        lines.append(f"    {nid}{shape}")
        cats.setdefault(nt.category, "")
        lines.append(f"    class {nid} cat_{nt.category}")
    for e in pipeline.edges:
        label = ""
        if outs_count.get(e.src_node, 1) > 1 and ins_count.get(e.dst_node, 1) > 1:
            label = f"{e.src_port} → {e.dst_port}"
        elif outs_count.get(e.src_node, 1) > 1:
            label = e.src_port
        elif ins_count.get(e.dst_node, 1) > 1:
            label = e.dst_port
        arrow = f'-->|"{_q(label)}"|' if label else "-->"
        src = f"n_{e.src_node}".replace("-", "_")
        dst = f"n_{e.dst_node}".replace("-", "_")
        lines.append(f"    {src} {arrow} {dst}")
    for cat in cats:
        fill, stroke = _MERMAID_STYLE.get(cat, ("#222", "#888"))
        lines.append(f"    classDef cat_{cat} fill:{fill},stroke:{stroke},color:#f3f0ff")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- python script
def _module_source(module: Any, extra_drop: tuple[str, ...] = ()) -> str:
    keep = [
        ln
        for ln in inspect.getsource(module).splitlines()
        if not ln.startswith(("from __future__", "from .", *extra_drop))
    ]
    return "\n".join(keep)


def _lit(value: Any) -> str:
    return repr(value)


def _indent(lines: list[str], n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln else ln for ln in lines)


def _body(type_name: str, node_id: str, p: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    """Python statements implementing one node (variables: ``client``, ``i``)."""
    if type_name == "input":
        return [f"return {{'value': INPUTS[{(p.get('name') or node_id)!r}]}}"]
    if type_name == "output":
        return ["return {'value': i.get('value')}"]
    if type_name == "prompt":
        return [
            f"messages = build_messages({str(p.get('system') or '')!r}, {str(p.get('template') or '')!r}, i)",
            f"model = pick_model({p.get('model')!r}, {p.get('role', 'chat')!r})",
            "res = await client.chat(model, messages, "
            f"temperature={float(p.get('temperature', 0.2))!r}, max_tokens={int(p.get('max_tokens', 1024))!r})",
            "return {'text': res.content.strip()}",
        ]
    if type_name == "json":
        return [
            f"messages = build_messages({str(p.get('system') or '')!r}, {str(p.get('template') or '')!r}, i)",
            f"model = pick_model({p.get('model')!r}, {p.get('role', 'chat')!r})",
            f"data = await ask_json(client, model, messages, {_lit(p.get('schema') or {'type': 'object'})}, "
            f"temperature={float(p.get('temperature', 0.0))!r}, max_tokens={int(p.get('max_tokens', 1024))!r}, "
            f"retries={int(p.get('repair_attempts', 2))!r})",
            "return {'data': data}",
        ]
    if type_name == "embed":
        thr = float(p.get("dedupe_threshold") or 0)
        return [
            "texts = as_texts(i.get('texts'))",
            f"model = pick_model({p.get('model')!r}, {p.get('role', 'embed')!r})",
            f"vectors = await client.embed(model, texts, input_type={p.get('input_type', 'passage')!r})",
            f"keep = dedupe_indices(vectors, {thr!r}, cosine) if {thr!r} > 0 else list(range(len(texts)))",
            "unique = [texts[k] for k in keep]",
            "return {'vectors': vectors, 'unique': unique, 'count': len(unique)}",
        ]
    if type_name == "rerank":
        return [
            f"model = pick_model({p.get('model')!r}, {p.get('role', 'rerank')!r})",
            "passages = as_texts(i.get('passages'))",
            f"top_n = {int(p['top_n']) if p.get('top_n') else None!r}",
            "ranked = await client.rerank(model, stringify(i.get('query')), passages, top_n=top_n)",
            "return {'ranked': [r.text for r in ranked], 'scores': [round(r.score, 4) for r in ranked]}",
        ]
    if type_name == "branch":
        return [
            f"taken = bool(evaluate({str(p.get('condition') or 'value')!r}, i))",
            "return {'then' if taken else 'else': i.get('value')}",
        ]
    if type_name == "merge":
        return ["live = i.get('in') or []", "return {'out': live[0] if live else None}"]
    if type_name == "join":
        sep = str(p.get("separator", "\\n\\n")).encode().decode("unicode_escape")
        return [
            "items = []",
            "for value in i.get('in') or []:",
            "    items.extend(value if isinstance(value, list) else [value])",
            f"return {{'text': {sep!r}.join(stringify(v) for v in items)}}",
        ]
    if type_name == "template":
        return [f"return {{'text': render({str(p.get('template') or '')!r}, i)}}"]
    if type_name == "regex":
        return [
            f"return regex_apply(stringify(i.get('text')), {str(p.get('pattern') or '')!r}, "
            f"{p.get('mode', 'search')!r}, {int(p.get('group', 0))!r}, {str(p.get('flags') or '')!r}, "
            f"{str(p.get('replacement', ''))!r})"
        ]
    if type_name == "chunk":
        return [
            f"chunks = chunk_text(stringify(i.get('text')), {int(p.get('max_chars', 1200))!r}, "
            f"{int(p.get('overlap', 100))!r})",
            "return {'chunks': chunks, 'count': len(chunks)}",
        ]
    if type_name == "python_fn":
        return [
            "from nimlab.loom.functions import get_function  # registered functions only",
            f"fn = get_function({str(p.get('fn') or '')!r})",
            f"if fn is None:\n        raise SystemExit('function {p.get('fn')} is not registered')",
            f"result = fn(i.get('value'), **{_lit(p.get('args') or {})})",
            "if inspect.isawaitable(result):\n        result = await result",
            "return {'result': result}",
        ]
    if type_name == "http_get":
        return [
            f"url = render({str(p.get('url') or '')!r}, i).strip()",
            f"body, status = await http_fetch(url, {float(p.get('timeout', 15))!r}, {int(p.get('max_bytes', 200000))!r})",
            "return {'body': body, 'status': status}",
        ]
    if type_name == "council":
        kw = {k: int(p[k]) for k in ("rounds", "layers", "n") if p.get(k)}
        models = [str(m) for m in (p.get("models") or []) if str(m).strip()] or None
        return [
            "from nimlab.council import Council",
            f"res = await Council(client).run(stringify(i.get('question')).strip(), strategy={p.get('strategy', 'moa')!r}, "
            f"models={models!r}, **{_lit(kw)})",
            "return {'answer': res.answer, 'agreement': round(res.agreement, 4) if res.agreement is not None else 0, "
            "'rationale': res.rationale or ''}",
        ]
    if type_name == "atlas_ask":
        return [
            "from nimlab.atlas import Atlas",
            "from nimlab.atlas.config import DEFAULT_DB_PATH",
            f"atlas = Atlas(client, {p.get('db') or None!r} or os.environ.get('NIMLAB_ATLAS_DB') or DEFAULT_DB_PATH)",
            "try:",
            f"    ans = await atlas.ask(stringify(i.get('question')).strip(), {str(p.get('collection') or 'default')!r}, "
            f"mode={p.get('mode', 'simple')!r}, verify={bool(p.get('verify', False))!r})",
            "finally:",
            "    atlas.close()",
            "return {'answer': ans.text, 'citations': [c.model_dump(mode='json') for c in ans.citations]}",
        ]
    raise ValueError(f"cannot export node type {type_name!r}")


def _node_function(n: Any) -> str:
    nt = get_node_type(n.type)
    fid = n.id.replace("-", "_")
    if n.type == "map":
        step = n.params.get("step") or {}
        st = get_node_type(str(step.get("type") or "prompt"))
        sp = dict(step.get("params") or {})
        _, souts = ports_for(st, sp)
        item_port = next((q.name for q in st.inputs if q.required), None)
        primary = souts[0].name if souts else "text"
        step_fn = (
            f"async def n_{fid}__step(client, i):\n"
            + _indent(_body(st.name, n.id, sp, n.defaults))
            + "\n\n"
        )
        c = int(n.params.get("concurrency", 4))
        on_error = str(n.params.get("on_error", "fail"))
        extra = f"        inp[{item_port!r}] = item\n" if item_port else ""
        return (
            step_fn
            + f"async def n_{fid}(client, i):  # map\n"
            + "    items = coerce(i.get('items'), 'list') or []\n"
            + f"    sem = asyncio.Semaphore({c})\n"
            + "    shared = {k: v for k, v in i.items() if k != 'items'}\n\n"
            + "    async def one(k, item):\n"
            + "        async with sem:\n"
            + "            inp = {**shared, 'item': item, 'index': k}\n"
            + (extra.replace("        inp", "            inp") if extra else "")
            + "            try:\n"
            + f"                return (await n_{fid}__step(client, inp)).get({primary!r})\n"
            + "            except Exception:\n"
            + f"                if {on_error!r} == 'fail':\n"
            + "                    raise\n"
            + f"                return SKIP if {on_error!r} == 'skip' else None\n\n"
            + "    results = [r for r in await asyncio.gather(*(one(k, it) for k, it in enumerate(items))) if r is not SKIP]\n"
            + "    return {'results': results, 'count': len(results)}\n"
        )
    body = _body(n.type, n.id, n.params, n.defaults)
    doc = f"  # {nt.title}"
    return f"async def n_{fid}(client, i):{doc}\n" + _indent(body) + "\n"


_PRELUDE = """
import argparse
import asyncio
import inspect
import json
import os
import re
import sys
import time
from typing import Any

from nimlab.core import NIMClient, NIMConfig, chunk_text, cosine, default_registry, extract_json
from nimlab.core.mock import JSON_SCHEMA_CLOSE, JSON_SCHEMA_OPEN  # tags the offline mock reads

SKIP = object()  # marks a jack that carried nothing (the untaken side of a branch)


class LoomError(Exception):
    pass


class TemplateError(LoomError):
    pass


class ExprError(LoomError):
    pass


class InputError(LoomError):
    pass
"""

_HELPERS = '''
def pick_model(model, role):
    return (model or "").strip() or default_registry().pick(role)


def build_messages(system, template, scope):
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": render(system, scope).strip()})
    messages.append({"role": "user", "content": render(template, scope)})
    return messages


def as_texts(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [stringify(v) for v in value] if isinstance(value, list) else [stringify(value)]


async def ask_json(client, model, messages, schema, *, temperature, max_tokens, retries):
    """Ask for JSON matching ``schema``; re-prompt on parse errors (validated if jsonschema exists)."""
    note = ("Respond with ONLY a single valid JSON value that conforms to the JSON Schema below. "
            "No explanations and no markdown fences.\\n\\n"
            + JSON_SCHEMA_OPEN + "\\n" + json.dumps(schema, indent=2) + "\\n" + JSON_SCHEMA_CLOSE)
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] += "\\n\\n" + note
    else:
        msgs.insert(0, {"role": "system", "content": note})
    last = None
    for _ in range(retries + 1):
        res = await client.chat(model, msgs, temperature=temperature, max_tokens=max_tokens)
        try:
            data = extract_json(res.content)
            try:
                import jsonschema
                jsonschema.validate(data, schema)
            except ImportError:
                pass
            return data
        except Exception as exc:  # parse or schema error: ask again with the message
            last = exc
            msgs += [{"role": "assistant", "content": res.content or "(empty)"},
                     {"role": "user", "content": f"That was invalid ({exc}). Return ONLY the corrected JSON."}]
    raise LoomError(f"model did not return valid JSON: {last}")


async def http_fetch(url, timeout, max_bytes):
    import httpx
    check_http_url(url, http_allowlist())
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as http:
        async with http.stream("GET", url) as resp:
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) >= max_bytes:
                    break
            return bytes(body[:max_bytes]).decode("utf-8", errors="replace"), resp.status_code
'''

_RUNNER = """
def collect(nid, state, skipped):
    spec = SPEC[nid]
    inputs, dead_required, live, wired = {}, None, False, False
    for port, (ptype, required, multi) in spec["ports"].items():
        sources = spec["wires"].get(port, [])
        values = [coerce(state[s], ptype) for s in sources if s in state and s.split(".")[0] not in skipped]
        wired = wired or bool(sources)
        if multi:
            inputs[port] = values
            live = live or bool(values)
            if sources and not values and required:
                dead_required = f"every input on {port!r} was skipped"
        elif values:
            inputs[port], live = values[0], True
        elif sources:
            inputs[port] = None
            if required:
                dead_required = f"input {port!r} was not produced upstream"
        elif port in spec["defaults"]:
            inputs[port], live = coerce(spec["defaults"][port], ptype), True
        else:
            inputs[port] = None
    if spec["any_input"]:
        return inputs, ("no live input" if wired and not live else None)
    return inputs, dead_required


async def execute(client):
    state, skipped = {}, set()

    async def go(nid):
        inputs, skip = collect(nid, state, skipped)
        if skip:
            skipped.add(nid)
            print(f"  skip  {nid}: {skip}", file=sys.stderr)
            return
        t0 = time.monotonic()
        out = await IMPL[nid](client, inputs)
        for port, value in out.items():
            state[f"{nid}.{port}"] = value
        print(f"  done  {nid} ({time.monotonic() - t0:.2f}s)", file=sys.stderr)

    for level in LEVELS:
        await asyncio.gather(*(go(nid) for nid in level))
    return {name: state[f"{nid}.value"] for name, nid in OUTPUTS.items() if nid not in skipped}


def parse_inputs(pairs):
    values = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or key not in INPUT_SPEC:
            raise SystemExit(f"--set expects name=value with one of: {', '.join(INPUT_SPEC)}")
        values[key] = coerce(raw, INPUT_SPEC[key]["type"])
    for key, spec in INPUT_SPEC.items():
        if key not in values:
            if spec["required"]:
                raise SystemExit(f"missing input {key!r}: pass --set {key}=...")
            values[key] = coerce(spec["default"], spec["type"]) if spec["default"] is not None else ""
    return values


async def main():
    ap = argparse.ArgumentParser(description=PIPELINE_TITLE)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a pipeline input")
    ap.add_argument("--mock", action="store_true", help="offline mock backend (no API key needed)")
    ap.add_argument("--json", action="store_true", help="print the outputs as JSON")
    args = ap.parse_args()
    INPUTS.update(parse_inputs(args.set))
    config = NIMConfig.from_env(**({"mock": True} if args.mock else {}))
    async with NIMClient(config) as client:
        outputs = await execute(client)
    if args.json:
        print(json.dumps(outputs, indent=2, ensure_ascii=False, default=str))
    else:
        for name, value in outputs.items():
            print(f"== {name} ==")
            print(stringify(value))


INPUTS: dict[str, Any] = {}

if __name__ == "__main__":
    asyncio.run(main())
"""


def to_python(pipeline: Pipeline) -> str:
    """A standalone script (uses ``nimlab.core``) that runs the pipeline from the command line."""
    levels = topo_levels(pipeline)
    spec: dict[str, Any] = {}
    input_spec: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    for n in pipeline.nodes:
        nt = get_node_type(n.type)
        ins, outs = ports_for(nt, n.params)
        wires: dict[str, list[str]] = {}
        for e in pipeline.incoming(n.id):
            wires.setdefault(e.dst_port, []).append(e.source)
        spec[n.id] = {
            "ports": {p.name: [p.type, p.required, p.multi] for p in ins},
            "wires": wires,
            "defaults": n.defaults,
            "any_input": nt.any_input,
        }
        if n.type == "input":
            name = str(n.params.get("name") or n.id)
            input_spec[name] = {
                "type": outs[0].type,
                "default": n.params.get("default"),
                "required": "default" not in n.params,
            }
        if n.type == "output":
            outputs[str(n.params.get("name") or n.id)] = n.id
    funcs = "\n\n".join(_node_function(n) for n in pipeline.nodes)
    impl = ", ".join(f"{n.id!r}: n_{n.id.replace('-', '_')}" for n in pipeline.nodes)
    title = pipeline.title or pipeline.name
    parts = [
        "#!/usr/bin/env python3",
        f'"""{_q_doc(title)}\n\nGenerated by `loom export` from the pipeline {pipeline.name!r}. '
        "Run it with:\n\n    python this_script.py --mock --set name=value\n\n"
        'Edit the node functions freely; the helpers below the line are copied from nimlab.loom."""',
        "from __future__ import annotations\n",
        _PRELUDE.strip("\n"),
        "\n# "
        + "-" * 70
        + "\n# helpers copied from nimlab.loom (templates, safe expressions, types)\n# "
        + "-" * 70,
        _module_source(_template),
        _module_source(_expr),
        _module_source(_datatypes),
        _module_source(_textops),
        _HELPERS.strip("\n"),
        "\n# " + "-" * 70 + f"\n# pipeline: {pipeline.name}\n# " + "-" * 70,
        f"PIPELINE_TITLE = {title!r}",
        f"LEVELS = {json.dumps(levels)}",
        f"OUTPUTS = {outputs!r}",
        f"INPUT_SPEC = {input_spec!r}",
        f"SPEC = {spec!r}\n",
        funcs,
        f"\nIMPL = {{{impl}}}",
        _RUNNER.strip("\n"),
    ]
    return "\n".join(parts) + "\n"


def _q_doc(text: str) -> str:
    return text.replace('"""', "'''").replace("\\", "\\\\")
