# Loom: patchable pipelines for NIM models

Loom lets you wire NIM models, text tools and the other NIM Lab apps into a **pipeline**: a JSON
document that describes a DAG of modules joined by cables. You can run it from Python, from the CLI,
or build it visually in NIM Lab Studio on a patch bay, where nodes are rack modules with jacks and
cables carry signal pulses while a run is live. Everything works offline in mock mode.

```bash
nimlab loom list                              # bundled example pipelines
nimlab loom run research-brief --mock         # run one
nimlab loom run support-triage --mock --set ticket="My invoice is wrong" --json
nimlab loom lint my-pipeline.json             # static checks
nimlab loom graph my-pipeline.json            # mermaid diagram
nimlab loom export my-pipeline.json -o run_it.py   # standalone script that uses nimlab.core
nimlab loom new "Release notes"               # scaffold ./release-notes.json
```

```python
import asyncio
from nimlab.loom import Pipeline, run_pipeline

pipe = Pipeline.load("my-pipeline.json")          # or run_pipeline("research-brief", ...)
result = asyncio.run(run_pipeline(pipe, {"topic": "tidal power"}, mock=True))
print(result.ok, result.outputs["brief"], result.tokens)
```

## Contents

[Pipeline format](#pipeline-format) | [Node types](#node-types) | [Templates and expressions](#templates-and-expressions) |
[Engine](#engine) | [Events](#events) | [Lint](#lint) | [Export](#export) | [CLI](#cli) | [Python API](#python-api) |
[Extending Loom](#extending-loom) | [Studio](#studio) | [Bundled pipelines](#bundled-pipelines) | [Testing](#testing)

## Pipeline format

A pipeline is a versioned, pydantic-validated JSON document (`SCHEMA_VERSION = 1`; a newer version is
refused with a clear message).

```json
{
  "version": 1,
  "name": "demo",
  "title": "Demo",
  "description": "What this pipeline does.",
  "nodes": [
    {"id": "q", "type": "input", "params": {"type": "string", "default": "hello"}, "pos": {"x": 40, "y": 60}},
    {"id": "a", "type": "prompt", "params": {"template": "Answer: {{q}}", "role": "chat"}, "retries": 1, "timeout": 60},
    {"id": "out", "type": "output", "params": {"name": "answer"}}
  ],
  "edges": [
    {"from": "q.value", "to": "a.q"},
    {"from": "a.text", "to": "out.value"}
  ]
}
```

| Node field | |
| --- | --- |
| `id` | letters, digits, `_`, `-` (max 48); unique |
| `type` | a registered node type |
| `label` | display name in the editor (optional) |
| `params` | the type's parameters (see below) |
| `defaults` | values for input jacks that have no cable, e.g. `{"tone": "friendly"}` |
| `retries` | extra attempts after a failure (0 to 10), exponential backoff from 0.5 s |
| `timeout` | seconds per attempt (`asyncio.wait_for`); a timeout counts as a failure |
| `cache` | `true`/`false`; omitted = the type's default (model nodes cache, pure nodes do not) |
| `pos` | `{x, y}` for the editor; missing positions are laid out automatically |

An **edge** goes from an output jack to an input jack: `"node.port"` to `"node.port"`. Unknown keys are
rejected, duplicate ids, edges to unknown nodes and malformed endpoints are schema errors.

### Jacks and types

Every jack has a data type: `text`, `number`, `bool`, `json` (any structured value), `list`, `vectors`
(embeddings) or `any`. Compatible jacks (`text` to `any`, `number` to `text`, `list` to `json`...) patch
cleanly, lossy pairs are converted at run time and warned about by lint (`text` to `number`, `json` to
`text`...), clearly incompatible pairs (`vectors` to `text`, `list` to `number`) are lint errors and the
editor refuses to patch them.

Input jacks of `prompt`, `json`, `template`, `http_get` nodes are **derived from the `{{variables}}`** in
their templates; `branch` jacks come from the identifiers in its condition; `map` jacks from its step.
Edit the template and the jacks follow. A jack that takes several cables (`merge`, `join`) is `multi`.

## Node types

`GET /api/loom/node-types` (and `nimlab.loom.node_types()`) lists everything below with JSON-Schema
parameter descriptions; the Studio inspector renders its forms from them.

| Type | Jacks (in, out) | What it does |
| --- | --- | --- |
| `input` | none, `value` | A named, typed pipeline parameter: `type` (`string number boolean json list`), `default`, `name`. No `default` key means the input is required. |
| `prompt` | `{{vars}}`, `text` | Renders `template` (and optional `system`) and calls a model. `model` or `role`, `temperature`, `max_tokens`. Streams `node.token` events when someone listens. |
| `json` | `{{vars}}`, `data` | `chat_json` against an inline JSON `schema`; invalid replies are re-prompted (`repair_attempts`). |
| `embed` | `texts`, `vectors` `unique` `count` | Embeds a list. `dedupe_threshold` (cosine, 0 = off) also returns the near-duplicate-free texts. |
| `rerank` | `query` `passages`, `ranked` `scores` | Orders passages for a query (`top_n`). |
| `branch` | `value` + condition vars, `then` `else` | Evaluates a [safe expression](#templates-and-expressions); the value leaves on exactly one side. |
| `map` | `items` + step vars, `results` `count` | Runs one `step` (any non-control node: `prompt`, `json`, `template`, `regex`, `python_fn`...) for every item with `concurrency` at a time. `{{item}}` and `{{index}}` are available; other variables of the step become jacks shared by all items. `on_error`: `fail`, `skip` or `null`; `max_items` guards runaway fan-out. |
| `merge` | `in` (many), `out` | First live value: joins the arms of a branch. |
| `join` | `in` (many), `text` | Concatenates every live input (lists are flattened) with `separator`. |
| `template` | `{{vars}}`, `text` | Pure string formatting. |
| `regex` | `text`, `match` `matches` `found` | `search`, `findall`, `split` or `sub`; `group`, `flags` (`i m s`), `replacement`. Patterns over 500 characters and texts over 1 MB are refused. |
| `chunk` | `text`, `chunks` `count` | Overlapping chunks (`nimlab.core.chunk_text`). |
| `python_fn` | `value`, `result` | Calls a **registered** function with `args`. Arbitrary code is never accepted. |
| `http_get` | `{{vars}}` in `url`, `body` `status` | **Disabled by default.** Only hosts allowed by `NIMLAB_LOOM_HTTP_ALLOW` (comma separated, `*.example.com` ok) or the `http_allow` run option; http(s) only, no credentials in the URL, no redirects, size-capped. |
| `council` | `question`, `answer` `agreement` `rationale` | Calls `nimlab.council.Council` (`strategy`, `models`, `rounds`, `layers`, `n`). |
| `atlas_ask` | `question`, `answer` `citations` | Asks an Atlas collection (`collection`, `mode`, `verify`, `db`). |
| `output` | `value`, none | Collects a result under `name` (default: the node id). |

Registered functions (`python_fn`): `flatten`, `pluck`, `flatten_field`, `get_field`, `unique`, `sort`,
`top_k`, `count`, `join_lines`, `split_lines`, `sentences`, `word_count`, `truncate`, `to_json`,
`from_json`, `lower`, `numbered`. A function gets the `value` input first and `args` as keyword
arguments. Add your own with `register_function`.

## Templates and expressions

**Templates** (`{{name}}`, `{{plan.steps.0}}`, `{{items | bullets}}`) never execute code. Paths walk
dict keys and list indexes. Filters: `json upper lower title trim len lines bullets numbered first last
join`. A list of strings is rendered one per line, other structures as indented JSON; text between
braces that is not a variable path (a JSON example like `{{"a": 1}}`) is left alone.

**Expressions** (`branch` conditions) are parsed with `ast` and interpreted over a strict whitelist:
literals, names (the node's jacks), `and or not`, comparisons including `in`/`not in`, `+ - * / // %`,
`a if c else b`, list/dict literals, subscripts and slices (`value["category"] == "billing"`,
`score >= 7 and "refund" in text`). **Attribute access, calls, lambdas, comprehensions, f-strings and
`**` are rejected** with an explanation. Expressions are capped at 500 characters and 200 nodes, repeats
(`"x" * n`) at 10 000 elements, and there is no `eval` or `exec` anywhere. `true`/`false`/`null` work
as literals.

## Engine

`run_pipeline` builds one asyncio task per node. A node waits for the nodes it reads from and then
starts immediately, so independent branches run **at the same time** (bounded by `max_parallel`, default
16, and by the client's own concurrency limit); a diamond runs its middle in parallel and joins after.

* **Branches work by absence.** `branch` emits on one side only. Everything that needs a value from the
  silent side is `skipped` (`node.skipped`), transitively. `merge` and `join` run as soon as one input is
  live and are skipped only when every input is dead. An `output` behind a skipped arm is simply absent
  from the result.
* **Retries and timeouts** are per node. Each failed attempt emits `node.retry` and waits `0.5 s * 2^n`.
* **Failure is fail-fast.** After the retries are spent the node emits `node.error`, running siblings
  are cancelled and `run_pipeline` returns `RunResult(ok=False, error=..., failed_node=...)`.
  Static problems (cycle, lint errors, unknown or missing inputs) raise before anything runs:
  `CycleError` ("pipeline has a cycle: a -> b -> a"), `PipelineError`, `InputError`.
* **Caching** (`cache=True` or a directory): results are stored under `.nimlab/loom/cache`
  (`$NIMLAB_LOOM_CACHE`) as `<sha256>.json`. The key covers the node type, its resolved params, every
  input value and whether the backend is mock or live, so editing a prompt or an upstream value misses
  and mock results never answer live runs. Model, `map`, `embed`, `council` and `atlas_ask` nodes cache
  by default, set `"cache": false` on a node to opt out. `map` caches per node, not per item. The CLI and
  Studio turn the cache on, the Python API leaves it off.
* **Tokens** are exact for non-streamed chat calls and estimated (4 characters per token) for streamed
  text, `json`, embeddings and rerank.

`RunResult`: `ok`, `outputs` (by output name), `nodes` (`NodeReport` with `status`
`done|cached|skipped|error|pending`, `outputs`, `latency_ms`, `tokens`, `attempts`, `model`, `error`),
`tokens`, `elapsed_s`, `cache_hits`, `error`, `failed_node`; `to_dict()` is JSON-ready.

## Events

`run_pipeline(..., on_event=callback)` receives plain dicts, each with `type` and `t` (ms since start):

| `type` | Fields |
| --- | --- |
| `run.plan` | `nodes` (execution order), `levels`, `inputs`, `allow_http` |
| `node.start` | `node`, `type` |
| `node.token` | `node`, `text` (coalesced to about 25 events per second) |
| `node.progress` | `node`, `done`, `total` (`map`) |
| `node.retry` | `node`, `attempt`, `error` |
| `node.end` | `node`, `preview`, `outputs` (preview per jack), `emitted` (jacks that carried a value), `latency_ms`, `tokens`, `cached`, `model` |
| `node.skipped` | `node`, `reason` |
| `node.error` | `node`, `error`, `attempts` |
| `run.done` | `ok`, `outputs` (previews), `tokens`, `elapsed_ms`, `cache_hits`, `error`, `failed_node` |

A listener that raises is ignored; it can never break a run.

## Lint

`nimlab loom lint FILE` (`nimlab.loom.lint(pipeline_or_dict)`) checks everything that can be known
without calling a model. It never raises on bad input and reports problems with a node and jack so an
editor can light an LED. Exit code 1 on errors (`--strict`: also on warnings; `--json` for tools).

| Code | Severity | Meaning |
| --- | --- | --- |
| `schema` | error | the document does not validate |
| `cycle` | error | `cycle: a -> b -> a` |
| `unknown-type`, `unknown-function` | error | unregistered node type or `python_fn` function |
| `param` | error / warning | missing required parameter, wrong type or range, unknown parameter |
| `expr`, `template` | error | bad branch expression (attribute access, calls...), unknown template filter |
| `unknown-port`, `multiple-edges` | error | a cable to a jack that does not exist; two cables into a single-cable jack |
| `unbound` | error | **unknown variable**: an input jack has no cable and no default |
| `type-mismatch` | error / warning | incompatible jack types / converted at run time |
| `path-type` | warning | `{{x.y}}` reads inside a value that is plain text or a number |
| `unreachable` | warning | the node never contributes to an output |
| `no-output`, `duplicate-name` | error | no output node; two inputs or outputs share a name |
| `http-disabled` | warning | an `http_get` node while no host is allowlisted |
| `unused-default` | warning | a `defaults` entry for a jack that does not exist |

`run_pipeline` refuses to start when lint reports errors (`validate=False` skips this).

## Export

* **Mermaid**: `nimlab loom graph FILE [-o file.mmd]` / `to_mermaid(pipeline)`: a `flowchart LR` with
  stadiums for inputs and outputs, diamonds for branches, subroutines for `map`, cables labelled with
  their jacks and one colour class per node category.
* **Python**: `nimlab loom export FILE -o script.py` / `to_python(pipeline)`: a standalone script
  (`python script.py --mock --set topic=tea --json`) whose model calls use `nimlab.core`
  (`NIMClient`, mock mode included). The pure helpers (templates, the safe evaluator, type coercion,
  regex and URL policy) are copied verbatim into the script, so branch and skip semantics match the
  engine; each dependency level runs concurrently. `council`, `atlas_ask` and `python_fn` import from
  `nimlab`. The tests run every exported bundled pipeline and compare outputs with the engine.

## CLI

`nimlab loom ...` (also `python -m nimlab.loom.cli`; the app is `nimlab.loom.cli.app`).

| Command | |
| --- | --- |
| `list [--json]` | bundled pipelines plus those saved in `~/NIMLab/loom/` |
| `run FILE\|name [--set k=v] [--json] [--mock] [--no-cache] [--allow-http HOST] [-q]` | run and print outputs. `--set k=@file.txt` reads a file; values are coerced to the input's type. Progress goes to stderr. Exit 1 if a node failed, 2 for usage or pipeline errors |
| `lint FILE [--strict] [--json]` | static checks |
| `graph FILE [-o file]` | mermaid |
| `export FILE -o script.py` | standalone Python script |
| `new NAME [-o path] [--force]` | scaffold input, prompt and output |
| `cache-clear` | delete the node cache |

## Python API

```python
from nimlab.loom import (
    Pipeline, run_pipeline, lint, to_mermaid, to_python,
    register_function, register_node_type, NodeType, Port,
    list_pipelines, load_pipeline, new_pipeline,
)
```

`run_pipeline(pipeline, inputs=None, *, client=None, on_event=None, cache=False, stream=None,
max_parallel=16, http_allow=(), http_client=None, atlas_db=None, validate=True, mock=None)`.
`pipeline` may be a `Pipeline`, its dict, a file path or a bundled/saved name. Without `client` a
`NIMClient` is created from the environment (and closed afterwards). Any object with `chat`,
`chat_json`, `embed`, `rerank` (and optionally `stream_chat`, `mock`) works, which is how the tests
drive it with a fake client.

## Extending Loom

```python
from nimlab.loom import NodeType, Port, register_function, register_node_type
from nimlab.loom.context import NodeContext, NodeResult

register_function("title_case", lambda value, **_: str(value).title())

async def run(ctx: NodeContext) -> NodeResult:           # ctx.inputs, ctx.params, ctx.client
    return NodeResult({"text": str(ctx.inputs["text"])[::-1]})

register_node_type(NodeType("reverse", "Reverse", "data", "Reverse the text.", run,
                            inputs=(Port("text", "text"),), outputs=(Port("text", "text"),)))
```

A node returns values only for the jacks it wants to emit; leaving a jack out is how `branch` skips a
side. `params` is a dict of JSON-Schema properties (`title`, `description`, `enum`, `default`,
`minimum`/`maximum`, and `format`: `textarea`, `json`, `model`, `expr`, `code`) which drives validation
and the Studio form. See `examples/loom/custom_node.py`.

## Studio

The **Loom** screen is a modular-synth style node editor.

* **Modules** are rack units with input jacks on the left and output jacks on the right. Drag a module
  by its header to move it (several at once when selected; shift+drag on the background draws a
  selection box). **Drag from a jack to a compatible jack** to patch a sagging bezier cable, colour-coded
  by data type (text pink, json cyan, list lime, number orange, bool violet, vectors mint). Lift a cable
  off an input jack to re-patch it, drop it on empty canvas to unplug it, double-click a cable to remove
  it. Cables that would create a loop or join incompatible jacks are refused with a message; dropping on
  a module body picks a free compatible jack for you.
* **Palette** (left): search, click to add at the view centre or drag onto the canvas.
* **Inspector** (right): schema-driven forms for the selected module, unwired-input defaults,
  reliability options (retries, timeout, cache), problems and the last run. With nothing selected it
  shows the pipeline title, the run inputs, all lint problems and the shortcut list.
* **Running**: press *Run* (or Cmd/Ctrl+Enter). Modules glow while active, text streams onto their LCD
  screens, signal pulses travel along the cables that carry a value, skipped branches dim, retries and
  errors show on the module. The outputs tray (toolbar button) lists the results and an event log.
* **Lint** runs while you edit: an amber (warning) or red (error) LED on each module header with the
  messages in its tooltip, plus a count in the toolbar. A run is refused while errors exist.
* **View**: Ctrl/Cmd+scroll or the buttons to zoom, scroll or drag the background to pan, `F` to fit.
  Cables live in one SVG layer and modules are plain DOM positioned with transforms; jack positions
  are computed from the model, so a 30-module graph drags at full frame rate.
* **Keyboard**: Delete removes the selection (module or cable), Cmd/Ctrl+Z / Shift+Z undo and redo
  (a snapshot command stack; field edits coalesce), Cmd/Ctrl+S saves, +/- zoom, arrows nudge, Cmd/Ctrl+D
  duplicates, Cmd/Ctrl+A selects all, Escape deselects. Modules are focusable (Enter selects).
* **Open / Save / Export**: bundled pipelines are read-only (Save makes a copy). Saved pipelines live in
  `~/NIMLab/loom/*.json` (`$NIMLAB_LOOM_DIR` overrides; otherwise `<Studio home>/loom`). *Export* copies
  the mermaid diagram or the Python script to the clipboard. Unsaved work survives switching screens.

### HTTP API

Everything is under `/api/loom` (token-protected like the rest of Studio).

| Endpoint | |
| --- | --- |
| `GET /node-types` | every node type: `{type, title, category, description, glyph, params (JSON Schema), inputs, input_vars, outputs, cacheable}` |
| `GET /pipelines` | `[{id, name, title, description, source: "bundled"\|"user", nodes, readonly}]` |
| `GET /pipelines/{id}` | `{id, source, readonly, pipeline}` (positions are filled in) |
| `PUT /pipelines/{id}` | save `{pipeline: {...}}` (or the bare document); 409 for a bundled id, 422 for schema errors. Lint errors do not block saving; the reply carries counts |
| `DELETE /pipelines/{id}` | 403 for bundled, 404 if missing |
| `POST /lint` | `{pipeline}` gives `{ok, errors, warnings, issues: [{severity, code, message, node, port}], ports}` |
| `POST /export` | `{pipeline, format: "mermaid"\|"python"}` gives `{format, text}` |
| `POST /run` | `{pipeline \| id, inputs}` gives `{run_id}`; 422 (with `issues`) when lint finds errors or inputs are wrong |

A run is a normal Studio run (`app: "loom"`): follow `GET /api/runs/{run_id}/events`. The SSE events are
the [engine events](#events) (the manager adds `run.start`; the terminal `run.done` carries
`{ok, outputs, tokens, elapsed_s, cache_hits, nodes: {id: {status, latency_ms, tokens, ...}}, simulated}`;
a failing node ends the run with `run.failed` after its `node.error`). Studio runs use the shared
client (the offline mock when no key is set; the run is then marked `simulated`) and cache in
`<Studio home>/loom/cache`.

### Wiring it into Studio

`nimlab.studio.routes_loom` exports `router` (an `APIRouter(prefix="/api/loom")`) and `APP_INFO`
(`id: "loom"`, colour `#3DFFB4`, glyph `≋`). The router needs `app.state.studio` when mounted on the
real app and otherwise keeps its own run manager, so it also works on a bare `FastAPI()`.

1. **Server**: `app.include_router(routes_loom.router)` *before* the router that holds the `/api/{rest}`
   catch-all (routes match in order). Add `APP_INFO` next to the other apps (`state.APPS`, run id list).
2. **CLI**: add `"loom"` to `PLUGIN_NAMES` in `nimlab/cli.py` (the plugin contract exposes
   `nimlab.loom.cli.app`).
3. **Web**: add `{ id: 'loom', name: 'Loom', key: '6', cls: 'app-loom', icon: 'loom', app: 'loom' }` to
   `APPS` in `js/app.js` (and extend the `⌘1..5` key check); the screen module is `js/screens/loom.js`
   exporting `mount(el, params)`, which loads its own stylesheet `css/screens/loom.css` (a `<link>` in
   `index.html` is optional). Tokens: `--c-loom: #3DFFB4` (light theme `#00855A`) and
   `.app-loom { --accent: var(--c-loom); }`. Rail icon `loom`, 24 px grid:
   `<circle cx="4.5" cy="8" r="1.8"/><circle cx="19.5" cy="16" r="1.8"/><path d="M6.3 8C12 8 12 16 17.7 16"/><circle cx="4.5" cy="16" r="1.8"/><circle cx="19.5" cy="8" r="1.8"/><path d="M6.3 16C12 16 12 8 17.7 8"/>`.
   `go('loom', {pipeline: 'support-triage'})` opens a specific pipeline.

## Bundled pipelines

| Id | Shape |
| --- | --- |
| `research-brief` | `json` decomposes a topic into questions, `map` answers each in parallel, `prompt` synthesises one brief |
| `support-triage` | `json` classifies a ticket (enum category, urgency), two chained `branch`es pick a billing, technical or general responder, `merge` joins the reply |
| `doc-to-quiz` | `chunk` the document, `map` a `json` question generator over the chunks, flatten, `embed` with `dedupe_threshold`, number the unique questions |
| `translate-backcheck` | translate, translate back, a `json` judge scores the round trip, a `branch` keeps the translation or retries it with the judge's notes |

`examples/loom/` adds `quickstart.py` (build, lint and run in code), `custom_node.py` (register a
function and a node type) and `meeting-actions.json`.

## Testing

```bash
python -m pytest tests/loom tests/studio/test_routes_loom.py -q
ruff check src/nimlab/loom && ruff format --check src/nimlab/loom && mypy src/nimlab/loom
```

The suite needs no key or network: schema validation, cycle detection, parallel scheduling (a fake
client with delays records peak concurrency), branch / map / merge semantics, the safe evaluator
(including rejected attribute access and calls), caching, retries and timeouts, lint, mermaid and
Python export (every exported bundled pipeline is executed), every bundled pipeline end to end in
mock mode, the CLI through `CliRunner --mock` and the routes on a bare FastAPI app plus once on the
real Studio app with SSE.
