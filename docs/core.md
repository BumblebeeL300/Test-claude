# `nimlab.core` API reference

Shared foundation for every NIM Lab app: async NVIDIA NIM client, an offline mock backend, a model
registry, a tool-calling agent loop, and text utilities. Everything below is importable from
`nimlab.core` unless a module path is given. Python >= 3.10, pydantic v2, fully typed.

```python
from nimlab.core import (
    NIMClient, NIMConfig,                                    # client + config
    ChatResult, ToolCall, Usage, RerankResult, UsageTracker, # results
    NIMError, NIMAuthError, NIMRateLimitError, NIMResponseError, NIMTimeoutError, NIMJSONError,
    ModelRegistry, ModelInfo, Role, default_registry,        # registry
    tool, Tool, run_agent, AgentRun, AgentStep, AgentEvent, ToolResult,  # agent loop
    strip_think, extract_json, chunk_text, cosine, get_console,          # utilities
)
```

## Setup

```bash
pip install -e '.[dev]'      # or: uv pip install -e '.[dev]'
pytest tests/core            # core suite; `ruff check src tests` for lint
nimlab doctor                # config / mock / connectivity;  nimlab --mock models --live
```

**No API key? Everything still runs.** With no `NVIDIA_API_KEY` (or `NIMLAB_MOCK=1`) the client uses
the deterministic offline mock backend (see [Mock backend](#mock-backend)). Build and test every app
against mock mode; real NIM is a drop-in switch via env vars.

## Quick start

```python
import asyncio
from pydantic import BaseModel
from nimlab.core import NIMClient, Role, default_registry

class Verdict(BaseModel):
    winner: str
    score: int

async def main() -> None:
    reg = default_registry()
    async with NIMClient() as client:                       # NIMConfig.from_env()
        msgs = [{"role": "user", "content": "Which of A or B is better, 1-10?"}]
        r = await client.chat(reg.pick(Role.CHAT), msgs)
        print(r.content, r.usage.total_tokens)
        v = await client.chat_json(reg.pick(Role.REASONING), msgs, Verdict)   # -> Verdict
        vecs = await client.embed(reg.pick(Role.EMBED), ["a", "b"])
        print(client.usage.summary())

asyncio.run(main())
```

## Configuration — `NIMConfig`

Pydantic model. Unspecified fields are read from the environment, then from `./.env`
(only `NVIDIA_*`, `NIM_*`, `NIMLAB_*` keys are loaded; real env vars win; disable with
`NIMLAB_NO_DOTENV=1`). Explicit constructor arguments beat env.

| field | env var | default |
|---|---|---|
| `api_key: str \| None` | `NVIDIA_API_KEY` | `None` (never shown in `repr`) |
| `base_url: str` | `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `rerank_base_url: str` | `NIM_RERANK_URL` | `https://ai.api.nvidia.com/v1/retrieval` |
| `timeout: float` | `NIM_TIMEOUT` | `120` (seconds, per request) |
| `max_retries: int` | `NIM_MAX_RETRIES` | `5` (retries after the first attempt; `0` = no retry) |
| `max_concurrency: int` | `NIM_MAX_CONCURRENCY` | `8` (per client) |
| `mock: bool` | `NIMLAB_MOCK` | `True` if there is no api_key, else `False` |
| `mock_reason: str \| None` | – | `"explicit"`, `"no_api_key"` or `None` (informational) |

* `NIMConfig.from_env(**overrides)` == `NIMConfig(**overrides)`; both read env.
* `NIMLAB_MOCK=1` forces mock even with a key. `NIMLAB_MOCK=0` / `mock=False` forces real calls even
  without a key (self-hosted NIM without auth).
* When mock turns on only because the key is missing, one warning is logged per process
  (logger `nimlab.config`). `nimlab.core.config.quiet_config()` is a context manager that silences it.
* Unknown keyword arguments raise (`extra="forbid"`).

## Client — `NIMClient`

```python
NIMClient(config: NIMConfig | None = None, *, http_client: httpx.AsyncClient | None = None)
```

Async context manager (`async with NIMClient() as c:`) or call `await c.aclose()`. Attributes:
`c.config`, `c.usage` (a `UsageTracker`), `c.mock` (bool). One client may be shared by many
concurrent tasks (a semaphore of `max_concurrency` bounds in-flight HTTP requests; the slot is
released while backing off between retries). It tolerates reuse across sequential `asyncio.run`
calls, but is not thread-safe: use one client per event loop.

```python
async chat(model: str, messages: list[dict], *, temperature: float = 0.2, max_tokens: int = 1024,
           tools: list[dict] | None = None, tool_choice: Any = None,
           response_format: dict | None = None, extra: dict | None = None) -> ChatResult

async stream_chat(model: str, messages: list[dict], *, <same keyword args as chat>) -> AsyncIterator[str]

async chat_json(model: str, messages: list[dict], schema: type[T], *, retries: int = 2, **kw) -> T
                # T bound to BaseModel; **kw are chat() kwargs

async embed(model: str, texts: list[str], *, input_type: Literal["query", "passage"] = "passage")
            -> list[list[float]]

async rerank(model: str, query: str, passages: list[str], top_n: int | None = None)
            -> list[RerankResult]

async list_models() -> list[str]
```

Messages are OpenAI-format dicts (`{"role": "system"|"user"|"assistant"|"tool", "content": ...}`;
content may be a list of parts for vision). `extra` is merged into the request body (e.g.
`{"top_p": 0.9, "nvext": {...}}`).

* **chat** — result `content` never contains `<think>...</think>`; it is moved to `.reasoning`
  (also fills from a `reasoning_content` field). Tool calls come back parsed in `.tool_calls`
  (`content` is `""` then). Use `result.to_message()` to append the assistant turn to a conversation.
* **stream_chat** — SSE; yields text chunks only (think blocks filtered even when a tag is split
  across chunks; tool-call deltas ignored — use `chat` for tools). Retries happen only before the first
  chunk; a failure mid-stream raises. Token usage is estimated (chars/4) unless the mock reports it.
* **chat_json** — injects the schema in a system message (merged into your first system message if
  there is one) inside `<json_schema>...</json_schema>`, extracts JSON tolerantly (fences, prose,
  `<think>`, trailing commas), validates with pydantic, and on failure re-prompts with the validation
  errors up to `retries` more times, then raises `NIMJSONError` (has `.raw`). Each attempt is a real
  call and is counted in `usage`. Does **not** send `response_format`, so it works on any model.
  Prefer it over parsing free text yourself, especially in mock mode.
* **embed** — batches of 32 sent concurrently; order preserved; sends `input_type` and
  `truncate: "END"`. Use `input_type="query"` for search queries, `"passage"` (default) for documents.
  Vector size depends on the model (mock: 1024; `llama-3.2-nv-embedqa-1b-v2`: up to 2048) — never
  hard-code it, derive it from the first vector.
* **rerank** — `POST {rerank_base_url}/{model}/reranking` with
  `{"model", "query": {"text"}, "passages": [{"text"}]}`; reads `rankings[{index, logit}]`. Returns
  `RerankResult(index, score, text)` sorted by score descending (ties by index); `index` is the
  position in your `passages`. Over 256 passages are split into several requests and merged. Scores are
  raw logits (real) or overlap scores (mock): only compare within one call.
* **list_models** — sorted ids from `GET {base_url}/models` (mock: the curated registry ids).

### Retries and errors

Retried with exponential backoff + equal jitter (0.5s base, 30s cap): HTTP 429, 408, >= 500, timeouts
and connection errors. `Retry-After` (seconds or HTTP-date) is honoured as a minimum delay; if it is
longer than 120s the error is raised immediately. 401/403 and other 4xx are never retried.

```
NIMError                       .message .status_code .body .model
├─ NIMAuthError                401/403
├─ NIMRateLimitError           429 after retries (.retry_after)
├─ NIMResponseError            other 4xx, 5xx after retries, malformed body
│  └─ NIMJSONError             chat_json gave up (.raw)
└─ NIMTimeoutError             timeouts after retries
```

### Results (pydantic)

```python
Usage(prompt_tokens: int, completion_tokens: int, total_tokens: int)
ToolCall(id: str, name: str, arguments: dict[str, Any])
ChatResult(model: str, content: str, reasoning: str | None, tool_calls: list[ToolCall],
           finish_reason: str | None, usage: Usage, latency_s: float)   # .to_message() -> dict
RerankResult(index: int, score: float, text: str)
```

If a model returns malformed tool-call JSON, `ToolCall.arguments == {"_raw": "<the string>"}`.

### Usage — `client.usage: UsageTracker`

Thread- and async-safe. Every client call records calls / tokens / latency / errors per model
(embed and rerank record under their model id, `list_models` under `"(list_models)"`).

```python
client.usage.summary()
# {"models": {model: {"calls","errors","retries","tokens_in","tokens_out","total_tokens",
#                     "latency_s","avg_latency_s"}}, "total": {...same keys...}}
client.usage.rich_table() -> rich.table.Table      # get_console().print(...)
client.usage.total_calls / .total_tokens / .reset() / .record(model, usage, latency_s, error=False)
```

## Mock backend

Active when `config.mock` is true. No network, no randomness: the same input gives the same output.
It runs through the same parsing and usage-tracking code as real responses.

| call | behaviour |
|---|---|
| `chat` | Plausible prose built from a hash of (model, last user message), echoing keywords from it. Respects `max_tokens` (`finish_reason="length"`). Reasoning models (`deepseek-r1`, `qwq`) emit a `<think>` block so stripping is exercised. |
| `chat_json` / `response_format={"type":"json_schema",...}` | Valid JSON for the schema: str/int/float/bool, lists, nested models, enums, `Literal`, optional (populated, not null), `dict[str, X]`, `date`/`datetime`/`UUID`, min/max bounds, `uniqueItems`, recursive models (depth-limited). Strings read like `"Summary about <keywords>"`; fields named `query`/`question`/`text`/`prompt`... echo the user message. |
| `tools=[...]` | First turn (no `tool` message after the last `user` message): a tool call to the first tool (or the one forced by `tool_choice`; `"none"` disables) with schema-valid arguments, `finish_reason="tool_calls"`. Once a `tool` message follows: plain text `"Based on the tool results, here is the answer: <tool output>"`. |
| `stream_chat` | The `chat` text, in word chunks. |
| `embed` | Hashed bag-of-words (+bigrams, light stemming, stop-words dropped) unit vectors, dim 1024; identical text gives identical vectors, shared words give higher cosine, unrelated texts ~0. `input_type` has no effect. |
| `rerank` | Query-token coverage + small passage-precision term (0..1.1). |
| `list_models` | Curated registry ids. |

`NIMLAB_MOCK_LATENCY=0.2` adds artificial latency per call (demos). Helpers live in
`nimlab.core.mock` (`generate_json(schema_dict, seed, user_text)`, `embed_text`, ...). Mock text is
generic filler: assert on structure and flow in tests, not on wording. Real-world numbers (token
counts, scores, latency) are estimates in mock mode.

## Model registry

```python
class Role(str, Enum):  CHAT REASONING CODER VISION EMBED RERANK REWARD SAFETY FAST   # values lowercase
ModelInfo(id: str, roles: frozenset[Role], context_window: int, notes: str, cost_tier: int)  # 1..5, frozen
```

```python
reg = default_registry()            # process-wide ModelRegistry
reg.pick(Role.CODER)                # -> "qwen/qwen2.5-coder-32b-instruct"  (str roles OK: "coder")
reg.by_role(Role.REASONING)         # -> list[ModelInfo]; current pick first, then by cost_tier, id
reg.get("meta/llama-3.3-70b-instruct")   # ModelInfo; KeyError if not curated
reg.find(id) -> ModelInfo | None;  id in reg;  reg.ids();  reg.all();  len(reg)
reg.register(ModelInfo(...), default_for="coder")    # add custom models at runtime
```

Defaults: CHAT `meta/llama-3.3-70b-instruct`, REASONING `deepseek-ai/deepseek-r1`, CODER
`qwen/qwen2.5-coder-32b-instruct`, VISION `meta/llama-3.2-90b-vision-instruct`, EMBED
`nvidia/llama-3.2-nv-embedqa-1b-v2`, RERANK `nvidia/llama-3.2-nv-rerankqa-1b-v2`, REWARD
`nvidia/llama-3.1-nemotron-70b-reward`, SAFETY `nvidia/llama-3.1-nemoguard-8b-content-safety`, FAST
`meta/llama-3.1-8b-instruct`. `nimlab models [--role R] [--live]` prints the catalog.

`pick(role)` precedence (re-read on every call): env `NIMLAB_MODEL_<ROLE>` (e.g.
`NIMLAB_MODEL_CODER=...`) > `[models]` table in `./nimlab.toml` (or the file named by
`NIMLAB_CONFIG`) > built-in default. Overrides may name models that are not in the catalog.

```toml
# nimlab.toml
[models]
coder = "qwen/qwen3-coder-480b-a35b-instruct"
chat  = "meta/llama-3.1-405b-instruct"
```

The catalog is curated from the public NIM catalog and is **not verified against a live account**
(no network when it was written). `nimlab models --live` marks which entries your endpoint serves
and lists live ids the catalog lacks; `ModelRegistry.missing_from(live_ids)` does the same in code.
Models that exist but are not enabled for a key return an error from the API, not from the registry.

## Tool agent loop — `nimlab.core.tools`

```python
@tool                                   # or @tool(name="...", description="...")
def get_weather(city: str, days: int = 3, unit: Literal["c", "f"] = "c") -> dict:
    """Get a forecast.

    Args:
        city: City name.
        days: Days ahead.
    """
```

`@tool` returns a `Tool` (still callable as the original function). Works for sync and async
functions. The JSON schema comes from type hints (any pydantic-compatible type, nested models,
`Annotated[..., Field(description=...)]`), defaults (parameters without one are required) and the
docstring (first paragraph = description; Google `Args:` or Sphinx `:param x:` = parameter docs).

* `tool.schema()` -> OpenAI tool dict `{"type": "function", "function": {name, description, parameters}}`
  (also `.name`, `.description`, `.parameters`).
* `await tool.invoke(arguments: dict)` validates (raises `pydantic.ValidationError`) and runs;
  sync functions run in a worker thread so they never block the loop.

```python
async run_agent(client: NIMClient, model: str, messages: list[dict], tools: list[Tool],
                max_steps: int = 8, on_event: Callable[[AgentEvent], Awaitable | None] | None = None,
                *, temperature: float = 0.2, max_tokens: int = 1024, tool_choice=None,
                extra: dict | None = None) -> AgentRun
```

Calls the model, runs every requested tool call (concurrently within a turn), appends
`{"role": "tool", "tool_call_id", "name", "content"}` messages and repeats until the model answers
without tool calls. Tool failures (unknown tool, invalid arguments, exceptions) are fed back to the
model as `"Error: ..."` tool messages rather than raised. Results are serialised: `str` as is,
pydantic models/dicts/lists as JSON, `None` as `"null"`. Input `messages` is not mutated.

```python
AgentRun(final: str, steps: list[AgentStep], messages: list[dict], truncated: bool)  # .tool_calls
AgentStep(index: int, result: ChatResult, tool_results: list[ToolResult])
ToolResult(call_id, name, arguments, output: str, is_error: bool, duration_s: float)
AgentEvent(kind: "model" | "tool_call" | "tool_result" | "final", step: int,
           result: ChatResult | None, tool_call: ToolCall | None, tool_result: ToolResult | None)
```

`on_event` may be sync or async; exceptions it raises are logged and ignored. If `max_steps` model
calls are used up while the model still wants tools, `truncated=True`, `final` is that last
(possibly empty) content and the pending calls are not executed.

## Utilities

`nimlab.core.text` (the four main ones are re-exported from `nimlab.core`):

* `strip_think(text) -> (content, reasoning | None)` — handles several blocks, an unterminated
  `<think>` (all reasoning) and a lone `</think>`. Text without tags comes back unchanged.
* `extract_json(text) -> Any` — whole text, then fenced blocks, then the first `{...}`/`[...]` in prose;
  strips `<think>`; tolerates trailing commas. Raises `JSONExtractionError` (a `ValueError`).
* `chunk_text(text, max_chars=1200, overlap=150) -> list[str]` — paragraph, then sentence, then word
  aware; every chunk <= `max_chars`; later chunks start with up to `overlap` chars of the previous one
  (word-aligned). Raises `ValueError` unless `0 <= overlap < max_chars`.
* `cosine(a, b) -> float` — lists or numpy arrays; 0.0 for a zero vector; `ValueError` on size mismatch.
* `ThinkStreamFilter` — `feed(chunk) -> str` / `flush() -> str`, for filtering your own streams.

`nimlab.core.logging`: `get_console() -> rich.console.Console` (shared, stdout; 140 columns when
piped), `get_logger(name="nimlab")`, `configure_logging(level=None)` (rich stderr handler; level from
`NIMLAB_LOG_LEVEL`), `set_console(console)` (tests).

## Adding a CLI to an app

`nimlab/cli.py` imports `nimlab.<name>.cli` for `council`, `atlas`, `forge`, `arena`. If that module
has `app = typer.Typer(...)` it is mounted as `nimlab <name> ...`. A missing `nimlab.<name>` /
`nimlab.<name>.cli` module is skipped silently; any other import error (including a missing
dependency or a typo inside your module) is re-raised on purpose, so `nimlab` will fail loudly if your
CLI module is broken. Do **not** import `nimlab.cli` from your own CLI module (circular import).
`nimlab --mock <app> ...` sets `NIMLAB_MOCK=1` before your command runs. Each app creates its own
`NIMClient` inside the command (`asyncio.run(...)`), and should end with
`get_console().print(client.usage.rich_table())` if it wants to report cost.

## Testing conventions

`tests/conftest.py` (applies to every test directory) provides:

* an autouse fixture that clears `NVIDIA_*`/`NIM_*`/`NIMLAB_*` env vars, disables `.env` loading, runs
  each test in an empty temp cwd, and installs a wide color-free shared console, so tests are hermetic;
* an autouse `sleeps` fixture: backoff sleeping is faked (records delays in the returned list);
* `mock_config`, async `mock_client` (offline `NIMClient`), and `real_config`
  (`NIMConfig(api_key="nvapi-test", mock=False, max_retries=3)`) for `respx` tests.

pytest runs with `asyncio_mode=auto` (write `async def test_...` directly) and
`--import-mode=importlib` (same test file names in different folders are fine). Real-HTTP behaviour is
tested with `respx` (`respx_mock` fixture); see `tests/core/test_client_http.py` for patterns.
Use `monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")` to leave mock mode in a test.

## Gotchas

* Mock mode is the default without a key: apps must work end to end offline. Don't parse mock prose;
  use `chat_json` / tools for anything structured.
* Do not hard-code embedding dimensions, model ids or context sizes; use the registry (and
  `NIMLAB_MODEL_<ROLE>` overrides) and derive dimensions from returned vectors.
* `Role` values are lowercase strings; the env var uses the upper-case member name.
* `chat` returns `content=""` when the model only calls tools (check `.tool_calls` first).
* `max_tokens=1024` is the default; raise it for reasoning models (their `<think>` output counts) or
  `content` may be empty with `finish_reason="length"`.
* Never print or log `config.api_key`.
