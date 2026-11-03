"""The run engine: topological scheduling with maximum parallelism.

Every node becomes an asyncio task that waits for the nodes it depends on, so independent
branches run at the same time. Branches work by *absence*: a node that does not emit on an output
jack (``branch`` emits on exactly one of ``then``/``else``) leaves everything downstream of that
jack ``skipped`` (event ``node.skipped``); ``merge``/``join`` run as soon as one input is live.

Events (plain dicts with ``type``, ``t`` = ms since start and, for node events, ``node``):

========================  =================================================================
``run.plan``              ``nodes`` (execution order), ``levels``, ``inputs``
``node.start``            ``node``, ``type``
``node.token``            ``node``, ``text`` (streamed model text; coalesced)
``node.progress``         ``node``, ``done``, ``total`` (``map``)
``node.retry``            ``node``, ``attempt``, ``error``
``node.end``              ``node``, ``preview``, ``outputs`` (per jack preview), ``emitted``
                          (jacks that carried a value), ``latency_ms``, ``tokens``, ``cached``
``node.skipped``          ``node``, ``reason``
``node.error``            ``node``, ``error``, ``attempts``
``run.done``              ``ok``, ``outputs`` (previews), ``tokens``, ``elapsed_ms``,
                          ``cache_hits``, ``error``, ``failed_node``
========================  =================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import nodes as _nodes  # noqa: F401  (registers the built-in node types)
from .cache import NodeCache, cache_key
from .context import NodeContext, NodeResult, RunEnv
from .errors import InputError, NodeFailed, NodeTimeout, PipelineError
from .nodes import http_allowlist
from .registry import NodeType, apply_defaults, get_node_type, ports_for
from .schema import Pipeline, topo_levels
from .datatypes import coerce, preview

__all__ = ["NodeReport", "RunResult", "run_pipeline"]

_log = logging.getLogger("nimlab.loom")
RETRY_BASE_DELAY_S = 0.5


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass
class NodeReport:
    """What happened to one node."""

    id: str
    type: str
    status: str = "pending"  # pending | done | cached | skipped | error
    outputs: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    tokens: int = 0
    attempts: int = 0
    model: str | None = None
    error: str | None = None
    reason: str | None = None

    @property
    def preview(self) -> str:
        if not self.outputs:
            return ""
        return preview(next(iter(self.outputs.values())))


@dataclass
class RunResult:
    """Outcome of :func:`run_pipeline`. A failed node gives ``ok=False``, not an exception."""

    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, NodeReport] = field(default_factory=dict)
    error: str | None = None
    failed_node: str | None = None
    tokens: int = 0
    elapsed_s: float = 0.0
    cache_hits: int = 0

    def raise_for_error(self) -> RunResult:
        if not self.ok:
            raise NodeFailed(self.failed_node or "?", self.error or "failed")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outputs": self.outputs,
            "error": self.error,
            "failed_node": self.failed_node,
            "tokens": self.tokens,
            "elapsed_s": round(self.elapsed_s, 3),
            "cache_hits": self.cache_hits,
            "nodes": {
                nid: {
                    "type": r.type,
                    "status": r.status,
                    "latency_ms": r.latency_ms,
                    "tokens": r.tokens,
                    "attempts": r.attempts,
                    "model": r.model,
                    "error": r.error,
                    "reason": r.reason,
                    "preview": r.preview,
                }
                for nid, r in self.nodes.items()
            },
        }


def _input_defs(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
    """``{input name: {node, type, default, required}}`` for the input nodes."""
    out: dict[str, dict[str, Any]] = {}
    for node in pipeline.nodes:
        if node.type != "input":
            continue
        params = node.params
        name = str(params.get("name") or node.id)
        _, outs = ports_for(get_node_type("input"), params)
        out[name] = {
            "node": node.id,
            "type": outs[0].type,
            "default": params.get("default"),
            "required": "default" not in params,
            "description": params.get("description", ""),
        }
    return out


def prepare_inputs(pipeline: Pipeline, supplied: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and coerce run inputs by name; fill defaults. Raises :class:`InputError`."""
    defs = _input_defs(pipeline)
    given = dict(supplied or {})
    unknown = sorted(set(given) - set(defs))
    if unknown:
        raise InputError(
            f"unknown input(s) {', '.join(unknown)}; this pipeline takes: "
            f"{', '.join(defs) or '(no inputs)'}"
        )
    values: dict[str, Any] = {}
    for name, d in defs.items():
        if name in given:
            raw = given[name]
        elif not d["required"]:
            raw = d["default"]
            if raw is None and d["type"] == "text":
                raw = ""
        else:
            raise InputError(f"missing required input {name!r}")
        try:
            values[name] = coerce(raw, d["type"])
        except InputError as exc:
            raise InputError(f"input {name!r}: {exc}") from exc
    return values


class _Engine:
    def __init__(
        self,
        pipeline: Pipeline,
        env: RunEnv,
        cache: NodeCache | None,
        max_parallel: int,
    ) -> None:
        self.p = pipeline
        self.env = env
        self.cache = cache
        self.sem = asyncio.Semaphore(max(1, max_parallel))
        self.reports: dict[str, NodeReport] = {n.id: NodeReport(n.id, n.type) for n in pipeline.nodes}
        self.done: dict[str, asyncio.Event] = {n.id: asyncio.Event() for n in pipeline.nodes}
        self.failure: tuple[str, str] | None = None
        self.tasks: list[asyncio.Task[None]] = []
        self.backend = "mock" if getattr(env.client, "mock", False) else "live"

    # ------------------------------------------------------------------ inputs
    def _collect(self, node_id: str, ntype: NodeType, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Resolve input jacks. Returns ``(inputs, skip_reason)``."""
        node = self.p.node(node_id)
        ins, _ = ports_for(ntype, params)
        inputs: dict[str, Any] = {}
        dead_required: str | None = None
        any_live = False
        any_wired = False
        for port in ins:
            edges = [e for e in self.p.incoming(node_id) if e.dst_port == port.name]
            values: list[Any] = []
            dead = 0
            for e in edges:
                src = self.reports[e.src_node]
                if src.status in ("skipped",) or e.src_port not in src.outputs:
                    dead += 1
                else:
                    values.append(coerce(src.outputs[e.src_port], port.type))
            any_wired = any_wired or bool(edges)
            if port.multi:
                inputs[port.name] = values
                any_live = any_live or bool(values)
                if edges and not values and port.required:
                    dead_required = dead_required or f"every input on {port.name!r} was skipped"
                continue
            if values:
                inputs[port.name] = values[0]
                any_live = True
            elif edges:
                if port.required:
                    dead_required = dead_required or f"input {port.name!r} was not produced upstream"
                inputs[port.name] = None
            elif port.name in node.defaults:
                inputs[port.name] = coerce(node.defaults[port.name], port.type)
                any_live = True
            else:
                inputs[port.name] = None
                if port.required:
                    raise PipelineError(f"node {node_id!r}: input {port.name!r} is not connected")
        if ntype.any_input:
            if any_wired and not any_live:
                return inputs, "no live input"
            return inputs, None
        return inputs, dead_required

    # ------------------------------------------------------------------ one node
    async def _node(self, node_id: str) -> None:
        report = self.reports[node_id]
        node = self.p.node(node_id)
        try:
            for dep in {e.src_node for e in self.p.incoming(node_id)}:
                await self.done[dep].wait()
            if self.failure:
                return
            ntype = get_node_type(node.type)
            params = apply_defaults(ntype, node.params)
            inputs, skip = self._collect(node_id, ntype, params)
            if skip:
                report.status, report.reason = "skipped", skip
                self.env.event("node.skipped", node=node_id, reason=skip)
                return
            async with self.sem:
                if self.failure:
                    return
                await self._execute(node_id, ntype, params, inputs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(node_id, exc)
        finally:
            self.done[node_id].set()

    def _fail(self, node_id: str, exc: BaseException) -> None:
        report = self.reports[node_id]
        message = str(exc) or type(exc).__name__
        report.status, report.error = "error", message
        if self.failure is None:
            self.failure = (node_id, message)
            self.env.event("node.error", node=node_id, error=message, attempts=report.attempts)
            for task in self.tasks:
                if not task.done() and task is not asyncio.current_task():
                    task.cancel()

    async def _execute(
        self, node_id: str, ntype: NodeType, params: dict[str, Any], inputs: dict[str, Any]
    ) -> None:
        node = self.p.node(node_id)
        report = self.reports[node_id]
        self.env.event("node.start", node=node_id, type=node.type)
        t0 = time.monotonic()
        use_cache = self.cache is not None and (
            node.cache if node.cache is not None else ntype.cacheable
        )
        key = ""
        if use_cache and self.cache is not None:
            key = cache_key(
                node.type, params, {**node.defaults, **inputs}, self.backend
            )
            hit = self.cache.get(key)
            if hit is not None:
                self._finish(report, hit.outputs, hit.tokens, hit.model, t0, cached=True)
                return
        attempts = node.retries + 1
        result: NodeResult | None = None
        last: BaseException | None = None
        ctx = NodeContext(node, params, inputs, self.env)
        for attempt in range(1, attempts + 1):
            report.attempts = attempt
            try:
                coro = ntype.run(ctx)
                result = await (asyncio.wait_for(coro, node.timeout) if node.timeout else coro)
                break
            except asyncio.TimeoutError:
                last = NodeTimeout(f"timed out after {node.timeout:g}s")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = exc
            ctx.flush_tokens()
            if attempt < attempts:
                self.env.event("node.retry", node=node_id, attempt=attempt, error=str(last))
                await _sleep(RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
        if result is None:
            assert last is not None
            raise last
        if use_cache and self.cache is not None:
            self.cache.put(key, result.outputs, result.tokens, result.model)
        self._finish(report, result.outputs, result.tokens, result.model, t0, cached=False)

    def _finish(
        self,
        report: NodeReport,
        outputs: dict[str, Any],
        tokens: int,
        model: str | None,
        t0: float,
        *,
        cached: bool,
    ) -> None:
        report.status = "cached" if cached else "done"
        report.outputs = outputs
        report.tokens = 0 if cached else tokens
        report.model = model
        report.latency_ms = int((time.monotonic() - t0) * 1000)
        previews = {k: preview(v) for k, v in outputs.items()}
        self.env.event(
            "node.end",
            node=report.id,
            preview=next(iter(previews.values()), ""),
            outputs=previews,
            emitted=list(outputs),
            latency_ms=report.latency_ms,
            tokens=report.tokens,
            cached=cached,
            model=model,
        )

    # ------------------------------------------------------------------ whole run
    async def run(self) -> RunResult:
        t0 = time.monotonic()
        self.tasks = [asyncio.ensure_future(self._node(n.id)) for n in self.p.nodes]
        await asyncio.gather(*self.tasks, return_exceptions=True)
        elapsed = time.monotonic() - t0
        outputs: dict[str, Any] = {}
        if not self.failure:
            for n in self.p.nodes:
                rep = self.reports[n.id]
                if n.type == "output" and rep.status in ("done", "cached"):
                    outputs[str(n.params.get("name") or n.id)] = rep.outputs.get("value")
        tokens = sum(r.tokens for r in self.reports.values())
        hits = sum(1 for r in self.reports.values() if r.status == "cached")
        failed, error = self.failure or (None, None)
        result = RunResult(
            ok=self.failure is None,
            outputs=outputs,
            nodes=self.reports,
            error=error,
            failed_node=failed,
            tokens=tokens,
            elapsed_s=elapsed,
            cache_hits=hits,
        )
        self.env.event(
            "run.done",
            ok=result.ok,
            outputs={k: preview(v, 400) for k, v in outputs.items()},
            tokens=tokens,
            elapsed_ms=int(elapsed * 1000),
            cache_hits=hits,
            error=error,
            failed_node=failed,
        )
        return result


def _coerce_pipeline(pipeline: Pipeline | Mapping[str, Any] | str | Path) -> Pipeline:
    if isinstance(pipeline, Pipeline):
        return pipeline
    if isinstance(pipeline, Mapping):
        return Pipeline.from_data(dict(pipeline))
    from .library import load_pipeline

    return load_pipeline(pipeline)


async def run_pipeline(
    pipeline: Pipeline | Mapping[str, Any] | str | Path,
    inputs: Mapping[str, Any] | None = None,
    *,
    client: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cache: bool | str | Path = False,
    stream: bool | None = None,
    max_parallel: int = 16,
    http_allow: Sequence[str] = (),
    http_client: Any | None = None,
    atlas_db: str | Path | None = None,
    validate: bool = True,
    mock: bool | None = None,
) -> RunResult:
    """Run a pipeline and return a :class:`RunResult`.

    ``pipeline`` is a :class:`Pipeline`, its dict form, a file path or the name of a bundled
    pipeline. ``inputs`` maps input names to values (strings are coerced to the input's type).
    ``client`` is a ``NIMClient`` (or anything with ``chat``/``chat_json``/``embed``/``rerank``);
    without one a client is created from the environment (``mock=True`` forces the offline
    backend). ``cache`` is ``True`` (default directory), ``False`` or a directory.
    ``stream`` defaults to on when ``on_event`` is given. Static problems (cycles, bad inputs,
    lint errors) raise :class:`~nimlab.loom.errors.LoomError`; a node that fails after its retries
    gives ``ok=False`` with ``error`` and ``failed_node``.
    """
    pipe = _coerce_pipeline(pipeline)
    topo_levels(pipe)  # raises CycleError
    if validate:
        from .linter import lint

        errors = [i for i in lint(pipe).issues if i.severity == "error"]
        if errors:
            raise PipelineError(
                "pipeline has errors:\n" + "\n".join(f"  - {i.format()}" for i in errors)
            )
    values = prepare_inputs(pipe, inputs)
    levels = topo_levels(pipe)

    owned = client is None
    if client is None:
        from nimlab.core import NIMClient, NIMConfig

        client = NIMClient(NIMConfig.from_env(**({"mock": True} if mock else {})))
    node_cache: NodeCache | None = None
    if cache:
        node_cache = NodeCache(None if cache is True else cache)
    env = RunEnv(
        client=client,
        input_values=values,
        emit=_safe(on_event),
        stream=bool(on_event) if stream is None else stream,
        http_allow=list(http_allow),
        http_client=http_client,
        atlas_db=atlas_db,
    )
    env.event(
        "run.plan",
        nodes=[nid for level in levels for nid in level],
        levels=levels,
        inputs=sorted(values),
        allow_http=bool(http_allowlist(http_allow)),
    )
    try:
        return await _Engine(pipe, env, node_cache, max_parallel).run()
    finally:
        if owned:
            await client.aclose()


def _safe(fn: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None] | None:
    if fn is None:
        return None

    def emit(event: dict[str, Any]) -> None:
        try:
            fn(event)
        except Exception:  # a broken listener must never break a run
            _log.debug("event listener failed", exc_info=True)

    return emit

