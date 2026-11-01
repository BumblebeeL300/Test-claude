"""``nimlab atlas ...`` command line interface."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nimlab.core import NIMConfig, NIMError, get_console

from .config import DEFAULT_COLLECTION, DEFAULT_DB_PATH, AtlasConfig
from .engine import Atlas
from .evaluation import EvalReport, load_qa, run_eval
from .models import Answer, AtlasError, IngestReport

app = typer.Typer(
    help="Atlas: agentic, citation-verified RAG over your documents.",
    no_args_is_help=True,
    add_completion=False,
)

T = TypeVar("T")


class ModeChoice(str, Enum):
    simple = "simple"
    agentic = "agentic"


DbOption = Annotated[
    Path,
    typer.Option("--db", envvar="NIMLAB_ATLAS_DB", help="SQLite database path.", show_default=True),
]
CollectionOption = Annotated[
    str, typer.Option("--collection", "-c", help="Collection (namespace) to use.")
]
MockOption = Annotated[
    bool,
    typer.Option(
        "--mock", help="Use the offline mock backend (also automatic without an API key)."
    ),
]


def _run(coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _fail(message: str, code: int = 1) -> typer.Exit:
    get_console().print(f"[bold red]error:[/bold red] {message}", highlight=False)
    return typer.Exit(code)


async def _with_atlas(
    db: Path,
    mock: bool,
    fn: Callable[[Atlas], Awaitable[T]],
    config: AtlasConfig | None = None,
    *,
    usage: bool = False,
) -> T:
    nim_config = NIMConfig.from_env(**({"mock": True} if mock else {}))
    async with Atlas.open(db, config=config, nim_config=nim_config) as atlas:
        result = await fn(atlas)
        if usage:
            get_console().print(atlas.client.usage.rich_table())
        return result


def _mock_note(atlas: Atlas) -> None:
    if atlas.mock:
        get_console().print(
            "[yellow]mock mode:[/yellow] model output is synthetic; retrieval is real.",
            highlight=False,
        )


# --------------------------------------------------------------------------- ingest


def _render_ingest(report: IngestReport) -> None:
    console = get_console()
    table = Table(title=f"Ingest into '{report.collection}'", title_justify="left")
    for col in ("File", "Status", "Chunks", "Dups", "Embedded", "Note"):
        table.add_column(col, justify="right" if col in {"Chunks", "Dups", "Embedded"} else "left")
    style = {
        "added": "green",
        "updated": "cyan",
        "unchanged": "dim",
        "skipped": "yellow",
        "error": "red",
    }
    for f in report.files:
        table.add_row(
            f.name,
            f"[{style[f.status]}]{f.status}[/{style[f.status]}]",
            str(f.chunks),
            str(f.duplicates),
            str(f.embedded),
            f.message,
        )
    console.print(table)
    removed = f", {len(report.removed)} removed" if report.removed else ""
    console.print(
        f"{report.added} added, {report.updated} updated, {report.unchanged} unchanged, "
        f"{report.errors} failed{removed}; {report.chunks} chunks in {report.elapsed_s:.2f}s",
        highlight=False,
    )


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory (.md .txt .html .pdf).")],
    collection: CollectionOption = DEFAULT_COLLECTION,
    db: DbOption = DEFAULT_DB_PATH,
    mock: MockOption = False,
    chunk_size: Annotated[int, typer.Option(help="Maximum characters per chunk.")] = 1200,
    overlap: Annotated[int, typer.Option(help="Characters shared between chunks.")] = 150,
    force: Annotated[bool, typer.Option("--force", help="Re-ingest unchanged files too.")] = False,
    prune: Annotated[
        bool, typer.Option("--prune", help="Remove documents whose files no longer exist.")
    ] = False,
    recursive: Annotated[bool, typer.Option(help="Descend into sub-directories.")] = True,
) -> None:
    """Chunk, embed and index a file or directory."""
    try:
        config = AtlasConfig(
            chunk_max_chars=chunk_size,
            chunk_overlap=overlap,
            chunk_min_chars=min(200, chunk_size // 2),
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    async def go(atlas: Atlas) -> IngestReport:
        _mock_note(atlas)
        return await atlas.ingest(path, collection, force=force, prune=prune, recursive=recursive)

    try:
        report = _run(_with_atlas(db, mock, go, config))
    except (AtlasError, NIMError) as exc:
        raise _fail(str(exc)) from exc
    _render_ingest(report)
    if report.errors:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- ask


def _render_answer(
    answer: Answer, *, show_sources: bool, show_trace: bool, show_claims: bool = True
) -> None:
    console = get_console()
    if show_trace and answer.trace:
        for step in answer.trace:
            console.print(f"[dim]- {step.kind}:[/dim] {step.message}", highlight=False)
    text = Text(answer.text)
    text.highlight_regex(r"\[\d+\]", "bold cyan")
    title = "Answer (blocked)" if answer.blocked else f"Answer ({answer.mode})"
    console.print(
        Panel(
            text, title=title, title_align="left", border_style="red" if answer.blocked else "cyan"
        )
    )
    if show_sources and answer.sources:
        cited = {c.n for c in answer.citations}
        table = Table(title="Sources", title_justify="left")
        for col in ("#", "Document", "Section", "Page", "Score", "Cited"):
            table.add_column(col)
        for s in answer.sources:
            table.add_row(
                str(s.n),
                s.source,
                " > ".join(s.heading_path) or "-",
                "-" if s.page is None else str(s.page),
                f"{s.score:.3f}",
                "yes" if s.n in cited else "",
            )
        console.print(table)
    v = answer.verification
    if v is not None:
        if v.groundedness is None:
            console.print("Verification: no checkable claims in the answer.", highlight=False)
        else:
            color = (
                "green" if v.groundedness >= 0.8 else ("yellow" if v.groundedness >= 0.5 else "red")
            )
            console.print(
                f"Groundedness: [{color}]{v.groundedness:.0%}[/{color}]  "
                f"({v.supported} supported, {v.partial} partial, {v.unsupported} unsupported; "
                f"method={v.method}{', revised' if v.revised else ''})",
                highlight=False,
            )
        if show_claims and v.claims:
            table = Table(title="Claims", title_justify="left")
            for col in ("#", "Verdict", "Claim", "Cites"):
                table.add_column(col)
            colors = {"supported": "green", "partial": "yellow", "unsupported": "red"}
            for c in v.claims:
                table.add_row(
                    str(c.id),
                    f"[{colors[c.verdict]}]{c.verdict}[/{colors[c.verdict]}]",
                    Text(c.text),
                    ",".join(map(str, c.cited)) or "-",
                )
            console.print(table)
    for w in answer.warnings:
        console.print(f"[yellow]note:[/yellow] {w}", highlight=False)
    console.print(
        f"[dim]{answer.latency_s:.2f}s, {answer.usage.get('calls', 0)} model calls[/dim]",
        highlight=False,
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    collection: CollectionOption = DEFAULT_COLLECTION,
    mode: Annotated[
        ModeChoice, typer.Option("--mode", "-m", help="simple or agentic.")
    ] = ModeChoice.simple,
    verify: Annotated[
        bool, typer.Option("--verify", help="Check every claim against its sources.")
    ] = False,
    revise: Annotated[
        bool, typer.Option("--revise", help="Auto-revise unsupported claims (implies --verify).")
    ] = False,
    guard: Annotated[
        bool, typer.Option("--guard", help="Screen the question and answer with NemoGuard.")
    ] = False,
    hyde: Annotated[
        bool, typer.Option("--hyde", help="Embed a hypothetical answer (HyDE).")
    ] = False,
    k: Annotated[
        int, typer.Option("--k", "-k", min=1, max=50, help="Passages given to the model.")
    ] = 6,
    rerank: Annotated[bool, typer.Option(help="Use the reranker.")] = True,
    mmr: Annotated[bool, typer.Option(help="Diversify results with MMR.")] = True,
    sources: Annotated[
        bool, typer.Option("--sources/--no-sources", help="Show the sources table.")
    ] = True,
    trace: Annotated[bool, typer.Option("--trace", help="Show the pipeline steps.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full answer as JSON.")] = False,
    usage: Annotated[bool, typer.Option("--usage", help="Print the model usage table.")] = False,
    db: DbOption = DEFAULT_DB_PATH,
    mock: MockOption = False,
) -> None:
    """Ask a question and get a cited answer."""
    config = AtlasConfig(use_rerank=rerank, use_mmr=mmr)

    async def go(atlas: Atlas) -> Answer:
        return await atlas.ask(
            question,
            collection,
            mode=mode.value,  # type: ignore[arg-type]
            k=k,
            verify=verify,
            revise=revise,
            guard=guard,
            hyde=hyde or None,
        )

    try:
        answer = _run(_with_atlas(db, mock, go, config, usage=usage and not as_json))
    except (AtlasError, NIMError) as exc:
        raise _fail(str(exc)) from exc
    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    _render_answer(answer, show_sources=sources, show_trace=trace)


# --------------------------------------------------------------------------- serve / stats


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8765,
    db: DbOption = DEFAULT_DB_PATH,
    mock: MockOption = False,
    reload: Annotated[bool, typer.Option(hidden=True)] = False,
) -> None:
    """Start the web UI and HTTP API."""
    import uvicorn

    from .server import create_app

    nim_config = NIMConfig.from_env(**({"mock": True} if mock else {}))
    web = create_app(db, nim_config=nim_config)
    get_console().print(
        f"Atlas on [bold]http://{host}:{port}[/bold]  "
        f"(db: {db}, {'mock' if nim_config.mock else 'live'} mode)",
        highlight=False,
    )
    uvicorn.run(web, host=host, port=port, log_level="info", reload=reload)


@app.command()
def stats(
    db: DbOption = DEFAULT_DB_PATH,
    docs: Annotated[bool, typer.Option("--docs", help="List every document.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Show collections, documents and chunk counts."""
    from .store import AtlasStore

    path = Path(db)
    if not path.exists():
        raise _fail(f"no database at {db}; run 'nimlab atlas ingest' first")
    store = AtlasStore(path)
    try:
        data = store.stats()
        documents = store.list_documents()
    finally:
        store.close()
    if as_json:
        typer.echo(json.dumps({**data, "document_list": [d.__dict__ for d in documents]}, indent=2))
        return
    console = get_console()
    table = Table(title=f"Atlas database: {db}", title_justify="left")
    for col in ("Collection", "Documents", "Chunks", "Dim", "Embedding model"):
        table.add_column(col, justify="right" if col in {"Documents", "Chunks", "Dim"} else "left")
    for c in data["collections"]:
        table.add_row(
            c["name"],
            str(c["documents"]),
            str(c["chunks"]),
            str(c["dim"] or "-"),
            c["embed_model"] or "-",
        )
    console.print(table)
    kib = data["db_bytes"] / 1024
    console.print(
        f"{data['documents']} documents, {data['chunks']} chunks, {kib:.1f} KiB", highlight=False
    )
    if docs:
        dt = Table(title="Documents", title_justify="left")
        for col in ("Collection", "Name", "Type", "Chunks"):
            dt.add_column(col)
        for d in documents:
            dt.add_row(d.collection, d.name, d.media_type, str(d.n_chunks))
        console.print(dt)


@app.command()
def delete(
    collection: Annotated[str, typer.Argument(help="Collection to delete.")],
    db: DbOption = DEFAULT_DB_PATH,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete a collection and all of its documents."""
    from .store import AtlasStore

    if not Path(db).exists():
        raise _fail(f"no database at {db}")
    if not yes:
        typer.confirm(f"Delete collection '{collection}' and everything in it?", abort=True)
    store = AtlasStore(db)
    try:
        removed = store.delete_collection(collection)
    finally:
        store.close()
    if not removed:
        raise _fail(f"unknown collection '{collection}'")
    get_console().print(f"Deleted collection '{collection}'.", highlight=False)


@app.command()
def reindex(
    collection: CollectionOption = DEFAULT_COLLECTION,
    db: DbOption = DEFAULT_DB_PATH,
    mock: MockOption = False,
) -> None:
    """Re-embed every chunk of a collection with the current embedding model."""

    async def go(atlas: Atlas) -> int:
        return await atlas.reindex(collection)

    try:
        n = _run(_with_atlas(db, mock, go))
    except (AtlasError, NIMError) as exc:
        raise _fail(str(exc)) from exc
    get_console().print(f"Re-embedded {n} chunks in '{collection}'.", highlight=False)


# --------------------------------------------------------------------------- eval


def _render_eval(report: EvalReport) -> None:
    console = get_console()
    table = Table(title=f"Evaluation: {report.name or 'QA set'}", title_justify="left")
    for col in ("Question", "Rank", "Recall@k", "Keywords", "Grounded", "Sec"):
        table.add_column(col, overflow="fold")
    top = report.ks[-1]
    for it in report.items:
        question = it.question if len(it.question) <= 60 else it.question[:57] + "..."
        if it.error:
            table.add_row(question, "[red]error[/red]", it.error, "", "", f"{it.latency_s:.2f}")
            continue
        rank = f"{it.first_rank}" if it.first_rank else "[red]miss[/red]"
        table.add_row(
            question,
            rank,
            f"{it.recall.get(top, 0.0):.0%}" if it.expected else "-",
            "-" if it.keyword_coverage is None else f"{it.keyword_coverage:.0%}",
            "-" if it.groundedness is None else f"{it.groundedness:.0%}",
            f"{it.latency_s:.2f}",
        )
    console.print(table)
    metrics = Table(title="Summary", title_justify="left")
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")
    for k in report.ks:
        metrics.add_row(f"recall@{k}", f"{report.recall_at_k[k]:.3f}")
    metrics.add_row("MRR", f"{report.mrr:.3f}")
    metrics.add_row("hit rate", f"{report.hit_rate:.3f}")
    if report.keyword_coverage is not None:
        metrics.add_row("keyword coverage", f"{report.keyword_coverage:.3f}")
    if report.mean_groundedness is not None:
        metrics.add_row("mean groundedness", f"{report.mean_groundedness:.3f}")
    metrics.add_row("questions / time", f"{report.n_questions} / {report.elapsed_s:.1f}s")
    console.print(metrics)
    if report.mock and report.answered:
        console.print(
            "[yellow]mock mode:[/yellow] retrieval metrics are real; groundedness reflects "
            "synthetic placeholder answers.",
            highlight=False,
        )


@app.command(name="eval")
def eval_cmd(
    file: Annotated[Path, typer.Argument(help="QA JSON file (see examples/atlas/qa.json).")],
    collection: CollectionOption = DEFAULT_COLLECTION,
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=50, help="Passages per question.")] = 6,
    mode: Annotated[
        ModeChoice, typer.Option("--mode", "-m", help="simple or agentic.")
    ] = ModeChoice.simple,
    answer: Annotated[
        bool,
        typer.Option(
            "--answer/--no-answer", help="Generate and verify answers (else retrieval only)."
        ),
    ] = True,
    rerank: Annotated[bool, typer.Option(help="Use the reranker.")] = True,
    mmr: Annotated[bool, typer.Option(help="Diversify results with MMR.")] = True,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the full report as JSON.")
    ] = None,
    min_recall: Annotated[
        float | None, typer.Option("--min-recall", help="Exit with status 1 if recall@k is lower.")
    ] = None,
    db: DbOption = DEFAULT_DB_PATH,
    mock: MockOption = False,
) -> None:
    """Evaluate retrieval recall@k, MRR and answer groundedness on a QA set."""
    try:
        name, items = load_qa(file)
    except AtlasError as exc:
        raise _fail(str(exc)) from exc
    config = AtlasConfig(use_rerank=rerank, use_mmr=mmr)

    async def go(atlas: Atlas) -> EvalReport:
        return await run_eval(
            atlas,
            items,
            collection=collection,
            k=k,
            mode=mode.value,  # type: ignore[arg-type]
            answer=answer,
            verify=answer,
            name=name,
        )

    try:
        report = _run(_with_atlas(db, mock, go, config))
    except (AtlasError, NIMError) as exc:
        raise _fail(str(exc)) from exc
    _render_eval(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        get_console().print(f"Report written to {out}", highlight=False)
    if min_recall is not None and report.recall_at_k[report.ks[-1]] < min_recall:
        raise _fail(
            f"recall@{report.ks[-1]} {report.recall_at_k[report.ks[-1]]:.3f} < {min_recall}"
        )
