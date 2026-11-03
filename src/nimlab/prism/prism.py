"""The :class:`Prism` engine: load images, pick a mode, run it, batch it."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nimlab.core import NIMClient, Role, default_registry

from .errors import PrismError
from .images import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_SIDE,
    PrismImage,
    list_images,
    load_image,
)
from .modes import Mode, get_mode
from .session import Observer, Session

__all__ = ["BatchItem", "ImageSource", "Prism", "PrismResult", "write_jsonl"]

ImageSource = str | Path | bytes | PrismImage


class PrismResult(BaseModel):
    """What a mode produced. ``output`` is mode-specific; ``text`` is its plain-text rendering."""

    mode: str
    model: str
    simulated: bool
    images: list[dict[str, Any]]
    output: dict[str, Any]
    text: str
    elapsed_s: float
    calls: int
    notes: list[str] = Field(default_factory=list)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


class BatchItem(BaseModel):
    """One image of a batch: either a result or an error message."""

    index: int
    name: str
    ok: bool
    result: PrismResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def write_jsonl(items: Sequence[BatchItem], path: str | Path) -> None:
    """One JSON object per line: ``{index, name, ok, result|error}``."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def _expand(sources: str | Path | Sequence[ImageSource], recursive: bool) -> list[ImageSource]:
    if isinstance(sources, (str, Path)):
        if Path(sources).is_dir():
            return list(list_images(sources, recursive=recursive))
        return [sources]
    return list(sources)


class Prism:
    """Vision workbench on NIM vision-language models.

    ::

        async with Prism() as prism:
            r = await prism.describe("photo.jpg")
            r = await prism.extract("receipt.png", schema="receipt")
            r = await prism.ui2code("shot.png", refine=2)

    ``client`` defaults to ``NIMClient()`` (mock mode when there is no API key). In mock mode the
    model is never called: Prism returns deterministic, mode-appropriate output and flags the
    result ``simulated``. ``simulate=True/False`` overrides that detection.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str | None = None,
        simulate: bool | None = None,
        max_side: int = DEFAULT_MAX_SIDE,
        max_bytes: int = DEFAULT_MAX_BYTES,
        sim_delay: float = 0.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self.model = model
        self._simulate = simulate
        self.max_side = max_side
        self.max_bytes = max_bytes
        self.sim_delay = sim_delay

    # ------------------------------------------------------------------ lifecycle
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = NIMClient()
        return self._client

    @property
    def simulated(self) -> bool:
        if self._simulate is not None:
            return self._simulate
        return bool(getattr(self.client, "mock", False))

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Prism:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ images
    def load(self, source: ImageSource, *, name: str | None = None) -> PrismImage:
        return load_image(source, name=name, max_side=self.max_side, max_bytes=self.max_bytes)

    # ------------------------------------------------------------------ running
    async def run(
        self,
        mode: str,
        images: ImageSource | Sequence[ImageSource],
        *,
        model: str | None = None,
        observer: Observer | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **options: Any,
    ) -> PrismResult:
        """Run ``mode`` on one image (or two for ``compare``) and return a :class:`PrismResult`.

        ``options`` are the mode's options (``schema=``, ``refine=``, ``question=``, ``detail=``,
        ``focus=``). ``observer(type, data)`` receives ``stage.start`` / ``stage.end`` /
        ``token`` / ``result`` events; with an observer, text modes stream from the model.
        """
        m = get_mode(mode)
        sources = [images] if isinstance(images, (str, Path, bytes, PrismImage)) else list(images)
        opts = m.validate(len(sources), options)
        t0 = time.monotonic()
        if observer is not None:
            observer("stage.start", {"stage": "prepare"})
        loaded = [
            s if isinstance(s, PrismImage) else await asyncio.to_thread(self.load, s)
            for s in sources
        ]
        if observer is not None:
            observer("stage.end", {"stage": "prepare", "seconds": round(time.monotonic() - t0, 3)})
        result = await self._run_loaded(m, loaded, opts, model, observer, max_tokens, temperature)
        if observer is not None:
            observer("result", result.model_dump(mode="json"))
        return result

    async def _run_loaded(
        self,
        m: Mode,
        images: list[PrismImage],
        opts: dict[str, Any],
        model: str | None,
        observer: Observer | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> PrismResult:
        chosen = model or self.model or default_registry().pick(Role.VISION)
        simulated = self.simulated
        ses = Session(
            self.client,
            chosen,
            simulated=simulated,
            observer=observer,
            sim_delay=self.sim_delay,
            max_tokens=max_tokens or m.max_tokens,
            temperature=m.temperature if temperature is None else temperature,
        )
        t0 = time.monotonic()
        output = await m.run(ses, images, opts)
        notes = [n for img in images for n in img.notes]
        if simulated:
            from .sim import SIM_NOTE

            notes.insert(0, SIM_NOTE)
        return PrismResult(
            mode=m.name,
            model=chosen,
            simulated=simulated,
            images=[img.info() for img in images],
            output=output,
            text=m.render(output),
            elapsed_s=round(time.monotonic() - t0, 3),
            calls=ses.calls,
            notes=notes,
        )

    # ------------------------------------------------------------------ one-liners
    async def describe(self, image: ImageSource, **kw: Any) -> PrismResult:
        return await self.run("describe", image, **kw)

    async def ocr(self, image: ImageSource, **kw: Any) -> PrismResult:
        return await self.run("ocr", image, **kw)

    async def extract(
        self, image: ImageSource, schema: str | Path | dict[str, Any] = "receipt", **kw: Any
    ) -> PrismResult:
        return await self.run("extract", image, schema=schema, **kw)

    async def ui2code(self, image: ImageSource, refine: int = 0, **kw: Any) -> PrismResult:
        return await self.run("ui2code", image, refine=refine, **kw)

    async def compare(self, a: ImageSource, b: ImageSource, **kw: Any) -> PrismResult:
        return await self.run("compare", [a, b], **kw)

    async def chart(self, image: ImageSource, **kw: Any) -> PrismResult:
        return await self.run("chart", image, **kw)

    async def ask(self, image: ImageSource, question: str, **kw: Any) -> PrismResult:
        return await self.run("ask", image, question=question, **kw)

    # ------------------------------------------------------------------ batch
    async def batch(
        self,
        sources: str | Path | Sequence[ImageSource],
        mode: str,
        *,
        concurrency: int = 4,
        recursive: bool = False,
        observer: Observer | None = None,
        **kw: Any,
    ) -> list[BatchItem]:
        """Run a single-image ``mode`` over a folder (or a list of images) concurrently.

        One bad image never stops the batch: it becomes an item with ``ok=False`` and an
        ``error``. Results keep the input order. Observer events carry ``index`` and ``name``.
        """
        m = get_mode(mode)
        if m.min_images != 1 or m.max_images != 1:
            raise PrismError(f"{m.name} takes {m.max_images} images and cannot be batched")
        items = _expand(sources, recursive)
        if not items:
            raise PrismError("no images found (png, jpg, webp or gif)")
        m.validate(1, kw)  # fail fast on bad options, before any work starts
        gate = asyncio.Semaphore(max(1, concurrency))

        def label(i: int, src: ImageSource) -> str:
            if isinstance(src, PrismImage):
                return src.name
            return Path(src).name if isinstance(src, (str, Path)) else f"image-{i + 1}"

        async def one(i: int, src: ImageSource) -> BatchItem:
            name = label(i, src)

            def tagged(type_: str, data: dict[str, Any]) -> None:
                if observer is not None:
                    observer(type_, {**data, "index": i, "name": name})

            async with gate:
                try:
                    res = await self.run(mode, src, observer=tagged if observer else None, **kw)
                except Exception as exc:
                    return BatchItem(
                        index=i, name=name, ok=False, error=str(exc) or type(exc).__name__
                    )
            return BatchItem(index=i, name=name, ok=True, result=res)

        return list(await asyncio.gather(*(one(i, s) for i, s in enumerate(items))))
