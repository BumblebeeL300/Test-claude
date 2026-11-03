"""Prism engine plumbing: client ownership, events and the concurrent batch."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nimlab.core import NIMClient, NIMConfig
from nimlab.prism import BatchItem, Prism, PrismError, write_jsonl


def make_dir(tmp_path: Path, png: Callable[..., bytes], n: int = 4) -> Path:
    folder = tmp_path / "shots"
    folder.mkdir()
    for i in range(n):
        (folder / f"img{i}.png").write_bytes(png(20 + i, 10, fill=(10 * i, 40, 200)))
    return folder


class TestBatch:
    async def test_folder_is_processed_in_order(
        self, prism: Prism, tmp_path: Path, png: Callable[..., bytes]
    ) -> None:
        items = await prism.batch(make_dir(tmp_path, png), "describe")
        assert [i.name for i in items] == [f"img{n}.png" for n in range(4)]
        assert all(i.ok and i.result and i.result.simulated for i in items)
        assert [i.index for i in items] == [0, 1, 2, 3]
        widths = [i.result.images[0]["width"] for i in items if i.result]
        assert widths == [20, 21, 22, 23]

    async def test_runs_concurrently_up_to_the_limit(
        self, tmp_path: Path, png: Callable[..., bytes], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = peak = 0

        async def slow(self: Any, *a: Any, **k: Any) -> Any:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1
            return await orig(self, *a, **k)

        orig = Prism._run_loaded
        monkeypatch.setattr(Prism, "_run_loaded", slow)
        async with NIMClient(NIMConfig(mock=True)) as client:
            folder = make_dir(tmp_path, png, 6)
            await Prism(client).batch(folder, "ocr", concurrency=3)
            assert peak == 3
            peak = 0
            await Prism(client).batch(folder, "ocr", concurrency=1)
            assert peak == 1

    async def test_one_bad_image_does_not_stop_the_batch(
        self, prism: Prism, tmp_path: Path, png: Callable[..., bytes]
    ) -> None:
        folder = make_dir(tmp_path, png, 2)
        (folder / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\nnot really")
        items = await prism.batch(folder, "ocr")
        by_name = {i.name: i for i in items}
        assert by_name["img0.png"].ok and by_name["img1.png"].ok
        assert not by_name["broken.png"].ok and "corrupt PNG" in (by_name["broken.png"].error or "")
        assert by_name["broken.png"].result is None

    async def test_options_reach_every_image(
        self, prism: Prism, tmp_path: Path, png: Callable[..., bytes]
    ) -> None:
        items = await prism.batch(make_dir(tmp_path, png, 2), "ask", question="What size?")
        assert all(i.result and i.result.output["question"] == "What size?" for i in items)

    async def test_accepts_a_list_of_mixed_sources(
        self, prism: Prism, sample_ui: Path, png: Callable[..., bytes]
    ) -> None:
        items = await prism.batch([sample_ui, png(8, 8), str(sample_ui)], "describe")
        assert [i.name for i in items] == ["sample_ui.png", "image-2", "sample_ui.png"]

    async def test_single_file_path(self, prism: Prism, sample_ui: Path) -> None:
        assert len(await prism.batch(sample_ui, "describe")) == 1

    async def test_upfront_errors(self, prism: Prism, tmp_path: Path, sample_ui: Path) -> None:
        with pytest.raises(PrismError, match="no images found"):
            await prism.batch(tmp_path, "ocr")
        with pytest.raises(PrismError, match="cannot be batched"):
            await prism.batch([sample_ui], "compare")
        with pytest.raises(PrismError, match="'question' is required"):
            await prism.batch([sample_ui], "ask")

    async def test_observer_events_are_tagged_with_the_image(
        self, prism: Prism, tmp_path: Path, png: Callable[..., bytes]
    ) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        await prism.batch(
            make_dir(tmp_path, png, 3), "ocr", observer=lambda t, d: events.append((t, d))
        )
        results = [d for t, d in events if t == "result"]
        assert sorted(d["index"] for d in results) == [0, 1, 2]
        assert all("name" in d and d["mode"] == "ocr" for d in results)
        assert all("index" in d for _, d in events)

    async def test_jsonl_output(
        self, prism: Prism, tmp_path: Path, png: Callable[..., bytes]
    ) -> None:
        folder = make_dir(tmp_path, png, 2)
        (folder / "bad.png").write_bytes(b"junk")
        out = tmp_path / "out.jsonl"
        write_jsonl(await prism.batch(folder, "ocr"), out)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 3 and [r["ok"] for r in rows] == [False, True, True]
        assert rows[0]["error"] and rows[1]["result"]["output"]["text"]
        assert BatchItem.model_validate(rows[1]).result is not None


class TestEngine:
    async def test_events_for_a_plain_run(self, prism: Prism, sample_ui: Path) -> None:
        seen: list[str] = []
        await prism.ask(sample_ui, "hi", observer=lambda t, d: seen.append(t))
        assert seen[:2] == ["stage.start", "stage.end"]  # prepare
        assert seen[2] == "stage.start" and seen[-1] == "result"
        assert "token" in seen

    async def test_no_observer_means_no_streaming_and_no_crash(
        self, prism: Prism, sample_ui: Path
    ) -> None:
        assert (await prism.ocr(sample_ui)).output["text"]

    async def test_simulate_flag_overrides_detection(self, sample_ui: Path) -> None:
        async with NIMClient(NIMConfig(mock=True)) as client:
            assert Prism(client, simulate=False).simulated is False
            assert Prism(client).simulated is True

    async def test_default_client_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, sample_ui: Path
    ) -> None:
        monkeypatch.setenv("NIMLAB_MOCK", "1")
        async with Prism() as p:
            assert (await p.describe(sample_ui)).simulated
        assert p._client is None  # an owned client is closed with the engine

    async def test_a_borrowed_client_is_left_open(self, sample_ui: Path) -> None:
        async with NIMClient(NIMConfig(mock=True)) as client:
            async with Prism(client) as p:
                await p.describe(sample_ui)
            assert p.client is client

    async def test_default_model_is_the_registry_vision_pick(
        self, prism: Prism, sample_ui: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (
            await prism.describe(sample_ui)
        ).model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        monkeypatch.setenv("NIMLAB_MODEL_VISION", "microsoft/phi-3.5-vision-instruct")
        assert (await prism.describe(sample_ui)).model == "microsoft/phi-3.5-vision-instruct"
        assert (await Prism(prism.client, model="x/y").describe(sample_ui)).model == "x/y"

    async def test_image_notes_surface_in_the_result(self, prism: Prism) -> None:
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (2000, 100)).save(buf, "PNG")
        r = await Prism(prism.client, max_side=500).describe(buf.getvalue())
        assert any("downscaled" in n for n in r.notes)
        assert r.images[0]["width"] == 500 and r.images[0]["original_width"] == 2000
