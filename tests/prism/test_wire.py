"""What actually goes over HTTP: OpenAI-style ``image_url`` content parts (respx, no network)."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from nimlab.core import NIMAuthError, NIMClient, NIMConfig
from nimlab.prism import Prism, PrismResult
from nimlab.prism.images import image_size

CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
VISION = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


def sse(*texts: str) -> httpx.Response:
    chunks = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": t}}]}) + "\n\n"
        for t in texts
    ]
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(chunks) + "data: [DONE]\n\n",
    )


def body(call: Any) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(call.request.content)
    return parsed


def parts_of(call: Any) -> list[dict[str, Any]]:
    user = [m for m in body(call)["messages"] if m["role"] == "user"][-1]
    assert isinstance(user["content"], list)
    parts: list[dict[str, Any]] = user["content"]
    return parts


def decode(part: dict[str, Any]) -> tuple[str, bytes]:
    url = part["image_url"]["url"]
    head, _, b64 = url.partition(",")
    assert head.startswith("data:") and head.endswith(";base64")
    return head[5:-7], base64.b64decode(b64)


@pytest.fixture
async def live() -> AsyncIterator[Prism]:
    cfg = NIMConfig(api_key="nvapi-test", mock=False, max_retries=0)
    async with NIMClient(cfg) as client:
        yield Prism(client)


async def test_core_mock_accepts_list_content_unchanged(sample_ui: Path) -> None:
    """NIMClient.chat in mock mode must not choke on image content parts."""
    from nimlab.prism.images import load_image

    img = load_image(sample_ui)
    messages = [{"role": "user", "content": [{"type": "text", "text": "what is this"}, img.part()]}]
    async with NIMClient(NIMConfig(mock=True)) as client:
        result = await client.chat(VISION, messages)
        streamed = "".join([c async for c in client.stream_chat(VISION, messages)])
    assert result.content and streamed
    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_describe_sends_the_image_as_an_image_url_part(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        return_value=completion(
            json.dumps({"description": "A mock app.", "alt_text": "App screen", "tags": ["UI"]})
        )
    )
    r = await live.describe(sample_ui)
    assert isinstance(r, PrismResult) and r.simulated is False and r.calls == 1
    assert r.output == {"description": "A mock app.", "alt_text": "App screen", "tags": ["ui"]}
    assert not any("Simulated" in n for n in r.notes)

    sent = body(route.calls.last)
    assert sent["model"] == VISION and sent["stream"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    parts = parts_of(route.calls.last)
    assert parts[0]["type"] == "text" and "alt_text" in parts[0]["text"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    mime, raw = decode(parts[1])
    assert mime == "image/png" and raw == sample_ui.read_bytes()
    assert "<json_schema>" in sent["messages"][0]["content"]  # chat_json's schema injection
    assert route.calls.last.request.headers["authorization"] == "Bearer nvapi-test"


async def test_compare_sends_both_images_in_order(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path, sample_chart: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        return_value=completion(
            json.dumps(
                {
                    "summary": "Different screens.",
                    "similarity": 0.1,
                    "differences": [
                        {"area": "layout", "a": "cards", "b": "bars", "severity": "HIGH"}
                    ],
                }
            )
        )
    )
    r = await live.compare(sample_ui, sample_chart)
    assert r.output["differences"][0]["severity"] == "high"
    parts = parts_of(route.calls.last)
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert decode(parts[1])[1] == sample_ui.read_bytes()
    assert decode(parts[2])[1] == sample_chart.read_bytes()


async def test_streaming_text_mode_requests_sse_and_emits_tokens(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(return_value=sse("Hello ", "wor", "ld"))
    events: list[tuple[str, dict[str, Any]]] = []
    r = await live.ocr(sample_ui, observer=lambda t, d: events.append((t, d)))
    assert r.output["text"] == "Hello world"
    sent = body(route.calls.last)
    assert sent["stream"] is True and sent["temperature"] == 0.0
    assert parts_of(route.calls.last)[1]["type"] == "image_url"
    assert [d["text"] for t, d in events if t == "token"] == ["Hello ", "wor", "ld"]
    assert [t for t, _ in events][-1] == "result"


async def test_non_streaming_text_mode_without_an_observer(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(return_value=completion("```\nTOTAL  4.50\n```"))
    r = await live.ocr(sample_ui)
    assert r.output["text"] == "TOTAL  4.50"  # a wrapping code fence is removed
    assert body(route.calls.last)["stream"] is False


async def test_extract_retries_until_the_schema_is_satisfied(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    good = {"merchant": "Cafe", "date": "2024-01-02", "items": [], "total": 3.5}
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[completion('{"merchant": "Cafe"}'), completion(json.dumps(good))]
    )
    r = await live.extract(sample_ui, schema="receipt")
    assert r.output["valid"] and r.output["data"] == good and r.calls == 1
    assert route.call_count == 2
    retry = body(route.calls.last)["messages"]
    assert "missing required property 'date'" in retry[-1]["content"]
    assert any(isinstance(m["content"], list) for m in retry)  # the image travels with the retry
    assert "<json_schema>" in retry[0]["content"] and '"title": "Receipt"' in retry[0]["content"]


async def test_extract_reports_invalid_output_instead_of_hiding_it(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    respx_mock.post(CHAT_URL).mock(return_value=completion('{"merchant": 5}'))
    o = (await live.extract(sample_ui, schema="receipt")).output
    assert o["valid"] is False and o["data"] == {"merchant": 5}
    assert any("merchant" in e for e in o["errors"])


async def test_extract_with_a_custom_schema_sends_that_schema(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    schema = {
        "title": "Plate",
        "type": "object",
        "properties": {"plate": {"type": "string"}},
        "required": ["plate"],
    }
    route = respx_mock.post(CHAT_URL).mock(return_value=completion('{"plate": "7ABC123"}'))
    r = await live.extract(sample_ui, schema=schema)
    assert r.output["data"] == {"plate": "7ABC123"} and r.output["schema_name"] == "Plate"
    assert '"plate"' in body(route.calls.last)["messages"][0]["content"]


async def test_ui2code_refine_loop_sends_the_screenshot_every_time(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    draft = "<!doctype html><html><head><title>t</title></head><body><p>one</p></body></html>"
    better = draft.replace("one", "two")
    critique = {"score": 0.5, "issues": ["Heading is missing"], "done": False}
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            completion(f"Here:\n```html\n{draft}\n```"),
            completion(json.dumps(critique)),
            completion(f"```html\n{better}\n```"),
        ]
    )
    r = await live.ui2code(sample_ui, refine=1)
    assert route.call_count == 3 and r.calls == 3
    assert r.output["html"].strip() == better and r.output["drafts"] == 2 and r.output["valid"]
    assert r.output["critiques"] == [{"round": 1, "score": 0.5, "issues": ["Heading is missing"]}]
    for call in route.calls:
        assert parts_of(call)[1]["type"] == "image_url"
    critique_prompt = parts_of(route.calls[1])[0]["text"]
    assert draft in critique_prompt and "score (0-1" in critique_prompt
    revise_prompt = parts_of(route.calls[2])[0]["text"]
    assert "- Heading is missing" in revise_prompt and draft in revise_prompt


async def test_ui2code_stops_refining_when_the_critique_is_clean(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            completion("<!doctype html><html><head></head><body>x</body></html>"),
            completion('{"score": 0.97, "issues": [], "done": true}'),
        ]
    )
    r = await live.ui2code(sample_ui, refine=3)
    assert route.call_count == 2 and r.output["drafts"] == 1


async def test_chart_pads_short_rows(
    live: Prism, respx_mock: respx.MockRouter, sample_chart: Path
) -> None:
    respx_mock.post(CHAT_URL).mock(
        return_value=completion(
            json.dumps(
                {
                    "chart_type": "bar",
                    "title": "T",
                    "columns": ["Q", "A", "B"],
                    "rows": [["Q1", 1], ["Q2", 2, 3.5]],
                    "summary": "up",
                }
            )
        )
    )
    o = (await live.chart(sample_chart)).output
    assert o["rows"] == [["Q1", 1, None], ["Q2", 2, 3.5]] and o["csv"] == "Q,A,B\nQ1,1,\nQ2,2,3.5\n"


async def test_ask_sends_the_question_as_text(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(return_value=completion("Three cards."))
    r = await live.ask(sample_ui, "How many cards?", model="microsoft/phi-3.5-vision-instruct")
    assert r.output["answer"] == "Three cards." and r.model == "microsoft/phi-3.5-vision-instruct"
    assert parts_of(route.calls.last)[0] == {"type": "text", "text": "How many cards?"}
    assert body(route.calls.last)["model"] == "microsoft/phi-3.5-vision-instruct"


async def test_the_normalised_image_is_what_gets_sent(
    live: Prism, respx_mock: respx.MockRouter
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    exif = Image.Exif()
    exif[0x010F] = "SecretCam"
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (1.0, 2.0, 3.0), "E", (4.0, 5.0, 6.0)
    buf = io.BytesIO()
    Image.new("RGB", (3200, 1600), (20, 80, 200)).save(buf, "JPEG", exif=exif)
    assert b"SecretCam" in buf.getvalue()

    route = respx_mock.post(CHAT_URL).mock(return_value=completion("ok"))
    await live.ask(buf.getvalue(), "?")
    mime, raw = decode(parts_of(route.calls.last)[1])
    assert mime == "image/jpeg" and image_size(raw) == (1568, 784)
    assert b"SecretCam" not in raw and b"Exif" not in raw


async def test_upstream_errors_propagate(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path
) -> None:
    respx_mock.post(CHAT_URL).mock(return_value=httpx.Response(401, json={"detail": "bad key"}))
    with pytest.raises(NIMAuthError):
        await live.describe(sample_ui)


async def test_batch_makes_one_request_per_image(
    live: Prism, respx_mock: respx.MockRouter, sample_ui: Path, sample_chart: Path
) -> None:
    route = respx_mock.post(CHAT_URL).mock(return_value=completion("fine"))
    items = await live.batch([sample_ui, sample_chart], "ask", question="ok?", concurrency=2)
    assert all(i.ok for i in items) and route.call_count == 2
    sent = sorted(decode(parts_of(c)[1])[1] for c in route.calls)
    assert sent == sorted([sample_ui.read_bytes(), sample_chart.read_bytes()])


def test_part_shape_matches_the_openai_spec(png: Callable[..., bytes]) -> None:
    from nimlab.prism import load_image

    part = load_image(png(4, 4)).part()
    assert set(part) == {"type", "image_url"} and set(part["image_url"]) == {"url"}
