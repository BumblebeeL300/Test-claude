"""Every mode in mock (simulation) mode, schema extraction and generated-HTML validity."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from nimlab.prism import (
    BUILTIN_SCHEMAS,
    Mode,
    Prism,
    PrismError,
    UnknownModeError,
    check_html,
    extract_html,
    get_mode,
    list_modes,
    mode_names,
    register_mode,
    resolve_schema,
    validate_instance,
)
from nimlab.prism import sim as simmod
from nimlab.prism.images import load_image
from nimlab.prism.modes import schema_model, table_csv


class TestRegistry:
    def test_all_seven_modes_are_registered(self) -> None:
        assert set(mode_names()) == {
            "describe", "ocr", "extract", "ui2code", "compare", "chart", "ask"
        }  # fmt: skip
        assert [m.name for m in list_modes()] == mode_names()

    def test_unknown_mode(self) -> None:
        with pytest.raises(UnknownModeError, match="choose one of describe"):
            get_mode("sketch")

    def test_info_describes_options(self) -> None:
        info = get_mode("ui2code").info()
        assert info["options"][0]["name"] == "refine" and info["options"][0]["max"] == 5
        assert get_mode("compare").info()["min_images"] == 2

    def test_custom_mode_can_be_registered(self, prism_factory: Callable[..., Prism]) -> None:
        @register_mode
        class Shout(Mode):
            name = "shout"
            title = "Shout"

            async def run(self, ses: Any, images: Any, opts: Any) -> dict[str, Any]:
                return {"loud": images[0].name.upper()}

        try:
            assert "shout" in mode_names()
        finally:
            from nimlab.prism import modes

            modes._REGISTRY.pop("shout")


class TestOptions:
    @pytest.mark.parametrize(
        ("mode", "options", "match"),
        [
            ("ask", {}, "'question' is required"),
            ("ui2code", {"refine": 9}, "between 0 and 5"),
            ("ui2code", {"refine": "x"}, "integer"),
            ("describe", {"detail": "huge"}, "one of brief, rich"),
        ],
    )
    async def test_bad_options(
        self, prism: Prism, sample_ui: Path, mode: str, options: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(PrismError, match=match):
            await prism.run(mode, sample_ui, **options)

    async def test_image_counts_are_enforced(self, prism: Prism, sample_ui: Path) -> None:
        with pytest.raises(PrismError, match="exactly 2"):
            await prism.run("compare", sample_ui)
        with pytest.raises(PrismError, match="exactly 1"):
            await prism.run("describe", [sample_ui, sample_ui])


class TestSimulatedModes:
    async def test_describe(self, prism: Prism, sample_ui: Path) -> None:
        r = await prism.describe(sample_ui)
        assert r.simulated and r.mode == "describe" and r.calls == 1
        out = r.output
        assert "256 by 160" in out["description"] and out["alt_text"] and out["tags"]
        assert len(out["alt_text"]) <= 125
        assert r.text.endswith(out["alt_text"])
        assert "Simulated output" in r.notes[0]

    async def test_describe_brief(self, prism: Prism, sample_ui: Path) -> None:
        r = await prism.describe(sample_ui, detail="brief")
        assert r.output["description"] == r.output["alt_text"]

    async def test_ocr_keeps_layout(self, prism: Prism, sample_ui: Path) -> None:
        r = await prism.ocr(sample_ui)
        lines = r.output["text"].splitlines()
        assert r.output["lines"] == len(lines) > 6
        assert any(line.startswith("   ") for line in lines)  # centred header
        assert any(line.rstrip()[-1].isdigit() for line in lines)  # right-aligned amounts

    async def test_ocr_and_extract_agree_for_the_same_image(
        self, prism: Prism, sample_ui: Path
    ) -> None:
        data = (await prism.extract(sample_ui, schema="receipt")).output["data"]
        text = (await prism.ocr(sample_ui)).output["text"]
        assert data["merchant"] in text and f"{data['total']:.2f}" in text

    async def test_ask_answers_about_what_it_can_measure(
        self, prism: Prism, sample_ui: Path
    ) -> None:
        r = await prism.ask(sample_ui, "What colors dominate?")
        assert r.output["question"] == "What colors dominate?"
        assert r.output["answer"].startswith("[simulated]") and "navy" in r.output["answer"]
        size = await prism.ask(sample_ui, "What size is it?")
        assert "256 by 160" in size.output["answer"]

    async def test_chart_gives_a_table_csv_and_summary(
        self, prism: Prism, sample_chart: Path
    ) -> None:
        o = (await prism.chart(sample_chart)).output
        assert o["columns"][0] == "Period" and len(o["rows"]) >= 3
        assert all(len(r) == len(o["columns"]) for r in o["rows"])
        rows = list(csv.reader(io.StringIO(o["csv"])))
        assert rows[0] == o["columns"] and len(rows) == len(o["rows"]) + 1
        assert o["summary"] and o["chart_type"] in {"bar", "line", "area"}

    async def test_compare_two_different_images(
        self, prism: Prism, sample_ui: Path, sample_chart: Path
    ) -> None:
        r = await prism.compare(sample_ui, sample_chart)
        areas = {d["area"] for d in r.output["differences"]}
        assert {"dimensions", "palette"} <= areas
        assert 0 <= r.output["similarity"] < 1 and "- [high] dimensions" in r.text

    async def test_compare_identical_images(self, prism: Prism, sample_ui: Path) -> None:
        r = await prism.compare(sample_ui, sample_ui)
        assert r.output["similarity"] == 1.0 and r.output["differences"] == []
        assert "identical" in r.output["summary"]

    async def test_compare_same_size_other_pixels(
        self, prism: Prism, png: Callable[..., bytes]
    ) -> None:
        r = await prism.compare(png(20, 20, fill=(10, 10, 10)), png(20, 20, fill=(12, 10, 10)))
        assert r.output["differences"][0]["area"] == "pixel data"

    async def test_results_are_deterministic(self, prism: Prism, sample_ui: Path) -> None:
        for mode, kw in [("describe", {}), ("ocr", {}), ("chart", {}), ("ui2code", {"refine": 1})]:
            a = await prism.run(mode, sample_ui, **kw)
            b = await prism.run(mode, sample_ui, **kw)
            assert a.output == b.output

    async def test_result_serialises(self, prism: Prism, sample_ui: Path, tmp_path: Path) -> None:
        r = await prism.describe(sample_ui)
        r.save_json(tmp_path / "r.json")
        loaded = json.loads((tmp_path / "r.json").read_text())
        assert loaded["mode"] == "describe" and loaded["simulated"] is True
        assert loaded["images"][0]["name"] == "sample_ui.png"

    async def test_mock_usage_is_recorded(self, prism: Prism, sample_ui: Path) -> None:
        await prism.ocr(sample_ui)
        stats = prism.client.usage.summary()["models"]["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"]
        assert stats["calls"] == 1 and stats["tokens_out"] > 0

    async def test_model_override(self, prism: Prism, sample_ui: Path) -> None:
        r = await prism.describe(sample_ui, model="microsoft/phi-3.5-vision-instruct")
        assert r.model == "microsoft/phi-3.5-vision-instruct"


class TestExtract:
    @pytest.mark.parametrize("name", list(BUILTIN_SCHEMAS))
    async def test_builtin_schemas_give_valid_data(
        self, prism: Prism, sample_ui: Path, name: str
    ) -> None:
        r = await prism.extract(sample_ui, schema=name)
        o = r.output
        assert o["valid"] and o["errors"] == [] and o["schema_name"] == name
        assert list(Draft202012Validator(BUILTIN_SCHEMAS[name].schema).iter_errors(o["data"])) == []
        assert validate_instance(o["data"], BUILTIN_SCHEMAS[name].schema) == []

    async def test_receipt_numbers_add_up(self, prism: Prism, sample_ui: Path) -> None:
        d = (await prism.extract(sample_ui, schema="receipt")).output["data"]
        assert round(sum(i["total"] for i in d["items"]), 2) == d["subtotal"]
        assert round(d["subtotal"] + d["tax"], 2) == d["total"]

    async def test_custom_schema_from_a_dict(self, prism: Prism, sample_ui: Path) -> None:
        schema = {
            "title": "Badge",
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 100, "maximum": 200},
                "kind": {"type": "string", "enum": ["staff", "guest"]},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "active": {"type": "boolean"},
            },
            "required": ["id", "kind", "tags"],
        }
        o = (await prism.extract(sample_ui, schema=schema)).output
        assert o["valid"] and o["schema_name"] == "Badge"
        assert 100 <= o["data"]["id"] <= 200 and o["data"]["kind"] in ("staff", "guest")
        assert list(Draft202012Validator(schema).iter_errors(o["data"])) == []

    async def test_schema_from_file_and_inline_json(
        self, prism: Prism, sample_ui: Path, tmp_path: Path
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        path = tmp_path / "person.json"
        path.write_text(json.dumps(schema))
        from_file = await prism.extract(sample_ui, schema=path)
        inline = await prism.extract(sample_ui, schema=json.dumps(schema))
        assert from_file.output["schema_name"] == "person"
        assert from_file.output["data"]["name"] and inline.output["valid"]

    async def test_default_schema_is_receipt(self, prism: Prism, sample_ui: Path) -> None:
        assert (await prism.run("extract", sample_ui)).output["schema_name"] == "receipt"

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("nonsense", "unknown schema 'nonsense'"),
            ("{not json", "not valid JSON"),
            ({"type": "array"}, "root must describe an object"),
            ("[1, 2]", "unknown schema"),
        ],
    )
    async def test_bad_schemas(self, prism: Prism, sample_ui: Path, spec: Any, match: str) -> None:
        with pytest.raises(PrismError, match=match):
            await prism.extract(sample_ui, schema=spec)

    def test_resolve_accepts_spelling_variants(self) -> None:
        assert resolve_schema("Business-Card")[0] == "business_card"
        assert resolve_schema(None)[0] == "receipt"

    def test_validator_catches_what_it_should(self) -> None:
        schema = BUILTIN_SCHEMAS["receipt"].schema
        errors = validate_instance({"merchant": 3, "items": [{}], "total": "x"}, schema)
        joined = "\n".join(errors)
        assert "$.merchant: expected string" in joined
        assert "missing required property 'date'" in joined
        assert "$.items[0]: missing required property 'description'" in joined
        assert "$.total: expected number" in joined

    def test_dynamic_model_reports_schema_and_rejects_bad_json(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        model = schema_model(schema)
        assert model.model_json_schema() == schema
        assert model.model_validate({"n": 3, "extra": 1}).model_dump() == {"n": 3, "extra": 1}
        with pytest.raises(ValueError, match="missing required property 'n'"):
            model.model_validate({})


class TestUi2Code:
    class _Tags(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: list[str] = []
            self.style = ""
            self._in_style = False

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            self.tags.append(tag)
            self._in_style = tag == "style"

        def handle_endtag(self, tag: str) -> None:
            self._in_style = False

        def handle_data(self, data: str) -> None:
            if self._in_style:
                self.style += data

    async def test_produces_a_valid_self_contained_page(
        self, prism: Prism, sample_ui: Path
    ) -> None:
        r = await prism.ui2code(sample_ui)
        o = r.output
        assert o["valid"] and o["problems"] == [] and o["drafts"] == 1 and o["critiques"] == []
        html = o["html"]
        assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
        parser = self._Tags()
        parser.feed(html)
        assert {"html", "head", "body", "style", "header", "main", "footer"} <= set(parser.tags)
        assert "http://" not in html and "https://" not in html  # self-contained
        assert "<script" not in html
        assert "SIMULATED" in html  # flagged inside the artefact itself
        assert "256 &times; 160" in html and "max-width: 256px" not in html  # min width floor
        assert r.simulated

    async def test_uses_the_images_own_colours(
        self, prism: Prism, png: Callable[..., bytes]
    ) -> None:
        html = (await prism.ui2code(png(400, 300, fill=(250, 10, 10)))).output["html"]
        assert "--bg: #FA0A0A" in html and "max-width: 400px" in html

    async def test_refine_loop_critiques_and_improves(self, prism: Prism, sample_ui: Path) -> None:
        r = await prism.ui2code(sample_ui, refine=2)
        o = r.output
        assert o["drafts"] == 3 and len(o["critiques"]) == 2 and r.calls == 5
        scores = [c["score"] for c in o["critiques"]]
        assert scores == sorted(scores) and scores[0] < scores[-1]
        assert all(c["issues"] for c in o["critiques"])
        assert "draft 3" in o["html"] and ":focus-visible" in o["html"] and o["valid"]
        assert o["html"] != (await prism.ui2code(sample_ui)).output["html"]

    async def test_refine_stops_when_the_critique_is_clean(
        self, prism: Prism, sample_ui: Path
    ) -> None:
        r = await prism.ui2code(sample_ui, refine=5)
        assert r.output["drafts"] == 5 and r.output["critiques"][-1]["issues"] == []
        assert r.output["critiques"][-1]["round"] == 5

    async def test_stages_are_reported(self, prism: Prism, sample_ui: Path) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        await prism.ui2code(sample_ui, refine=1, observer=lambda t, d: events.append((t, d)))
        starts = [d["stage"] for t, d in events if t == "stage.start"]
        assert starts == ["prepare", "draft", "critique 1", "refine 1"]
        ends = [d["stage"] for t, d in events if t == "stage.end"]
        assert ends == starts
        tokens = [d for t, d in events if t == "token"]
        assert {d["stage"] for d in tokens} == {"draft", "refine 1"}
        assert "".join(d["text"] for d in tokens if d["stage"] == "draft").startswith("<!doctype")
        assert events[-1][0] == "result"


class TestHtmlHelpers:
    def test_extract_html_from_fence_with_prose(self) -> None:
        reply = (
            "Sure! Here you go:\n```html\n<!doctype html><html><body>hi</body></html>\n```\nEnjoy"
        )
        assert extract_html(reply) == "<!doctype html><html><body>hi</body></html>\n"

    def test_extract_html_trims_text_around_a_raw_document(self) -> None:
        html = extract_html("blah <html><body>x</body></html> trailing")
        assert html.startswith("<html>") and html.endswith("</html>\n")

    def test_fragments_are_wrapped_into_a_document(self) -> None:
        html = extract_html("<div>Hello</div>")
        assert html.startswith("<!doctype html>") and "<div>Hello</div>" in html
        assert check_html(html) == []

    def test_unterminated_fence(self) -> None:
        assert "<p>x</p>" in extract_html("```html\n<p>x</p>")

    def test_check_html_finds_problems(self) -> None:
        problems = check_html(
            '<html><head></head><body><div><span>x</div><script src="https://cdn.x/a.js"></script>'
            "</body></html></p>"
        )
        joined = "\n".join(problems)
        assert "not closed before </div>" in joined and "external resource" in joined
        assert "stray </p>" in joined and "missing <!doctype html>" in joined
        assert "never closed" in "\n".join(
            check_html("<!doctype html><html><head></head><body><b>")
        )

    def test_check_html_flags_remote_css(self) -> None:
        page = (
            "<!doctype html><html><head><style>@import url('https://x.dev/a.css');</style></head>"
            "<body></body></html>"
        )
        assert any("@import" in p for p in check_html(page))

    def test_void_and_optional_tags_are_fine(self) -> None:
        page = (
            "<!doctype html><html><head><meta charset=utf-8><link rel=icon href=data:,></head>"
            "<body><ul><li>a<li>b</ul><br><img alt=x src=data:image/png;base64,AA></body></html>"
        )
        assert check_html(page) == []


class TestSimHelpers:
    def test_color_names(self) -> None:
        assert simmod.color_name("#102060") == "navy"
        assert simmod.color_name("#F5F5F7") == "white"

    def test_table_csv_handles_none_and_commas(self) -> None:
        assert table_csv(["a", "b"], [["x,y", None]]) == 'a,b\n"x,y",\n'

    def test_sim_critique_eventually_clean(self) -> None:
        assert simmod.ui2code_critique(0)["issues"] and simmod.ui2code_critique(9)["done"]

    def test_loaded_image_object_is_accepted(self, png: Callable[..., bytes]) -> None:
        assert load_image(load_image(png(4, 4))).width == 4
