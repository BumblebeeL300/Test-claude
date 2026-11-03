"""Prism modes: a small registry of things you can ask a vision model to do with an image."""

from __future__ import annotations

import csv
import functools
import io
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nimlab.core import NIMJSONError, extract_json

from . import sim
from .errors import PrismError, UnknownModeError
from .images import PrismImage
from .schemas import resolve_schema, schema_names, validate_instance
from .session import Session

__all__ = [
    "Mode",
    "Option",
    "check_html",
    "extract_html",
    "get_mode",
    "list_modes",
    "mode_names",
    "register_mode",
    "schema_model",
]


@dataclass(frozen=True)
class Option:
    """One user-facing option of a mode (also what ``GET /api/prism/modes`` describes)."""

    name: str
    kind: Literal["text", "int", "bool", "choice", "schema"]
    default: Any = None
    help: str = ""
    choices: tuple[str, ...] = ()
    lo: int | None = None
    hi: int | None = None
    required: bool = False

    def coerce(self, value: Any) -> Any:
        if value is None or value == "":
            if self.required:
                raise PrismError(f"option {self.name!r} is required")
            return self.default
        if self.kind == "int":
            try:
                n = int(value)
            except (TypeError, ValueError):
                raise PrismError(f"option {self.name!r} must be an integer") from None
            if (self.lo is not None and n < self.lo) or (self.hi is not None and n > self.hi):
                raise PrismError(f"option {self.name!r} must be between {self.lo} and {self.hi}")
            return n
        if self.kind == "bool":
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        if self.kind == "choice":
            text = str(value).strip().lower()
            if text not in self.choices:
                raise PrismError(f"option {self.name!r} must be one of {', '.join(self.choices)}")
            return text
        if self.kind == "schema":
            return value
        return str(value).strip()

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "help": self.help,
            "choices": list(self.choices),
            "min": self.lo,
            "max": self.hi,
            "required": self.required,
        }


class Mode:
    """Base class. Subclasses set the class attributes and implement :meth:`run`."""

    name = ""
    title = ""
    description = ""
    min_images = 1
    max_images = 1
    options: tuple[Option, ...] = ()
    max_tokens = 1024
    temperature = 0.2

    def validate(self, n_images: int, options: dict[str, Any]) -> dict[str, Any]:
        if n_images < self.min_images or n_images > self.max_images:
            want = (
                f"exactly {self.min_images}"
                if self.min_images == self.max_images
                else f"{self.min_images} to {self.max_images}"
            )
            raise PrismError(f"{self.name} needs {want} image(s), got {n_images}")
        return {o.name: o.coerce(options.get(o.name)) for o in self.options}

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, output: dict[str, Any]) -> str:
        """The mode's primary output as plain text (what the CLI prints)."""
        return json.dumps(output, indent=2, ensure_ascii=False)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "min_images": self.min_images,
            "max_images": self.max_images,
            "options": [o.info() for o in self.options],
        }


_REGISTRY: dict[str, Mode] = {}


def register_mode(mode: Mode | type[Mode]) -> Any:
    """Register a mode (instance or class; usable as a class decorator)."""
    instance = mode() if isinstance(mode, type) else mode
    if not instance.name:
        raise ValueError("a mode needs a name")
    _REGISTRY[instance.name] = instance
    return mode


def get_mode(name: str) -> Mode:
    try:
        return _REGISTRY[str(name).strip().lower()]
    except KeyError:
        raise UnknownModeError(
            f"unknown mode {name!r}; choose one of {', '.join(_REGISTRY)}"
        ) from None


def list_modes() -> list[Mode]:
    return list(_REGISTRY.values())


def mode_names() -> list[str]:
    return list(_REGISTRY)


# --------------------------------------------------------------------------- helpers

_FENCE = re.compile(r"```(?:html|HTML)?\s*\n(.*?)```", re.DOTALL)
_VOID = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()  # noqa: SIM905
)
_OPTIONAL_END = frozenset(
    "html head body p li dt dd tr td th thead tbody tfoot colgroup option".split()  # noqa: SIM905
)


def extract_html(text: str) -> str:
    """The page from a model reply: fenced block or raw document, always a full document."""
    body = text.strip()
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    elif "```" in body:  # unterminated fence
        body = re.sub(r"^```\w*\s*\n?", "", body).strip()
    start = re.search(r"<!doctype html|<html[\s>]", body, re.IGNORECASE)
    if start:
        body = body[start.start() :]
        end = body.lower().rfind("</html>")
        if end != -1:
            body = body[: end + len("</html>")]
        return body.strip() + "\n"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Prism</title>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


class _Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.problems: list[str] = []
        self.seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.seen.add(tag)
        a = {k: (v or "") for k, v in attrs}
        for key in ("src", "href"):
            url = a.get(key, "")
            if url.startswith(("http://", "https://", "//")) and (
                tag in ("script", "img", "iframe", "video", "audio", "source")
                or (tag == "link" and "stylesheet" in a.get("rel", ""))
            ):
                self.problems.append(
                    f"external resource <{tag} {key}={url[:60]!r}>: not self-contained"
                )
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.seen.add(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if tag not in self.stack:
            self.problems.append(f"stray </{tag}>")
            return
        while self.stack:
            top = self.stack.pop()
            if top == tag:
                break
            if top not in _OPTIONAL_END:
                self.problems.append(f"<{top}> is not closed before </{tag}>")


def check_html(html: str) -> list[str]:
    """Structural problems of a page (empty list = well-formed, self-contained)."""
    problems: list[str] = []
    checker = _Checker()
    checker.feed(html)
    checker.close()
    problems += checker.problems
    problems += [f"<{t}> is never closed" for t in checker.stack if t not in _OPTIONAL_END]
    for tag in ("html", "head", "body"):
        if tag not in checker.seen:
            problems.append(f"missing <{tag}>")
    if re.search(r"@import\s+(?:url\()?['\"]?https?:", html):
        problems.append("CSS @import of a remote stylesheet: not self-contained")
    if not html.lstrip().lower().startswith("<!doctype html"):
        problems.append("missing <!doctype html>")
    return problems


def schema_model(schema: dict[str, Any]) -> type[BaseModel]:
    """A pydantic class whose JSON Schema *is* ``schema`` and whose validation enforces it.

    This lets ``NIMClient.chat_json`` drive extraction with a user-provided schema: it injects the
    schema into the prompt and, if the model's JSON breaks it, re-prompts with the error list.
    """

    class Extraction(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return schema

        @model_validator(mode="before")
        @classmethod
        def _conforms(cls, data: Any) -> Any:
            if isinstance(data, dict):
                errors = validate_instance(data, schema)
                if errors:
                    raise ValueError("; ".join(errors[:8]))
            return data

    return Extraction


# --------------------------------------------------------------------------- mode outputs


class Description(BaseModel):
    description: str
    alt_text: str
    tags: list[str] = Field(default_factory=list)


class Difference(BaseModel):
    area: str
    a: str = ""
    b: str = ""
    severity: str = "medium"

    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, v: Any) -> str:
        v = str(v).strip().lower()
        return v if v in ("low", "medium", "high") else "medium"

    @field_validator("a", "b", "area", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return "" if v is None else str(v)


class Comparison(BaseModel):
    summary: str
    similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    differences: list[Difference] = Field(default_factory=list)


class ChartTable(BaseModel):
    chart_type: str = "other"
    title: str = ""
    columns: list[str]
    rows: list[list[str | float | int | None]]
    summary: str = ""


class Critique(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    done: bool = False


# --------------------------------------------------------------------------- the modes


@register_mode
class Describe(Mode):
    name = "describe"
    title = "Describe"
    description = "A rich description of the image plus concise alt text and tags."
    max_tokens = 1024
    temperature = 0.3
    options = (Option("detail", "choice", "rich", "How much to write", choices=("brief", "rich")),)

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        img = images[0]
        brief = opts["detail"] == "brief"
        prompt = (
            "Describe this image. Return JSON with: description ("
            + ("one sentence" if brief else "2-4 paragraphs: subject, layout, colours, text, mood")
            + "), alt_text (under 125 characters, for a screen reader, no 'image of'), and tags "
            "(3-8 lowercase keywords)."
        )
        with ses.stage("describe"):
            out = await ses.json(
                prompt, [img], Description, system="You are a precise visual describer.",
                sim=lambda: sim.describe(img, opts["detail"]),
            )  # fmt: skip
        return {
            "description": out.description.strip(),
            "alt_text": out.alt_text.strip(),
            "tags": [t.strip().lower() for t in out.tags if t.strip()],
        }

    def render(self, output: dict[str, Any]) -> str:
        return f"{output['description']}\n\nAlt text: {output['alt_text']}"


@register_mode
class Ocr(Mode):
    name = "ocr"
    title = "OCR"
    description = "Extract all text, keeping line breaks, columns and tables."
    max_tokens = 2048
    temperature = 0.0

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        img = images[0]
        prompt = (
            "Transcribe every piece of text in this image exactly as written. Preserve the layout: "
            "keep line breaks and reading order, align columns with spaces, and render tables "
            "as Markdown tables. Do not translate, summarise or add commentary. "
            "If there is no text, reply with exactly: [no text]"
        )
        with ses.stage("read"):
            text = await ses.text(
                prompt, [img], system="You are an exact OCR engine.", sim=lambda: sim.ocr(img)
            )
        text = _strip_fence(text)
        return {
            "text": text,
            "lines": len(text.splitlines()),
            "has_text": text.strip() != "[no text]",
        }

    def render(self, output: dict[str, Any]) -> str:
        return str(output["text"])


def _strip_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```[\w-]*\n(.*?)\n?```\s*", text, re.DOTALL)
    return (m.group(1) if m else text).rstrip()


@register_mode
class Extract(Mode):
    name = "extract"
    title = "Extract"
    description = "Structured data from the image, validated against a JSON Schema."
    max_tokens = 2048
    temperature = 0.0
    options = (
        Option(
            "schema",
            "schema",
            "receipt",
            "A built-in schema ("
            + ", ".join(schema_names())
            + "), a JSON file path, or inline JSON",
        ),
    )

    def validate(self, n_images: int, options: dict[str, Any]) -> dict[str, Any]:
        opts = super().validate(n_images, options)
        opts["_resolved"] = resolve_schema(opts["schema"])
        return opts

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        img = images[0]
        name, schema = opts["_resolved"]
        prompt = (
            f"Extract the data from this image into JSON for the '{name}' schema. Use null or omit "
            "a field that is not visible; never invent values. Numbers are plain numbers without "
            "currency symbols."
        )
        errors: list[str] = []
        try:
            out = await ses.json(
                prompt,
                [img],
                schema_model(schema),
                system="You extract structured data from images.",
                sim=lambda: sim.extract(img, name, schema),
            )
            data = out.model_dump(mode="json")
        except NIMJSONError as exc:  # keep what the model said, but say it is not valid
            try:
                data = extract_json(exc.raw)
            except ValueError:
                raise PrismError(f"the model did not return JSON: {exc}") from exc
            errors = (
                validate_instance(data, schema) if isinstance(data, dict) else ["not an object"]
            )
        return {
            "schema_name": name,
            "schema": schema,
            "data": data,
            "valid": not errors,
            "errors": errors,
        }

    def render(self, output: dict[str, Any]) -> str:
        return json.dumps(output["data"], indent=2, ensure_ascii=False)


@register_mode
class Ui2Code(Mode):
    name = "ui2code"
    title = "UI to code"
    description = (
        "Turn a screenshot into one self-contained HTML/CSS file, with optional "
        "critique-and-refine rounds against the screenshot."
    )
    max_tokens = 6000
    temperature = 0.2
    options = (
        Option("refine", "int", 0, "Critique-and-fix rounds against the screenshot", lo=0, hi=5),
    )

    SYSTEM = (
        "You are a front-end engineer who reproduces user interfaces from screenshots as clean, "
        "semantic, accessible HTML and CSS."
    )

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        img = images[0]
        with ses.stage("draft"):
            raw = await ses.text(
                "Reproduce this screenshot as ONE self-contained HTML file: inline <style>, no "
                "JavaScript frameworks, no external fonts, images or scripts (use CSS shapes or "
                "inline SVG instead). Match layout, spacing, colours and typography; use "
                "placeholder text where the screenshot's text is unreadable. Reply with only the "
                "code in a single ```html block.",
                [img], system=self.SYSTEM, sim=lambda: sim.ui2code(img, 0),
            )  # fmt: skip
        html = extract_html(raw)
        drafts = 1
        critiques: list[dict[str, Any]] = []
        for n in range(1, int(opts["refine"]) + 1):
            with ses.stage(f"critique {n}"):
                crit = await ses.json(
                    "Image 1 is the target screenshot. Below is the HTML written to reproduce it. "
                    "Act as a strict design reviewer: list the concrete visual discrepancies "
                    "(layout, spacing, colours, typography, missing elements). Return JSON with "
                    "score (0-1 visual fidelity), issues (short strings, empty if faithful) and "
                    f"done (true if no meaningful issues remain).\n\n```html\n{html[:24000]}\n```",
                    [img], Critique, system=self.SYSTEM,
                    sim=functools.partial(sim.ui2code_critique, n - 1),
                )  # fmt: skip
            critiques.append({"round": n, "score": crit.score, "issues": crit.issues})
            if crit.done or not crit.issues:
                break
            with ses.stage(f"refine {n}"):
                raw = await ses.text(
                    "Revise the HTML so it matches the screenshot more closely. Fix these issues:\n"
                    + "\n".join(f"- {i}" for i in crit.issues)
                    + f"\n\nCurrent HTML:\n```html\n{html[:24000]}\n```\n"
                    "Reply with the complete corrected file in a single ```html block.",
                    [img], system=self.SYSTEM, sim=functools.partial(sim.ui2code, img, n),
                )  # fmt: skip
            html = extract_html(raw)
            drafts += 1
        problems = check_html(html)
        return {
            "html": html,
            "drafts": drafts,
            "critiques": critiques,
            "problems": problems,
            "valid": not problems,
            "width": img.width,
            "height": img.height,
        }

    def render(self, output: dict[str, Any]) -> str:
        return str(output["html"])


@register_mode
class Compare(Mode):
    name = "compare"
    title = "Compare"
    description = "Diff two images and list what changed."
    min_images = 2
    max_images = 2
    max_tokens = 1536
    temperature = 0.1
    options = (
        Option("focus", "text", "", "Optional aspect to concentrate on, e.g. 'copy and colours'"),
    )

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        a, b = images
        focus = f" Concentrate on: {opts['focus']}." if opts["focus"] else ""
        prompt = (
            "Image 1 is A and image 2 is B. List every meaningful difference between them (added, "
            "removed or changed elements, text, colours, layout)." + focus + " Return JSON with: "
            "summary (2-3 sentences), similarity (0-1), differences (each with area, a: what A "
            "shows, b: what B shows, severity: low|medium|high). If they are identical, say so "
            "and return an empty differences list."
        )
        with ses.stage("compare"):
            out = await ses.json(
                prompt, [a, b], Comparison, system="You are a meticulous visual diff tool.",
                sim=lambda: sim.compare(a, b),
            )  # fmt: skip
        return out.model_dump(mode="json")

    def render(self, output: dict[str, Any]) -> str:
        lines = [output["summary"], ""]
        for d in output["differences"]:
            lines.append(f"- [{d['severity']}] {d['area']}: {d['a']} -> {d['b']}")
        return "\n".join(lines).rstrip()


def table_csv(columns: list[str], rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
    return buf.getvalue()


@register_mode
class Chart(Mode):
    name = "chart"
    title = "Chart to data"
    description = "Read a chart back into a data table with a short summary."
    max_tokens = 1536
    temperature = 0.1

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        img = images[0]
        prompt = (
            "This image is a chart. Recover its data as a table. Return JSON with: chart_type "
            "(bar|line|pie|scatter|area|other), title, columns (first is the x-axis/category "
            "label, then one per series), rows (each a list: label then numbers; estimate values "
            "from the axes when not printed), and summary (2-3 sentences on the main trend)."
        )
        with ses.stage("read chart"):
            out = await ses.json(
                prompt, [img], ChartTable, system="You digitise charts accurately.",
                sim=lambda: sim.chart(img),
            )  # fmt: skip
        width = len(out.columns)
        rows = [([*r, *([None] * width)][:width]) for r in out.rows]
        return {
            "chart_type": out.chart_type,
            "title": out.title,
            "columns": out.columns,
            "rows": rows,
            "summary": out.summary,
            "csv": table_csv(out.columns, rows),
        }

    def render(self, output: dict[str, Any]) -> str:
        return f"{output['summary']}\n\n{output['csv']}".rstrip()


@register_mode
class Ask(Mode):
    name = "ask"
    title = "Ask"
    description = "Ask any question about the image."
    max_tokens = 1536
    temperature = 0.3
    options = (Option("question", "text", "", "What do you want to know?", required=True),)

    async def run(
        self, ses: Session, images: list[PrismImage], opts: dict[str, Any]
    ) -> dict[str, Any]:
        img = images[0]
        question = opts["question"]
        with ses.stage("answer"):
            answer = await ses.text(
                question, [img],
                system="Answer questions about the image concisely and only from what is visible. "
                "Say so when something cannot be determined.",
                sim=lambda: sim.ask(img, question),
            )  # fmt: skip
        return {"question": question, "answer": answer.strip()}

    def render(self, output: dict[str, Any]) -> str:
        return str(output["answer"])
