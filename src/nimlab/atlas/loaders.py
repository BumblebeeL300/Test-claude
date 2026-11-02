"""Document loaders: Markdown, plain text, HTML (stdlib parser) and PDF (pypdf, page-aware).

Every loader turns raw bytes into a list of :class:`~nimlab.atlas.models.Section` objects that
carry the heading breadcrumb (and the page for PDFs); the chunker works from those sections.
"""

from __future__ import annotations

import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from .models import LoadedDocument, LoaderError, MediaType, Section

SUPPORTED_EXTENSIONS: dict[str, MediaType] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".rst": "text",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}

_SKIP_DIRS = {"__pycache__", "node_modules"}

MAX_FILE_BYTES = 50 * 1024 * 1024


def media_type_for(name: str) -> MediaType | None:
    return SUPPORTED_EXTENSIONS.get(PurePosixPath(name.replace("\\", "/")).suffix.lower())


def discover_files(root: Path, *, recursive: bool = True) -> list[tuple[Path, str]]:
    """Supported files under ``root`` as ``(path, display_name)``; name is relative to ``root``.

    A single file yields itself. Hidden files/directories are skipped. Order is deterministic.
    """
    root = Path(root)
    if root.is_file():
        return [(root, root.name)] if media_type_for(root.name) else []
    if not root.is_dir():
        raise LoaderError(f"path does not exist: {root}")
    found: list[tuple[Path, str]] = []
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        rel = p.relative_to(root)
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.is_file() and media_type_for(p.name):
            found.append((p, rel.as_posix()))
    return found


# --------------------------------------------------------------------------- markdown

_ATX_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_LISTISH_RE = re.compile(r"^\s*([-*+]\s|\d+[.)]\s|[|>])")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _clean_heading(text: str) -> str:
    text = _LINK_RE.sub(r"\1", text)
    text = re.sub(r"[`*]|(?<!\w)_|_(?!\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _front_matter_title(lines: list[str]) -> tuple[str | None, int]:
    """(title, index of first body line) for an optional leading ``---`` YAML block."""
    if not lines or lines[0].strip() != "---":
        return None, 0
    for j in range(1, min(len(lines), 80)):
        if lines[j].strip() in ("---", "..."):
            title = None
            for raw in lines[1:j]:
                m = re.match(r"title:\s*(.+?)\s*$", raw.strip(), re.IGNORECASE)
                if m:
                    title = m.group(1).strip("'\"")
            return title, j + 1
    return None, 0


def parse_markdown(text: str) -> tuple[str | None, list[Section]]:
    """Split Markdown into heading-delimited sections. Fenced code never produces headings."""
    lines = text.splitlines()
    title, i = _front_matter_title(lines)
    body_start = i
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    sections: list[Section] = []
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(body, tuple(h for _, h in stack)))
        buf.clear()

    def push_heading(level: int, heading: str) -> None:
        nonlocal title
        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        heading = heading or "(untitled)"
        stack.append((level, heading))
        if title is None and level == 1:
            title = heading

    n = len(lines)
    while i < n:
        line = lines[i]
        if fence is not None:
            buf.append(line)
            close = _FENCE_CLOSE_RE.match(line)
            if close and close.group(1)[0] == fence[0] and len(close.group(1)) >= len(fence):
                fence = None
            i += 1
            continue
        opening = _FENCE_RE.match(line)
        if opening:
            fence = opening.group(1)
            buf.append(line)
            i += 1
            continue
        atx = _ATX_RE.match(line)
        if atx:
            push_heading(len(atx.group(1)), _clean_heading(atx.group(2)))
            i += 1
            continue
        if (
            line.strip()
            and i + 1 < n
            and (i == body_start or not lines[i - 1].strip())
            and not _LISTISH_RE.match(line)
            and not line.startswith("    ")
        ):
            setext = _SETEXT_RE.match(lines[i + 1])
            if setext:
                level = 1 if setext.group(1)[0] == "=" else 2
                push_heading(level, _clean_heading(line))
                i += 2
                continue
        buf.append(line)
        i += 1
    flush()
    return title, sections


# --------------------------------------------------------------------------- plain text


def parse_text(text: str) -> tuple[str | None, list[Section]]:
    body = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return None, ([Section(body)] if body else [])


# --------------------------------------------------------------------------- html

_HTML_SKIP = {"script", "style", "noscript", "template", "svg", "iframe", "nav", "head"}
_HTML_BLOCK = {
    "p", "div", "section", "article", "main", "header", "footer", "aside", "ul", "ol", "dl",
    "table", "blockquote", "figure", "figcaption", "form", "fieldset", "details", "summary",
    "dt", "dd", "address",
}  # fmt: skip
_HTML_VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source"}
_HEADING_TAGS = {f"h{i}" for i in range(1, 7)}


def _tidy_text(raw: str) -> str:
    """Normalise whitespace of extracted HTML text, leaving fenced ``pre`` blocks untouched."""
    out: list[str] = []
    in_fence = False
    blank = False
    for line in raw.split("\n"):
        if line.strip() == "```":
            in_fence = not in_fence
            out.append("```")
            blank = False
            continue
        if in_fence:
            out.append(line.rstrip())
            continue
        line = re.sub(r"[ \t\N{NO-BREAK SPACE}]+", " ", line).strip()
        if not line:
            if not blank and out:
                out.append("")
            blank = True
            continue
        blank = False
        out.append(line)
    return "\n".join(out).strip()


class _HTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.sections: list[Section] = []
        self._stack: list[tuple[int, str]] = []
        self._buf: list[str] = []
        self._skip: list[str] = []
        self._in_title = False
        self._title_buf: list[str] = []
        self._pre = 0
        self._heading_level: int | None = None
        self._heading_buf: list[str] = []
        self._row_cells = 0
        self._first_h1: str | None = None

    # -- helpers
    def _emit(self, text: str) -> None:
        if self._heading_level is not None:
            self._heading_buf.append(text)
        else:
            self._buf.append(text)

    def _flush(self) -> None:
        body = _tidy_text("".join(self._buf))
        if body:
            self.sections.append(Section(body, tuple(h for _, h in self._stack)))
        self._buf = []

    # -- parser callbacks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title" and not self._skip_inside_non_head():
            self._in_title = True
            return
        if tag == "body":
            self._skip.clear()
            return
        if self._skip:
            if tag in _HTML_SKIP and tag not in _HTML_VOID:
                self._skip.append(tag)
            return
        if tag in _HTML_SKIP:
            self._skip.append(tag)
            return
        if tag in _HEADING_TAGS:
            self._flush()
            self._heading_level = int(tag[1])
            self._heading_buf = []
        elif tag == "pre":
            self._pre += 1
            self._emit("\n```\n")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "tr":
            self._row_cells = 0
            self._emit("\n")
        elif tag in ("td", "th"):
            if self._row_cells:
                self._emit(" | ")
            self._row_cells += 1
        elif tag == "br":
            self._emit("\n")
        elif tag == "hr" or tag in _HTML_BLOCK:
            self._emit("\n\n")

    def _skip_inside_non_head(self) -> bool:
        return bool(self._skip) and self._skip[-1] != "head"

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HTML_VOID:
            self.handle_starttag(tag, attrs)
        else:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            if self._in_title:
                self._in_title = False
                text = re.sub(r"\s+", " ", "".join(self._title_buf)).strip()
                if text and self.title is None:
                    self.title = text
            return
        if self._skip:
            if tag in self._skip:
                while self._skip:
                    if self._skip.pop() == tag:
                        break
            return
        if tag in _HEADING_TAGS:
            if self._heading_level is not None:
                heading = re.sub(r"\s+", " ", "".join(self._heading_buf)).strip()
                level = self._heading_level
                self._heading_level = None
                self._heading_buf = []
                if heading:
                    while self._stack and self._stack[-1][0] >= level:
                        self._stack.pop()
                    self._stack.append((level, heading))
                    if level == 1 and self._first_h1 is None:
                        self._first_h1 = heading
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._emit("\n```\n")
        elif tag in _HTML_BLOCK:
            self._emit("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buf.append(data)
            return
        if self._skip:
            return
        self._emit(data if self._pre else re.sub(r"\s+", " ", data))

    def finish(self) -> tuple[str | None, list[Section]]:
        self.close()
        self._flush()
        return self.title or self._first_h1, self.sections


def parse_html(text: str) -> tuple[str | None, list[Section]]:
    parser = _HTMLExtractor()
    parser.feed(text)
    return parser.finish()


# --------------------------------------------------------------------------- pdf

_BULLET_RE = re.compile(r"^([-*\N{BULLET}\N{BLACK CIRCLE}]|\d+[.)])\s")


def reflow_pdf_text(text: str) -> str:
    """Re-join hard-wrapped PDF lines into paragraphs (best effort)."""
    text = text.replace("\r", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = [ln.strip() for ln in text.split("\n")]
    longest = max((len(ln) for ln in lines), default=0)
    paras: list[str] = []
    cur: list[str] = []

    def end_para() -> None:
        if cur:
            paras.append(" ".join(cur))
            cur.clear()

    for line in lines:
        if not line:
            end_para()
            continue
        if cur and _BULLET_RE.match(line):
            end_para()
        cur.append(line)
        if re.search(r"[.!?:]$", line) and len(line) < 0.6 * longest:
            end_para()
    end_para()
    return "\n\n".join(paras)


def parse_pdf(data: bytes) -> tuple[str | None, list[Section]]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as exc:  # pragma: no cover - pypdf is a hard dependency
        raise LoaderError("pypdf is required to read PDF files") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise LoaderError("PDF is encrypted and could not be opened without a password")
        title: str | None = None
        try:
            meta_title = reader.metadata.title if reader.metadata else None
            title = (
                meta_title.strip() if isinstance(meta_title, str) and meta_title.strip() else None
            )
        except Exception:  # metadata is optional and often malformed
            title = None
        sections: list[Section] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:  # one bad page must not sink the document
                continue
            body = reflow_pdf_text(raw).strip()
            if body:
                sections.append(Section(body, (), number))
    except LoaderError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, TypeError) as exc:
        raise LoaderError(f"could not parse PDF: {exc}") from exc
    if not sections:
        raise LoaderError("PDF contains no extractable text (scanned documents need OCR)")
    return title, sections


# --------------------------------------------------------------------------- entry points


def decode_text(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def load_bytes(data: bytes, *, name: str, path: str | None = None) -> LoadedDocument:
    """Parse raw bytes into a :class:`LoadedDocument`. ``name`` decides the format by extension."""
    media = media_type_for(name)
    if media is None:
        raise LoaderError(f"unsupported file type: {name}")
    if len(data) > MAX_FILE_BYTES:
        raise LoaderError(f"file too large ({len(data)} bytes, limit {MAX_FILE_BYTES})")
    if media == "pdf":
        title, sections = parse_pdf(data)
    else:
        text = decode_text(data)
        if media == "markdown":
            title, sections = parse_markdown(text)
        elif media == "html":
            title, sections = parse_html(text)
        else:
            title, sections = parse_text(text)
    if not sections:
        raise LoaderError("document is empty")
    stem = PurePosixPath(name.replace("\\", "/")).stem
    return LoadedDocument(
        path=path or name,
        name=name,
        title=title or stem,
        media_type=media,
        content_hash=hashlib.sha256(data).hexdigest(),
        size=len(data),
        sections=sections,
    )


def load_file(path: Path, *, name: str | None = None) -> LoadedDocument:
    """Read and parse one file. ``name`` is the display name (default: the file name)."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LoaderError(f"cannot read {path}: {exc}") from exc
    return load_bytes(data, name=name or path.name, path=str(path.resolve()))
