"""Deterministic offline outputs for mock mode.

Every function is pure: the result depends only on the image bytes (hash, size, format, dominant
colours) and the options. Outputs are derived from those facts, so they stay truthful ("an 800x600
PNG dominated by navy") instead of pretending to read content a mock cannot see. Callers flag the
result as simulated.
"""

# ruff: noqa: E501  (the simulated page is an HTML/CSS template)

from __future__ import annotations

import html as _html
import re
from typing import Any

from .images import PrismImage
from .schemas import BUILTIN_SCHEMAS, _receipt_sample, chart_series

__all__ = [
    "SIM_NOTE",
    "ask",
    "chart",
    "color_name",
    "compare",
    "describe",
    "extract",
    "ocr",
    "ui2code",
    "ui2code_critique",
]

SIM_NOTE = "Simulated output: no NVIDIA API key, so Prism used its offline mock."

_NAMED = {
    "black": (16, 16, 20), "charcoal": (54, 56, 64), "grey": (128, 128, 134),
    "silver": (192, 194, 200), "white": (246, 246, 248), "red": (214, 48, 49),
    "orange": (240, 140, 40), "yellow": (240, 210, 60), "green": (60, 170, 90),
    "teal": (30, 160, 160), "cyan": (60, 210, 230), "blue": (50, 100, 220),
    "navy": (20, 36, 96), "purple": (120, 70, 200), "magenta": (210, 60, 160),
    "pink": (240, 150, 190), "brown": (120, 80, 50), "beige": (222, 205, 175),
}  # fmt: skip


def _rgb(hex_: str) -> tuple[int, int, int]:
    return int(hex_[1:3], 16), int(hex_[3:5], 16), int(hex_[5:7], 16)


def color_name(hex_: str) -> str:
    """Nearest of a small set of everyday colour names."""
    r, g, b = _rgb(hex_)
    return min(
        _NAMED, key=lambda n: sum((x - y) ** 2 for x, y in zip(_NAMED[n], (r, g, b), strict=True))
    )


def luminance(hex_: str) -> float:
    r, g, b = _rgb(hex_)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def _on(hex_: str) -> str:
    return "#111218" if luminance(hex_) > 0.55 else "#F7F7FB"


def seed_of(img: PrismImage) -> int:
    return int(img.sha[:8], 16)


def _shape(img: PrismImage) -> str:
    ratio = img.width / img.height
    return "square" if 0.92 <= ratio <= 1.08 else ("landscape" if ratio > 1 else "portrait")


def _names(img: PrismImage) -> list[str]:
    out: list[str] = []
    for c in img.colors(4):
        n = color_name(c)
        if n not in out:
            out.append(n)
    return out or ["grey"]


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


# --------------------------------------------------------------------------- text modes


def describe(img: PrismImage, detail: str = "rich") -> dict[str, Any]:
    names, shape = _names(img), _shape(img)
    tone = "bright, high-key" if luminance(img.colors(1)[0]) > 0.55 else "dark, low-key"
    alt = f"A {shape} {img.format.upper()} image, {img.width} by {img.height} pixels, mostly {_join(names[:2])}."
    if detail == "brief":
        return {"description": alt, "alt_text": alt, "tags": [shape, *names[:2]]}
    pick = (
        "an interface or document layout",
        "a graphic composition",
        "a scene with a clear focal area",
    )
    guess = pick[seed_of(img) % len(pick)]
    description = (
        f"This is a {shape} {img.format.upper()} image of {img.width} by {img.height} pixels "
        f"(aspect ratio {img.width / img.height:.2f}:1). Its palette is led by {_join(names)}, "
        f"giving it a {tone} overall tone. The structure is consistent with {guess}. "
        "In simulation mode Prism can measure size, format and colour but cannot read the content; "
        "add an NVIDIA API key for a real description."
    )
    return {"description": description, "alt_text": alt, "tags": [shape, img.format, *names[:3]]}


def _receipt_text(data: dict[str, Any]) -> str:
    width = 38
    lines = [
        data["merchant"].center(width).rstrip(),
        data["address"][:width].center(width).rstrip(),
        "",
    ]
    lines.append(f"{data['date']}   {data['payment_method']}")
    lines.append("-" * width)
    for it in data["items"]:
        left = f"{it['quantity']:g} x {it['description']}"
        lines.append(f"{left:<{width - 9}}{it['total']:>9.2f}")
    lines.append("-" * width)
    lines.append(f"{'Subtotal':<{width - 9}}{data['subtotal']:>9.2f}")
    lines.append(f"{'Tax':<{width - 9}}{data['tax']:>9.2f}")
    lines.append(f"{'TOTAL':<{width - 9}}{data['total']:>9.2f}")
    lines += ["", "Thank you!".center(width).rstrip()]
    return "\n".join(lines)


def ocr(img: PrismImage) -> str:
    """A receipt laid out in columns; the same image yields the same data as :func:`extract`."""
    return _receipt_text(_receipt_sample(seed_of(img)))


def extract(img: PrismImage, name: str, schema: dict[str, Any]) -> dict[str, Any]:
    seed = seed_of(img)
    if name in BUILTIN_SCHEMAS and BUILTIN_SCHEMAS[name].schema is schema:
        return BUILTIN_SCHEMAS[name].sample(seed)
    from nimlab.core.mock import generate_json

    value = generate_json(schema, seed=f"prism|{img.sha}|{name}", user_text=f"{name} {img.name}")
    return value if isinstance(value, dict) else {}


def ask(img: PrismImage, question: str) -> str:
    names = _names(img)
    q = question.strip().rstrip("?")
    low = q.lower()
    if re.search(r"\bcolou?rs?\b", low):
        body = f"The dominant colours are {_join(names)}."
    elif re.search(r"\b(size|dimension|resolution|big|large)\b", low):
        body = f"The image is {img.width} by {img.height} pixels ({img.format.upper()}, {len(img.data) // 1024 or 1} KiB)."
    elif re.search(r"\b(how many|count)\b", low):
        body = "I cannot count objects in simulation mode."
    else:
        body = f"It is a {_shape(img)} image with a {_join(names[:2])} palette."
    return (
        f"[simulated] {body} "
        "Simulation mode only measures the file, so it cannot answer questions about the content; "
        "add an NVIDIA API key for a real answer."
    )


# --------------------------------------------------------------------------- compare


def compare(a: PrismImage, b: PrismImage) -> dict[str, Any]:
    diffs: list[dict[str, str]] = []
    if a.sha == b.sha:
        return {
            "summary": "The two images are byte-for-byte identical.",
            "similarity": 1.0,
            "differences": [],
        }
    if (a.width, a.height) != (b.width, b.height):
        diffs.append(
            {
                "area": "dimensions",
                "a": f"{a.width}x{a.height}",
                "b": f"{b.width}x{b.height}",
                "severity": "high",
            }
        )
    if a.format != b.format:
        diffs.append({"area": "format", "a": a.format, "b": b.format, "severity": "low"})
    ca, cb = a.colors(3), b.colors(3)
    dist = sum(
        sum(abs(x - y) for x, y in zip(_rgb(p), _rgb(q), strict=True))
        for p, q in zip(ca, cb, strict=False)
    ) / (3 * 765)
    if dist > 0.04:
        diffs.append(
            {
                "area": "palette",
                "a": _join([color_name(c) for c in ca]),
                "b": _join([color_name(c) for c in cb]),
                "severity": "medium" if dist > 0.15 else "low",
            }
        )
    if not diffs:
        diffs.append({"area": "pixel data", "a": a.sha[:8], "b": b.sha[:8], "severity": "low"})
    similarity = round(max(0.0, 1.0 - dist - 0.15 * (len(diffs) - 1)), 2)
    summary = (
        f"Found {len(diffs)} measurable difference{'s' if len(diffs) != 1 else ''} "
        f"({', '.join(d['area'] for d in diffs)}). Simulation mode compares file facts, not content."
    )
    return {"summary": summary, "similarity": similarity, "differences": diffs}


# --------------------------------------------------------------------------- chart


def chart(img: PrismImage) -> dict[str, Any]:
    title, xs, series, kind = chart_series(seed_of(img))
    rows: list[list[Any]] = [[x, *(s[1][i] for s in series)] for i, x in enumerate(xs)]
    top = max(range(len(xs)), key=lambda i: series[0][1][i])
    first, last = series[0][1][0], series[0][1][-1]
    trend = "up" if last > first else "down"
    summary = (
        f"This {kind} chart tracks {series[0][0].lower()} over {len(xs)} periods. Values trend {trend} "
        f"from {first:g} to {last:g}, peaking at {series[0][1][top]:g} in {xs[top]}. "
        f"The {series[1][0].lower()} series stays below it throughout."
    )
    return {
        "chart_type": kind,
        "title": title,
        "columns": ["Period", *(s[0] for s in series)],
        "rows": rows,
        "summary": summary,
    }


# --------------------------------------------------------------------------- ui2code

_CRITIQUES = (
    ["Header spacing is tighter than the screenshot.", "Card corner radius is too small.", "Body text is lighter than the original."],
    ["Footer should be pinned to the bottom edge.", "Button hover state is missing."],
    ["Heading weight is slightly off.", "Add a visible keyboard focus style."],
    ["Line height of paragraphs is a little tight."],
)  # fmt: skip


def ui2code_critique(round_: int) -> dict[str, Any]:
    """Critique of draft ``round_`` (0 = first draft): falling issue counts, rising score."""
    issues = _CRITIQUES[min(round_, len(_CRITIQUES) - 1)] if round_ < len(_CRITIQUES) else []
    score = round(min(0.97, 0.6 + 0.14 * (round_ + 1)), 2)
    return {"score": score, "issues": list(issues), "done": not issues}


def ui2code(img: PrismImage, round_: int = 0) -> str:
    """A small, valid, self-contained page sized and coloured like ``img``.

    Round ``n`` applies the fixes the simulated critique of round ``n-1`` asked for.
    """
    c = [*img.colors(3), "#2B2F4A", "#6C7CFF"]
    bg, fg_panel, accent = c[0], c[1], c[2]
    ink = _on(bg)
    panel_ink = _on(fg_panel)
    accent_ink = _on(accent)
    width = min(max(img.width, 320), 1200)
    radius = 6 + 4 * min(round_, 2)
    gap = 16 + 4 * min(round_, 2)
    name = _html.escape(img.name)
    extra = []
    if round_ >= 1:
        extra.append(
            "footer { margin-top: auto; }\n    .btn:hover { filter: brightness(1.08); transform: translateY(-1px); }"
        )
    if round_ >= 2:
        extra.append(
            "a:focus-visible, .btn:focus-visible { outline: 3px solid currentColor; outline-offset: 2px; }\n    h1 { font-weight: 800; }"
        )
    if round_ >= 3:
        extra.append("p { line-height: 1.65; }")
    extra_css = "\n    ".join(extra)
    cards = "\n".join(
        f'      <article class="card"><h2>{t}</h2><p>{d}</p></article>'
        for t, d in (
            ("Overview", "A summary block sized from the source screenshot."),
            ("Details", "Supporting content laid out in a responsive grid."),
            ("Actions", "Primary and secondary controls live here."),
        )
    )
    return f"""<!doctype html>
<!-- SIMULATED by NIM Lab Prism (offline mock). Pass a real NVIDIA API key for model-written code. -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="nimlab-prism (simulated), draft {round_ + 1}">
  <title>Simulated UI from {name}</title>
  <style>
    :root {{ --bg: {bg}; --panel: {fg_panel}; --accent: {accent}; --ink: {ink}; --gap: {gap}px; --radius: {radius}px; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; display: flex; flex-direction: column; background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; }}
    header, main, footer {{ width: 100%; max-width: {width}px; margin: 0 auto; padding: var(--gap); }}
    header {{ display: flex; align-items: center; justify-content: space-between; }}
    nav a {{ color: inherit; margin-left: var(--gap); text-decoration: none; opacity: .85; }}
    .hero {{ padding: calc(var(--gap) * 2) 0; }}
    .hero h1 {{ margin: 0 0 .25em; font-size: clamp(1.8rem, 4vw, 2.8rem); }}
    .btn {{ display: inline-block; padding: .6em 1.2em; border-radius: var(--radius); background: var(--accent); color: {accent_ink}; text-decoration: none; font-weight: 600; transition: filter .15s, transform .15s; }}
    .grid {{ display: grid; gap: var(--gap); grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
    .card {{ background: var(--panel); color: {panel_ink}; padding: var(--gap); border-radius: var(--radius); }}
    .card h2 {{ margin-top: 0; font-size: 1.1rem; }}
    footer {{ opacity: .7; font-size: .875rem; }}
    {extra_css}
  </style>
</head>
<body>
  <header>
    <strong>Prism</strong>
    <nav><a href="#">Home</a><a href="#">Docs</a><a href="#">Contact</a></nav>
  </header>
  <main>
    <section class="hero">
      <h1>{img.width} &times; {img.height} layout</h1>
      <p>Generated offline from the image's size and colours. This page is a placeholder for the real conversion.</p>
      <a class="btn" href="#">Get started</a>
    </section>
    <section class="grid">
{cards}
    </section>
  </main>
  <footer>Simulated output &middot; draft {round_ + 1}</footer>
</body>
</html>
"""
