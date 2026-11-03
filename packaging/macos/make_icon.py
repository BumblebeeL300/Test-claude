#!/usr/bin/env python3
"""Generate the NIM Lab Studio app icon: a 1024 px "N" mark on deep ink.

Outputs (next to this script, or into --out):

* ``AppIcon.png``   1024 x 1024 master (always written)
* ``AppIcon.iconset/``  the standard sizes (when Pillow is available)
* ``AppIcon.icns``  built with ``iconutil`` on macOS, or by Pillow's ICNS writer elsewhere

The glyph is an acid-chartreuse to magenta gradient on #0B0A12. Pillow gives anti-aliased output;
without Pillow a small pure-Python rasteriser writes a (hard-edged) PNG so the build never fails
just because of the icon.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

SIZE = 1024
INK = (11, 10, 18)
CHARTREUSE = (198, 255, 61)
GOLD = (255, 214, 102)  # mid stop: keeps the blend vivid instead of muddy brown
MAGENTA = (255, 62, 165)
TILE_RADIUS = 0.2237  # macOS "squircle" corner radius as a fraction of the tile (approximation)
TILE_INSET = 0.098  # Apple's icon grid leaves ~10% margin around the tile

Point = tuple[float, float]


def glyph_polygons(size: int) -> list[list[Point]]:
    """The "N": two vertical bars and a diagonal, plus a small spark. Coordinates in pixels."""
    s = size / 1024
    left, right = 330 * s, 694 * s
    top, bottom = 290 * s, 734 * s
    bar = 92 * s
    n_left = [(left, top), (left + bar, top), (left + bar, bottom), (left, bottom)]
    n_right = [(right - bar, top), (right, top), (right, bottom), (right - bar, bottom)]
    diag = [(left, top), (left + bar * 1.25, top), (right, bottom), (right - bar * 1.25, bottom)]
    cx, cy, r = 742 * s, 262 * s, 54 * s  # spark: a diamond above the right bar
    spark = [(cx, cy - r * 1.5), (cx + r, cy), (cx, cy + r * 1.5), (cx - r, cy)]
    return [n_left, diag, n_right, spark]


def gradient(t: float) -> tuple[int, int, int]:
    """Chartreuse -> gold -> magenta along ``t`` in [0, 1]."""
    t = min(1.0, max(0.0, t))
    a, b, u = (CHARTREUSE, GOLD, t * 2) if t < 0.5 else (GOLD, MAGENTA, t * 2 - 1)
    return tuple(round(x + (y - x) * u) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def glyph_t(size: int):  # type: ignore[no-untyped-def]
    """``t(x, y)``: a diagonal ramp across the glyph's bounding box (so both ends are saturated)."""
    s = size / 1024
    lo, hi = (330 + 290) * s, (694 + 734) * s
    return lambda x, y: (x + y - lo) / (hi - lo)


def render_pillow(size: int):  # type: ignore[no-untyped-def]
    from PIL import Image, ImageDraw, ImageFilter

    ss = 3  # supersample for smooth edges
    big = size * ss
    tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    inset = round(big * TILE_INSET)
    box = (inset, inset, big - inset, big - inset)
    radius = round((big - 2 * inset) * TILE_RADIUS)
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle(box, radius=radius, fill=(*INK, 255))
    # soft magenta glow behind the glyph so the tile is not a flat slab (clipped to the tile)
    glow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        (big * 0.22, big * 0.22, big * 0.78, big * 0.78), fill=(*MAGENTA, 70)
    )
    glow = glow.filter(ImageFilter.GaussianBlur(big * 0.07))
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    tile = Image.composite(Image.alpha_composite(tile, glow), tile, mask)

    # the glyph, filled with a diagonal chartreuse -> magenta gradient
    gmask = Image.new("L", (big, big), 0)
    gdraw = ImageDraw.Draw(gmask)
    for poly in glyph_polygons(big):
        gdraw.polygon(poly, fill=255)
    ramp = glyph_t(big)
    grad = Image.new("RGBA", (big, big))
    px = grad.load()
    ys = range(round(big * 0.1), round(big * 0.85))
    for y in ys:
        for x in range(round(big * 0.25), round(big * 0.85)):
            px[x, y] = (*gradient(ramp(x, y)), 255)
    tile.paste(grad, (0, 0), gmask)
    return tile.resize((size, size), Image.LANCZOS)


def _fill_polygon(rows: list[bytearray], poly: list[Point], colour_at, size: int) -> None:  # type: ignore[no-untyped-def]
    """Even-odd scanline fill into RGBA rows (no anti-aliasing)."""
    ys = [p[1] for p in poly]
    for y in range(max(0, int(min(ys))), min(size, int(max(ys)) + 1)):
        xs: list[float] = []
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1], strict=True):
            if (y0 <= y + 0.5 < y1) or (y1 <= y + 0.5 < y0):
                xs.append(x0 + (y + 0.5 - y0) * (x1 - x0) / (y1 - y0))
        xs.sort()
        for xa, xb in zip(xs[::2], xs[1::2], strict=False):
            for x in range(max(0, round(xa)), min(size, round(xb))):
                r, g, b = colour_at(x, y)
                rows[y][x * 4 : x * 4 + 4] = bytes((r, g, b, 255))


def render_fallback_png(size: int) -> bytes:
    """Pure-Python PNG: ink squircle-ish tile with the gradient glyph."""
    inset = round(size * TILE_INSET)
    radius = round((size - 2 * inset) * TILE_RADIUS)
    rows = [bytearray(size * 4) for _ in range(size)]
    lo, hi = inset, size - inset
    for y in range(lo, hi):
        for x in range(lo, hi):
            dx = max(lo + radius - x, 0, x - (hi - radius - 1))
            dy = max(lo + radius - y, 0, y - (hi - radius - 1))
            if dx * dx + dy * dy <= radius * radius:
                rows[y][x * 4 : x * 4 + 4] = bytes((*INK, 255))
    ramp = glyph_t(size)
    for poly in glyph_polygons(size):
        _fill_polygon(rows, poly, lambda x, y: gradient(ramp(x, y)), size)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


ICONSET_SIZES = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32), ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64), ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512), ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).parent, help="output directory")
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    png = out / "AppIcon.png"

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow not found: writing a basic PNG only (pip install pillow for the full icon).")
        png.write_bytes(render_fallback_png(SIZE))
        print(f"wrote {png}")
        return 0

    master = render_pillow(SIZE)
    master.save(png)
    iconset = out / "AppIcon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    from PIL import Image

    for name, px in ICONSET_SIZES:
        master.resize((px, px), Image.LANCZOS).save(iconset / name)
    icns = out / "AppIcon.icns"
    if sys.platform == "darwin" and shutil.which("iconutil"):
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
        print(f"wrote {icns} (iconutil)")
    else:
        try:
            master.save(icns)  # Pillow's ICNS writer: good enough to test the build off-macOS
            print(f"wrote {icns} (Pillow; on macOS this script uses iconutil)")
        except Exception as exc:  # pragma: no cover
            print(f"could not write {icns}: {exc}; the PNG is still available")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
