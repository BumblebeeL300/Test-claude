#!/usr/bin/env python3
"""Build the NIM Lab Studio app icon from its SVG master.

The master is ``src/nimlab/studio/web/app-icon.svg`` (1024 x 1024: an ink faceplate on the macOS
Big Sur+ grid with three patch cables converging on a glowing jack in a hex core, risograph grain
and a cyan/magenta misregistration on the core). ``src/nimlab/studio/web/favicon.svg`` is the
simplified geometry; it is used for the 16 and 32 px renders (so those are not downscales).

Outputs, written into ``--out`` (default: next to this script):

* ``AppIcon-1024.png``   the 1024 master render
* ``AppIcon.iconset/``   every iconset size (16 .. 512 plus @2x)
* ``AppIcon.icns``       ``iconutil`` on macOS, otherwise a built-in pure-Python ICNS writer
* ``docs/images/icon.png``  256 px, optimised (``--no-docs`` skips)

Renderers, tried in order (``--renderer`` forces one):

1. ``playwright``  headless Chromium screenshot with a transparent background (``pip install
   playwright && playwright install chromium``). Handles the grain filter and blend modes.
2. ``cairosvg``    ``pip install cairosvg``. Fallback: it ignores ``feTurbulence`` and blend modes,
   so the grain and the misregistration glow are lost, but the icon still builds.
3. ``rsvg-convert`` / ``resvg`` on PATH: same limitation as cairosvg.

Nothing here is a runtime dependency of nimlab; run it by hand when the artwork changes (the
generated files are committed). Pillow is used only to optimise ``docs/images/icon.png``.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WEB = REPO / "src" / "nimlab" / "studio" / "web"
MASTER = WEB / "app-icon.svg"
SMALL = WEB / "favicon.svg"
SMALL_MAX = 32  # renders at or below this many pixels use the hand-simplified favicon geometry

ICONSET_SIZES = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32), ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64), ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512), ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]  # fmt: skip
# ICNS PNG-payload element types -> pixel size (the OSType names the logical size and scale).
ICNS_TYPES = [
    (b"icp4", 16), (b"icp5", 32), (b"icp6", 64), (b"ic07", 128), (b"ic08", 256), (b"ic09", 512),
    (b"ic10", 1024), (b"ic11", 32), (b"ic12", 64), (b"ic13", 256), (b"ic14", 512),
]  # fmt: skip


def _page(svg: str, size: int, pad: int) -> str:
    return (
        '<!doctype html><html><body style="margin:0;background:transparent">'
        f"<style>svg{{position:absolute;left:{pad}px;top:{pad}px;"
        f"width:{size - 2 * pad}px;height:{size - 2 * pad}px}}</style>{svg}</body></html>"
    )


class PlaywrightRenderer:
    name = "playwright"

    def __init__(self) -> None:
        from playwright.sync_api import sync_playwright

        self._cm = sync_playwright()
        pw = self._pw = self._cm.start()
        try:
            self._browser = pw.chromium.launch()
        except Exception:  # browsers installed for another playwright version: find any chromium
            exe = next(
                (
                    p
                    for root in (os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "/opt/pw-browsers")
                    for p in glob.glob(f"{root}/chromium*/*/chrome*")
                    if root and os.path.isfile(p) and os.access(p, os.X_OK)
                ),
                None,
            )
            if exe is None:
                raise
            self._browser = pw.chromium.launch(executable_path=exe)

    def render(self, svg: str, size: int, pad: int = 0) -> bytes:
        page = self._browser.new_page(viewport={"width": size, "height": size})
        page.set_content(_page(svg, size, pad))
        page.wait_for_timeout(150)
        png = page.screenshot(omit_background=True)
        page.close()
        return png

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()


class CairoRenderer:
    name = "cairosvg"

    def __init__(self) -> None:
        import cairosvg  # noqa: F401

    def render(self, svg: str, size: int, pad: int = 0) -> bytes:
        import cairosvg

        inner = size - 2 * pad
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=inner, output_height=inner)
        if not pad:
            return png
        from io import BytesIO

        from PIL import Image

        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(Image.open(BytesIO(png)), (pad, pad))
        buf = BytesIO()
        canvas.save(buf, "PNG")
        return buf.getvalue()

    def close(self) -> None:
        pass


class CliRenderer:
    name = "rsvg/resvg"

    def __init__(self) -> None:
        self.exe = shutil.which("rsvg-convert") or shutil.which("resvg")
        if not self.exe:
            raise ImportError("neither rsvg-convert nor resvg on PATH")

    def render(self, svg: str, size: int, pad: int = 0) -> bytes:
        inner = size - 2 * pad
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td, "i.svg"), Path(td, "o.png")
            src.write_text(svg)
            if Path(self.exe).name == "rsvg-convert":
                cmd = [self.exe, "-w", str(inner), "-h", str(inner), "-o", str(dst), str(src)]
            else:
                cmd = [self.exe, "-w", str(inner), "-h", str(inner), str(src), str(dst)]
            subprocess.run(cmd, check=True)
            png = dst.read_bytes()
        if not pad:
            return png
        from io import BytesIO

        from PIL import Image

        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(Image.open(BytesIO(png)), (pad, pad))
        buf = BytesIO()
        canvas.save(buf, "PNG")
        return buf.getvalue()

    def close(self) -> None:
        pass


def get_renderer(prefer: str | None):  # type: ignore[no-untyped-def]
    order = {"playwright": PlaywrightRenderer, "cairosvg": CairoRenderer, "cli": CliRenderer}
    names = [prefer] if prefer else list(order)
    errors = []
    for n in names:
        try:
            return order[n]()
        except Exception as exc:  # any failure means "try the next renderer"
            errors.append(f"{n}: {exc}")
    raise SystemExit("no SVG renderer available:\n  " + "\n  ".join(errors))


def render_size(r, size: int, master: str, small: str) -> bytes:  # type: ignore[no-untyped-def]
    """Full artwork above SMALL_MAX; the simplified geometry (with a hinted 1 px margin) below."""
    if size <= SMALL_MAX:
        return r.render(small, size, pad=1)
    return r.render(master, size)


def write_icns(path: Path, pngs: dict[int, bytes]) -> None:
    """Pure-Python ICNS: each element is an OSType, a big-endian length and a PNG payload."""
    body = b"".join(t + struct.pack(">I", 8 + len(pngs[px])) + pngs[px] for t, px in ICNS_TYPES)
    path.write_bytes(b"icns" + struct.pack(">I", 8 + len(body)) + body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=HERE, help="output directory")
    ap.add_argument("--renderer", choices=["playwright", "cairosvg", "cli"])
    ap.add_argument("--no-docs", action="store_true", help="skip docs/images/icon.png")
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    master, small = MASTER.read_text(), SMALL.read_text()

    r = get_renderer(args.renderer)
    print(f"renderer: {r.name}")
    try:
        cache: dict[int, bytes] = {}
        for px in sorted({p for _, p in ICONSET_SIZES} | {p for _, p in ICNS_TYPES}):
            cache[px] = render_size(r, px, master, small)
    finally:
        r.close()

    (out / "AppIcon-1024.png").write_bytes(cache[1024])
    iconset = out / "AppIcon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    for name, px in ICONSET_SIZES:
        (iconset / name).write_bytes(cache[px])

    icns = out / "AppIcon.icns"
    if sys.platform == "darwin" and shutil.which("iconutil"):
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
        print(f"wrote {icns} (iconutil)")
    else:
        write_icns(icns, cache)
        print(f"wrote {icns} (built-in writer; on macOS this script uses iconutil)")
    shutil.rmtree(iconset)  # the .icns is the deliverable; the iconset is regenerable scratch
    print(f"wrote {out / 'AppIcon-1024.png'}")

    docs = REPO / "docs" / "images"
    if not args.no_docs and docs.is_dir() and out == HERE:
        png = docs / "icon.png"
        try:
            from io import BytesIO

            from PIL import Image

            Image.open(BytesIO(cache[256])).save(png, optimize=True)
        except ImportError:
            png.write_bytes(cache[256])
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
