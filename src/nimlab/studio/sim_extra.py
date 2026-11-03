"""More simulated Forge projects: a URL shortener API, markdown tables, unit conversion, passwords,
a text adventure, CSV statistics and a notes CLI, plus a generic module named after the prompt.

Same contract as :mod:`nimlab.studio.sim_projects`: the correct module, one planted bug, and a test
suite in which exactly one test catches it. Sources are raw strings so regexes stay readable.
"""

from __future__ import annotations

import re

from .sim_projects import ProjectTemplate

__all__ = ["EXTRA_RULES", "generic_template", "prompt_words", "short_prompt"]

# --------------------------------------------------------------------------- URL shortener
_SHORTENER = ProjectTemplate(
    key="shortener",
    module="shortener",
    goal="A URL shortener with a stdlib HTTP API: POST /shorten and GET /<code> redirects",
    steps=(
        ("Design the API: POST /shorten, GET /<code> -> 302, validation rules", ("README.md",)),
        (
            "Implement base62 codes, the Shortener store and the http.server handler",
            ("shortener.py",),
        ),
        (
            "Test encoding, validation, de-duplication and a live server round trip",
            ("test_shortener.py",),
        ),
    ),
    source=r'''"""A tiny URL shortener with a standard-library HTTP API."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode(n: int) -> str:
    """Base62-encode a non-negative integer."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return ALPHABET[0]
    digits = []
    while n:
        n, rest = divmod(n, len(ALPHABET))
        digits.append(ALPHABET[rest])
    return "".join(reversed(digits))


class Shortener:
    """In-memory store mapping short codes to URLs."""

    def __init__(self) -> None:
        self._by_code: dict[str, str] = {}
        self._by_url: dict[str, str] = {}
        self._next = 1

    def shorten(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"not an http(s) URL: {url!r}")
        if url in self._by_url:
            return self._by_url[url]
        code = encode(self._next)
        self._next += 1
        self._by_code[code] = url
        self._by_url[url] = code
        return code

    def resolve(self, code: str) -> str:
        return self._by_code[code]


def make_server(shortener: Shortener, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """An HTTP server (port 0 picks a free one) exposing ``shortener``."""

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/shorten":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                url = json.loads(self.rfile.read(length) or b"{}")["url"]
                code = shortener.shorten(url)
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": "send JSON like {\"url\": \"https://...\"}"})
            self._json(201, {"code": code, "short": f"/{code}"})

        def do_GET(self) -> None:
            try:
                target = shortener.resolve(self.path.lstrip("/"))
            except KeyError:
                return self._json(404, {"error": "unknown code"})
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    server = make_server(Shortener(), port=8000)
    print("listening on", server.server_address)
    server.serve_forever()
''',
    good='    return "".join(reversed(digits))\n',
    bad='    return "".join(digits)\n',
    tests=r"""import http.client
import json
import threading

import pytest

from shortener import Shortener, encode, make_server


def test_encode_small_values():
    assert encode(0) == "0"
    assert encode(10) == "a"
    assert encode(61) == "Z"


def test_encode_multi_digit_is_most_significant_first():
    assert encode(62) == "10"
    assert encode(62 * 62 + 1) == "101"


def test_shorten_is_stable_and_deduplicated():
    s = Shortener()
    first = s.shorten("https://example.com/a")
    assert s.shorten("https://example.com/a") == first
    assert s.shorten("https://example.com/b") != first
    assert s.resolve(first) == "https://example.com/a"


def test_rejects_non_http_urls():
    with pytest.raises(ValueError):
        Shortener().shorten("ftp://example.com/file")
    with pytest.raises(ValueError):
        Shortener().shorten("not a url")


def test_http_round_trip():
    server = make_server(Shortener())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/shorten", body=json.dumps({"url": "https://example.com/x"}))
        resp = conn.getresponse()
        code = json.loads(resp.read())["code"]
        assert resp.status == 201
        conn.request("GET", "/" + code)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 302 and resp.getheader("Location") == "https://example.com/x"
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 404
        conn.request("POST", "/shorten", body="{}")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 400
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
""",
    failing_test="test_shortener.py::test_encode_multi_digit_is_most_significant_first",
    readme='# URL shortener\n\nStandard-library only.\n\n- `POST /shorten` with `{"url": "https://..."}` returns `{"code": ...}`\n- `GET /<code>` answers `302` with a `Location` header\n\nRun `python shortener.py`, test with `pytest`.\n',
    risks=("base62 digits must come out most significant first", "only http(s) URLs are accepted"),
    review_note="The API surface is tiny, validated and tested end to end against a live server.",
)

# --------------------------------------------------------------------------- markdown tables
_MDTABLE = ProjectTemplate(
    key="mdtable",
    module="mdtable",
    goal="A markdown table parser: parse_table() to row dicts, alignments() and to_markdown()",
    steps=(
        (
            "Pin down the GitHub table syntax we support (alignment row, escaped pipes)",
            ("README.md",),
        ),
        ("Implement split_row(), parse_table(), alignments() and to_markdown()", ("mdtable.py",)),
        (
            "Test plain tables, alignment, escaped pipes, short rows and errors",
            ("test_mdtable.py",),
        ),
    ),
    source=r'''"""Parse and render GitHub-flavoured markdown tables."""

from __future__ import annotations

import re

_SEPARATOR = re.compile(r"\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*")


def split_row(line: str) -> list[str]:
    """Split one table row into trimmed cells; ``\|`` is a literal pipe."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def _lines(text: str) -> list[str]:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2 or not _SEPARATOR.fullmatch(lines[1]):
        raise ValueError("not a markdown table: the second line must be a --- separator row")
    return lines


def parse_table(text: str) -> list[dict[str, str]]:
    """Rows as dicts keyed by header; short rows are padded, long rows truncated."""
    lines = _lines(text)
    headers = split_row(lines[0])
    rows = []
    for line in lines[2:]:
        cells = (split_row(line) + [""] * len(headers))[: len(headers)]
        rows.append(dict(zip(headers, cells)))
    return rows


def alignments(text: str) -> list[str]:
    """``left``, ``right`` or ``center`` for each column (``left`` is the default)."""
    out = []
    for cell in split_row(_lines(text)[1]):
        if cell.startswith(":") and cell.endswith(":"):
            out.append("center")
        elif cell.endswith(":"):
            out.append("right")
        else:
            out.append("left")
    return out


def to_markdown(rows: list[dict[str, str]]) -> str:
    """Render rows back to an aligned markdown table."""
    if not rows:
        raise ValueError("no rows to render")
    headers = list(rows[0])
    esc = lambda value: str(value).replace("|", "\\|")  # noqa: E731
    widths = [max(len(h), *(len(esc(r.get(h, ""))) for r in rows)) for h in headers]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"  # noqa: E731
    out = [line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    out += [line([esc(r.get(h, "")) for h in headers]) for r in rows]
    return "\n".join(out)
''',
    good='    return [cell.strip().replace("\\\\|", "|") for cell in cells]\n',
    bad="    return [cell.strip() for cell in cells]\n",
    tests=r'''import pytest

from mdtable import alignments, parse_table, split_row, to_markdown

TABLE = """
| Name  | Qty | Price |
| :---- | --: | :---: |
| Apple |   3 |  1.50 |
| Pear  |  10 |  0.75 |
"""


def test_parse_rows():
    rows = parse_table(TABLE)
    assert rows == [
        {"Name": "Apple", "Qty": "3", "Price": "1.50"},
        {"Name": "Pear", "Qty": "10", "Price": "0.75"},
    ]


def test_alignments():
    assert alignments(TABLE) == ["left", "right", "center"]


def test_escaped_pipe_is_a_literal_pipe():
    assert split_row(r"| a \| b | c |") == ["a | b", "c"]


def test_short_and_long_rows():
    rows = parse_table("a | b\n--- | ---\n1\n1 | 2 | 3")
    assert rows == [{"a": "1", "b": ""}, {"a": "1", "b": "2"}]


def test_rejects_text_without_separator():
    with pytest.raises(ValueError):
        parse_table("just\nsome text")


def test_round_trip():
    rows = parse_table(TABLE)
    assert parse_table(to_markdown(rows)) == rows
''',
    failing_test="test_mdtable.py::test_escaped_pipe_is_a_literal_pipe",
    readme="# mdtable\n\nParse markdown tables into dicts and render them back.\n\n```python\nfrom mdtable import parse_table\nrows = parse_table(open('table.md').read())\n```\n",
    risks=("escaped pipes inside cells", "rows with too few or too many cells"),
    review_note="Parser and renderer round-trip, and the error path is explicit.",
)

# --------------------------------------------------------------------------- unit converter
_UNITS = ProjectTemplate(
    key="units",
    module="units",
    goal="A unit converter for length, mass, volume and temperature with clear errors",
    steps=(
        (
            "List the supported units and the conversion approach (base unit per family)",
            ("README.md",),
        ),
        ("Implement convert() with linear families and special-cased temperature", ("units.py",)),
        ("Test known values, round trips, temperature and error cases", ("test_units.py",)),
    ),
    source=r'''"""Convert between common units."""

from __future__ import annotations

_LINEAR: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254},
    "mass": {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125},
    "volume": {"l": 1.0, "ml": 0.001, "gal": 3.785411784, "cup": 0.2365882365},
}
_TEMPERATURE = ("c", "f", "k")


def _family(unit: str) -> str:
    if unit in _TEMPERATURE:
        return "temperature"
    for name, table in _LINEAR.items():
        if unit in table:
            return name
    raise ValueError(f"unknown unit: {unit!r}")


def _to_celsius(value: float, unit: str) -> float:
    if unit == "f":
        return (value - 32) * 5 / 9
    if unit == "k":
        return value - 273.15
    return value


def _from_celsius(value: float, unit: str) -> float:
    if unit == "f":
        return value * 9 / 5 + 32
    if unit == "k":
        return value + 273.15
    return value


def convert(value: float, source: str, target: str) -> float:
    """Convert ``value`` from ``source`` to ``target`` (unit names are case-insensitive)."""
    source, target = source.lower(), target.lower()
    family = _family(source)
    if family != _family(target):
        raise ValueError(f"cannot convert {family} ({source}) to {_family(target)} ({target})")
    if family == "temperature":
        return _from_celsius(_to_celsius(value, source), target)
    table = _LINEAR[family]
    return value * table[source] / table[target]
''',
    good="        return (value - 32) * 5 / 9\n",
    bad="        return value - 32 * 5 / 9\n",
    tests=r"""import pytest

from units import convert


def test_length():
    assert convert(1, "km", "m") == pytest.approx(1000)
    assert convert(1, "mi", "km") == pytest.approx(1.609344)
    assert convert(12, "in", "ft") == pytest.approx(1)


def test_mass_and_volume():
    assert convert(1, "lb", "g") == pytest.approx(453.59237)
    assert convert(1, "gal", "ml") == pytest.approx(3785.411784)


def test_fahrenheit_to_celsius():
    assert convert(212, "F", "C") == pytest.approx(100)
    assert convert(-40, "f", "c") == pytest.approx(-40)


def test_celsius_to_fahrenheit_and_kelvin():
    assert convert(100, "c", "f") == pytest.approx(212)
    assert convert(0, "c", "k") == pytest.approx(273.15)


def test_round_trip():
    assert convert(convert(42.5, "ft", "m"), "m", "ft") == pytest.approx(42.5)


def test_errors():
    with pytest.raises(ValueError):
        convert(1, "kg", "m")
    with pytest.raises(ValueError):
        convert(1, "parsec", "m")
""",
    failing_test="test_units.py::test_fahrenheit_to_celsius",
    readme="# units\n\n```python\nfrom units import convert\nconvert(212, 'F', 'C')   # 100.0\nconvert(1, 'mi', 'km')   # 1.609344\n```\n\nFamilies: length, mass, volume, temperature.\n",
    risks=("temperature is not a pure scale factor", "mixing families must fail loudly"),
    review_note="Linear families share one code path; temperature is isolated and tested both ways.",
)

# --------------------------------------------------------------------------- password generator
_PASSWORD = ProjectTemplate(
    key="password",
    module="passgen",
    goal="A password generator that guarantees every enabled character class, plus an entropy estimate",
    steps=(
        ("Decide the options (length, classes) and what 'strong' means", ("README.md",)),
        ("Implement generate(), entropy_bits() and strength()", ("passgen.py",)),
        ("Test length, class coverage, option errors and the entropy math", ("test_passgen.py",)),
    ),
    source=r'''"""Generate random passwords and estimate their strength."""

from __future__ import annotations

import math
import random
import secrets
import string

SYMBOLS = "!@#$%^&*()-_=+[]{};:,.?"


def generate(
    length: int = 16,
    *,
    lower: bool = True,
    upper: bool = True,
    digits: bool = True,
    symbols: bool = True,
    rng: random.Random | None = None,
) -> str:
    """A random password with at least one character from every enabled class."""
    pools = [
        pool
        for pool, enabled in (
            (string.ascii_lowercase, lower),
            (string.ascii_uppercase, upper),
            (string.digits, digits),
            (SYMBOLS, symbols),
        )
        if enabled
    ]
    if not pools:
        raise ValueError("enable at least one character class")
    if length < len(pools):
        raise ValueError(f"length must be at least {len(pools)} for the chosen classes")
    rng = rng or secrets.SystemRandom()
    chars = [rng.choice(pool) for pool in pools]
    allowed = "".join(pools)
    chars += [rng.choice(allowed) for _ in range(length - len(chars))]
    rng.shuffle(chars)
    return "".join(chars)


def entropy_bits(password: str) -> float:
    """Rough entropy: length * log2(size of the character classes used)."""
    size = 0
    if any(c in string.ascii_lowercase for c in password):
        size += 26
    if any(c in string.ascii_uppercase for c in password):
        size += 26
    if any(c in string.digits for c in password):
        size += 10
    if any(c not in string.ascii_letters + string.digits for c in password):
        size += len(SYMBOLS)
    return len(password) * math.log2(size) if size else 0.0


def strength(password: str) -> str:
    bits = entropy_bits(password)
    if bits < 40:
        return "weak"
    if bits < 60:
        return "fair"
    if bits < 80:
        return "strong"
    return "excellent"
''',
    good="    chars += [rng.choice(allowed) for _ in range(length - len(chars))]\n",
    bad="    chars += [rng.choice(allowed) for _ in range(length)]\n",
    tests=r"""import random
import string

import pytest

from passgen import SYMBOLS, entropy_bits, generate, strength


def test_exact_length():
    for n in (4, 8, 16, 40):
        assert len(generate(n, rng=random.Random(n))) == n


def test_every_enabled_class_is_present():
    for seed in range(20):
        pw = generate(6, rng=random.Random(seed))
        assert any(c in string.ascii_lowercase for c in pw)
        assert any(c in string.ascii_uppercase for c in pw)
        assert any(c in string.digits for c in pw)
        assert any(c in SYMBOLS for c in pw)


def test_disabled_classes_are_absent():
    pw = generate(30, symbols=False, digits=False, rng=random.Random(1))
    assert pw.isalpha()


def test_same_seed_same_password():
    assert generate(rng=random.Random(5)) == generate(rng=random.Random(5))


def test_option_errors():
    with pytest.raises(ValueError):
        generate(8, lower=False, upper=False, digits=False, symbols=False)
    with pytest.raises(ValueError):
        generate(2)


def test_entropy_and_strength():
    assert entropy_bits("abcd") == pytest.approx(4 * 4.700439718141092)
    assert entropy_bits("") == 0.0
    assert strength("abc") == "weak"
    assert strength(generate(24, rng=random.Random(2))) == "excellent"
""",
    failing_test="test_passgen.py::test_exact_length",
    readme="# passgen\n\n```python\nfrom passgen import generate, strength\npw = generate(20)\nprint(pw, strength(pw))\n```\n\nUses `secrets` by default; pass `rng=random.Random(seed)` for tests.\n",
    risks=(
        "each enabled class must appear at least once",
        "never use random.random() for real passwords",
    ),
    review_note="Uses the secrets module by default and keeps the seedable path for tests.",
)

# --------------------------------------------------------------------------- text adventure
_ADVENTURE = ProjectTemplate(
    key="adventure",
    module="adventure",
    goal="A tiny text adventure: three rooms, a key, a locked door and a win condition",
    steps=(
        ("Sketch the map, items and commands (look, go, take, inventory)", ("README.md",)),
        ("Implement the Game state machine and a command loop", ("adventure.py",)),
        ("Test movement, picking things up, the locked door and winning", ("test_adventure.py",)),
    ),
    source=r'''"""A three-room text adventure."""

from __future__ import annotations

ROOMS = {
    "cellar": {"desc": "A damp cellar. Stairs lead up.", "exits": {"up": "hall"}, "items": ["lamp"]},
    "hall": {
        "desc": "A dusty hall. An oak door leads east.",
        "exits": {"down": "cellar", "east": "vault"},
        "items": ["key"],
    },
    "vault": {"desc": "The vault glitters.", "exits": {"west": "hall"}, "items": []},
}
COMMANDS = {
    "look": "_look",
    "l": "_look",
    "go": "_go",
    "take": "_take",
    "get": "_take",
    "inventory": "_inventory",
    "i": "_inventory",
}


class Game:
    def __init__(self) -> None:
        self.room = "cellar"
        self.inventory: list[str] = []
        self.items = {name: list(room["items"]) for name, room in ROOMS.items()}
        self.won = False

    def do(self, command: str) -> str:
        """Run one command and return the text to show the player."""
        words = command.lower().split()
        if not words:
            return "Say something."
        verb, arg = words[0], " ".join(words[1:])
        handler = COMMANDS.get(verb)
        if handler is None:
            return f"I don't understand '{verb}'."
        return getattr(self, handler)(arg)

    def _look(self, arg: str) -> str:
        room = ROOMS[self.room]
        text = str(room["desc"])
        if self.items[self.room]:
            text += " You see: " + ", ".join(self.items[self.room]) + "."
        return text + " Exits: " + ", ".join(sorted(room["exits"])) + "."

    def _go(self, arg: str) -> str:
        exits = ROOMS[self.room]["exits"]
        if arg not in exits:
            return f"You can't go {arg or 'nowhere'}."
        destination = exits[arg]
        if destination == "vault" and "key" not in self.inventory:
            return "The oak door is locked."
        self.room = destination
        if destination == "vault":
            self.won = True
            return "The door creaks open. Treasure! You win!"
        return self._look("")

    def _take(self, arg: str) -> str:
        if arg not in self.items[self.room]:
            return f"There is no {arg or 'thing'} here."
        self.items[self.room].remove(arg)
        self.inventory.append(arg)
        return f"Taken: {arg}."

    def _inventory(self, arg: str) -> str:
        return "You carry: " + ", ".join(self.inventory) + "." if self.inventory else "You carry nothing."


def main() -> None:
    game = Game()
    print(game.do("look"))
    while not game.won:
        try:
            print(game.do(input("> ")))
        except EOFError:
            break


if __name__ == "__main__":
    main()
''',
    good="        self.items[self.room].remove(arg)\n        self.inventory.append(arg)\n",
    bad="        self.inventory.append(arg)\n",
    tests=r"""from adventure import Game


def play(*commands):
    game = Game()
    return game, [game.do(c) for c in commands]


def test_look_lists_items_and_exits():
    _, out = play("look")
    assert "lamp" in out[0] and "up" in out[0]


def test_movement_and_bad_direction():
    game, out = play("go north", "go up")
    assert "can't go north" in out[0]
    assert game.room == "hall"


def test_taking_moves_the_item_into_the_inventory():
    game, out = play("take lamp", "inventory", "look", "take lamp")
    assert out[0] == "Taken: lamp."
    assert "lamp" in out[1]
    assert "lamp" not in out[2].split("Exits")[0].replace("A damp cellar", "")
    assert "no lamp" in out[3]


def test_locked_door_needs_the_key():
    game, out = play("go up", "go east")
    assert "locked" in out[1] and game.room == "hall" and not game.won


def test_winning():
    game, out = play("go up", "take key", "go east")
    assert game.won and "win" in out[2]


def test_unknown_and_empty_commands():
    _, out = play("dance", "")
    assert "don't understand" in out[0] and out[1] == "Say something."
""",
    failing_test="test_adventure.py::test_taking_moves_the_item_into_the_inventory",
    readme="# adventure\n\nThree rooms, one key, one locked door.\n\n```bash\npython adventure.py\n```\n\nCommands: `look`, `go <dir>`, `take <item>`, `inventory`.\n",
    risks=(
        "taking an item must remove it from the room",
        "the vault must stay locked without the key",
    ),
    review_note="The game is a plain state machine, so every rule is directly testable.",
)

# --------------------------------------------------------------------------- CSV statistics
_CSVSTATS = ProjectTemplate(
    key="csvstats",
    module="csvstats",
    goal="A CSV stats tool: detect numeric columns and report count, mean, median, min, max and stdev",
    steps=(
        ("Decide how numeric columns are detected and how blanks are treated", ("README.md",)),
        ("Implement read_columns(), median(), describe() and a CLI report", ("csvstats.py",)),
        (
            "Test parsing, odd and even medians, blanks, text columns and the report",
            ("test_csvstats.py",),
        ),
    ),
    source=r'''"""Summary statistics for the numeric columns of a CSV file."""

from __future__ import annotations

import csv
import io
import statistics
import sys


def read_columns(text: str) -> dict[str, list[float]]:
    """Numeric columns of CSV ``text``; blanks are skipped, text columns are dropped."""
    reader = csv.DictReader(io.StringIO(text))
    raw: dict[str, list[str]] = {name: [] for name in reader.fieldnames or []}
    for row in reader:
        for name in raw:
            raw[name].append((row.get(name) or "").strip())
    columns: dict[str, list[float]] = {}
    for name, cells in raw.items():
        present = [c for c in cells if c]
        try:
            values = [float(c) for c in present]
        except ValueError:
            continue
        if values:
            columns[name] = values
    return columns


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no values")
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def report(text: str) -> str:
    lines = []
    for name, values in read_columns(text).items():
        stats = describe(values)
        lines.append(f"{name}: " + "  ".join(f"{k}={v:.2f}" for k, v in stats.items()))
    return "\n".join(lines) or "no numeric columns"


if __name__ == "__main__":
    print(report(open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()))
''',
    good="    return (ordered[mid - 1] + ordered[mid]) / 2\n",
    bad="    return ordered[mid]\n",
    tests=r"""import pytest

from csvstats import describe, median, read_columns, report

CSV = "name,age,score\nann,30,1.5\nbob,25,\ncy,41,4.5\n"


def test_read_columns_keeps_only_numeric_and_skips_blanks():
    assert read_columns(CSV) == {"age": [30.0, 25.0, 41.0], "score": [1.5, 4.5]}


def test_median_odd():
    assert median([5, 1, 3]) == 3


def test_median_even_averages_the_middle_pair():
    assert median([1, 2, 3, 4]) == 2.5


def test_describe():
    stats = describe([2.0, 4.0, 6.0])
    assert stats["count"] == 3 and stats["mean"] == 4.0 and stats["median"] == 4.0
    assert stats["min"] == 2.0 and stats["max"] == 6.0
    assert stats["stdev"] == pytest.approx(2.0)
    assert describe([7.0])["stdev"] == 0.0


def test_describe_empty_raises():
    with pytest.raises(ValueError):
        describe([])


def test_report():
    out = report(CSV)
    assert out.splitlines()[0].startswith("age: count=3.00")
    assert report("a,b\nx,y\n") == "no numeric columns"
""",
    failing_test="test_csvstats.py::test_median_even_averages_the_middle_pair",
    readme="# csvstats\n\n```bash\npython csvstats.py data.csv\n```\n\nPrints count, mean, median, min, max and stdev for every numeric column.\n",
    risks=("the median of an even-sized column is the mean of the middle pair", "blank cells"),
    review_note="Numeric detection is conservative and the report degrades cleanly on text-only files.",
)

# --------------------------------------------------------------------------- notes CLI
_NOTES = ProjectTemplate(
    key="notes",
    module="notes",
    goal="A notes CLI with a JSON store: add, list, search and delete notes from the terminal",
    steps=(
        ("Define the commands and the on-disk JSON format", ("README.md",)),
        ("Implement the store functions and the argparse CLI", ("notes.py",)),
        ("Test the store, case-insensitive search and the CLI end to end", ("test_notes.py",)),
    ),
    source=r'''"""A small notes CLI backed by a JSON file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def save(path: Path, notes: list[dict]) -> None:
    path.write_text(json.dumps(notes, indent=2))


def add(path: Path, text: str, tags: list[str] | None = None) -> dict:
    if not text.strip():
        raise ValueError("a note cannot be empty")
    notes = load(path)
    note = {"id": max((n["id"] for n in notes), default=0) + 1, "text": text.strip(), "tags": tags or []}
    save(path, [*notes, note])
    return note


def search(path: Path, query: str) -> list[dict]:
    """Notes whose text or tags contain ``query`` (case-insensitive)."""
    needle = query.lower()
    return [
        n for n in load(path) if needle in n["text"].lower() or any(needle in t.lower() for t in n["tags"])
    ]


def remove(path: Path, note_id: int) -> None:
    notes = load(path)
    kept = [n for n in notes if n["id"] != note_id]
    if len(kept) == len(notes):
        raise KeyError(note_id)
    save(path, kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="notes")
    parser.add_argument("--file", type=Path, default=Path(os.environ.get("NOTES_FILE", "notes.json")))
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add")
    p_add.add_argument("text")
    p_add.add_argument("--tag", action="append", default=[])
    sub.add_parser("list")
    sub.add_parser("search").add_argument("query")
    sub.add_parser("delete").add_argument("id", type=int)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "add":
            print(f"added #{add(args.file, args.text, args.tag)['id']}")
        elif args.cmd == "delete":
            remove(args.file, args.id)
            print(f"deleted #{args.id}")
        else:
            notes = load(args.file) if args.cmd == "list" else search(args.file, args.query)
            for n in notes:
                tags = " ".join(f"#{t}" for t in n["tags"])
                print(f"{n['id']:>3}  {n['text']}  {tags}".rstrip())
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    good="    needle = query.lower()\n",
    bad="    needle = query\n",
    tests=r"""import pytest

from notes import add, load, main, remove, search


def test_add_assigns_increasing_ids(tmp_path):
    f = tmp_path / "n.json"
    assert add(f, "buy milk")["id"] == 1
    assert add(f, "call mum", ["family"])["id"] == 2
    assert [n["text"] for n in load(f)] == ["buy milk", "call mum"]


def test_empty_note_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        add(tmp_path / "n.json", "   ")


def test_search_matches_text_and_tags(tmp_path):
    f = tmp_path / "n.json"
    add(f, "buy milk")
    add(f, "call mum", ["family"])
    assert [n["id"] for n in search(f, "milk")] == [1]
    assert [n["id"] for n in search(f, "family")] == [2]


def test_search_is_case_insensitive(tmp_path):
    f = tmp_path / "n.json"
    add(f, "Buy Milk")
    assert len(search(f, "MILK")) == 1
    assert len(search(f, "buy milk")) == 1


def test_remove(tmp_path):
    f = tmp_path / "n.json"
    add(f, "x")
    remove(f, 1)
    assert load(f) == []
    with pytest.raises(KeyError):
        remove(f, 1)


def test_cli(tmp_path, capsys):
    f = str(tmp_path / "n.json")
    assert main(["--file", f, "add", "water plants", "--tag", "home"]) == 0
    assert main(["--file", f, "list"]) == 0
    assert "water plants  #home" in capsys.readouterr().out
    assert main(["--file", f, "delete", "9"]) == 1
""",
    failing_test="test_notes.py::test_search_is_case_insensitive",
    readme="# notes\n\n```bash\npython notes.py add 'buy milk' --tag home\npython notes.py list\npython notes.py search MILK\npython notes.py delete 1\n```\n\nNotes live in `notes.json` (or `$NOTES_FILE`).\n",
    risks=("search must ignore case", "ids must never be reused within a run"),
    review_note="Store functions are pure file operations, so the CLI stays a thin layer.",
)

EXTRA_RULES: tuple[tuple[str, ProjectTemplate], ...] = (
    (r"shorten|short ?link|http|rest\b|api\b|endpoint|server|web service", _SHORTENER),
    (r"markdown|table|\.md\b", _MDTABLE),
    (r"\bunits?\b|convert|metric|imperial|celsius|fahrenheit", _UNITS),
    (r"password|passphrase|secret generator", _PASSWORD),
    (r"adventure|text game|game|dungeon|\brpg\b|quest", _ADVENTURE),
    (r"csv|spreadsheet|statistic|stats|dataset|median", _CSVSTATS),
    (r"note|journal|memo|diary", _NOTES),
)

# --------------------------------------------------------------------------- generic fallback
_STOP = frozenset(
    """a an the and or of for to in on with without that this these those is are be it its my our your
    please can could would should will make build create write generate implement need want simple small
    tiny basic python program script tool app application module library function functions class using
    use which what how into from by as at about some any new""".split()
)


def prompt_words(prompt: str, limit: int = 3) -> list[str]:
    """Distinct content words of ``prompt`` as safe lowercase identifiers (at most ``limit``)."""
    out: list[str] = []
    for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", prompt.lower()):
        if word in _STOP or len(word) < 3 or word in out:
            continue
        out.append(word)
        if len(out) == limit:
            break
    return out or ["item"]


def short_prompt(prompt: str, limit: int = 70) -> str:
    text = re.sub(r"[^A-Za-z0-9 ,.'()+/-]+", " ", prompt)
    text = re.sub(r"\s+", " ", text).strip() or "your request"
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def generic_template(prompt: str) -> ProjectTemplate:
    """A small keyword-analysis module named and documented after ``prompt``."""
    words = prompt_words(prompt)
    name = "_".join(w for w in words) + "_kit"
    said = short_prompt(prompt)
    funcs = "\n\n".join(
        f'def count_{w}(text: str) -> int:\n    """How often "{w}" appears in ``text`` '
        f'(whole words only), for the request: {said}"""\n    return _count({w!r}, text)\n'
        for w in words
    )
    source = f'''"""{name}: a small helper toolkit for the request "{said}"."""

from __future__ import annotations

import re

PROMPT = {said!r}
KEYWORDS = {tuple(w for w in words)!r}


def _count(word: str, text: str) -> int:
    pattern = rf"\\b{{re.escape(word)}}\\b"
    return len(re.findall(pattern, text.lower()))


{funcs}

def find_keywords(text: str) -> list[str]:
    """The request's keywords that occur in ``text``, in keyword order."""
    return [k for k in KEYWORDS if _count(k, text)]


def coverage(text: str) -> float:
    """Fraction (0..1) of the request's keywords that ``text`` mentions."""
    return len(find_keywords(text)) / len(KEYWORDS)


def outline() -> str:
    """A one-line description of what this module was built for."""
    return f"{{PROMPT}} -> " + ", ".join(KEYWORDS)
'''
    first = words[0].rstrip("_")
    tests = f'''import pytest

import {name} as m


def test_each_keyword_is_counted_case_insensitively():
    for word in m.KEYWORDS:
        counter = getattr(m, "count_" + word)
        assert counter("A " + word.upper() + " and another " + word + ".") == 2


def test_counts_whole_words_only():
    assert m.count_{words[0]}("{first}ing is not {first}") == 1


def test_find_keywords_and_coverage():
    text = " ".join(m.KEYWORDS[:2])
    assert m.find_keywords(text) == list(m.KEYWORDS[:2])
    assert m.coverage(text) == pytest.approx(min(2, len(m.KEYWORDS)) / len(m.KEYWORDS))
    assert m.coverage("nothing relevant") == 0


def test_outline_mentions_the_request():
    assert m.PROMPT in m.outline()
'''
    good = "    return len(re.findall(pattern, text.lower()))\n"
    bad = "    return text.lower().count(word)\n"
    readme = (
        f"# {name}\n\nGenerated for: **{said}**\n\n"
        f"Count keywords from the request (`{', '.join('count_' + w for w in words)}`), "
        "list them with `find_keywords()` and score a text with `coverage()`.\n\nRun `pytest`.\n"
    )
    return ProjectTemplate(
        key="generic",
        module=name,
        goal=f'A helper module for "{said}": keyword counting, matching and coverage',
        steps=(
            (
                f'Read the request "{said}" and pick its key terms: {", ".join(words)}',
                ("README.md",),
            ),
            (
                f"Implement {', '.join('count_' + w for w in words)}, find_keywords() and coverage()",
                (f"{name}.py",),
            ),
            ("Test whole-word matching, case folding and coverage", (f"test_{name}.py",)),
        ),
        source=source,
        good=good,
        bad=bad,
        tests=tests,
        failing_test=f"test_{name}.py::test_counts_whole_words_only",
        readme=readme,
        risks=("substring matches would overcount (weathering vs weather)", f'scope: "{said}"'),
        review_note=f'Stays within the request ("{said}") and keeps every helper small.',
    )
