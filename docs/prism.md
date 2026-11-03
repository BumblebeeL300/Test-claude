# Prism: a vision workbench on NVIDIA NIM

Prism points NIM vision-language models (`Role.VISION`, by default `meta/llama-3.2-90b-vision-instruct`;
`microsoft/phi-3.5-vision-instruct` and any other vision model work through `--model`) at images and
returns something useful: a description with alt text, the text with its layout, structured data in a
schema you choose, a working HTML page, a diff of two images, a chart as a table, or an answer. It runs
fully offline in mock mode with deterministic, clearly flagged **simulated** output.

```bash
nimlab --mock prism describe examples/prism/sample_ui.png
nimlab --mock prism ui2code examples/prism/sample_ui.png -o out.html --refine 2
```

(`prism ... --mock` works too, and so does `python -m nimlab.prism.cli ...` before the root CLI lists
Prism.)

## Contents

[Modes](#modes) | [CLI](#cli) | [Python API](#python-api) | [Images](#image-handling) |
[Extract schemas](#extract-schemas) | [UI to code](#ui-to-code) | [Batch](#batch) |
[Simulation](#simulation-mode) | [Wire format](#wire-format) | [Studio](#nim-lab-studio) |
[Extending](#adding-a-mode) | [Testing](#testing)

## Modes

| Mode | Images | Options | Output (`result.output`) |
| --- | --- | --- | --- |
| `describe` | 1 | `detail=rich\|brief` | `description`, `alt_text` (under 125 chars), `tags` |
| `ocr` | 1 | | `text` (layout kept), `lines`, `has_text` |
| `extract` | 1 | `schema=receipt\|invoice\|business_card\|chart_data\|file.json\|{inline}` | `data`, `schema_name`, `schema`, `valid`, `errors` |
| `ui2code` | 1 | `refine=0..5` | `html`, `drafts`, `critiques`, `valid`, `problems`, `width`, `height` |
| `compare` | 2 | `focus=text` | `summary`, `similarity` (0-1), `differences` (`area`, `a`, `b`, `severity`) |
| `chart` | 1 | | `chart_type`, `title`, `columns`, `rows`, `summary`, `csv` |
| `ask` | 1 | `question=text` (required) | `question`, `answer` |

Every result is a `PrismResult`: `mode`, `model`, `simulated`, `images` (what was sent: sizes, format,
notes such as "downscaled"), `output`, `text` (the primary output as plain text), `elapsed_s`, `calls`
and `notes`. `result.save_json(path)` writes it.

## CLI

```bash
prism describe IMG [--detail brief] [--json]
prism ocr IMG [-o text.txt]
prism extract IMG --schema receipt|invoice|business_card|chart_data|schema.json [-o data.json]
prism ui2code IMG -o out.html --refine 2
prism compare A B
prism chart IMG --csv out.csv
prism ask IMG "How many cards are there?"
prism batch DIR --mode ocr --out results.jsonl [-c 4] [-R] [-q QUESTION] [-s SCHEMA]
prism modes                                       # modes and built-in schemas
```

Common options: `--model/-m ID`, `--json` (the full `PrismResult`), `--mock`. Errors exit with status 1
(`extract` exits 2 when the model's output does not match the schema; the data is still printed).
`app = typer.Typer()` lives in `nimlab.prism.cli`; the root `nimlab` CLI discovers it once `prism` is
added to its plugin list, giving `nimlab prism ...`.

## Python API

```python
from nimlab.prism import Prism

async with Prism() as prism:                       # NIMClient() from the environment
    r = await prism.describe("photo.jpg")
    print(r.output["alt_text"])
    r = await prism.extract("receipt.png", schema="receipt")
    r = await prism.extract("card.jpg", schema={"type": "object", "properties": {...}})
    r = await prism.ui2code("shot.png", refine=2)
    r = await prism.compare("before.png", "after.png")
    r = await prism.run("ask", "photo.jpg", question="What is on the sign?", model="x/y")
    items = await prism.batch("scans/", "ocr", concurrency=6)
```

`Prism(client=None, model=None, simulate=None, max_side=1568, max_bytes=4 MiB, sim_delay=0.0)`. A
borrowed `client` is never closed. Images may be paths, `bytes` or loaded `PrismImage` objects.
`run(..., observer=fn)` calls `fn(type, data)` for `stage.start`, `stage.end`, `token` and `result`
events; with an observer, text modes stream from the model.

## Image handling

`load_image(src)` (also `Prism.load`) accepts png, jpg, webp and gif, detected by content rather than
extension, and returns a `PrismImage` (`data`, `format`, `width`, `height`, `data_url`, `part()`,
`notes`).

* **Downscaling**: with Pillow, the longest side is limited to 1568 px. If the bytes are still over the
  cap (4 MiB), the image is re-encoded as JPEG and, if needed, scaled down further.
* **EXIF**: with Pillow, the orientation is applied and all EXIF (including GPS) is dropped. Without
  Pillow, JPEG EXIF segments are still removed with a small stdlib parser (orientation is then not applied).
* **GIF**: flattened to its first frame and sent as PNG (Pillow only; without Pillow GIFs pass through).
* **No Pillow**: nothing is resized. An image over `max_side` is passed through with a note if it fits
  the byte cap, and rejected with a clear message ("... pip install pillow, or resize the image first")
  if not. Sizes are read from file headers, so no decoding is needed.
* Pillow is optional; Prism adds no dependency.

## Extract schemas

Built-ins: `receipt`, `invoice`, `business_card`, `chart_data` (see `nimlab.prism.BUILTIN_SCHEMAS`).
A custom schema is a JSON Schema whose root is an object, given as a file path, inline JSON or a dict.
Extraction goes through `NIMClient.chat_json`: the schema is injected into the system prompt and the
model's JSON is validated against it (a small built-in validator covering `type`, `required`,
`properties`, `items`, `enum`, bounds, `anyOf`/`oneOf`/`allOf` and local `$ref`), so invalid output
triggers the usual re-prompt with the list of problems. If the model never conforms, you still get its
last JSON with `valid: false` and `errors`.

## UI to code

`ui2code` asks for one self-contained HTML file: inline CSS, no scripts, fonts or remote images.
`--refine N` adds up to N rounds of **critique, then revise**: the model sees the screenshot plus its
own HTML, scores fidelity and lists concrete discrepancies, then rewrites the page. It stops early when
the critique finds nothing left. The result includes `critiques` (`round`, `score`, `issues`) and a
structural check (`valid`, `problems`: unclosed tags, missing doctype, external resources). The model
only ever sees the HTML as text; it does not render it.

## Batch

`prism.batch(dir_or_list, mode, concurrency=4, recursive=False, **options)` runs a single-image mode
over a folder (or list) concurrently and returns `BatchItem`s in input order (`index`, `name`, `ok`,
`result` or `error`). One bad image never stops the rest. `write_jsonl(items, path)` writes one JSON
object per line. Concurrency is also bounded by the client's `max_concurrency`.

## Simulation mode

With no API key, `NIMLAB_MOCK=1` or `--mock`, Prism never calls a model; `result.simulated` is true and
`notes[0]` says so (the CLI prints a `SIMULATED` banner, Studio a badge). The output is derived from the
image itself, so it is deterministic and truthful about what a mock can know (size, format, dominant
colours from Pillow, or from a stdlib PNG decoder, or a stable hash palette):

* `describe` / `ask` report dimensions, orientation, tone and colour names and say they cannot read content.
* `ocr` and `extract` return the same sample document (a receipt, an invoice, a business card...), so
  the two agree; built-in schemas get realistic values, custom ones a schema-valid value from the core mock.
* `ui2code` returns a real, valid, self-contained page sized to the image and coloured with its
  palette, marked `SIMULATED` in the file; each refine round applies the fixes the simulated critique asked for.
* `compare` diffs file facts (identical bytes, size, format, palette); `chart` returns a small table.

The mode code is identical in both paths (stages, token events, refine loop), only the model call is swapped.

## Wire format

Images travel as OpenAI-style content parts, in the order given (the text first):

```json
{"role": "user", "content": [
  {"type": "text", "text": "Describe this image..."},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
]}
```

`NIMClient.chat`, `stream_chat` and `chat_json` pass list content through unchanged and the core mock
accepts it, so no core change was needed. NIM may cap inline image size; keep `max_bytes` modest.

## NIM Lab Studio

Files: `nimlab/studio/routes_prism.py`, `web/js/screens/prism.js`, `web/css/screens/prism.css`.

* `POST /api/prism/run`: `multipart/form-data` with `mode`, option fields (`question`, `schema`,
  `refine`, `detail`, `focus`), optional `model`, and one or more image files (any field name).
  Returns `{"run_id"}`. Compare takes exactly two images; any other mode given several images runs
  them as a concurrent batch. Bad mode, option or image is a `422`, a non-multipart body `415`, over
  40 MB `413`. For safety the `schema` field accepts built-in names or inline JSON only, never a file path.
* `GET /api/prism/modes`: `{modes, schemas, model, vision_models, pillow, simulated, max_images}`.
* Run events (SSE, via the shared run manager): `run.start`, `stage.start` / `stage.end`
  (`prepare`, then the mode's stages, e.g. `draft`, `critique 1`, `refine 1`), `token`
  (`{text, stage}`, streamed text), `result` (the `PrismResult`; in a batch also `index`, `name`), then
  `run.done` with `{mode, batch, simulated, items}` (items are results, or `{name, error}`).
* UI: the **optical bench**. Drop, paste or choose images on a lit table; a glass prism refracts a
  white beam into seven mode beams (one per mode, arrow keys switch); mode options appear below; the
  answer develops as a print. `extract` renders a form-like table (with a JSON tab), `ui2code` shows
  the code beside a sandboxed `<iframe sandbox>` preview (fit / tablet / phone), with the critique rounds.

### Wiring it in (integrator)

1. `server.py`: `from .routes_prism import router as prism_router` and
   `app.include_router(prism_router)` **before** the catch-all `/api/{rest:path}` router is included
   (include it, or insert the route, ahead of `app.include_router(api)`).
2. `runs.py`: add `"prism"` to `APP_IDS` so counts and last-run times are tracked from the start
   (the manager tolerates unknown apps, so this only affects the switchboard stats).
3. `state.py` / `/api/status`: add `routes_prism.APP_INFO` to the app list
   (`{id, name, tagline, color: "#B07CFF", glyph}`).
4. `app.js`: add `{ id: 'prism', name: 'Prism', key: '6', cls: 'app-prism', icon: 'eye', app: 'prism',
   hint: 'Put an image on the light table' }` to `APPS` (the screen module also exports `meta` and
   `route`; the module is lazily imported as `./screens/prism.js` and exports `mount(el, params)`
   returning an unmount function). Extend the number-key shortcut range if desired.
5. `index.html`: add `<link rel="stylesheet" href="css/screens/prism.css">`. (The screen also injects
   the stylesheet itself if it is missing, and `prism.css` defines `--c-prism` and `.app-prism`.)
6. `cli.py`: add `"prism"` to `PLUGIN_NAMES`.

## Adding a mode

```python
from nimlab.prism import Mode, Option, register_mode

@register_mode
class Haiku(Mode):
    name = "haiku"; title = "Haiku"; description = "A haiku about the image."
    options = (Option("mood", "text", "calm"),)

    async def run(self, ses, images, opts):
        text = await ses.text(f"Write a {opts['mood']} haiku.", images,
                              system="You are a poet.", sim=lambda: "Quiet pixels rest...")
        return {"haiku": text.strip()}

    def render(self, output): return output["haiku"]
```

`ses.text` / `ses.json` take the real prompt and a `sim` value; wrap steps in `with ses.stage("name"):`
to emit stage events.

## Testing

```bash
pytest tests/prism tests/studio/test_routes_prism.py
```

Covers loading and encoding (with and without Pillow, EXIF, downscaling), every mode in simulation,
schema extraction, `ui2code` HTML validity and the refine loop, the batch, the CLI via `CliRunner
--mock`, a respx test that the real request body contains `image_url` parts (plus streaming, retries and
the normalised image bytes), and the Studio routes on a bare FastAPI app with the real run manager.
`examples/prism/make_samples.py` regenerates the two tiny sample PNGs with only `struct` and `zlib`.
