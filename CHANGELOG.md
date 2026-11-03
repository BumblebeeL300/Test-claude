# Changelog

All notable changes to NIM Lab are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-11-03

### Added

- **Live model discovery**: with a key, the registry lists `GET /v1/models` once (cached for 24 h in
  `~/NIMLab/models_cache.json`) and `pick(role)` returns the first preferred default that is actually served,
  else the best name-heuristic match for the role (`embed`, `rerank`, `reward`, `guard`/`safety`, `vision`/`vl`,
  `coder`, `r1`/`reasoning`/`thinking`, `instruct`, size parsed from the name). Explicit overrides still win.
- **Runtime fallback**: a 404 / "model not found" / "not available" from chat, streaming chat, embeddings or
  rerank marks the model unavailable for the session and retries once with the next candidate for the role
  (new `NIMModelUnavailableError`), logging a warning that names both models.
- **Per-role pickers in Studio** (Settings > Models) with "Auto" as the default and a "Refresh" for the live
  list, a **per-app patch bay** on every app screen (Foundry PLANNER/CODER/REVIEWER, Atlas
  EMBED/RERANK/ANSWER, Prism VISION, ...) with a searchable dropdown, an "All models" toggle and "Reset to
  Auto", `⌘K` actions ("Switch model for <role>...", "Switch <app> model..."), and "Last run via ..."
  telemetry that also reports automatic substitutions. Every run endpoint accepts `model_overrides`
  (`{role: model}`); precedence is request > app > global > env > `nimlab.toml` > live auto-pick > curated
  default. Overrides persist in `~/NIMLab/studio_settings.json`.
- Switchboard shows an amber "N default models unavailable, auto-substituted" notice when the live catalog
  shows stale defaults; the catalog is refreshed after `POST /api/key`.
- `nimlab models --live` (each role's chosen model, curated ids missing from the live catalog, suggested
  replacements) and `nimlab models --refresh`.
- **In-app uninstall** (Settings > Danger zone, `POST /api/uninstall`, `nimlab uninstall`) with a plan of
  exactly what is removed (key, settings and caches, optionally projects and Loom pipelines, the Atlas
  database and the app itself), a typed `UNINSTALL` confirmation, a goodbye screen, and detached removal of
  the `.app` (to the Trash) or the pip package after exit. Nothing outside the known folders is touched and
  symlinks are never followed.
- **Update check** (Settings > About, `GET /api/update/check`, `nimlab update`): `git fetch` + compare with
  `claude/nimlab`, "Update now" (`git pull --ff-only` and `pip install -e '.[desktop]'`), clear messages for
  dirty trees, diverged history, offline, no git, pip and frozen installs.
- New app icon.

### Changed

- **Refreshed model catalog** (checked against NVIDIA's live catalog on 2026-11-03): chat
  `nvidia/nemotron-3-super-120b-a12b`, reasoning `nvidia/nemotron-3-ultra-550b-a55b`, coder
  `poolside/laguna-xs-2.1`, vision `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, fast
  `nvidia/nemotron-3.5-lightning-30b-a3b`, embed `nvidia/nemotron-3.5-embed-1b`, rerank
  `nvidia/nemotron-3.5-rerank-8b`, safety `nvidia/nemotron-3.5-content-safety`; reward falls back to the
  deprecated `nvidia/nemotron-4-340b-reward` (NVIDIA has deprecated all hosted reward models). Retired ids
  (Llama 3.x, DeepSeek R1, Qwen coder, Phi-4 mini, GPT-OSS 120B, NV-EmbedQA, ...) are mapped to replacements
  in `registry.RETIRED`; `ModelInfo` gained `deprecated`.
- Parsers handle the new endpoint shapes: `nemotron-3.5-content-safety` (plain conversation in, `User Safety:`
  lines out) next to the NemoGuard JSON template, the per-attribute reply of `nemotron-4-340b-reward` next to
  `reward:<float>`, and the `results` / `relevance_score` rerank reply next to `rankings` / `logit`.
- Package version is now 0.2.1 (`pyproject.toml` and `nimlab.__version__`).

## [0.2.0] - 2026-11-03

### Added

- **Original toolkit and apps**
  - `nimlab.core`: async NVIDIA NIM client (chat, streaming, `chat_json`, embeddings, rerank, retries,
    usage tracking), model registry with roles and overrides, tool-agent loop and a deterministic
    offline mock backend.
  - **Council**: multi-model deliberation (mixture-of-agents, debate, best-of-N, critic) with HTML traces.
  - **Atlas**: agentic RAG with hybrid retrieval, reranking, claim verification, FastAPI server and web UI.
  - **Forge**: test-driven coding agent working in a sandboxed workspace.
  - **Arena**: eval suites, graders, Bradley-Terry Elo, HTML leaderboard, smart router and an
    OpenAI-compatible routing proxy.
  - Root `nimlab` CLI (`doctor`, `models`) that auto-mounts every app's sub-commands.
- **NIM Lab Studio**, the macOS desktop app: token-protected FastAPI server on `127.0.0.1`, a synth-rack
  web UI (Switchboard, Foundry, Council, Atlas, Arena), run streaming over SSE, a project browser,
  Keychain key storage, a simulation mode that needs no key, `⌘K` command palette, light and dark themes,
  and PyInstaller packaging under `packaging/macos`.
- **Prism** (vision): describe, OCR, structured extraction, screenshot to HTML/CSS with refine rounds, image
  compare, chart to data, free-form questions and batch runs; `nimlab prism`.
- **Sentinel** (safety lab): 64 probes in seven suites, judges, scorecards with HTML reports, Shield
  (NemoGuard plus heuristics) and reversible PII redaction; `nimlab sentinel`.
- **Loom** (pipeline node editor): typed JSON pipelines, parallel scheduling, caching, linting, mermaid and
  standalone-Python export, four bundled pipelines and a canvas editor in Studio; `nimlab loom`.
- **Distill** (synthetic datasets): four recipes, multi-model generation, reward scoring, filters,
  near-duplicate removal, eval-set decontamination, resumable runs, JSONL export and a dataset card;
  `nimlab distill`.
- Studio now hosts all eight apps: a two-tier Switchboard around the NIM Core, a rail and `⌘1` to `⌘9`
  shortcuts for every module, and new command-palette actions (analyze an image, run a safety scan, new
  pipeline, generate a dataset).

### Changed

- Package version is now 0.2.0 (`pyproject.toml` and `nimlab.__version__`).
- The wheel and the macOS bundle include the new package data (Sentinel suites and templates, Loom
  pipelines, Distill recipes and templates).

### Fixed

- Studio Switchboard: the NIM Core dial rings were invisible because a Council stylesheet rule
  (`.ring`) leaked into the whole app.

### Known limitations

- **Not yet exercised against the live NIM API.** Everything was developed and tested offline (mock mode
  and HTTP-mocked tests); expect to fix small things on first contact with the real endpoints.
- **The macOS `.app` and the native window are untested on a real Mac.** The server, the web UI (driven
  with Playwright in a browser) and the PyInstaller spec were exercised on Linux only.
- **The Forge sandbox is not an isolation boundary.** It confines paths, scrubs the environment and
  allowlists commands, but code the agent writes still runs on your machine.
- **Sentinel judges are heuristic.** Refusal, citation and injection verdicts are rules (optionally an LLM
  judge) and will misjudge some replies; the probes are a regression battery, not a full red team.
- Mock and simulated output is synthetic; leaderboards, scorecards and datasets produced in mock mode
  only illustrate the tooling.
- No license has been chosen yet.
