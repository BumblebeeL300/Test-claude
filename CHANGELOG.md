# Changelog

All notable changes to NIM Lab are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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
