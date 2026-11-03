# Distill: a synthetic dataset foundry

Distill builds instruction-tuning, preference (DPO) and eval datasets from NVIDIA NIM models, the way
Nemotron-style synthetic-data pipelines do: generate with several models, score with the reward
model, filter, deduplicate, decontaminate, then export JSONL plus a dataset card and an HTML quality
report. Runs are checkpointed and resumable. Everything works offline: without `NVIDIA_API_KEY` (or
with `--mock`) a deterministic **simulator** stands in for the models and every output is labelled
*simulated*.

```python
from nimlab.distill import Distiller

result = await Distiller().run("instruction_diversity", n=200, out="data")
print(len(result.kept), result.report.funnel[-1].out, result.meta.cost_usd)
```

## Pipeline

| Stage | What happens |
|---|---|
| `seed` | `qa_from_docs`: documents are loaded and chunked with `nimlab.atlas`. Other recipes: a `Role.REASONING` or `Role.CHAT` model expands domains into topics, subtopics and prompts (round-robin, so neighbours differ). |
| `generate` | Several generator models in turn, temperatures cycled per the recipe. Bounded concurrency. |
| `score` | `Role.REWARD` (Nemotron) is asked for `reward:<float>`, parsed by `nimlab.council.reward.parse_reward`. If it errors or replies with junk, an LLM-judge rubric (0-10) scores instead. Reward and judge scales are never mixed inside a pair. |
| `filter` | Score threshold (per source), length bounds, refusal filter, language heuristic, repetition, and per-kind checks: preference margin, QA grounding, eval difficulty. |
| `dedup` | Exact (normalised hash), then near-duplicates: cosine similarity of `Role.EMBED` vectors at `threshold`, greedy clustering in numpy, best score represents each cluster. |
| `decontaminate` | Drops samples whose word n-grams overlap a user-provided eval set. |
| `finalize` | Quality report, dataset files, dataset card, HTML report. |

`n` is the number of *candidates* generated; the final dataset is smaller by design (the funnel shows
where samples were lost and why). Drop reasons: `error unscored low_score too_short too_long
short_prompt refusal language repetition low_margin ungrounded too_easy duplicate_exact
duplicate_near contaminated`.

## Recipes

JSON, validated by pydantic (`nimlab.distill.Recipe`). Bundled in `src/nimlab/distill/recipes/`:

* `qa_from_docs`: grounded question/answer pairs from a folder of documents; answers must overlap the excerpt.
* `instruction_diversity`: topic tree -> subtopics -> instructions -> responses.
* `preference_pairs`: two models answer each prompt, the reward model ranks them, `chosen`/`rejected` in DPO format (needs two different models).
* `eval_set`: hard questions with reference answers from a reasoning model; a weak model also answers and the reward gap becomes a 0-1 `difficulty`.

Custom recipes are any JSON with the same schema (`distill run my.json`, see
`examples/distill/recipe_faq_strict.json`). Fields: `name`, `kind`, `seed_role`, `generator_role`,
`temperatures`, `max_tokens`, `system_prompt`, `domains`, `topics_per_domain`, `subtopics_per_topic`,
`chunk_chars`, `chunk_overlap`, `default_format`, `include_meta`, and the groups `filters`, `dedup`
(`exact`, `near`, `threshold`, `field`) and `decontaminate` (`ngram`, `min_overlap`, `check`). Unknown
fields are rejected.

## CLI

`python -m nimlab.distill.cli ...` (a Typer app: `nimlab.distill.cli:app`).

```bash
distill recipes [--json] [--show NAME]
distill run RECIPE --n 200 --models a,b --out data/ --format openai-chat --docs DIR \
    [--eval-set FILE] [--domain D]... [--threshold 0.92] [--seed 0] [--concurrency 8] [--run-id ID] [--mock]
distill resume RUN [--out DIR] [--mock]
distill report RUN [--html-out FILE] [--json]
distill inspect FILE.jsonl --sample 5
distill dedup FILE.jsonl --threshold 0.92 [--field prompt|response|both] [--out FILE] [--mock]
```

Runs live in `.nimlab/distill/runs/<id>/` (`--runs-dir` or `NIMLAB_DISTILL_RUNS`). `--out data/`
writes `data/<run id>/{<recipe>.<format>.jsonl, README.md, quality_report.html}`.

## Resume

`run.json` (config, stages done, usage), `seeds.json` and `samples.jsonl` are written atomically at
every stage end and every `checkpoint_every` samples. Each stage works only on what is still
missing, so `distill resume RUN` after a crash or Ctrl-C continues mid-stage and produces the same
dataset as an uninterrupted run.

## Export formats

| `--format` | Record |
|---|---|
| `openai-chat` | `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` (+ optional system) |
| `alpaca` | `{"instruction", "input", "output"}` (`input` carries the QA excerpt) |
| `sharegpt` | `{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}` |
| `dpo` | `{"prompt", "chosen", "rejected"}` (preference recipe only) |

`include_meta` (recipe or option) adds a `metadata` object (id, model, score, topic, difficulty). The
dataset card has provenance (models, recipe, filters, seed), counts per stage, quality metrics, usage
and a licence placeholder (`license: TODO`).

## Quality report

Funnel counts per stage with drop reasons, reward and judge score histograms (all vs kept), a
diversity metric (mean pairwise embedding distance, distinct-1/2/3), per-model yield, top and bottom
kept samples, and usage with an estimated cost (from the registry `cost_tier`, not billing data; JSON
and embedding token counts are estimated). The HTML page is self-contained, light/dark aware, and
shows a banner when simulated.

## Simulation

`SimClient` produces topic trees, varied instructions, QA pairs grounded in the excerpt, answers
whose quality depends on the model's cost tier, and reward scores computed from the text. It injects
refusals, fragments, repetition, non-English text, duplicates and near-duplicates so each stage
visibly acts. Mock embeddings are lexical, so near-duplicate detection is lexical too.

## Studio

`nimlab.studio.routes_distill` exports `router` (prefix `/api/distill`) and `APP_INFO`
(`id: distill`, colour `#FFC83D`).

| Route | |
|---|---|
| `GET /recipes` | recipes with `needs_docs`, allowed `formats`, `min_models` |
| `POST /run` `{recipe, n, models?, options?, format?}` | starts a run on the Studio run manager -> `{run_id}`; options are `RunOptions` fields (e.g. `docs`, `eval_set`) |
| `GET /runs`, `GET /runs/{id}` | runs on disk; detail includes the result summary |
| `GET /runs/{id}/samples?stage=&offset=&limit=&reason=` | `stage`: a stage name, `final`, `dropped`, `all` |
| `GET /runs/{id}/export?format=` | JSONL download |
| `GET /runs/{id}/report`, `/card` | HTML report, dataset card |

Events on `/api/runs/{run_id}/events`: `run.config`, `seeds`, `stage.start`/`stage.end`,
`sample.generated`, `sample.scored` (score), `sample.dropped` (reason), `dedup.cluster`, `funnel`,
`export`, then `run.done` (result summary) or `run.failed`.

Wiring (shared files, done by the integrator): `app.include_router(routes_distill.router)` before the
`/api` router with its catch-all; add `APP_INFO` to the app list; add an `APPS` entry
`{id: 'distill', cls: 'app-distill', app: 'distill'}` in `app.js`; link
`css/screens/distill.css` in `index.html`. The screen is `js/screens/distill.js` (`export mount(el)`).
