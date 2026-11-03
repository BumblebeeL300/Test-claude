// Fake backend for ?demo=1 — scripted runs that mimic the SSE contract. Dev/screenshot use only.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => Date.now() / 1000;
const MODELS = [
  ['nvidia/nemotron-3-super-120b-a12b', ['chat', 'reasoning'], 3, 1048576], ['nvidia/nemotron-3-ultra-550b-a55b', ['reasoning', 'chat'], 5, 1048576],
  ['deepseek-ai/deepseek-v4.1-flash', ['reasoning', 'chat'], 2, 1048576], ['google/gemma-4-31b-it', ['chat', 'vision', 'fast'], 2, 131072],
  ['poolside/laguna-xs-2.1', ['coder'], 2, 262144], ['mistralai/codestral-22b-instruct-v0.1', ['coder'], 2, 32768],
  ['nvidia/nemotron-3.5-lightning-30b-a3b', ['fast', 'chat'], 1, 1048576], ['mistralai/mistral-large-4-1t-a52b', ['chat'], 5, 262144],
  ['nvidia/nemotron-3.5-embed-1b', ['embed'], 1, 8192], ['nvidia/nemotron-3.5-rerank-8b', ['rerank'], 2, 8192],
  ['nvidia/nemotron-4-340b-reward', ['reward'], 4, 4096], ['nvidia/nemotron-3.5-content-safety', ['safety'], 1, 131072],
  ['nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', ['vision', 'reasoning'], 2, 262144], ['nvidia/nemotron-3.5-bolt-e4b', ['fast', 'vision'], 1, 131072],
].map(([id, roles, cost_tier, context_window]) => ({ id, roles, cost_tier, context_window, notes: '' }));

const S = {
  key: false, mode: 'mock', nextId: 4,
  runs: [
    { id: 'run_1', app: 'forge', title: 'Pomodoro timer CLI', status: 'done', started_at: now() - 3600, finished_at: now() - 3520, simulated: true },
    { id: 'run_2', app: 'council', title: 'Is a tomato a fruit?', status: 'done', started_at: now() - 7200, finished_at: now() - 7150, simulated: true },
    { id: 'run_3', app: 'atlas', title: 'What does the SLA say about refunds?', status: 'done', started_at: now() - 90000, finished_at: now() - 89990, simulated: true },
  ],
  projects: [
    { id: 'pomodoro', name: 'pomodoro', path: '~/NIMLab/projects/pomodoro', files: 4, updated_at: now() - 3520, last_status: 'done' },
    { id: 'tiny-wiki', name: 'tiny-wiki', path: '~/NIMLab/projects/tiny-wiki', files: 7, updated_at: now() - 172800, last_status: 'failed' },
  ],
  files: {}, collections: [{ name: 'default', documents: 3 }, { name: 'contracts', documents: 2 }],
  streams: {},
};
const FILES = {
  'tasker/store.py': ['py', `"""A tiny JSON-backed task store."""
import json
from pathlib import Path


class Store:
    def __init__(self, path="tasks.json"):
        self.path = Path(path)
        self.tasks = json.loads(self.path.read_text()) if self.path.exists() else []

    def add(self, title, tags=None):
        task = {"id": len(self.tasks) + 1, "title": title, "done": False, "tags": tags or []}
        self.tasks.append(task)
        self._save()
        return task

    def complete(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                task["done"] = True
                self._save()
                return task
        raise KeyError(task_id)

    def pending(self):
        # only tasks that are not done yet
        return [t for t in self.tasks if not t["done"]]

    def _save(self):
        self.path.write_text(json.dumps(self.tasks, indent=2))
`],
  'tasker/cli.py': ['py', `import argparse
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tasker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("title")
    add.add_argument("--tag", action="append", default=[])
    done = sub.add_parser("done")
    done.add_argument("id", type=int)
    sub.add_parser("list")
    args = parser.parse_args(argv)
    store = Store()
    if args.cmd == "add":
        print(f"added #{store.add(args.title, args.tag)['id']}")
    elif args.cmd == "done":
        print(f"completed #{store.complete(args.id)['id']}")
    else:
        for t in store.pending():
            print(f"[ ] {t['id']:>3}  {t['title']}")


if __name__ == "__main__":
    main()
`],
  'tests/test_store.py': ['py', `from tasker.store import Store


def test_add_assigns_ids(tmp_path):
    s = Store(tmp_path / "t.json")
    assert s.add("a")["id"] == 1
    assert s.add("b")["id"] == 2


def test_complete_marks_done(tmp_path):
    s = Store(tmp_path / "t.json")
    s.add("a")
    assert s.complete(1)["done"] is True


def test_pending_hides_done(tmp_path):
    s = Store(tmp_path / "t.json")
    s.add("a"); s.add("b"); s.complete(1)
    assert [t["title"] for t in s.pending()] == ["b"]


def test_persists_between_instances(tmp_path):
    Store(tmp_path / "t.json").add("keep")
    assert len(Store(tmp_path / "t.json").tasks) == 1


def test_ids_survive_deletion(tmp_path):
    s = Store(tmp_path / "t.json")
    s.add("a"); s.add("b")
    s.tasks.pop(0)
    assert s.add("c")["id"] == 3
`],
  'README.md': ['md', `# tasker

A tiny command-line task list.

## Usage

- \`tasker add "write docs" --tag work\`
- \`tasker done 1\`
- \`tasker list\`

Tasks live in \`tasks.json\` next to where you run it.
`],
};
const FIXED_STORE = FILES['tasker/store.py'][1].replace('"id": len(self.tasks) + 1', '"id": max((t["id"] for t in self.tasks), default=0) + 1');
const lines = (s) => s.split('\n').length - 1;

function newRun(app, title) {
  const r = { id: 'run_' + S.nextId++, app, title, status: 'running', started_at: now(), finished_at: null, simulated: S.mode === 'mock', events: [], subs: new Set() };
  S.runs.unshift(r); return r;
}
function emit(r, type, data = {}, phase) {
  const ev = { seq: r.events.length + 1, ts: now(), run_id: r.id, type, data }; if (phase) ev.phase = phase;
  r.events.push(ev); r.subs.forEach((f) => f(ev));
  if (type === 'run.done' || type === 'run.failed') { r.status = type === 'run.done' ? 'done' : 'failed'; r.finished_at = now(); }
}
const stopped = (r) => r.status === 'cancelled';

async function forgeScript(r, prompt, project) {
  const ph = async (p, fn) => { if (stopped(r)) return; emit(r, 'phase.start', {}, p); await fn(); await sleep(250); emit(r, 'phase.end', {}, p); };
  const names = Object.keys(FILES);
  await ph('explore', async () => {
    await sleep(500); emit(r, 'tool.call', { name: 'list_dir', args: { path: '.' } }, 'explore');
    await sleep(450); emit(r, 'tool.result', { name: 'list_dir', ok: true, preview: 'empty workspace' }, 'explore');
    emit(r, 'log', { level: 'info', message: 'greenfield project, no existing conventions to follow' }, 'explore'); await sleep(500);
  });
  await ph('plan', async () => {
    await sleep(700);
    emit(r, 'plan', { steps: [
      { id: 1, description: 'Persistent JSON task store with auto ids', files: ['tasker/store.py'] },
      { id: 2, description: 'argparse CLI: add, done, list', files: ['tasker/cli.py'] },
      { id: 3, description: 'Unit tests covering store behaviour', files: ['tests/test_store.py'] },
      { id: 4, description: 'README with usage examples', files: ['README.md'] }] }, 'plan');
    emit(r, 'usage', { model: 'nvidia/nemotron-3-ultra-550b-a55b', prompt_tokens: 1840, completion_tokens: 612, calls: 1 });
    await sleep(600);
  });
  await ph('implement', async () => {
    for (const n of names) {
      await sleep(650); emit(r, 'tool.call', { name: 'write_file', args: { path: n } }, 'implement');
      await sleep(300); emit(r, 'file.write', { path: n, lines: lines(FILES[n][1]), language: FILES[n][0], bytes: FILES[n][1].length }, 'implement');
      emit(r, 'usage', { model: 'poolside/laguna-xs-2.1', prompt_tokens: 900 + Math.round(Math.random() * 600), completion_tokens: 300 + Math.round(Math.random() * 400), calls: 1 });
    }
  });
  const names1 = ['test_add_assigns_ids', 'test_complete_marks_done', 'test_pending_hides_done', 'test_persists_between_instances', 'test_ids_survive_deletion'];
  await ph('verify', async () => {
    emit(r, 'test.run', { command: 'pytest -q' }, 'verify'); await sleep(1100);
    emit(r, 'test.result', { passed: 4, failed: 1, tests: names1.map((name, i) => ({ name, outcome: i === 4 ? 'failed' : 'passed' })) }, 'verify');
    emit(r, 'log', { level: 'warn', message: 'tests/test_store.py::test_ids_survive_deletion  AssertionError: 2 != 3' }, 'verify');
  });
  await ph('debug', async () => {
    await sleep(700); emit(r, 'log', { level: 'info', message: 'root cause: ids derived from len(tasks), collide after deletion' }, 'debug');
    await sleep(700); emit(r, 'tool.call', { name: 'write_file', args: { path: 'tasker/store.py' } }, 'debug');
    FILES['tasker/store.py'][1] = FIXED_STORE;
    emit(r, 'file.write', { path: 'tasker/store.py', lines: lines(FIXED_STORE), language: 'py' }, 'debug');
    emit(r, 'usage', { model: 'deepseek-ai/deepseek-v4.1-flash', prompt_tokens: 2400, completion_tokens: 1100, calls: 1 });
  });
  await ph('verify', async () => {
    emit(r, 'test.run', { command: 'pytest -q' }, 'verify'); await sleep(1000);
    emit(r, 'test.result', { passed: 5, failed: 0, tests: names1.map((name) => ({ name, outcome: 'passed' })) }, 'verify');
  });
  await ph('review', async () => {
    await sleep(900);
    emit(r, 'review', { verdict: 'approve', issues: [{ severity: 'low', message: 'Consider file locking for concurrent writers.' }, { severity: 'info', message: 'README could document the tasks.json location.' }] }, 'review');
    emit(r, 'usage', { model: 'nvidia/nemotron-3-super-120b-a12b', prompt_tokens: 3100, completion_tokens: 420, calls: 1 });
  });
  await ph('report', async () => { await sleep(500); emit(r, 'log', { level: 'info', message: 'wrote REPORT.md' }, 'report'); });
  const pid = project || 'tasker';
  if (!S.projects.find((p) => p.id === pid)) S.projects.unshift({ id: pid, name: pid, path: '~/NIMLab/projects/' + pid, files: names.length, updated_at: now(), last_status: 'done' });
  S.files[pid] = JSON.parse(JSON.stringify(FILES));
  if (!stopped(r)) emit(r, 'run.done', { result: { project: pid, summary: `Built ${names.length} files, 5/5 tests green after 1 debug iteration.`, verdict: 'approve', iterations: 2 } });
}

async function councilScript(r, q, strategy) {
  const panel = ['nvidia/nemotron-3-super-120b-a12b', 'nvidia/nemotron-3-ultra-550b-a55b', 'deepseek-ai/deepseek-v4.1-flash', 'mistralai/mistral-large-4-1t-a52b', 'google/gemma-4-31b-it'];
  const takes = ['Botanically yes: it develops from the ovary of a flower and carries seeds.', 'By botanical definition a fruit; culinary convention files it with vegetables.', 'Fruit. The savoury use is a kitchen habit, not taxonomy.', 'It is a fruit, though the US Supreme Court once ruled otherwise for tariffs.', 'Fruit in botany, vegetable in the kitchen; both statements are true.'];
  const stage = async (name, who, round) => {
    emit(r, 'stage.start', { name, round }); await Promise.all(who.map(async (m, i) => { await sleep(i * 380); emit(r, 'model.start', { model: panel[m], stage: name }); await sleep(900 + Math.random() * 900); emit(r, 'model.end', { model: panel[m], stage: name, preview: takes[m], tokens: 120 + m * 20 }); }));
    emit(r, 'stage.end', { name });
  };
  await stage('propose', [0, 1, 2, 3, 4]);
  emit(r, 'consensus', { agreement: 0.46 }); await sleep(600);
  await stage('critique', [0, 1, 2, 3, 4]);
  emit(r, 'consensus', { agreement: 0.71 }); await sleep(500);
  await stage('judge', [1]);
  emit(r, 'consensus', { agreement: 0.88 });
  emit(r, 'run.done', { result: { answer: 'Yes. Botanically a tomato is a fruit, specifically a berry: it grows from a flower’s ovary and holds the seeds. Culinary and legal contexts call it a vegetable, which is a usage convention, not a contradiction.', rationale: 'Four of five panelists converged on the botanical definition after critique; the dissenting framing (tariff law) was folded in as context rather than as a competing answer.', agreement: 0.88, strategy, candidates: panel.map((m, i) => ({ model: m, text: takes[i] })) } });
}

async function atlasScript(r, q) {
  const cites = [
    { n: 1, source: 'master-services-agreement.pdf', heading: '9.2 Service credits', snippet: 'If monthly uptime falls below 99.9%, Customer is entitled to a service credit of 10% of the monthly fee for each full 0.1% shortfall, capped at 30%.' },
    { n: 2, source: 'master-services-agreement.pdf', heading: '9.4 Refunds', snippet: 'Credits are Customer’s sole remedy for availability failures. Refunds are issued only where the Agreement is terminated for repeated breach under Section 14.3.' },
    { n: 3, source: 'sla-addendum.md', heading: 'Exclusions', snippet: 'Scheduled maintenance announced 72 hours in advance does not count toward downtime.' }];
  await sleep(500); emit(r, 'sources', { sources: cites.map((c) => ({ source: c.source, heading: c.heading })) });
  const text = 'Refunds are narrow: availability failures are compensated with service credits of 10% of the monthly fee per 0.1% shortfall, capped at 30% [1]. Credits are the sole remedy, so a cash refund only arises if the agreement is terminated for repeated breach [2]. Announced maintenance windows do not count as downtime [3].';
  for (const w of text.split(/(?<=\s)/)) { if (stopped(r)) return; emit(r, 'token', { text: w }); await sleep(35 + Math.random() * 40); }
  emit(r, 'run.done', { result: { answer: text, citations: cites, groundedness: 0.94 } });
}

const ARENA = [['nvidia/nemotron-3-ultra-550b-a55b', 1184, .91, 2100], ['nvidia/nemotron-3-super-120b-a12b', 1131, .88, 1750], ['deepseek-ai/deepseek-v4.1-flash', 1112, .9, 3900], ['mistralai/mistral-large-4-1t-a52b', 1046, .8, 1200], ['google/gemma-4-31b-it', 1018, .76, 640], ['poolside/laguna-xs-2.1', 982, .71, 520], ['nvidia/nemotron-3.5-lightning-30b-a3b', 861, .52, 210], ['nvidia/nemotron-3.5-bolt-e4b', 826, .47, 180]];


// ---- model switching (demo): same shapes as /api/models/* ----
const APP_ROLES = { forge: [['reasoning', 'PLANNER'], ['coder', 'CODER'], ['chat', 'REVIEWER']], council: [['chat', 'PANEL'], ['reasoning', 'JUDGE']], atlas: [['embed', 'EMBED'], ['rerank', 'RERANK'], ['chat', 'ANSWER']], arena: [['chat', 'CHAT'], ['reasoning', 'REASONING'], ['coder', 'CODER'], ['fast', 'FAST']], prism: [['vision', 'VISION']], sentinel: [['chat', 'TARGET'], ['safety', 'GUARD']], loom: [['chat', 'CHAT'], ['fast', 'FAST'], ['reasoning', 'REASON'], ['embed', 'EMBED']], distill: [['chat', 'GENERATE'], ['fast', 'WEAK'], ['reward', 'REWARD'], ['embed', 'EMBED']] };
const DEF = { chat: 'nvidia/nemotron-3-super-120b-a12b', reasoning: 'nvidia/nemotron-3-ultra-550b-a55b', coder: 'poolside/laguna-xs-2.1', vision: 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', fast: 'nvidia/nemotron-3.5-lightning-30b-a3b', embed: 'nvidia/nemotron-3.5-embed-1b', rerank: 'nvidia/nemotron-3.5-rerank-8b', reward: 'nvidia/nemotron-4-340b-reward', safety: 'nvidia/nemotron-3.5-content-safety' };
S.override = {}; S.appOverride = {};
function modelsPayload() {
  const eff = (role, app) => (app && S.appOverride[app]?.[role] ? [S.appOverride[app][role], 'app'] : S.override[role] ? [S.override[role], 'studio'] : [DEF[role], 'default']);
  return { live: S.key, mock: !S.key, note: S.key ? '' : 'Simulation mode: showing the curated catalog. The live model list appears once a key is added.', catalog: S.key ? { source: 'network', count: 126, age_s: 40, fetched_at: now() } : null, stale: [], unavailable: {}, substitutions: [],
    roles: Object.keys(DEF).map((role) => ({ role, model: eff(role)[0], source: eff(role)[1], default: DEF[role], substituted: false, override: S.override[role] || null, pinned_by: null })),
    apps: Object.fromEntries(Object.entries(APP_ROLES).map(([app, slots]) => [app, { overridden: !!Object.keys(S.appOverride[app] || {}).length, roles: slots.map(([role, label]) => ({ role, label, model: eff(role, app)[0], source: eff(role, app)[1], override: S.appOverride[app]?.[role] || null })) }])) };
}
const setOv = (tbl, ov) => { for (const [k, v] of Object.entries(ov || {})) { if (v) tbl[k] = v; else delete tbl[k]; } };
export async function request(method, path, body) {
  await sleep(60 + Math.random() * 80);
  const [p, q = ''] = path.split('?'); const sp = new URLSearchParams(q);
  const m = (re) => p.match(re);
  if (p === '/status') return { version: '0.1.0-demo', mode: S.mode, base_url: 'https://integrate.api.nvidia.com/v1', network: S.key ? 'ok' : 'unknown',
    key: { set: S.key, source: S.key ? 'keychain' : 'none', masked: S.key ? 'nvapi-••••••••••••k3Zq' : '' },
    apps: ['council', 'atlas', 'forge', 'arena'].map((id) => ({ id, name: id[0].toUpperCase() + id.slice(1), color: '', glyph: id,
      tagline: { council: 'Many models, one verdict', atlas: 'Answers grounded in your documents', forge: 'Plan, build, test, ship', arena: 'Benchmark, rank and route' }[id],
      status: S.runs.some((r) => r.app === id && r.status === 'running') ? 'busy' : 'ready', runs: S.runs.filter((r) => r.app === id).length + { council: 11, atlas: 24, forge: 6, arena: 3 }[id],
      last_run_at: (S.runs.find((r) => r.app === id) || {}).started_at || now() - 86400 })) };
  if (p === '/key' && method === 'POST') { await sleep(900); const ok = /^nvapi-/.test(body.api_key || ''); if (!ok) throw Object.assign(new Error('That does not look like an NVIDIA key'), { status: 400 }); S.key = true; S.mode = 'live'; return { ok: true, verified: !/unverified/.test(body.api_key), model_count: 126, error: null, key: { set: true, source: 'keychain', masked: 'nvapi-••••••••••••' + body.api_key.slice(-4) } }; }
  if (p === '/key' && method === 'DELETE') { S.key = false; S.mode = 'mock'; return { ok: true }; }
  if (p === '/key/test') { await sleep(700); return { ok: S.key, verified: S.key, model_count: 126, error: S.key ? null : 'no key', latency_ms: 212 }; }
  if (p === '/models') return MODELS;
  if (p === '/usage') return { totals: { calls: 214, tokens: 482300, cost: 0 }, models: ARENA.slice(0, 7).map(([model], i) => ({ model, calls: [64, 41, 38, 29, 22, 12, 8][i], tokens: [120e3, 98e3, 87e3, 60e3, 41e3, 26e3, 9e3][i], latency_ms: [2100, 1750, 3900, 1200, 640, 520, 210][i] })) };
  if (p === '/runs') return S.runs.map(({ events, subs, ...r }) => r);
  if (m(/^\/runs\/([^/]+)\/cancel$/)) { const r = S.runs.find((x) => x.id === m(/^\/runs\/([^/]+)\//)[1]); if (r && r.status === 'running') { r.status = 'cancelled'; emit(r, 'run.failed', { error: 'cancelled' }); r.status = 'cancelled'; } return { ok: true }; }
  if (m(/^\/runs\/([^/]+)$/)) { const r = S.runs.find((x) => x.id === m(/^\/runs\/([^/]+)$/)[1]); if (!r) throw Object.assign(new Error('not found'), { status: 404 }); const done = r.events?.find((e) => e.type === 'run.done'); return { ...r, events: undefined, subs: undefined, result: done?.data?.result || null }; }
  if (p === '/projects') return S.projects;
  if (m(/^\/projects\/([^/]+)\/tree$/)) { const id = m(/^\/projects\/([^/]+)\/tree$/)[1]; const f = S.files[id] || FILES; return Object.keys(f).map((path) => ({ path, lines: lines(f[path][1]), language: f[path][0] })); }
  if (m(/^\/projects\/([^/]+)\/file$/)) { const id = m(/^\/projects\/([^/]+)\//)[1]; const f = (S.files[id] || FILES)[sp.get('path')]; return { path: sp.get('path'), content: f ? f[1] : '# (empty)\n', language: f?.[0] }; }
  if (m(/^\/projects\/([^/]+)\/diff$/)) { const id = m(/^\/projects\/([^/]+)\//)[1]; const f = S.files[id] || FILES; return { diff: Object.keys(f).map((path) => { const ls = f[path][1].replace(/\n$/, '').split('\n'); return `diff --git a/${path} b/${path}\nnew file mode 100644\n--- /dev/null\n+++ b/${path}\n@@ -0,0 +1,${ls.length} @@\n` + ls.map((l) => '+' + l).join('\n'); }).join('\n') + '\n' }; }
  if (m(/^\/projects\/([^/]+)\/reveal$/)) return { ok: true };
  if (p === '/forge/build') { const r = newRun('forge', body.prompt.slice(0, 80)); const pid = body.project || 'tasker'; forgeScript(r, body.prompt, pid); return { run_id: r.id, project: pid }; }
  if (p === '/council/run') { const r = newRun('council', body.question.slice(0, 80)); councilScript(r, body.question, body.strategy); return { run_id: r.id }; }
  if (p === '/atlas/collections') return S.collections;
  if (p === '/atlas/ingest') { await sleep(700); const n = body instanceof FormData ? body.getAll('files').length : 1; const c = S.collections[0]; c.documents += n; return { ok: true, ingested: n, chunks: n * 14, collection: c.name }; }
  if (p === '/atlas/ask') { const r = newRun('atlas', body.question.slice(0, 80)); atlasScript(r, body.question); return { run_id: r.id }; }
  if (p === '/arena/runs') return [{ id: 'arena_1', started_at: now() - 86400, status: 'done', result: { leaderboard: ARENA.map(([model, elo, quality, latency_ms]) => ({ model, elo, quality, latency_ms, ci: 18 })) } }];
  if (p === '/arena/route') {
    await sleep(500); const code = /code|python|bug|function|regex/i.test(body.prompt);
    const c = (code ? [['poolside/laguna-xs-2.1', .88, 520, 3], ['mistralai/codestral-22b-instruct-v0.1', .84, 410, 2], ['google/gemma-4-31b-it', .79, 640, 3], ['nvidia/nemotron-3.5-lightning-30b-a3b', .6, 210, 1]] : [['nvidia/nemotron-3-super-120b-a12b', .9, 1750, 5], ['google/gemma-4-31b-it', .82, 640, 3], ['mistralai/mistral-large-4-1t-a52b', .8, 1200, 4], ['nvidia/nemotron-3.5-lightning-30b-a3b', .58, 210, 1]]).map(([model, quality, latency, cost]) => ({ model, quality, latency, cost, utility: quality - latency / 12000 - cost * .02 }));
    c.sort((a, b) => b.utility - a.utility);
    return { category: code ? 'coding' : 'general reasoning', chosen: c[0].model, candidates: c, reasons: [`Classified as ${code ? 'coding' : 'general reasoning'} by embedding kNN (0.${code ? 83 : 71})`, 'Utility = quality - latency penalty - cost penalty', `${c[0].model.split('/').pop()} leads the leaderboard for this category`] };
  }
  if (p === '/arena/run') { const r = newRun('arena', 'Mini arena'); setTimeout(() => emit(r, 'run.done', { result: {} }), 1500); return { run_id: r.id }; }
  if (p === '/models/roles') return modelsPayload();
  if (p === '/models/refresh') return modelsPayload();
  if (p === '/models/options') { const role = sp.get('role'); const list = MODELS.filter((x) => sp.get('all') === '1' || x.roles.includes(role)); return { role, live: S.key, all: sp.get('all') === '1', note: S.key ? '' : 'Simulation mode: showing the curated catalog. The live model list appears once a key is added.', models: list.map((x) => ({ id: x.id, live: S.key, curated: true, preferred: x.id === DEF[role], deprecated: false, unavailable: false })) }; }
  if (p === '/models/overrides' && method === 'PUT') { setOv(S.override, body.overrides); return modelsPayload(); }
  if (m(/^\/models\/app\/([^/]+)$/)) { const app = m(/^\/models\/app\/([^/]+)$/)[1]; S.appOverride[app] ||= {}; if (method === 'DELETE') S.appOverride[app] = {}; else setOv(S.appOverride[app], body.overrides); return modelsPayload(); }
  if (p === '/update/check') return { version: '0.2.1', kind: 'git', state: 'up-to-date', message: 'NIM Lab 0.2.1 is up to date (origin/claude/nimlab).', commits: [], can_update: false, behind: 0, ahead: 0, dirty: false };
  if (p === '/uninstall') return { dry_run: true, home: '~/NIMLab', config_dir: '~/.config/nimlab', install: { kind: 'dev', instructions: 'Demo: nothing is ever removed.' }, outside: [], items: [['key', 'API key', 1, 0], ['data', 'Settings and caches', 12, 48e3], ['projects', 'Projects', 2, 9e5], ['atlas', 'Atlas database', 1, 2e6], ['app', 'Application', 0, 0]].map(([id, label, count, bytes]) => ({ id, label, paths: [], count, bytes, selected: false, note: '' })) };
  throw Object.assign(new Error('demo: unknown route ' + p), { status: 404 });
}
export function stream(runId, onEvent) {
  const r = S.runs.find((x) => x.id === runId); if (!r) return;
  r.events.slice().forEach(onEvent);
  if (r.status !== 'running' && r.events.some((e) => e.type === 'run.done' || e.type === 'run.failed')) return;
  r.subs.add(onEvent); return () => r.subs.delete(onEvent);
}
