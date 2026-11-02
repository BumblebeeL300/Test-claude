// Switchboard: four modules + the NIM Core, patched together with live cables.
import { h, clear, ago, fmtNum, shortModel, pop, reduced, clamp } from '../util.js';
import { icon } from '../icons.js';
import { store, on, go, api, appStatus, isSim, refresh } from '../app.js';

const MODS = [
  { id: 'forge', name: 'Forge', route: 'foundry', n: '01', key: '2', tag: 'Plan, build, test, ship', verb: 'Open Foundry', area: 'a' },
  { id: 'council', name: 'Council', route: 'council', n: '02', key: '3', tag: 'Many models, one verdict', verb: 'Convene', area: 'b' },
  { id: 'atlas', name: 'Atlas', route: 'atlas', n: '03', key: '4', tag: 'Answers grounded in your documents', verb: 'Ask your docs', area: 'c' },
  { id: 'arena', name: 'Arena', route: 'arena', n: '04', key: '5', tag: 'Benchmark, rank and route', verb: 'Enter the arena', area: 'd' },
];

function normUsage(u) {
  if (!u) return { rows: [], totals: {} };
  let m = u.models || u.by_model || u.per_model || u.model_usage || [];
  if (!Array.isArray(m)) m = Object.entries(m).map(([model, v]) => ({ model, ...v }));
  const rows = m.map((r) => ({ model: r.model || r.id || r.name, calls: r.calls ?? r.requests ?? 0, tokens: r.tokens ?? r.total_tokens ?? ((r.prompt_tokens || r.tokens_in || 0) + (r.completion_tokens || r.tokens_out || 0)), lat: r.latency_ms ?? r.avg_latency_ms ?? (r.avg_latency_s != null ? r.avg_latency_s * 1000 : null) ?? r.latency ?? null }));
  return { rows: rows.sort((a, b) => b.calls - a.calls), totals: u.totals || u.total || {} };
}

export function mount(el) {
  const offs = [];
  const mods = {}; // id -> refs
  const svg = h('svg.cables', { 'aria-hidden': 'true' });
  const core = h('button.module.core', { 'aria-label': 'NIM Core engine: open settings', on: { click: () => import('./settings.js').then((m) => m.openSettings()) } });
  const grid = h('div.rack-grid', core, MODS.map((m) => {
    const card = h('button.module', { class: 'app-' + m.id, style: { gridArea: m.area }, 'aria-label': `${m.name}: ${m.tag}`, on: { click: () => go(m.route) } });
    mods[m.id] = { card, m };
    return card;
  }));
  const rack = h('div.rack', svg, grid);
  const tape = h('div.tape.hidden-scroll', { role: 'list', 'aria-label': 'Recent runs' });
  const usage = h('div.usage');

  el.append(h('div.sb',
    h('header.sb-h', h('div', h('div.label', 'Switchboard'), h('h1', 'Everything on the bench, ', h('em', 'patched in.'))), h('div.sb-sub.mono', { id: 'sb-sub' })),
    rack,
    h('section.sb-sec', h('div.sec-h', h('span.label', 'Recent runs'), h('span.rule')), tape),
    h('section.sb-sec', h('div.sec-h', h('span.label', 'Signal level · calls per model'), h('span.rule'), h('span.label#usage-tot')), usage)));

  // ---- module skeletons ----
  for (const { card, m } of Object.values(mods)) {
    const r = mods[m.id];
    r.leds = { ready: h('span.led'), busy: h('span.led'), key: h('span.led') };
    r.runs = h('b.num.big', '0'); r.last = h('span.num', '—'); r.state = h('span.label');
    r.vu = h('div.vu.idle', Array.from({ length: 14 }, (_, i) => h('i', { style: { '--i': i } })));
    card.append(
      h('i.screw.tl'), h('i.screw.tr'), h('i.screw.bl'), h('i.screw.br'),
      h('div.m-top', h('span.label', `MOD ${m.n}`), h('span.leds', h('span.lg', r.leds.ready, h('span.label', 'RDY')), h('span.lg', r.leds.busy, h('span.label', 'BSY')), h('span.lg', r.leds.key, h('span.label', 'KEY')))),
      h('div.m-main', h('div.bezel', icon(m.id === 'forge' ? 'forge' : m.id)), h('div', h('h2', m.name), h('p', m.tag))),
      h('div.m-stats', h('div', h('span.label', 'Runs'), r.runs), h('div', h('span.label', 'Last run'), r.last), r.vu),
      h('div.m-foot', r.state, h('span.go', m.verb, icon('chevron'), h('kbd', '⌘' + m.key))));
    r.jack = h('i.jack'); card.append(r.jack);
  }
  const c = { mode: h('span.mode'), key: h('code.mono'), models: h('b.num.big', '—'), net: h('span.led'), netl: h('span.label'), base: h('span.mono.dim') };
  core.append(h('i.screw.tl'), h('i.screw.tr'), h('i.screw.bl'), h('i.screw.br'),
    h('div.m-top', h('span.label', 'NIM CORE'), h('span.lg', c.net, c.netl)),
    h('div.core-dial', { html: '<svg viewBox="0 0 120 120" aria-hidden="true"><circle cx="60" cy="60" r="54" fill="none" stroke="currentColor" stroke-opacity=".2" stroke-width="1"/><circle class="ring" cx="60" cy="60" r="46" fill="none" stroke="var(--c-core)" stroke-width="3" stroke-dasharray="6 5" stroke-linecap="round"/><circle class="ring r2" cx="60" cy="60" r="34" fill="none" stroke="var(--c-atlas)" stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"/><circle cx="60" cy="60" r="8" fill="var(--c-core)"/></svg>' }),
    h('h2', 'NIM Engine'), c.mode,
    h('div.core-stats', h('div', h('span.label', 'Models'), c.models), h('div', h('span.label', 'Key'), c.key)), c.base,
    h('i.jack.j-core'));

  // ---- paint ----
  function paint() {
    const s = store.status; if (!s) return;
    if (!el.isConnected) return;
    document.getElementById('sb-sub').textContent = `${s.mode === 'live' ? 'LIVE' : 'SIMULATION'} · v${s.version || '?'} · ${store.runs.filter((r) => r.status === 'running').length} running`;
    for (const { m } of Object.values(mods)) {
      const a = appStatus(m.id), r = mods[m.id], st = a.status || 'ready';
      r.card.dataset.status = st; r.card.classList.toggle('busy', st === 'busy');
      r.leds.ready.className = 'led' + (st === 'ready' ? ' on' : ''); r.leds.busy.className = 'led' + (st === 'busy' ? ' busy' : ''); r.leds.key.className = 'led' + (st === 'needs-key' ? ' warn' : '');
      r.vu.classList.toggle('idle', st !== 'busy');
      r.state.textContent = st === 'busy' ? 'RUNNING' : st === 'needs-key' ? 'NEEDS KEY' : 'STANDING BY';
      r.last.textContent = a.last_run_at ? ago(a.last_run_at) : 'never';
      const to = a.runs ?? 0, from = +r.runs.dataset.v || 0; r.runs.dataset.v = to;
      if (from !== to) { const t0 = performance.now(); const tick = (now) => { const p = clamp((now - t0) / 700, 0, 1); r.runs.textContent = String(Math.round(from + (to - from) * (1 - (1 - p) ** 3))).padStart(2, '0'); if (p < 1 && el.isConnected) requestAnimationFrame(tick); }; requestAnimationFrame(tick); }
    }
    c.mode.className = 'mode ' + (isSim() ? 'sim' : 'live'); c.mode.replaceChildren(h('span.led', { class: isSim() ? 'warn' : 'on' }), isSim() ? 'SIMULATION' : 'LIVE');
    c.key.textContent = s.key?.set ? s.key.masked || 'nvapi-••••' : 'no key';
    c.models.textContent = store.models.length || '—';
    c.net.className = 'led ' + ({ ok: 'on', blocked: 'bad' }[s.network] || 'warn'); c.netl.textContent = { ok: 'ONLINE', blocked: 'BLOCKED' }[s.network] || 'UNPROBED';
    c.base.textContent = (s.base_url || '').replace(/^https?:\/\//, '');
    cables();
  }
  function paintRuns() {
    clear(tape);
    if (!store.runs.length) { tape.append(h('div.empty', { style: { width: '100%', padding: '18px' } }, icon('terminal'), h('h3', 'The tape is blank'), h('p', 'Start a build, a council or a question and it will print here.'))); return; }
    store.runs.slice(0, 24).forEach((r, i) => {
      const cell = h('button.cell', { role: 'listitem', class: 'app-' + r.app, on: { click: () => go(r.app === 'forge' ? 'foundry' : r.app, { run: r.id, ...(r.project ? { project: r.project } : {}) }) } },
        h('div.cell-top', h('span.label', r.app), h('span.st', { class: r.status }, h('span.led', { class: r.status === 'done' ? 'on' : r.status === 'running' ? 'busy' : 'bad' }), r.status), r.simulated && h('span.sim', 'SIM')),
        h('div.cell-t', r.title || r.id), h('div.cell-a.mono', ago(r.started_at)));
      tape.append(pop(cell, { delay: i * 35, y: 10, s: .96 }));
    });
  }
  function paintUsage() {
    const { rows, totals } = normUsage(store.usage); clear(usage); if (!el.isConnected) return;
    document.getElementById('usage-tot').textContent = totals.calls != null ? `${fmtNum(totals.calls)} calls · ${fmtNum(totals.tokens ?? totals.total_tokens)} tokens` : '';
    if (!rows.length) { usage.append(h('div.empty', { style: { gridColumn: '1/-1' } }, icon('arena'), h('h3', 'No signal yet'), h('p', 'Model usage shows up here as soon as something runs.'))); return; }
    const max = Math.max(...rows.map((r) => r.calls), 1), SEG = 24;
    rows.slice(0, 8).forEach((r, i) => {
      const lit = Math.max(1, Math.round((r.calls / max) * SEG));
      const bar = h('div.vu-bar', { role: 'img', 'aria-label': `${r.calls} calls` }, Array.from({ length: SEG }, (_, k) => h('i', { class: k < lit ? (k > SEG * .8 ? 'on hot' : k > SEG * .6 ? 'on mid' : 'on') : '', style: { '--d': `${k * 18 + i * 40}ms` } })));
      usage.append(h('div.vu-row', h('span.mn', shortModel(r.model)), bar, h('span.num', r.calls), h('span.num.dim', r.lat != null ? Math.round(r.lat) + 'ms' : '')));
    });
  }

  // ---- cables ----
  function cables() {
    const rr = rack.getBoundingClientRect(); if (!rr.width) return;
    const cr = core.getBoundingClientRect(); const P = (r) => r;
    svg.setAttribute('viewBox', `0 0 ${rr.width} ${rr.height}`); clear(svg);
    const defs = h('defs'); svg.append(defs);
    for (const { m, card } of Object.values(mods)) {
      const mr = card.getBoundingClientRect(), mx = mr.left - rr.left + mr.width / 2, my = mr.top - rr.top + mr.height / 2;
      const cl = cr.left - rr.left, cr_ = cl + cr.width, ct = cr.top - rr.top, cb = ct + cr.height;
      let a, b, d;
      if (mx < cl - 8 || mx > cr_ + 8) { // beside core
        const left = mx < cl; a = [left ? mr.right - rr.left : mr.left - rr.left, my]; b = [left ? cl : cr_, clamp(my, ct + 40, cb - 40)];
        const k = Math.abs(b[0] - a[0]) * .5; d = `M${a[0]} ${a[1]} C${a[0] + (left ? k : -k)} ${a[1]} ${b[0] + (left ? -k : k)} ${b[1]} ${b[0]} ${b[1]}`;
      } else { // stacked under core
        a = [mx, mr.top - rr.top]; b = [clamp(mx, cl + 30, cr_ - 30), cb]; const k = Math.abs(a[1] - b[1]) * .5; d = `M${a[0]} ${a[1]} C${a[0]} ${a[1] - k} ${b[0]} ${b[1] + k} ${b[0]} ${b[1]}`;
      }
      const busy = card.classList.contains('busy'), col = `var(--c-${m.id})`;
      svg.append(h('path.cab-base', { d }), h('path.cab-core', { d, stroke: col }), h('path.cab-flow', { d, stroke: col, class: busy ? 'on' : '' }));
      for (const [x, y] of [a, b]) svg.append(h('circle.plug', { cx: x, cy: y, r: 5, stroke: col }));
    }
  }
  const ro = new ResizeObserver(cables); ro.observe(rack); ro.observe(core);
  offs.push(() => ro.disconnect(), on('status', paint), on('runs', () => { paintRuns(); paint(); }), on('usage', paintUsage), on('models', paint));
  paint(); paintRuns(); paintUsage();
  refresh(['usage', 'models']).then(() => { paintUsage(); paint(); });
  requestAnimationFrame(() => { cables(); [...grid.children].forEach((c, i) => pop(c, { delay: i * 70, y: 16, s: .96 })); });
  setTimeout(cables, 450);
  return () => offs.forEach((f) => f && f());
}
