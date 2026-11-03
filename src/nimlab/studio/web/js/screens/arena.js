// Arena: scoreboard with odometer Elo, quality/latency scatter, and a router you can watch think.
import { h, clear, shortModel, pop, reduced, sleep, clamp, fmtNum } from '../util.js';
import { icon } from '../icons.js';
import { store, on, api, toast, refresh } from '../app.js';

const IDEAS = ['Fix this regex that matches emails', 'Explain quantum tunnelling to a teenager', 'Summarise this contract clause'];
const AR = { prompt: '' };

function normBoard(runs) {
  for (const r of [...runs].sort((a, b) => String(b.created_at || b.started_at || '').localeCompare(String(a.created_at || a.started_at || '')))) {
    const res = r.result || r; let lb = res.standings || res.leaderboard || res.ranking || res.rankings || res.rows || res.models;
    if (lb && !Array.isArray(lb)) lb = Object.entries(lb).map(([model, v]) => ({ model, ...(typeof v === 'object' ? v : { elo: v }) }));
    if (Array.isArray(lb) && lb.length) return lb.map((x) => ({ model: x.model || x.id || x.name, elo: x.elo ?? x.rating ?? null, q: x.quality ?? x.overall ?? x.score ?? x.mean_score ?? null, lat: x.latency_ms ?? x.latency_p50_ms ?? (x.latency_p50_s != null ? x.latency_p50_s * 1000 : null) ?? x.p50_ms ?? x.latency ?? null, ci: Array.isArray(x.ci) ? (x.ci[1] - x.ci[0]) / 2 : x.elo_hi != null && x.elo_lo != null ? (x.elo_hi - x.elo_lo) / 2 : x.ci ?? x.elo_ci ?? null })).filter((x) => x.model).sort((a, b) => (b.elo ?? b.q ?? 0) - (a.elo ?? a.q ?? 0));
  }
  return [];
}
function odometer(value, delay) {
  const s = String(Math.round(value)).padStart(4, ' '); const wrap = h('span.odo', { 'aria-label': String(Math.round(value)), role: 'text' });
  [...s].forEach((c, i) => { const col = h('span.dg', { 'aria-hidden': 'true' }, h('span.strip', Array.from({ length: 10 }, (_, d) => h('span', d)))); const strip = col.firstChild; if (c === ' ') col.style.visibility = 'hidden'; wrap.append(col); strip.style.transform = 'translateY(0)'; requestAnimationFrame(() => requestAnimationFrame(() => { strip.style.transitionDelay = `${delay + i * 90}ms`; strip.style.transform = `translateY(${-(+c || 0) * 10}%)`; })); });
  return wrap;
}

export function mount(el) {
  const offs = []; let board = [];
  const rows = h('div.sc-rows', { role: 'table', 'aria-label': 'Elo leaderboard' }); const scat = h('div.scat');
  const board_ = h('section.panel.sc', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Scoreboard · Elo'), h('span.dim.mono', { id: 'ar-src' })), h('div.sc-head.mono', h('span', '#'), h('span', 'Model'), h('span', 'Elo'), h('span', 'Quality'), h('span.r', 'p50')), rows);
  const scatP = h('section.panel.sp', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Quality vs latency'), h('span.dim.mono', 'up and left wins')), scat);

  // ---- router ----
  const inp = h('input.field.rt-in', { type: 'text', placeholder: 'Type a prompt and watch the router decide…', 'aria-label': 'Prompt to route', value: AR.prompt, on: { keydown: (e) => { if (e.key === 'Enter') route(); } } });
  const goBtn = h('button.btn.primary', { on: { click: route } }, icon('route'), 'Route');
  const flow = h('div.flow');
  const router = h('section.panel.rt', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Smart router')), h('div.rt-top', inp, goBtn), h('div.chips', IDEAS.map((t) => h('button.chip', { on: { click: () => { inp.value = t; route(); } } }, t))), flow);
  flow.append(h('p.dim.rt-idle', 'The router classifies the prompt, scores every candidate on quality minus latency and cost, then plugs in the winner.'));
  if (store.prefill?.arena) { inp.value = store.prefill.arena; delete store.prefill.arena; }

  async function route() {
    const prompt = inp.value.trim(); if (!prompt) { toast('Type a prompt to route', 'warn'); inp.focus(); return; }
    AR.prompt = prompt; goBtn.disabled = true; clear(flow); flow.append(h('div.skeleton', { style: { height: '90px' } }));
    try {
      const r = await api.post('/arena/route', { prompt }); r.chosen ||= r.model; const cand = (r.candidates || []).map((c) => ({ ...c, latency: c.latency ?? (c.latency_s != null ? c.latency_s * 1000 : null) })).sort((a, b) => (b.utility ?? -1) - (a.utility ?? -1) || (a.model === r.chosen ? -1 : 0)); const hasU = cand.some((c) => c.utility != null); clear(flow);
      const cat = h('div.step.s1', h('span.label', '1 · Classify'), h('div.cat', h('i'), r.category || 'unknown')); flow.append(pop(cat, { y: 10 }));
      await sleep(reduced() ? 0 : 450);
      const max = Math.max(...cand.map((c) => c.utility ?? 0), .01), min = Math.min(...cand.map((c) => c.utility ?? 0), 0);
      const list = h('div.step.s2', h('span.label', '2 · Score candidates'), cand.map((c) => h('div.cd', { 'data-m': c.model }, h('span.mn', shortModel(c.model)), h('div.ub', h('i', { style: { width: '0%' } })), h('span.num', c.utility != null ? c.utility.toFixed(2) : 'n/a'), h('span.num.dim', c.quality != null ? `q ${c.quality.toFixed(2)} \u00b7 ${Math.round(c.latency ?? 0)}ms` : `cost tier ${c.cost_tier ?? '?'} \u00b7 no leaderboard data yet`)))); flow.append(pop(list, { y: 10 }));
      await sleep(80);
      list.querySelectorAll('.cd').forEach((row, i) => { const c = cand[i]; setTimeout(() => (row.querySelector('i').style.width = (hasU ? clamp((((c.utility ?? min) - min) / (max - min || 1)) * 90 + 10, 6, 100) : c.model === r.chosen ? 100 : 55) + '%'), i * 160); });
      await sleep(reduced() ? 0 : cand.length * 160 + 400);
      const won = list.querySelector(`[data-m="${CSS.escape(r.chosen)}"]`) || list.querySelector('.cd'); won?.classList.add('won');
      const plate = h('div.step.s3', h('span.label', '3 · Routed to'), h('div.plate', icon('route'), h('b', shortModel(r.chosen)), h('small.mono', r.chosen)), r.reasons?.length ? h('ul.why', r.reasons.map((x) => h('li', x))) : null);
      flow.append(pop(plate, { y: 10, s: .9 })); scat.querySelectorAll('.pt').forEach((p) => p.classList.toggle('chosen', p.dataset.m === r.chosen));
    } catch (e) { clear(flow); flow.append(h('p.dim', 'Routing failed: ' + e.message)); toast('Routing failed: ' + e.message, 'bad'); } finally { goBtn.disabled = false; }
  }

  // ---- board ----
  function paintBoard() {
    clear(rows); document.getElementById('ar-src') && (document.getElementById('ar-src').textContent = board.length ? `${board.length} models` : '');
    if (!board.length) { rows.append(h('div.empty', icon('arena'), h('h3', 'No matches played yet'), h('p', 'Run a mini arena to pit models against each other and fill the scoreboard.'), h('button.btn.primary', { on: { click: runArena } }, icon('play'), 'Run mini arena'))); clear(scat); scat.append(h('div.empty', h('p', 'The scatter appears once there is a leaderboard.'))); return; }
    const mx = Math.max(...board.map((b) => b.q ?? 0), .01);
    board.forEach((b, i) => rows.append(h('div.sc-row', { role: 'row', 'data-m': b.model, class: i === 0 ? 'lead' : '', on: { mouseenter: () => hl(b.model, true), mouseleave: () => hl(b.model, false) } },
      h('span.rk.num', i + 1), h('span.md', h('b', shortModel(b.model)), h('small', b.model.includes('/') ? b.model.split('/')[0] : '')),
      h('span.el', b.elo != null ? odometer(b.elo, i * 70) : h('span.dim', '—'), b.ci != null && h('small.num', '±' + Math.round(b.ci))),
      h('span.qb', h('i', { style: { width: ((b.q ?? 0) / mx) * 100 + '%' } }), h('b.num', b.q != null ? b.q.toFixed(2) : '—')), h('span.num.r.dim', b.lat != null ? Math.round(b.lat) + 'ms' : '—'))));
    paintScatter();
  }
  const hl = (m, on_) => { document.querySelectorAll(`[data-m="${CSS.escape(m)}"]`).forEach((n) => n.classList.toggle('hl', on_)); };
  function paintScatter() {
    const pts = board.filter((b) => b.q != null && b.lat != null); clear(scat); if (pts.length < 2) { scat.append(h('p.dim', { style: { padding: '16px' } }, 'Needs quality and latency per model.')); return; }
    const W = 520, H = 330, m = { l: 46, r: 18, t: 16, b: 40 }, maxL = Math.max(...pts.map((p) => p.lat)) * 1.1, qs = pts.map((p) => p.q), q0 = Math.max(0, Math.min(...qs) - .08), q1 = Math.min(1, Math.max(...qs) + .05);
    const X = (l) => m.l + (l / maxL) * (W - m.l - m.r), Y = (q) => H - m.b - ((q - q0) / (q1 - q0 || 1)) * (H - m.t - m.b);
    const frontier = []; let best = -1; [...pts].sort((a, b) => a.lat - b.lat).forEach((p) => { if (p.q > best) { best = p.q; frontier.push(p); } });
    const elos = pts.map((p) => p.elo ?? 1000), e0 = Math.min(...elos), e1 = Math.max(...elos);
    let g = ''; for (let i = 0; i <= 4; i++) { const l = (maxL / 4) * i, q = q0 + ((q1 - q0) / 4) * i; g += `<line x1="${X(l)}" y1="${m.t}" x2="${X(l)}" y2="${H - m.b}" class="gl"/><text x="${X(l)}" y="${H - m.b + 16}" class="ax" text-anchor="middle">${l >= 1000 ? (l / 1000).toFixed(1) + 's' : Math.round(l) + 'ms'}</text><line x1="${m.l}" y1="${Y(q)}" x2="${W - m.r}" y2="${Y(q)}" class="gl"/><text x="${m.l - 8}" y="${Y(q) + 3}" class="ax" text-anchor="end">${q.toFixed(2)}</text>`; }
    const fl = frontier.map((p, i) => (i ? 'L' : 'M') + X(p.lat) + ' ' + Y(p.q)).join(' ');
    const dots = pts.map((p) => { const r = 5 + (((p.elo ?? 1000) - e0) / (e1 - e0 || 1)) * 6, f = frontier.includes(p); return `<g class="pt ${f ? 'fr' : ''}" data-m="${p.model}" tabindex="0"><title>${shortModel(p.model)}: quality ${p.q.toFixed(2)}, ${Math.round(p.lat)} ms${p.elo ? ', Elo ' + Math.round(p.elo) : ''}</title><circle cx="${X(p.lat)}" cy="${Y(p.q)}" r="${r}" /><text x="${X(p.lat) + r + 4}" y="${Y(p.q) + 3.5}" class="lb">${shortModel(p.model).replace(/-instruct|-v\d.*$/g, '').slice(0, 20)}</text></g>`; }).join('');
    scat.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Scatter plot of quality against latency">${g}<path d="${fl}" class="front" fill="none"/><text x="${W / 2}" y="${H - 4}" class="ax" text-anchor="middle">latency (p50)</text><text x="12" y="${H / 2}" class="ax" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">quality</text>${dots}</svg>`;
    scat.querySelectorAll('.pt').forEach((n) => { n.addEventListener('mouseenter', () => hl(n.dataset.m, true)); n.addEventListener('mouseleave', () => hl(n.dataset.m, false)); });
  }
  async function runArena() { try { const r = await api.post('/arena/run', {}); toast('Arena started. Matches stream in as they finish.'); api.stream(r.run_id, (ev) => { if (ev.type === 'item') { const t = document.getElementById('ar-src'); if (t) t.textContent = `playing ${ev.data.done}/${ev.data.total}`; } if (ev.type === 'run.done') { load(); refresh(['status', 'runs']); } if (ev.type === 'run.failed') toast('Arena failed: ' + (ev.data?.error || ''), 'bad'); }); } catch (e) { toast(e.message, 'bad'); } }
  async function load() { try { const r = await api.get('/arena/runs'); board = normBoard(Array.isArray(r) ? r : r.runs || []); } catch { board = []; } paintBoard(); }

  el.append(h('div.ar', h('header.ar-h', h('div.label', 'Arena'), h('h1', 'Who’s winning, ', h('em', 'and who gets the call.'))), h('div.ar-grid', board_, scatP), router));
  rowsSkeleton(); load();
  function rowsSkeleton() { for (let i = 0; i < 5; i++) rows.append(h('div.skeleton', { style: { height: '40px', margin: '6px 16px' } })); }
  offs.push(on('prefill', (d) => { if (d.screen === 'arena') { inp.value = d.text; route(); } }));
  if (inp.value && store.prefill == null) {}
  return () => offs.forEach((f) => f && f());
}
