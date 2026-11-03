// Distill: a refinery column. Samples rise as droplets (still -> condenser -> filter -> collector),
// rejects drip off to a waste tank with their reason, and the funnel and reward histogram fill live.
import { h, clear, pop, reduced, clamp, fmtNum, shortModel } from '../util.js';
import { icon } from '../icons.js';
import { api, toast, refresh } from '../app.js';
import { modelOverrides } from '../components/models.js';
import { token } from '../api.js';

const TIERS = [
  { id: 'collector', name: 'Collector flask', hint: 'final dataset', stages: ['finalize'] },
  { id: 'filter', name: 'Filter bed', hint: 'filters, dedup, decontamination', stages: ['filter', 'dedup', 'decontaminate'] },
  { id: 'condenser', name: 'Condenser', hint: 'reward scoring', stages: ['score'] },
  { id: 'still', name: 'Still', hint: 'seed and generate', stages: ['seed', 'generate'] },
];
const FUNNEL = [['generate', 'Generated'], ['score', 'Scored'], ['filter', 'Passed filters'], ['dedup', 'After dedup'], ['decontaminate', 'Decontaminated']];
const REASON_COLOR = { refusal: 'var(--bad)', error: 'var(--bad)', unscored: 'var(--bad)', low_score: 'var(--warn)', low_margin: 'var(--warn)', too_short: 'var(--c-arena)', too_long: 'var(--c-arena)', short_prompt: 'var(--c-arena)', language: 'var(--c-atlas)', too_easy: 'var(--c-atlas)', repetition: 'var(--c-core)', ungrounded: 'var(--c-core)', duplicate_exact: 'var(--c-council)', duplicate_near: 'var(--c-council)', contaminated: 'var(--c-forge)' };
const reasonColor = (r) => REASON_COLOR[r] || 'var(--dim)';
const WHY = { refusal: 'refusal', low_score: 'low score', too_short: 'too short', too_long: 'too long', short_prompt: 'short prompt', language: 'not English', repetition: 'repetitive', low_margin: 'pair too close', ungrounded: 'not grounded', too_easy: 'too easy', duplicate_exact: 'exact duplicate', duplicate_near: 'near duplicate', contaminated: 'eval overlap', error: 'call failed', unscored: 'unscored' };
const MAX_DROPLETS = 40;
const D = { runId: null, stop: null, status: 'idle', recipe: 'instruction_diversity', n: 60, models: '', docs: '', evalSet: '', fmt: null };

const tierOf = (stage) => TIERS.find((t) => t.stages.includes(stage))?.id || 'still';
const scoreTone = (s) => (s.score_source === 'judge' ? (s.score >= 7 ? 'ok' : s.score >= 6 ? 'warn' : 'bad') : s.score >= -2 ? 'ok' : s.score >= -8 ? 'warn' : 'bad');

export async function mount(el) {
  let recipes = [], runs = [], simulated = null, replayUntil = 0;
  let cards = [], cardIdx = 0, cardStage = 'final', flipped = false, runFormats = ['openai-chat'];
  let live = fresh();
  const offs = [];
  function fresh() { return { gen: 0, total: 0, scored: 0, drops: {}, waste: {}, scores: [], clusters: 0, stage: null, seen: new Set(), done: new Set(), funnel: null, final: undefined, active: 0 }; }

  // ------------------------------------------------------------------ controls
  const recipeBox = h('div.dt-recipes', { role: 'radiogroup', 'aria-label': 'Recipe' });
  const nOut = h('output.num', D.n);
  const nRange = h('input', { type: 'range', min: 20, max: 300, step: 10, value: D.n, 'aria-label': 'Number of candidate samples', on: { input: () => { D.n = +nRange.value; nOut.textContent = D.n; } } });
  const models = h('input.field', { type: 'text', placeholder: 'default panel (comma-separated ids)', value: D.models, 'aria-label': 'Generator models', on: { input: () => (D.models = models.value) } });
  const docs = h('input.field', { type: 'text', placeholder: '/path/to/docs (optional in simulation)', value: D.docs, 'aria-label': 'Documents folder', on: { input: () => (D.docs = docs.value) } });
  const docsRow = h('label.dt-f', h('span.label', 'Documents'), docs);
  const evalIn = h('input.field', { type: 'text', placeholder: '/path/to/eval.jsonl (optional)', value: D.evalSet, 'aria-label': 'Eval set to decontaminate against', on: { input: () => (D.evalSet = evalIn.value) } });
  const goBtn = h('button.btn.primary', { on: { click: start } }, icon('play'), 'Start distilling');
  const stopBtn = h('button.btn.danger', { hidden: true, on: { click: cancel } }, icon('stop'), 'Cancel');
  const status = h('p.dt-status.mono', { 'aria-live': 'polite' }, 'Idle. Pick a recipe and start.');
  const ctl = h('section.panel.dt-ctl', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Recipe')), recipeBox,
    h('div.dt-ctl-b', h('label.dt-f', h('span.label', 'Candidates'), h('div.dt-n', nRange, nOut)), h('label.dt-f', h('span.label', 'Models'), models), docsRow,
      h('label.dt-f', h('span.label', 'Decontaminate against'), evalIn), h('div.dt-go', goBtn, stopBtn), status));

  // ------------------------------------------------------------------ refinery column
  const drops = h('div.rf-drops', { 'aria-hidden': 'true' });
  const tierEls = {};
  const tierNodes = TIERS.map((t) => {
    const count = h('b.num', '0'), sub = h('small.mono', t.hint);
    const node = h('div.rf-tier', { 'data-tier': t.id }, h('div.rf-vessel', h('i.rf-liquid'), h('i.rf-glass')), h('div.rf-lab', h('span.label', t.name), count, sub));
    tierEls[t.id] = { node, count, sub, liquid: node.querySelector('.rf-liquid') };
    return node;
  });
  const pipe = h('div.rf-pipe', h('i'));
  const wasteList = h('ul.rf-wl');
  const wasteCount = h('b.num', '0');
  const waste = h('div.rf-waste', h('div.rf-tank', h('i.rf-liquid')), h('div.rf-wh', h('span.label', 'Waste'), wasteCount), wasteList);
  const column = h('section.panel.dt-col', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Distillation column'), h('span.dim.mono#dt-src')),
    h('div.rf', { role: 'img', 'aria-label': 'Distillation column: samples rise from the still through the condenser and filter into the collector; rejects drip to waste' }, pipe, h('div.rf-tiers', tierNodes), waste, drops));

  // ------------------------------------------------------------------ funnel + histogram
  const funnelRows = FUNNEL.map(([id, label]) => {
    const bar = h('i'), n = h('b.num', '0'), why = h('small.mono');
    return { id, node: h('div.fn-row', h('span.fn-l', label, n), h('div.fn-t', bar), why), bar, n, why };
  });
  const funnel = h('section.panel.dt-fn', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Live funnel')), h('div.fn', funnelRows.map((r) => r.node)));
  const hist = h('div.dt-hist', { role: 'img', 'aria-label': 'Histogram of reward scores' });
  const histNote = h('p.dim.dt-hn', 'Scores appear as the condenser works.');
  const histP = h('section.panel.dt-hp', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Reward histogram')), hist, histNote);

  // ------------------------------------------------------------------ inspector + export
  const stack = h('div.dt-stack', { tabindex: 0, role: 'group', 'aria-label': 'Sample cards. Use the arrow keys to flip through.', on: { keydown: (e) => { if (e.key === 'ArrowRight') step(1); else if (e.key === 'ArrowLeft') step(-1); else if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); flip(); } } } });
  const counter = h('span.num.dim', '0 / 0');
  const stageChips = [['final', 'Kept'], ['dropped', 'Dropped'], ['all', 'All']].map(([id, label]) => h('button.chip', { 'aria-pressed': String(id === cardStage), 'data-s': id, on: { click: () => loadCards(id) } }, label));
  const nav = h('div.dt-nav', h('button.btn.sm.icon', { 'aria-label': 'Previous sample', on: { click: () => step(-1) } }, h('span', { style: { display: 'inline-flex', transform: 'rotate(180deg)' } }, icon('chevron'))), counter, h('button.btn.sm.icon', { 'aria-label': 'Next sample', on: { click: () => step(1) } }, icon('chevron')), h('button.btn.sm', { on: { click: flip } }, icon('refresh'), 'Flip'));
  const fmtBox = h('div.chips', { role: 'radiogroup', 'aria-label': 'Export format' });
  const exportA = h('a.btn.primary', { href: '#', download: '' }, icon('upload'), 'Export JSONL');
  const reportA = h('a.btn', { href: '#', target: '_blank', rel: 'noopener' }, icon('doc'), 'Quality report');
  const cardA = h('a.btn', { href: '#', target: '_blank', rel: 'noopener' }, icon('file'), 'Dataset card');
  const exportBox = h('div.dt-exp', { hidden: true }, h('span.label', 'Export'), fmtBox, h('div.dt-exp-b', exportA, reportA, cardA));
  const insp = h('section.panel.dt-insp', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Sample inspector'), h('div.chips', stageChips)), stack, nav, exportBox);

  const pastList = h('div.dt-past');
  const past = h('section.panel.dt-pastp', h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', 'Earlier runs')), pastList);

  el.append(h('div.dt', h('header.dt-h', h('div.label', 'Distill'), h('h1', 'Refine the noise ', h('em', 'into signal.')), h('p.dim', 'Generate with several models, score with the reward model, filter, deduplicate and decontaminate, then export.')),
    h('div.dt-grid', ctl, column, h('div.dt-side', funnel, histP)), insp, past));

  // ------------------------------------------------------------------ painting helpers
  let raf = 0;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; paintLive(); }); };
  function paintLive() {
    tierEls.still.count.textContent = fmtNum(live.gen);
    tierEls.condenser.count.textContent = fmtNum(live.scored);
    const dropFilter = (live.drops.filter || 0) + (live.drops.dedup || 0) + (live.drops.decontaminate || 0);
    const reachedFilter = live.seen.has('filter');
    tierEls.filter.count.textContent = reachedFilter ? fmtNum(Math.max(0, live.scored - dropFilter)) : '0';
    const total = Math.max(live.total, 1);
    const level = (id, v) => (tierEls[id].liquid.style.height = clamp((v / total) * 100, 0, 100) + '%');
    level('still', live.gen); level('condenser', live.scored); level('filter', reachedFilter ? live.scored - dropFilter : 0);
    const outs = { generate: live.gen, score: live.scored, filter: reachedFilter ? live.scored - (live.drops.filter || 0) : 0 };
    outs.dedup = live.seen.has('dedup') ? outs.filter - (live.drops.dedup || 0) : 0;
    outs.decontaminate = live.seen.has('decontaminate') ? outs.dedup - (live.drops.decontaminate || 0) : 0;
    const f = live.funnel;
    funnelRows.forEach((r, i) => {
      const exact = f && live.done.has(r.id);
      const out = exact ? f[i].out : outs[r.id], dropped = exact ? f[i].dropped : (live.drops[r.id] || 0);
      r.n.textContent = out; r.bar.style.width = clamp((out / total) * 100, 0, 100) + '%';
      const reasons = exact ? Object.entries(f[i].reasons || {}) : [];
      r.why.textContent = dropped ? `−${dropped}${reasons.length ? ' · ' + reasons.map(([k, v]) => `${WHY[k] || k} ${v}`).join(', ') : ''}` : '';
    });
    const wasteTotal = Object.values(live.waste).reduce((a, b) => a + b, 0);
    wasteCount.textContent = fmtNum(wasteTotal);
    waste.querySelector('.rf-tank .rf-liquid').style.height = clamp((wasteTotal / total) * 100 * 1.6, 0, 100) + '%';
    clear(wasteList);
    Object.entries(live.waste).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => wasteList.append(h('li', { style: { '--c': reasonColor(k) } }, h('i'), h('span', WHY[k] || k), h('b.num', v))));
    paintHist(live.final);
  }
  function paintHist(final) {
    const W = 320, H = 130, m = { l: 8, r: 8, t: 8, b: 20 };
    let edges, all, kept;
    if (final) ({ edges, all, kept } = final);
    else if (live.scores.length) {
      const reward = live.scores.filter((s) => s.src !== 'judge').map((s) => s.v), vals = reward.length ? reward : live.scores.map((s) => s.v);
      const lo = Math.min(...vals), hi = Math.max(...vals, lo + 1), bins = 14; edges = Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins);
      all = new Array(bins).fill(0); vals.forEach((v) => { all[Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins))]++; }); kept = null;
    }
    if (!edges) { hist.replaceChildren(); return; }
    const top = Math.max(...all, 1), bw = (W - m.l - m.r) / all.length, bh = H - m.t - m.b;
    let svg = '';
    all.forEach((c, i) => {
      const x = m.l + i * bw, ha = (c / top) * bh, hk = kept ? (kept[i] / top) * bh : 0;
      svg += `<rect class="hb-all" x="${(x + 1).toFixed(1)}" y="${(m.t + bh - ha).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${ha.toFixed(1)}" rx="2"><title>${edges[i].toFixed(1)} to ${edges[i + 1].toFixed(1)}: ${c} scored${kept ? ', ' + kept[i] + ' kept' : ''}</title></rect>`;
      if (kept) svg += `<rect class="hb-kept" x="${(x + 1).toFixed(1)}" y="${(m.t + bh - hk).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${hk.toFixed(1)}" rx="2"/>`;
    });
    [0, 0.5, 1].forEach((p) => { svg += `<text class="hx" x="${(m.l + p * (W - m.l - m.r)).toFixed(1)}" y="${H - 5}" text-anchor="${p === 0 ? 'start' : p === 1 ? 'end' : 'middle'}">${(edges[0] + p * (edges[edges.length - 1] - edges[0])).toFixed(1)}</text>`; });
    hist.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${svg}</svg>`;
    histNote.textContent = final ? 'Bright bars are the scores that survived every filter.' : `${live.scores.length} scored so far.`;
  }

  // droplets: animate between two tiers (or to the waste tank); skipped under reduced motion or load
  function pos(target) { const box = drops.getBoundingClientRect(), r = target.getBoundingClientRect(); return { x: r.left - box.left + r.width / 2, y: r.top - box.top + r.height / 2 }; }
  function droplet(fromId, toId, tone, label, color) {
    if (reduced() || performance.now() < replayUntil || live.active >= MAX_DROPLETS || !drops.isConnected) return;
    const a = pos(tierEls[fromId].node.querySelector('.rf-vessel')), b = toId === 'waste' ? pos(waste.querySelector('.rf-tank')) : pos(tierEls[toId].node.querySelector('.rf-vessel'));
    const jx = (Math.random() - 0.5) * 34, d = h('i.rf-drop', { 'data-tone': tone || '' }, label ? h('span.num', label) : null);
    d.style.left = a.x + 'px'; d.style.top = a.y + 'px'; if (color) d.style.setProperty('--c', color); drops.append(d); live.active++;
    const dx = b.x - a.x + (toId === 'waste' ? 0 : jx), dy = b.y - a.y;
    const frames = toId === 'waste'
      ? [{ transform: 'translate(0,0) scale(1)', opacity: 1 }, { transform: `translate(${dx * 0.35}px,${dy * 0.15 - 14}px) scale(1)`, opacity: 1, offset: 0.35 }, { transform: `translate(${dx}px,${dy + 28}px) scale(.6)`, opacity: 0.2 }]
      : [{ transform: 'translate(0,0) scale(.4)', opacity: 0 }, { transform: `translate(${jx * 0.5}px,${dy * 0.5}px) scale(1)`, opacity: 1, offset: 0.4 }, { transform: `translate(${dx}px,${dy}px) scale(.7)`, opacity: 0.1 }];
    const anim = d.animate(frames, { duration: toId === 'waste' ? 900 : 1100 + Math.random() * 400, easing: 'cubic-bezier(.3,.7,.3,1)' });
    anim.onfinish = anim.oncancel = () => { d.remove(); live.active--; };
  }
  const flash = (tierId, text) => { const n = tierEls[tierId].sub; n.textContent = text; n.classList.add('hot'); setTimeout(() => { n.classList.remove('hot'); n.textContent = TIERS.find((t) => t.id === tierId).hint; }, 1400); };
  const setStage = (stage) => {
    live.stage = stage; live.seen.add(stage);
    const tid = tierOf(stage);
    Object.entries(tierEls).forEach(([id, t]) => t.node.classList.toggle('active', id === tid));
    pipe.classList.toggle('flow', D.status === 'running');
  };

  // ------------------------------------------------------------------ events
  function apply(ev) {
    const d = ev.data || {};
    const replay = ev.ts && Date.now() - Date.parse(ev.ts) > 2500;
    if (replay) replayUntil = performance.now() + 400;
    switch (ev.type) {
      case 'run.config': simulated = !!d.simulated; live.total = d.n || live.total; src(d); break;
      case 'seeds': live.total = d.count; status.textContent = `Seeded ${d.count} prompts across ${d.topics} topics.`; break;
      case 'stage.start': setStage(d.stage); if (d.total) live.total = Math.max(live.total, d.total); status.textContent = `Stage: ${d.stage}…`; break;
      case 'stage.end': live.done.add(d.stage); break;
      case 'sample.generated': live.gen++; droplet('still', 'still', 'raw'); break;
      case 'sample.scored': live.scored++; if (d.score != null) live.scores.push({ v: d.score, src: d.source }); droplet('still', 'condenser', 'scored', d.score != null ? d.score.toFixed(1) : ''); break;
      case 'sample.dropped': {
        live.drops[d.stage] = (live.drops[d.stage] || 0) + 1; live.waste[d.reason] = (live.waste[d.reason] || 0) + 1;
        if (d.stage === 'generate') live.gen++;
        droplet(tierOf(d.stage), 'waste', 'waste', WHY[d.reason] || d.reason, reasonColor(d.reason));
        break;
      }
      case 'dedup.cluster': live.clusters++; flash('filter', `${d.size} → 1 (${d.kind})`); break;
      case 'funnel': live.funnel = d.stages.map((s) => ({ out: s.out, dropped: s.dropped, reasons: s.reasons })); tierEls.collector.count.textContent = fmtNum(d.kept); tierEls.collector.liquid.style.height = clamp((d.kept / Math.max(d.total, 1)) * 100, 0, 100) + '%'; break;
      case 'export': break;
      case 'run.done': finished(d.result); break;
      case 'run.failed': D.status = d.cancelled ? 'cancelled' : 'failed'; idle(d.cancelled ? 'Cancelled. The checkpoint is kept; resume it with `distill resume`.' : 'Run failed: ' + (d.error || 'unknown error')); if (!d.cancelled) toast('Distill failed: ' + (d.error || ''), 'bad'); refresh(['status', 'runs']); break;
      default: break;
    }
    schedule();
  }
  function src(d) { const s = document.getElementById('dt-src'); if (s) s.textContent = `${d.recipe} · ${(d.models || []).map(shortModel).join(', ')}${d.simulated ? ' · SIMULATED' : ''}`; }
  function idle(msg) { goBtn.hidden = false; stopBtn.hidden = true; goBtn.disabled = false; status.textContent = msg; pipe.classList.remove('flow'); Object.values(tierEls).forEach((t) => t.node.classList.remove('active')); }
  function finished(result) {
    D.status = 'done'; idle(`Done: ${result.n_final} of ${result.n_requested} samples kept${result.simulated ? ' (simulated)' : ''}. Estimated cost $${(result.cost_usd || 0).toFixed(4)}.`);
    live.funnel = result.funnel.map((s) => ({ out: s.out, dropped: s.dropped, reasons: s.reasons })); FUNNEL.forEach(([id]) => live.done.add(id));
    live.total = result.funnel[0]?.in ?? live.total; live.gen = result.funnel[0]?.out ?? 0; live.scored = result.funnel[1]?.out ?? 0; live.waste = { ...(result.drop_reasons || {}) };
    live.drops = Object.fromEntries(result.funnel.map((s) => [s.stage, s.dropped])); ['filter', 'dedup', 'decontaminate'].forEach((x) => live.seen.add(x));
    tierEls.collector.count.textContent = fmtNum(result.n_final); tierEls.collector.liquid.style.height = clamp((result.n_final / Math.max(result.n_requested, 1)) * 100, 0, 100) + '%';
    const hh = result.histograms.reward || result.histograms.judge; live.final = hh ? { edges: hh.edges, all: hh.all, kept: hh.kept } : undefined;
    openExport(result); loadCards('final'); loadRuns(); refresh(['status', 'runs']); schedule();
  }

  // ------------------------------------------------------------------ run control
  function reset() {
    live = fresh(); clear(drops); Object.values(tierEls).forEach((t) => { t.count.textContent = '0'; t.liquid.style.height = '0%'; t.node.classList.remove('active'); });
    exportBox.hidden = true; cards = []; paintStack(); schedule();
  }
  async function start() {
    const rec = recipes.find((r) => r.name === D.recipe); if (!rec) return;
    const options = {}; if (D.docs.trim()) options.docs = D.docs.trim(); if (D.evalSet.trim()) options.eval_set = D.evalSet.trim();
    const ids = D.models.split(',').map((s) => s.trim()).filter(Boolean);
    goBtn.disabled = true; reset();
    try {
      const r = await api.post('/distill/run', { recipe: D.recipe, n: D.n, models: ids.length ? ids : undefined, options, model_overrides: await modelOverrides('distill') });
      D.runId = r.run_id; D.status = 'running'; runFormats = rec.formats; D.fmt = r.format; attach(r.run_id);
      goBtn.hidden = true; stopBtn.hidden = false; status.textContent = 'Warming the still…'; refresh(['status', 'runs']);
    } catch (e) { goBtn.disabled = false; toast(e.message, 'bad'); status.textContent = e.message; }
  }
  function attach(runId) { D.stop?.(); D.stop = api.stream(runId, apply); }
  async function cancel() { try { await api.post(`/runs/${D.runId}/cancel`); } catch (e) { toast(e.message, 'bad'); } }

  // ------------------------------------------------------------------ inspector
  async function loadCards(stage) {
    cardStage = stage; stageChips.forEach((c) => c.setAttribute('aria-pressed', String(c.dataset.s === stage)));
    if (!D.runId) return;
    try { const r = await api.get(`/distill/runs/${encodeURIComponent(D.runId)}/samples?stage=${stage}&limit=60`); cards = r.items; cardIdx = 0; flipped = false; paintStack(r.total); } catch (e) { toast(e.message, 'bad'); }
  }
  const chip = (text, tone) => h('span.dt-chip', { 'data-tone': tone || '' }, text);
  function face(s) {
    const chips = h('div.dt-chips', s.score != null && chip(`${s.score_source} ${s.score.toFixed(2)}`, scoreTone(s)), chip(shortModel(s.model)), chip(`t ${s.temperature}`), s.topic && chip(s.topic.split(' / ').pop()),
      s.difficulty != null && chip(`difficulty ${s.difficulty.toFixed(2)}`), s.cluster && chip(`cluster ${s.cluster}`), !s.alive && chip(`${WHY[s.reason] || s.reason} @ ${s.dropped_at}`, 'bad'));
    const body = s.rejected != null
      ? h('div.dt-vs', h('div.ch', h('span.label', `chosen · ${s.score?.toFixed(1)}`), h('p', s.response)), h('div.rj', h('span.label', `rejected · ${s.rejected_score?.toFixed(1)} · ${shortModel(s.rejected_model || '')}`), h('p', s.rejected)))
      : h('p.dt-resp', s.response || '(no response)');
    return [chips, h('h4.dt-q', s.prompt), body, s.detail && h('p.dt-why.mono', s.detail)];
  }
  function back(s) {
    const rows = [['id', s.id], ['model', s.model], ['temperature', s.temperature], ['topic', s.topic], ['source', s.source], ['stage reached', s.alive ? 'kept' : s.dropped_at], ['reason', s.reason], ['detail', s.detail]].filter(([, v]) => v);
    return [h('span.label', 'Provenance'), h('dl', rows.map(([k, v]) => [h('dt.mono', k), h('dd', String(v))])), s.context && h('details', { open: '' }, h('summary.label', 'Source excerpt'), h('p.dt-ctx', s.context))];
  }
  function paintStack(total) {
    clear(stack);
    if (!cards.length) { stack.append(h('div.empty', icon('doc'), h('h3', 'No samples yet'), h('p', 'Finished runs list their samples here: kept ones, and the ones that were dropped with their reason.'))); counter.textContent = '0 / 0'; return; }
    for (let k = Math.min(2, cards.length - cardIdx - 1); k >= 0; k--) {
      const s = cards[cardIdx + k], top = k === 0;
      const c = h('article.dt-card', { class: top ? (flipped ? 'top flipped' : 'top') : '', style: { '--k': k }, 'data-alive': String(s.alive), on: top ? { click: flip } : {} }, h('div.dt-face.fr', face(s)), h('div.dt-face.bk', back(s)));
      stack.append(c);
    }
    counter.textContent = `${cardIdx + 1} / ${total ?? cards.length}`;
  }
  function step(dir) {
    if (!cards.length) return; const next = clamp(cardIdx + dir, 0, cards.length - 1); if (next === cardIdx) return;
    const top = stack.querySelector('.top'); const go = () => { cardIdx = next; flipped = false; paintStack(); };
    if (top && !reduced() && top.animate) { top.animate([{ transform: 'none', opacity: 1 }, { transform: `translateX(${dir > 0 ? -120 : 120}%) rotate(${dir > 0 ? -9 : 9}deg)`, opacity: 0 }], { duration: 260, easing: 'ease-in' }).onfinish = go; } else go();
  }
  function flip() { if (!cards.length) return; flipped = !flipped; stack.querySelector('.top')?.classList.toggle('flipped', flipped); }

  function openExport(result) {
    const rec = recipes.find((r) => r.name === result.recipe); const formats = rec?.formats || runFormats; D.fmt = formats.includes(D.fmt) ? D.fmt : result.format;
    const link = () => { const q = `format=${encodeURIComponent(D.fmt)}&t=${encodeURIComponent(token)}`; exportA.href = `/api/distill/runs/${D.runId}/export?${q}`; exportA.setAttribute('download', `${result.recipe}.${D.fmt}.jsonl`); reportA.href = `/api/distill/runs/${D.runId}/report?t=${encodeURIComponent(token)}`; cardA.href = `/api/distill/runs/${D.runId}/card?t=${encodeURIComponent(token)}`; };
    const paintFmt = () => { clear(fmtBox); formats.forEach((f) => fmtBox.append(h('button.chip.mono', { role: 'radio', 'aria-pressed': String(f === D.fmt), on: { click: () => { D.fmt = f; paintFmt(); link(); } } }, f))); };
    paintFmt(); link(); exportBox.hidden = false; pop(exportBox, { y: 8 });
  }

  // ------------------------------------------------------------------ recipes + history
  function paintRecipes() {
    clear(recipeBox);
    recipes.forEach((r) => recipeBox.append(h('button.dt-rec', { role: 'radio', 'aria-checked': String(r.name === D.recipe), title: r.description, on: { click: () => { D.recipe = r.name; paintRecipes(); } } }, h('b', r.name.replace(/_/g, ' ')), h('small', r.description))));
    docsRow.hidden = !recipes.find((r) => r.name === D.recipe)?.needs_docs;
  }
  async function loadRuns() {
    try { runs = await api.get('/distill/runs'); } catch { runs = []; }
    clear(pastList);
    if (!runs.length) { pastList.append(h('p.dim', { style: { padding: '4px 16px 14px' } }, 'No runs yet.')); return; }
    runs.slice(0, 8).forEach((r) => pastList.append(h('button.dt-run', { on: { click: () => openRun(r.id) } }, h('span.led', { class: r.status === 'done' ? 'on' : r.status === 'failed' ? 'bad' : 'warn' }), h('b', r.recipe), h('span.dim.mono', `n=${r.n}`), r.simulated && h('span.dt-chip', 'simulated'), h('span.dim.mono', r.status))));
  }
  async function openRun(id) {
    try {
      const detail = await api.get(`/distill/runs/${encodeURIComponent(id)}`); reset(); D.runId = id; D.status = detail.status === 'done' ? 'done' : 'idle'; D.fmt = detail.format;
      live.total = detail.n; const res = detail.result;
      if (res) { replayUntil = performance.now() + 60000; finished(res); status.textContent = `Loaded run ${id} (${detail.status}).`; }
      else status.textContent = `Run ${id} is ${detail.status}; resume it from the CLI.`;
    } catch (e) { toast(e.message, 'bad'); }
  }

  paintStack(); paintHist();
  try { recipes = await api.get('/distill/recipes'); paintRecipes(); }
  catch (e) { recipeBox.append(h('div.empty', icon('bolt'), h('h3', 'Distill is not wired in yet'), h('p', String(e.message || e)))); goBtn.disabled = true; }
  loadRuns();
  if (D.runId && D.status === 'running') { attach(D.runId); goBtn.hidden = true; stopBtn.hidden = false; }
  return () => { offs.forEach((f) => f && f()); cancelAnimationFrame(raf); if (D.status !== 'running') { D.stop?.(); D.stop = null; } };
}
