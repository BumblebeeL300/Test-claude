// Council: a round table. Seats light up as models speak; lines converge as they agree.
import { h, clear, shortModel, pop, reduced, clamp } from '../util.js';
import { icon } from '../icons.js';
import { store, on, api, toast, refresh } from '../app.js';

const STRATS = [
  { id: 'moa', n: 'MoA', d: 'Proposers answer in parallel, layers refine, one synthesises' },
  { id: 'debate', n: 'Debate', d: 'Answer, critique peers, revise; a judge decides' },
  { id: 'best-of-n', n: 'Best-of-N', d: 'Sample many, a reward model keeps the best' },
  { id: 'critic', n: 'Critic', d: 'Draft, critique, revise until clean' },
];
const IDEAS = ['Is a tomato a fruit?', 'Should a small team adopt microservices?', 'Explain the Monty Hall problem to a ten-year-old'];
const C = { runId: null, events: [], status: 'idle', stop: null, ui: null, strategy: 'debate', question: '' };

function attach(runId) {
  C.stop?.(); Object.assign(C, { runId, events: [], status: 'running' });
  C.stop = api.stream(runId, (ev) => { C.events.push(ev); if (ev.type === 'run.done') C.status = 'done'; if (ev.type === 'run.failed') C.status = 'failed'; C.ui?.apply(ev, false); if (ev.type.startsWith('run.')) refresh(['status', 'runs']); }, () => (C.stop = null));
}

export function mount(el, params) {
  const offs = [];
  let seats = [], fresh = true; // {model, el, state}
  let agreement = 0, replaying = false;

  // ---- controls ----
  const q = h('textarea.field.q', { rows: 4, placeholder: 'Put a question to the table…', 'aria-label': 'Question for the council', on: { keydown: (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); convene(); } } } });
  q.value = C.question || (store.prefill?.council ?? ''); if (store.prefill) delete store.prefill.council;
  const knobSvg = h('svg.knob', { viewBox: '-100 -100 200 200', 'aria-hidden': 'true', html: '<circle r="62" fill="var(--raised)" stroke="var(--hair-2)" stroke-width="2"/><circle r="50" fill="var(--panel)" stroke="var(--hair)"/><g class="pointer"><path d="M0 0 L0 -50" stroke="var(--accent)" stroke-width="5" stroke-linecap="round"/><circle cy="-44" r="4" fill="var(--accent)"/></g><circle r="8" fill="var(--raised-2)"/>' });
  const pointer = knobSvg.querySelector('.pointer');
  const angle = (i) => -78 + i * 52;
  const sw = h('div.rotary', { role: 'radiogroup', 'aria-label': 'Strategy' }, knobSvg, STRATS.map((s, i) => {
    const a = (angle(i) - 90) * Math.PI / 180, r = 104;
    return h('button.rot-l', { role: 'radio', 'aria-checked': 'false', 'data-id': s.id, title: s.d, style: { left: `calc(50% + ${Math.cos(a) * r}px)`, top: `calc(54% + ${Math.sin(a) * r * .95}px)` }, on: { click: () => setStrat(s.id) } }, s.n);
  }));
  const stratDesc = h('p.strat-d');
  function setStrat(id) { C.strategy = id; const i = STRATS.findIndex((s) => s.id === id); pointer.style.transform = `rotate(${angle(i)}deg)`; [...sw.querySelectorAll('.rot-l')].forEach((b) => b.setAttribute('aria-checked', b.dataset.id === id)); stratDesc.textContent = STRATS[i].d; }
  sw.addEventListener('keydown', (e) => { const i = STRATS.findIndex((s) => s.id === C.strategy); if (e.key === 'ArrowRight' || e.key === 'ArrowDown') setStrat(STRATS[(i + 1) % 4].id); else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') setStrat(STRATS[(i + 3) % 4].id); else return; e.preventDefault(); });
  const goBtn = h('button.btn.primary.big', { on: { click: convene } }, icon('council'), 'Convene', h('kbd', '⌘↵'));
  const ctl = h('aside.cn-ctl', h('div.label', 'Question'), q, h('div.chips.col', IDEAS.map((t) => h('button.chip', { on: { click: () => { q.value = t; q.focus(); } } }, t))), h('div.label', { style: { marginTop: '10px' } }, 'Strategy'), sw, stratDesc, goBtn);

  // ---- table ----
  const lines = h('svg.lines', { viewBox: '0 0 100 100', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
  const hubPct = h('b.num', '—'), hubStage = h('span.label', 'Awaiting a question');
  const hub = h('div.hub', h('span.label', 'Agreement'), hubPct, hubStage);
  const table = h('div.table', h('div.disc'), lines, hub);
  const verdict = h('section.verdict', { hidden: true });
  const stageWrap = h('div.cn-stage', h('div.cn-in', h('div.table-wrap', table), verdict));

  function defaultPanel() {
    const roles = store.models.filter((m) => (m.roles || []).some((r) => r === 'chat' || r === 'reasoning')).map((m) => m.id);
    const base = ['meta/llama-3.1-405b-instruct', 'nvidia/llama-3.1-nemotron-ultra-253b-v1', 'deepseek-ai/deepseek-r1', 'mistralai/mistral-large-2-instruct', 'meta/llama-3.3-70b-instruct'];
    return (roles.length >= 5 ? base.filter((b) => roles.includes(b)).concat(roles) : base).filter((v, i, a) => a.indexOf(v) === i).slice(0, 5);
  }
  function seat(model) {
    if (seats.find((s) => s.model === model)) return seats.find((s) => s.model === model);
    const name = shortModel(model), org = model.includes('/') ? model.split('/')[0] : '';
    const e = h('div.seat', { 'data-state': 'idle' }, h('div.chair', h('span.ini', name.replace(/[^a-z0-9]/gi, '').slice(0, 2).toUpperCase()), h('i.ring')), h('div.sname', h('b', name), h('small', org)), h('p.say'));
    const s = { model, el: e, say: e.querySelector('.say'), state: 'idle' }; seats.push(s); table.append(e); place(); return s;
  }
  function place() {
    const n = seats.length, r = 41 - agreement * 4;
    seats.forEach((s, i) => { const a = (i / n) * Math.PI * 2 - Math.PI / 2; s.x = 50 + Math.cos(a) * r; s.y = 50 + Math.sin(a) * r; s.el.style.left = s.x + '%'; s.el.style.top = s.y + '%'; s.el.dataset.side = Math.cos(a) > .3 ? 'r' : Math.cos(a) < -.3 ? 'l' : Math.sin(a) > 0 ? 'b' : 't'; });
    drawLines();
  }
  function drawLines() {
    clear(lines); const w = .5 + agreement ** 1.4 * 4.5;
    seats.forEach((s) => { const alive = s.state === 'spoke' || s.state === 'think';
      const dx = 50 - s.x, dy = 50 - s.y, L = Math.hypot(dx, dy) || 1, ex = 50 - (dx / L) * 13, ey = 50 - (dy / L) * 13, sx = s.x + (dx / L) * 7, sy = s.y + (dy / L) * 7;
      lines.append(h('line', { x1: sx, y1: sy, x2: ex, y2: ey, class: 'ln' + (s.state === 'think' ? ' hot' : ''), style: { strokeWidth: alive ? w : .25, opacity: alive ? .25 + agreement * .75 : .25 }, 'vector-effect': 'non-scaling-stroke' })); });
  }
  function setSeat(model, state, text) {
    const s = seat(model); s.state = state; s.el.dataset.state = state;
    if (text != null) { s.say.textContent = text.length > 84 ? text.slice(0, 82).trimEnd() + '\u2026' : text; s.say.title = text; if (!replaying && !reduced()) s.say.animate([{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }], { duration: 400, easing: 'ease-out' }); }
    drawLines();
  }
  function reseat(models) { seats.forEach((s) => s.el.remove()); seats = []; models.forEach(seat); }

  function apply(ev, isReplay) {
    replaying = isReplay; const d = ev.data || {};
    switch (ev.type) {
      case 'stage.start': hubStage.textContent = String(d.stage || d.name || 'stage').replace(/^Round (\d+) · /, 'R$1 · '); table.classList.add('live'); break;
      case 'stage.end': break;
      case 'model.start': if (fresh) { fresh = false; reseat([]); } setSeat(d.model, 'think', ''); seats.find((s) => s.model === d.model).say.textContent = 'thinking…'; break;
      case 'model.end': setSeat(d.model, 'spoke', d.preview ?? d.text ?? ''); break;
      case 'consensus': { const a = d.agreement ?? d.score ?? 0; agreement = clamp(a, 0, 1); hubPct.textContent = Math.round(agreement * 100) + '%'; table.style.setProperty('--ag', agreement); place(); break; }
      case 'run.done': finish(d.result || d); break;
      case 'run.failed': goBtn.disabled = false; table.classList.remove('live'); hubStage.textContent = 'adjourned'; toast('Council failed: ' + (d.error || d.message || 'unknown'), 'bad'); break;
    }
  }
  function finish(r) {
    goBtn.disabled = false; table.classList.remove('live'); hubStage.textContent = 'verdict reached'; table.classList.add('done');
    if (r.agreement != null) { agreement = clamp(r.agreement, 0, 1); hubPct.textContent = Math.round(agreement * 100) + '%'; place(); }
    seats.forEach((s) => { if (s.state === 'think') { s.state = 'spoke'; s.el.dataset.state = 'spoke'; } });
    verdict.hidden = false;
    const cands = r.candidates || [];
    verdict.replaceChildren(h('div.v-h', h('span.label', 'Verdict'), h('span.v-ag', h('span.led.on'), `${Math.round((r.agreement ?? agreement) * 100)}% agreement`), r.strategy && h('span.chip.mono', { style: { pointerEvents: 'none' } }, r.strategy)),
      h('p.v-ans.selectable', r.answer || r.final || ''), r.rationale && h('div.v-why', h('span.label', 'Rationale'), h('p.selectable', r.rationale)),
      cands.length ? h('details.v-c', h('summary', `${cands.length} candidate answers`), cands.map((c) => h('div.cand', h('b.mono', shortModel(c.model)), h('p.selectable', c.text || c.answer || '')))) : null);
    if (!replaying && !reduced()) verdict.animate([{ opacity: 0, transform: 'translateY(14px) scale(.98)' }, { opacity: 1, transform: 'none' }], { duration: 550, easing: 'cubic-bezier(.34,1.56,.64,1)' });
  }
  function resetTable() { fresh = true; agreement = 0; hubPct.textContent = '—'; hubStage.textContent = 'Awaiting a question'; table.classList.remove('live', 'done'); table.style.setProperty('--ag', 0); verdict.hidden = true; reseat(defaultPanel()); place(); }

  async function convene() {
    const question = q.value.trim(); if (!question) { toast('Put a question to the table first', 'warn'); q.focus(); return; }
    goBtn.disabled = true;
    try { const r = await api.post('/council/run', { question, strategy: C.strategy }); C.question = question; resetTable(); hubStage.textContent = 'seating…'; C.ui = ui; attach(r.run_id); refresh(['status', 'runs']); }
    catch (e) { toast('Couldn’t convene: ' + e.message, 'bad'); goBtn.disabled = false; }
  }

  el.append(h('div.cn', ctl, stageWrap));
  const ui = { apply }; C.ui = ui;
  setStrat(C.strategy);
  resetTable();
  offs.push(on('prefill', (d) => { if (d.screen === 'council') { q.value = d.text; q.focus(); } }), on('models', () => { if (C.status === 'idle') resetTable(); }));
  if (params.run && params.run !== C.runId) { C.question = store.runs.find((r) => r.id === params.run)?.title || ''; q.value = C.question; attach(params.run); }
  else if (C.runId && C.events.length) { q.value = C.question; replaying = true; C.events.slice().forEach((ev) => apply(ev, true)); replaying = false; if (C.status === 'running') goBtn.disabled = true; }
  return () => { offs.forEach((f) => f && f()); if (C.ui === ui) C.ui = null; };
}
