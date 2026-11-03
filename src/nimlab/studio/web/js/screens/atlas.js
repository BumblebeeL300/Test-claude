// Atlas: specimen tray in, grounded answers out. Citations are cyan superscript chips.
import { h, clear, esc, pop, reduced, clamp, autogrow } from '../util.js';
import { icon } from '../icons.js';
import { store, on, api, toast, refresh } from '../app.js';

const A = { runId: null, events: [], status: 'idle', stop: null, ui: null, collection: '', question: '', mode: 'agentic', verify: true, shelf: {} };
const IDEAS = ['What are the key obligations in these documents?', 'Summarise anything that mentions refunds or credits', 'Where do these documents contradict each other?'];

function attach(runId) {
  A.stop?.(); Object.assign(A, { runId, events: [], status: 'running' });
  A.stop = api.stream(runId, (ev) => { A.events.push(ev); if (ev.type === 'run.done') A.status = 'done'; if (ev.type === 'run.failed') A.status = 'failed'; A.ui?.apply(ev); if (ev.type.startsWith('run.')) refresh(['status', 'runs']); }, () => (A.stop = null));
}

export function mount(el, params) {
  const offs = []; let cols = [], cites = [], text = '', docsCount = 0;

  // ---- tray ----
  const colSel = h('select.field', { 'aria-label': 'Collection', on: { change: () => { A.collection = colSel.value; paintTray(); } } });
  const slots = h('div.slots', { 'aria-hidden': 'true' });
  const shelfList = h('ul.shelf');
  const fileIn = h('input', { type: 'file', multiple: true, hidden: true, on: { change: () => { ingest([...fileIn.files]); fileIn.value = ''; } } });
  const drop = h('div.drop', { tabindex: '0', role: 'button', 'aria-label': 'Drop documents here or press Enter to choose files', on: {
    click: () => fileIn.click(), keydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } },
    dragover: (e) => { e.preventDefault(); drop.classList.add('over'); }, dragleave: () => drop.classList.remove('over'),
    drop: (e) => { e.preventDefault(); drop.classList.remove('over'); ingest([...e.dataTransfer.files]); } } },
    slots, h('div.drop-t', icon('upload'), h('b', 'Specimen tray'), h('span', 'Drop PDFs, Markdown or text. Or click to choose.')), h('i.scan'));
  const folderBtn = h('button.btn.sm', { hidden: !window.pywebview?.api?.pick_folder, on: { click: async () => { try { const f = await window.pywebview.api.pick_folder(); if (!f) return; drop.classList.add('busy'); const r = await api.post('/atlas/ingest', { path: f, collection: A.collection }); toast(`Ingested ${r.ingested ?? '?'} files from folder`); await loadCols(); } catch (e) { toast('Folder ingest failed: ' + e.message, 'bad'); } finally { drop.classList.remove('busy'); } } } }, icon('folder'), 'Choose a folder\u2026');
  const tray = h('aside.at-tray', h('div.label', 'Collection'), colSel, drop, fileIn, folderBtn, h('div.label', 'On the tray'), shelfList);
  function paintTray() {
    const c = cols.find((x) => x.name === A.collection) || cols[0]; docsCount = c?.documents ?? c?.docs ?? c?.count ?? 0;
    clear(slots); const total = 12; for (let i = 0; i < total; i++) slots.append(h('i.slot', { class: i < docsCount ? 'full' : '', style: { '--i': i } }));
    clear(shelfList); const mine = A.shelf[A.collection] || [];
    const names = (c?.files || []).map((f) => f.name || f).filter(Boolean); mine.forEach((n) => { if (!names.includes(n)) names.push(n); });
    if (!docsCount && !names.length) shelfList.append(h('li.dim', 'Empty tray. Nothing to cite yet.'));
    else { if (docsCount) shelfList.append(h('li', h('b.num', docsCount), ' document' + (docsCount === 1 ? '' : 's') + ' indexed')); names.slice(-8).forEach((f) => shelfList.append(h('li.f', icon('doc'), h('span', f)))); }
  }
  async function loadCols() { try { const r = await api.get('/atlas/collections'); cols = (Array.isArray(r) ? r : r.collections || []).map((c) => (typeof c === 'string' ? { name: c } : c)); } catch { cols = []; }
    clear(colSel); if (!cols.length) cols = [{ name: 'default', documents: 0 }]; cols.forEach((c) => colSel.append(h('option', { value: c.name }, c.name))); if (!cols.find((c) => c.name === A.collection)) A.collection = cols[0].name; colSel.value = A.collection; paintTray(); }
  async function ingest(files) {
    if (!files.length) return; drop.classList.add('busy');
    const fd = new FormData(); files.forEach((f) => fd.append('files', f)); fd.append('collection', A.collection);
    try { const r = await api.post('/atlas/ingest', fd); (A.shelf[A.collection] ||= []).push(...files.map((f) => f.name)); toast(`Ingested ${r.ingested ?? files.length} file${files.length > 1 ? 's' : ''}${r.chunks ? ` · ${r.chunks} chunks` : ''}`); await loadCols(); }
    catch (e) { toast('Ingest failed: ' + e.message, 'bad'); } finally { drop.classList.remove('busy'); }
  }

  // ---- ask ----
  const q = h('textarea.field.q', { rows: 2, placeholder: 'Ask your documents…', 'aria-label': 'Question', on: { keydown: (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); ask(); } } } });
  q.value = A.question || store.prefill?.atlas || ''; if (store.prefill) delete store.prefill.atlas; autogrow(q, 160);
  const modeSel = h('select.field.sm', { 'aria-label': 'Mode', on: { change: () => (A.mode = modeSel.value) } }, ['agentic', 'simple'].map((m) => h('option', { value: m, selected: m === A.mode }, m === 'agentic' ? 'Agentic (multi-hop)' : 'Simple (one pass)')));
  const verSw = h('button.switch', { role: 'switch', 'aria-checked': String(A.verify), 'aria-label': 'Verify claims', on: { click: () => { A.verify = !A.verify; verSw.setAttribute('aria-checked', A.verify); } } });
  const askBtn = h('button.btn.primary', { on: { click: ask } }, icon('atlas'), 'Ask', h('kbd', '⌘↵'));
  const ans = h('div.ans.selectable', { 'aria-live': 'polite' });
  const srcs = h('div.srcs');
  const gauge = h('div.gauge', { hidden: true });
  const chips = h('div.chips', IDEAS.map((t) => h('button.chip', { on: { click: () => { q.value = t; q.dispatchEvent(new Event('input')); q.focus(); } } }, t)));
  const out = h('div.at-out');
  const empty = () => h('div.empty', icon('atlas'), h('h3', 'Nothing on the slide yet'), h('p', 'Drop a few documents on the tray, then ask them something awkward. Every claim comes back with a source you can open.'));
  const main = h('section.at-main', h('div.q-row', q), h('div.q-ctl', h('label.ctl', h('span.label', 'Mode'), modeSel), h('label.ctl.row', verSw, h('span.label', 'Verify claims')), h('span.grow'), askBtn), chips, out);
  el.append(h('div.at', tray, main));
  out.append(empty());

  function renderAnswer(final) {
    const html = esc(text).replace(/\[(\d+)\]/g, (_, n) => `<button class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`);
    ans.innerHTML = html + (final ? '' : '<span class="caret"></span>');
  }
  ans.addEventListener('click', (e) => { const b = e.target.closest('.cite'); if (!b) return; toggleSource(+b.dataset.n); });
  const clip = (t) => (t && t.length > 420 ? t.slice(0, 420).trimEnd() + '\u2026' : t);
  function paintSources(final) {
    clear(srcs); if (!cites.length) return; srcs.append(h('div.label', 'Sources'));
    cites.forEach((c, i) => { const n = c.n ?? i + 1; const d = h('div.src', { id: 'src-' + n, 'data-n': n }, h('button.src-h', { 'aria-expanded': 'false', on: { click: () => toggleSource(n) } }, h('span.cite.static', n), h('span.src-n', c.source || c.title || 'source'), h('span.src-s', c.heading || c.section || (Array.isArray(c.heading_path) ? c.heading_path.join(' \u203a ') : '') || (c.title && c.title !== c.source ? c.title : '')), icon('chevron', 'chev')), h('blockquote.src-b', clip(c.snippet || c.text || c.excerpt) || (final ? 'No snippet returned.' : 'Retrieving\u2026'))); srcs.append(d); });
  }
  function toggleSource(n) {
    const d = srcs.querySelector(`[data-n="${n}"]`); if (!d) return; const open = !d.classList.contains('open'); d.classList.toggle('open', open); d.querySelector('.src-h').setAttribute('aria-expanded', open);
    if (open) { d.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' }); d.animate([{ background: 'color-mix(in srgb, var(--accent) 30%, transparent)' }, { background: 'transparent' }], { duration: 900 }); }
  }
  function paintGauge(g) {
    g = clamp(g, 0, 1); const R = 52, C_ = Math.PI * R; const col = g > .8 ? 'var(--ok)' : g > .55 ? 'var(--warn)' : 'var(--bad)';
    gauge.hidden = false; gauge.innerHTML = `<svg viewBox="0 0 140 84" role="img" aria-label="Groundedness ${Math.round(g * 100)} percent"><path d="M18 74 A52 52 0 0 1 122 74" fill="none" stroke="var(--raised-2)" stroke-width="10" stroke-linecap="round"/><path class="g-arc" d="M18 74 A52 52 0 0 1 122 74" fill="none" stroke="${col}" stroke-width="10" stroke-linecap="round" stroke-dasharray="${C_}" stroke-dashoffset="${C_}" style="filter:drop-shadow(0 0 6px ${col})"/></svg><div class="g-v"><b class="num" style="color:${col}">${Math.round(g * 100)}</b><span class="label">Grounded</span></div>`;
    const arc = gauge.querySelector('.g-arc'); requestAnimationFrame(() => { arc.style.transition = reduced() ? 'none' : 'stroke-dashoffset 1.2s cubic-bezier(.2,1,.3,1)'; arc.style.strokeDashoffset = C_ * (1 - g); });
  }
  function apply(ev) {
    const d = ev.data || {};
    if (ev.type === 'sources') { cites = (d.sources || []).map((s, i) => ({ n: i + 1, ...s })); paintSources(false); }
    else if (ev.type === 'token') { text += d.text ?? d.token ?? ''; renderAnswer(false); }
    else if (ev.type === 'run.done') { const r = d.result || d; if (r.answer) text = r.answer; renderAnswer(true); if (r.citations?.length) cites = r.citations; paintSources(true); const gr = r.groundedness ?? r.verification?.groundedness ?? r.verification?.score; if (gr != null) paintGauge(gr); askBtn.disabled = false; }
    else if (ev.type === 'run.failed') { askBtn.disabled = false; toast('Atlas failed: ' + (d.error || d.message || 'unknown'), 'bad'); ans.append(h('p.dim', 'The run stopped before an answer came back.')); }
  }
  function present() { clear(out); text = ''; cites = []; gauge.hidden = true; clear(srcs); out.append(h('div.ans-card', h('div.ans-h', h('span.label', 'Answer'), gauge), ans), srcs); ans.innerHTML = '<span class="caret"></span>'; }
  async function ask() {
    const question = q.value.trim(); if (!question) { toast('Ask something first', 'warn'); q.focus(); return; }
    askBtn.disabled = true;
    try { const r = await api.post('/atlas/ask', { question, collection: A.collection || undefined, mode: A.mode, verify: A.verify }); A.question = question; present(); A.ui = ui; attach(r.run_id); refresh(['status', 'runs']); }
    catch (e) { toast('Couldn’t ask: ' + e.message, 'bad'); askBtn.disabled = false; }
  }
  const ui = { apply }; A.ui = ui;
  loadCols();
  offs.push(on('prefill', (d) => { if (d.screen === 'atlas') { q.value = d.text; q.focus(); } }));
  if (params.run && params.run !== A.runId) { q.value = store.runs.find((r) => r.id === params.run)?.title || ''; present(); attach(params.run); }
  else if (A.runId && A.events.length) { q.value = A.question; present(); A.events.slice().forEach(apply); if (A.status === 'running') askBtn.disabled = true; }
  return () => { offs.forEach((f) => f && f()); if (A.ui === ui) A.ui = null; };
}
