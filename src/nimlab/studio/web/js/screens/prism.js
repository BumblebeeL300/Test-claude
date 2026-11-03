// Prism: an optical bench. Put an image on the light table, turn the prism to a mode (a white beam
// refracts into seven coloured ones), and the answer develops as a print.
import { h, clear, reduced, shortModel } from '../util.js';
import { icon } from '../icons.js';
import { api, toast, refresh } from '../app.js';
import { codeBlock } from '../highlight.js';

/** Registration data for the integrator (rail button + router + palette). */
export const route = 'prism';
export const meta = { id: 'prism', name: 'Prism', cls: 'app-prism', icon: 'eye', app: 'prism', hint: 'Put an image on the light table' };

const BEAM = { describe: 'var(--bad)', ocr: 'var(--c-arena)', extract: 'var(--warn)', ui2code: 'var(--ok)', compare: 'var(--c-atlas)', chart: 'var(--c-core)', ask: 'var(--c-prism)' };
const LABEL = { describe: 'Describe', ocr: 'OCR', extract: 'Extract', ui2code: 'UI → code', compare: 'Compare', chart: 'Chart → data', ask: 'Ask' };
const FALLBACK = [
  { name: 'describe', description: 'A rich description plus alt text.', min_images: 1, max_images: 1, options: [{ name: 'detail', kind: 'choice', choices: ['rich', 'brief'], default: 'rich', help: 'How much to write' }] },
  { name: 'ocr', description: 'Extract text, keeping its layout.', min_images: 1, max_images: 1, options: [] },
  { name: 'extract', description: 'Structured data from a schema.', min_images: 1, max_images: 1, options: [{ name: 'schema', kind: 'schema', default: 'receipt', help: 'Built-in name or inline JSON Schema' }] },
  { name: 'ui2code', description: 'Screenshot to one HTML/CSS file.', min_images: 1, max_images: 1, options: [{ name: 'refine', kind: 'int', default: 0, min: 0, max: 5, help: 'Critique-and-fix rounds' }] },
  { name: 'compare', description: 'Diff two images.', min_images: 2, max_images: 2, options: [{ name: 'focus', kind: 'text', help: 'Optional focus' }] },
  { name: 'chart', description: 'Chart to data table.', min_images: 1, max_images: 1, options: [] },
  { name: 'ask', description: 'Ask about the image.', min_images: 1, max_images: 1, options: [{ name: 'question', kind: 'text', required: true, help: 'What do you want to know?' }] },
];
const IDEAS = { describe: 'Alt text for a product photo', ocr: 'Read a receipt, keep the columns', extract: 'Pull a receipt into JSON', ui2code: 'Rebuild this screenshot as HTML', ask: 'What is the main call to action?' };
const MAX_FILES = 12;

// State outlives the screen so leaving and coming back keeps your table, mode and print.
const P = { mode: 'describe', files: [], active: 0, opts: {}, schemaText: '', info: null, runId: null, events: [], status: 'idle', stop: null, ui: null };

const human = (k) => String(k).replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2');
const fmtMs = (s) => (s < 1 ? Math.round(s * 1000) + ' ms' : s.toFixed(1) + ' s');
const modeOf = (name) => (P.info?.modes || FALLBACK).find((m) => m.name === name) || FALLBACK[0];

function ensureCss() {
  if (!document.querySelector('link[href$="screens/prism.css"]')) document.head.append(h('link', { rel: 'stylesheet', href: 'css/screens/prism.css' }));
}
function download(name, mime, text) {
  const a = h('a', { href: URL.createObjectURL(new Blob([text], { type: mime })), download: name });
  document.body.append(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
const copyBtn = (get, label = 'Copy') => h('button.btn.sm', { type: 'button', on: { click: async () => { try { await navigator.clipboard.writeText(get()); toast('Copied'); } catch { toast('Clipboard blocked: select the text and press ⌘C', 'warn'); } } } }, icon('copy'), label);
const saveBtn = (name, mime, get, label) => h('button.btn.sm', { type: 'button', on: { click: () => download(name, mime, get()) } }, icon('upload', 'flip'), label || 'Save');

// ------------------------------------------------------------------ result renderers
function valueNode(v) {
  if (v == null) return h('span.nil', 'null');
  if (Array.isArray(v)) {
    if (!v.length) return h('span.nil', 'none');
    if (v.every((x) => x === null || typeof x !== 'object')) return h('span.chips', v.map((x) => h('span.chip.mono', String(x))));
    return gridTable(v);
  }
  if (typeof v === 'object') return formTable(v);
  return h('span', { class: typeof v === 'number' ? 'num n' : '' }, typeof v === 'number' ? String(v) : String(v));
}
function formTable(obj) {
  return h('table.ft', h('tbody', Object.entries(obj).map(([k, v]) => h('tr', h('th', { scope: 'row' }, human(k)), h('td', valueNode(v))))));
}
function gridTable(rows) {
  const objs = rows.map((r) => (r && typeof r === 'object' && !Array.isArray(r) ? r : { value: r }));
  const cols = [...new Set(objs.flatMap((r) => Object.keys(r)))];
  return h('table.ft.grid', h('thead', h('tr', cols.map((c) => h('th', { scope: 'col' }, human(c))))), h('tbody', objs.map((r) => h('tr', cols.map((c) => h('td', c in r ? valueNode(r[c]) : h('span.nil', '·')))))));
}

const RENDER = {
  describe(o) {
    return h('div.rd-describe', h('div.caption', h('span.label', 'Alt text'), h('p.selectable', o.alt_text), copyBtn(() => o.alt_text)),
      h('div.prose.selectable', String(o.description).split(/\n{2,}|(?<=\.)\s(?=[A-Z][^.]{40,})/).filter(Boolean).map((p) => h('p', p))),
      o.tags?.length ? h('div.chips', o.tags.map((t) => h('span.chip.mono', t))) : null);
  },
  ocr(o, it) {
    return h('div.rd-ocr', h('div.tools', h('span.label', `${o.lines} line${o.lines === 1 ? '' : 's'}`), h('span.grow'), copyBtn(() => o.text), saveBtn('prism-ocr.txt', 'text/plain', () => o.text)),
      h('pre.ocr.selectable', o.text));
  },
  extract(o) {
    const raw = JSON.stringify(o.data, null, 2);
    const view = h('div.ex-form', formTable(o.data)), json = h('div.ex-json', { hidden: true }, codeBlock(raw, 'json'));
    const tab = (label, el) => h('button.chip', { type: 'button', 'aria-pressed': String(el === view), on: { click: (e) => { view.hidden = el !== view; json.hidden = el !== json; e.currentTarget.parentNode.querySelectorAll('.chip').forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); } } }, label);
    return h('div.rd-extract', h('div.tools', h('span.label', o.schema_name), h('span.pr-verdict', h('span.led', { class: o.valid ? 'on' : 'bad' }), o.valid ? 'matches schema' : `${o.errors.length} schema problem${o.errors.length === 1 ? '' : 's'}`),
      h('span.grow'), h('span.pr-seg', tab('Form', view), tab('JSON', json)), copyBtn(() => raw), saveBtn('prism-extract.json', 'application/json', () => raw)),
      o.errors.length ? h('ul.errs', o.errors.map((e) => h('li', e))) : null, view, json);
  },
  ui2code(o) {
    const frame = h('iframe.pr-preview', { sandbox: '', srcdoc: o.html, title: 'Live preview of the generated page', referrerpolicy: 'no-referrer' });
    const stage = h('div.vp', { 'data-w': 'fit' }, frame);
    const vp = (w, label) => h('button.chip', { type: 'button', 'aria-pressed': String(w === 'fit'), on: { click: (e) => { stage.dataset.w = w; e.currentTarget.parentNode.querySelectorAll('.chip').forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); } } }, label);
    const crit = o.critiques?.length ? h('ol.rounds', o.critiques.map((c) => h('li', h('span.num', `Round ${c.round}`), h('span.bar', h('i', { style: { width: Math.round(c.score * 100) + '%' } })), h('span.num', c.score.toFixed(2)), h('small', c.issues.length ? c.issues.join(' · ') : 'no issues left')))) : null;
    return h('div.rd-u2c', h('div.tools', h('span.label', `${o.drafts} draft${o.drafts === 1 ? '' : 's'}`), h('span.pr-verdict', h('span.led', { class: o.valid ? 'on' : 'warn' }), o.valid ? 'valid, self-contained' : o.problems.length + ' html warning(s)'),
      h('span.grow'), h('span.pr-seg', vp('fit', 'Fit'), vp('768', 'Tablet'), vp('390', 'Phone')), copyBtn(() => o.html), saveBtn('prism-ui.html', 'text/html', () => o.html)),
      o.problems.length ? h('ul.errs', o.problems.map((e) => h('li', e))) : null, crit,
      h('div.split', h('div.pane.codepane', h('span.label', 'HTML'), codeBlock(o.html, 'html')), h('div.pane', h('span.label', 'Live preview · sandboxed'), stage)));
  },
  compare(o, it) {
    const sim = Math.round(o.similarity * 100);
    return h('div.rd-compare', h('div.pr-sim', h('b.num', sim + '%'), h('span.label', 'alike'), h('i', h('u', { style: { width: sim + '%' } }))), h('p.selectable', o.summary),
      o.differences.length ? h('table.ft.grid', h('thead', h('tr', ['Where', 'A', 'B', 'Weight'].map((c) => h('th', c)))), h('tbody', o.differences.map((d) => h('tr', h('th', d.area), h('td', d.a), h('td', d.b), h('td', h('span.pr-sev', { 'data-s': d.severity }, d.severity)))))) : h('p.dim', 'No differences found.'));
  },
  chart(o) {
    return h('div.rd-chart', h('div.tools', h('span.label', `${o.chart_type} · ${o.rows.length} rows`), h('span.grow'), copyBtn(() => o.csv, 'Copy CSV'), saveBtn('prism-chart.csv', 'text/csv', () => o.csv, 'CSV')),
      o.title ? h('h3', o.title) : null, h('p.selectable', o.summary),
      h('table.ft.grid', h('thead', h('tr', o.columns.map((c) => h('th', c)))), h('tbody', o.rows.map((r) => h('tr', r.map((v, i) => h('td', { class: i ? 'num n' : '' }, v == null ? '·' : String(v))))))));
  },
  ask(o) {
    return h('div.rd-ask', h('blockquote.selectable', o.question), h('div.answer.selectable', o.answer), h('div.tools', h('span.grow'), copyBtn(() => o.answer)));
  },
};

// ------------------------------------------------------------------ screen
export function mount(el, params) {
  ensureCss();
  const offs = [];
  let live = null, liveBuf = '', raf = 0, dev = 0, replaying = false, running = false;

  // ---- light table ----
  const fileIn = h('input', { type: 'file', accept: 'image/png,image/jpeg,image/webp,image/gif', multiple: true, hidden: true, on: { change: () => { addFiles([...fileIn.files]); fileIn.value = ''; } } });
  const slides = h('div.slides'), stageImgs = h('div.lt-imgs');
  const hint = h('div.lt-hint', icon('upload'), h('b', 'Put an image on the table'), h('span', 'Drop, paste (⌘V) or click to choose. PNG, JPG, WebP, GIF.'));
  const table = h('div.pr-lt', { tabindex: '0', role: 'button', 'aria-label': 'Light table: drop an image here, paste one, or press Enter to choose', on: {
    click: (e) => { if (!e.target.closest('button')) fileIn.click(); },
    keydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } },
    dragover: (e) => { e.preventDefault(); table.classList.add('over'); }, dragleave: () => table.classList.remove('over'),
    drop: (e) => { e.preventDefault(); table.classList.remove('over'); addFiles([...e.dataTransfer.files]); } } }, h('i.pr-rule'), stageImgs, hint, h('i.pr-crop.a'), h('i.pr-crop.b'), h('i.pr-crop.c'), h('i.pr-crop.d'));
  const onPaste = (e) => { const f = [...(e.clipboardData?.files || [])].filter((x) => x.type.startsWith('image/')); if (f.length) { e.preventDefault(); addFiles(f); } };
  document.addEventListener('paste', onPaste); offs.push(() => document.removeEventListener('paste', onPaste));

  function addFiles(list) {
    const ok = list.filter((f) => /^image\/(png|jpe?g|webp|gif)$/.test(f.type));
    if (ok.length < list.length) toast('Only PNG, JPG, WebP and GIF images can go on the table', 'warn');
    for (const f of ok) { if (P.files.length >= MAX_FILES) { toast(`The table holds ${MAX_FILES} images at most`, 'warn'); break; } P.files.push({ file: f, url: URL.createObjectURL(f) }); }
    if (ok.length) P.active = Math.max(0, P.files.length - ok.length);
    drawTable();
  }
  function removeFile(i) { URL.revokeObjectURL(P.files[i].url); P.files.splice(i, 1); P.active = Math.min(P.active, Math.max(0, P.files.length - 1)); drawTable(); }
  function drawTable() {
    const m = modeOf(P.mode), pair = m.min_images === 2;
    clear(stageImgs); clear(slides);
    const shown = pair ? P.files.slice(0, 2) : P.files.slice(P.active, P.active + 1);
    shown.forEach((f, i) => stageImgs.append(h('figure.slide', pair ? h('figcaption.num', i ? 'B' : 'A') : null, h('img', { src: f.url, alt: f.file.name, draggable: 'false' }))));
    if (pair && P.files.length === 1) stageImgs.append(h('figure.slide.ghost', h('figcaption.num', 'B'), h('span', 'Add a second image')));
    table.classList.toggle('empty', !P.files.length); table.classList.toggle('pair', pair); hint.hidden = P.files.length > 0;
    P.files.forEach((f, i) => slides.append(h('div.tn', { 'aria-current': String(!pair && i === P.active) }, h('button.tn-i', { type: 'button', 'aria-label': 'Show ' + f.file.name, on: { click: () => { P.active = i; drawTable(); } } }, h('img', { src: f.url, alt: '' })),
      h('button.tn-x', { type: 'button', 'aria-label': 'Remove ' + f.file.name, on: { click: () => removeFile(i) } }, icon('x')))));
    slides.hidden = P.files.length < 2 && !pair; devImg();
    updateGo();
  }

  // ---- the prism (mode switch) ----
  const modes = () => P.info?.modes || FALLBACK;
  const NS = 'http://www.w3.org/2000/svg', ROW = 38;
  const svg = h('svg.rays', { viewBox: `0 0 150 ${ROW * 7}`, preserveAspectRatio: 'none', 'aria-hidden': 'true' });
  const mid = (ROW * 7) / 2;
  svg.innerHTML = `<defs><linearGradient id="pr-white" x1="0" x2="1"><stop offset="0" stop-color="var(--text)" stop-opacity="0"/><stop offset="1" stop-color="var(--text)" stop-opacity=".95"/></linearGradient></defs>
    <path class="in" d="M0 ${mid} L70 ${mid}" stroke="url(#pr-white)"/>
    <path class="glass" d="M44 ${mid - 46} L44 ${mid + 46} L96 ${mid} Z"/>
    ${modes().map((m, i) => `<path class="beam" data-m="${m.name}" d="M96 ${mid} C 118 ${mid}, 120 ${ROW * i + ROW / 2}, 150 ${ROW * i + ROW / 2}" style="--b:${BEAM[m.name]}"/>`).join('')}`;
  const beams = () => [...svg.querySelectorAll('.beam')];
  const ports = h('div.ports', { role: 'radiogroup', 'aria-label': 'Mode' }, modes().map((m) => h('button.port', { type: 'button', role: 'radio', 'aria-checked': 'false', 'data-m': m.name, title: m.description, style: { '--b': BEAM[m.name] || 'var(--accent)' }, on: { click: () => setMode(m.name) } }, h('i.pr-dot'), LABEL[m.name] || m.name)));
  const sw = h('div.prism-sw', svg, ports);
  sw.addEventListener('keydown', (e) => {
    const names = modes().map((m) => m.name), i = names.indexOf(P.mode); let n = i;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') n = (i + 1) % names.length; else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') n = (i + names.length - 1) % names.length; else return;
    e.preventDefault(); setMode(names[n]); ports.querySelector(`[data-m="${names[n]}"]`)?.focus();
  });
  const modeNote = h('p.mode-note');
  function setMode(name) {
    P.mode = name; const m = modeOf(name);
    ports.querySelectorAll('.port').forEach((b) => { b.setAttribute('aria-checked', String(b.dataset.m === name)); b.tabIndex = b.dataset.m === name ? 0 : -1; });
    beams().forEach((b) => b.classList.toggle('on', b.dataset.m === name));
    modeNote.textContent = m.description; el.style.setProperty('--beam', BEAM[name] || 'var(--accent)');
    drawOptions(); drawTable();
  }

  // ---- options for the chosen mode ----
  const optsBox = h('div.opts');
  function drawOptions() {
    clear(optsBox); const m = modeOf(P.mode), o = (P.opts[P.mode] ||= {});
    for (const opt of m.options || []) {
      const id = `pr-${P.mode}-${opt.name}`, val = o[opt.name] ?? opt.default ?? '';
      let ctl;
      if (opt.name === 'question') ctl = h('textarea.field', { id, rows: 3, placeholder: IDEAS.ask, 'aria-required': 'true', on: { input: (e) => { o.question = e.target.value; updateGo(); }, keydown: (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); develop(); } } } });
      else if (opt.kind === 'choice') ctl = h('select.field', { id, on: { change: (e) => { o[opt.name] = e.target.value; } } }, opt.choices.map((c) => h('option', { value: c, selected: c === val }, c)));
      else if (opt.kind === 'int') { const out = h('b.num', String(val)); ctl = h('div.range', h('input', { id, type: 'range', min: opt.min ?? 0, max: opt.max ?? 5, step: 1, value: val, on: { input: (e) => { o[opt.name] = +e.target.value; out.textContent = e.target.value; } } }), out); }
      else if (opt.kind === 'schema') ctl = schemaPicker(o, id);
      else ctl = h('input.field', { id, type: 'text', placeholder: opt.help, value: val, on: { input: (e) => { o[opt.name] = e.target.value; } } });
      if (opt.name === 'question') ctl.value = val;
      optsBox.append(h('div.opt', h('label.label', { for: id }, human(opt.name)), ctl, opt.help && !['schema', 'text'].includes(opt.kind) ? h('small', opt.help) : null));
    }
  }
  function schemaPicker(o, id) {
    const names = (P.info?.schemas || [{ name: 'receipt' }, { name: 'invoice' }, { name: 'business_card' }, { name: 'chart_data' }]).map((s) => s.name);
    const area = h('textarea.field.mono', { rows: 6, placeholder: '{ "type": "object", "properties": { "title": { "type": "string" } } }', 'aria-label': 'Custom JSON Schema', value: P.schemaText, hidden: !(o.schema || '').startsWith('{'), on: { input: (e) => { P.schemaText = e.target.value; o.schema = e.target.value.trim() || 'receipt'; } } });
    area.value = P.schemaText;
    const sel = h('select.field', { id, on: { change: (e) => { if (e.target.value === '__custom') { area.hidden = false; o.schema = P.schemaText.trim() || '{"type":"object","properties":{}}'; area.focus(); } else { area.hidden = true; o.schema = e.target.value; } } } },
      names.map((n) => h('option', { value: n, selected: n === o.schema }, human(n))), h('option', { value: '__custom', selected: (o.schema || '').startsWith('{') }, 'Custom JSON Schema…'));
    return h('div.schema', sel, area);
  }

  const goBtn = h('button.btn.primary.big', { type: 'button', on: { click: develop } }, icon('play'), 'Develop', h('kbd', '⌘↵'));
  const ctl = h('aside.pr-ctl', h('div.label', 'Light table'), table, slides, fileIn, h('div.label', 'Prism'), sw, modeNote, optsBox, goBtn);
  function updateGo() {
    const m = modeOf(P.mode), need = m.min_images || 1, o = P.opts[P.mode] || {};
    const missing = P.files.length < need ? `Add ${need === 2 ? 'two images' : 'an image'} first` : (m.options || []).find((x) => x.required && !String(o[x.name] || '').trim()) ? 'Type a question first' : '';
    pr.dataset.running = String(running); goBtn.disabled = running;
    const many = m.max_images === 1 && P.files.length > 1; goBtn.replaceChildren(icon('play'), many ? `Develop ${P.files.length} prints` : 'Develop', h('kbd', '⌘↵')); goBtn.title = missing; goBtn.dataset.ready = String(!missing);
  }

  // ---- the print ----
  const head = h('header.pr-head');
  const devFig = h('figure.dev', { 'aria-hidden': 'true' });
  const ticker = h('div.ticker', { role: 'status', 'aria-live': 'polite' });
  const body = h('div.pr-body');
  const empty = h('div.pr-empty', h('svg', { viewBox: '0 0 120 70', 'aria-hidden': 'true', html: '<rect x="4" y="8" width="112" height="54" rx="5" fill="none" stroke="currentColor" stroke-dasharray="4 5" opacity=".5"/><path d="M0 35h44M60 35l44-26M60 35h56M60 35l44 26" stroke="currentColor" opacity=".35" fill="none"/><path d="M44 18v34l24-17z" fill="none" stroke="currentColor" stroke-width="2"/>' }),
    h('h3', 'The print develops here'), h('p', 'Choose a mode on the prism, put an image on the table and press Develop. Everything runs through NIM vision models'), h('p.dim', 'Offline? Simulation mode still develops a clearly marked demo print.'));
  const print = h('section.print', { 'aria-label': 'Developed print', hidden: true }, head, h('div.pr-dev', devFig, ticker), body);
  const main = h('div.pr-main', empty, print);

  function devImg() {
    clear(devFig); const pair = modeOf(P.mode).min_images === 2;
    (pair ? P.files.slice(0, 2) : P.files.slice(P.active, P.active + 1)).forEach((f) => devFig.append(h('img', { src: f.url, alt: '' })));
  }
  const setDev = (v) => { dev = v; devFig.style.setProperty('--dev', v.toFixed(3)); };

  function header(res) {
    clear(head);
    head.append(h('span.mode-chip', { style: { '--b': BEAM[res?.mode || P.mode] } }, h('i.pr-dot'), LABEL[res?.mode || P.mode]));
    if (res) head.append(h('span.chip.mono', { style: { pointerEvents: 'none' }, title: res.model }, shortModel(res.model)), res.simulated ? h('span.badge-sim', { title: 'Offline mock: this is a demonstration print, not a model reading your image' }, h('span.led.warn'), 'SIMULATED') : null, h('span.grow'), h('span.num.dim', `${fmtMs(res.elapsed_s)} · ${res.calls} call${res.calls === 1 ? '' : 's'}`));
  }
  function showItem(it, idx = null) {
    const card = h('article.item', idx != null ? h('h4.item-h', h('span.num', String(idx + 1).padStart(2, '0')), it.images?.map((i) => i.name).join(' + ') || it.name) : null);
    if (it.error) card.append(h('p.fail', icon('x'), `${it.name}: ${it.error}`));
    else {
      const notes = (it.notes || []).filter((n) => !n.startsWith('Simulated output'));
      card.append(RENDER[it.mode](it.output, it)); if (notes.length) card.append(h('ul.notes', notes.map((n) => h('li', n))));
    }
    return card;
  }

  // ---- events ----
  function flush() { raf = 0; if (live) { live.textContent = liveBuf; live.scrollTop = live.scrollHeight; } }
  function startLive(stage) {
    liveBuf = '';
    const textual = ['ocr', 'ask', 'ui2code'].includes(P.mode);
    if (!textual) { live = null; return; }
    live = h('pre.pr-live.selectable', { 'aria-label': 'Streaming output' }); body.replaceChildren(h('div.live-h', h('span.label', stage), h('i.pr-caret')), live);
  }
  function apply(ev, isReplay) {
    replaying = isReplay; const d = ev.data || {};
    switch (ev.type) {
      case 'run.start': print.hidden = false; empty.hidden = true; print.dataset.state = 'running'; setDev(0.12); clear(body); live = null; header(null); ticker.replaceChildren(h('span.led.busy'), 'Exposing…'); break;
      case 'stage.start': ticker.replaceChildren(h('span.led.busy'), d.stage === 'prepare' ? 'Exposing…' : `Developing · ${d.stage}`); setDev(Math.max(dev, d.stage === 'prepare' ? 0.15 : 0.3)); if (d.stage !== 'prepare' && !('index' in d)) startLive(d.stage); break;
      case 'token': if (!('index' in d)) { liveBuf += d.text; setDev(Math.min(0.92, dev + 0.0025)); if (live && !raf) raf = requestAnimationFrame(flush); } break;
      case 'result': if ('index' in d) ticker.replaceChildren(h('span.led.busy'), `Developed ${d.index + 1}…`); break;
      case 'run.done': finish(d.result); break;
      case 'run.failed': running = false; updateGo(); print.dataset.state = 'failed'; setDev(0); ticker.replaceChildren(h('span.led.bad'), d.cancelled ? 'Cancelled' : 'Failed');
        body.replaceChildren(h('p.fail', icon('x'), d.error || 'The run failed')); if (!d.cancelled) toast('Prism failed: ' + (d.error || 'unknown'), 'bad'); break;
    }
  }
  function finish(r) {
    running = false; updateGo(); print.dataset.state = 'done'; setDev(1); live = null;
    ticker.replaceChildren(h('span.led.on'), r.batch ? `${r.items.length} prints developed` : 'Fixed');
    const first = r.items.find((i) => !i.error);
    header(first && !r.batch ? first : first ? { ...first, calls: r.items.length, elapsed_s: r.items.reduce((s, i) => s + (i.elapsed_s || 0), 0) } : null);
    body.replaceChildren(...r.items.map((it, i) => showItem(it, r.batch ? i : null)));
    if (!replaying && !reduced()) body.animate([{ opacity: 0, transform: 'translateY(14px)' }, { opacity: 1, transform: 'none' }], { duration: 500, easing: 'cubic-bezier(.34,1.56,.64,1)' });
  }
  function attach(runId) {
    P.stop?.(); Object.assign(P, { runId, events: [], status: 'running' });
    P.stop = api.stream(runId, (ev) => { P.events.push(ev); if (ev.type === 'run.done') P.status = 'done'; if (ev.type === 'run.failed') P.status = 'failed'; P.ui?.apply(ev, false); if (ev.type.startsWith('run.')) refresh(['status', 'runs']); }, () => (P.stop = null));
  }

  async function develop() {
    const m = modeOf(P.mode), o = P.opts[P.mode] || {};
    if (goBtn.disabled) return;
    if (goBtn.title) { toast(goBtn.title, 'warn'); return; }
    const files = m.min_images === 2 ? P.files.slice(0, 2) : P.files;
    const fd = new FormData(); fd.append('mode', P.mode);
    for (const [k, v] of Object.entries(o)) if (v !== '' && v != null) fd.append(k, v);
    files.forEach((f) => fd.append('images', f.file, f.file.name));
    running = true; updateGo();
    try { const r = await api.post('/prism/run', fd); P.ui = ui; attach(r.run_id); refresh(['status', 'runs']); }
    catch (e) { running = false; updateGo(); toast('Couldn’t develop: ' + e.message, 'bad'); }
  }

  const pr = h('div.pr', ctl, main); el.append(pr);
  const ui = { apply }; P.ui = ui;
  el.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); develop(); } });
  setMode(P.mode); drawTable();
  api.get('/prism/modes').then((info) => { P.info = info; drawOptions(); drawTable(); }).catch(() => {});
  if (params?.run && params.run !== P.runId) { P.files = []; attach(params.run); }
  else if (P.runId && P.events.length) { replaying = true; P.events.slice().forEach((ev) => apply(ev, true)); replaying = false; running = P.status === 'running'; updateGo(); }
  return () => { offs.forEach((f) => f && f()); if (raf) cancelAnimationFrame(raf); if (P.ui === ui) P.ui = null; };
}
