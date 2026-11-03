// Loom: a patch bay for NIM pipelines. Nodes are rack modules with jacks, cables sag and carry signal pulses.
// One <svg> layer draws every cable; nodes are plain DOM positioned with transforms. Jack positions are
// computed from the model (fixed row heights), never measured, so dragging 20+ modules stays cheap.
import { h, clear, clamp, debounce, shortModel, reduced } from '../util.js';
import { icon } from '../icons.js';
import { store, api, toast } from '../app.js';

// ---------- geometry (must match loom.css) ----------
const W = 232, HEAD = 34, ROW = 24, LCD = 70, FOOT = 10, ZMIN = 0.2, ZMAX = 1.6;
const nodeH = (rows) => HEAD + Math.max(rows, 1) * ROW + LCD + FOOT;

// ---------- palette of kinds and data types ----------
const CAT = { io: 'core', model: 'council', logic: 'arena', data: 'forge', apps: 'atlas', net: 'loom' };
const CAT_NAME = { io: 'Input / output', model: 'Models', logic: 'Logic', data: 'Data', apps: 'NIM Lab apps', net: 'Network' };
const TCOL = { text: 'var(--c-council)', json: 'var(--c-atlas)', list: 'var(--c-forge)', number: 'var(--c-arena)', bool: 'var(--c-core)', vectors: 'var(--lm)', any: 'var(--dim)' };
const tcol = (t) => TCOL[t] || TCOL.any;
const INPUT_T = { string: 'text', number: 'number', boolean: 'bool', json: 'json', list: 'list' };
const STARTERS = {
  input: { type: 'string', default: '' }, prompt: { template: 'Write a short note about {{topic}}.' },
  json: { template: 'Extract the key facts from {{text}}.' }, template: { template: '{{text}}' },
  branch: { condition: 'value' }, regex: { pattern: '(\\w+)' }, python_fn: { fn: 'flatten' },
  map: { step: { type: 'prompt', params: { template: 'Summarise: {{item}}' } } }, http_get: { url: 'https://example.com' },
};
// jack compatibility, mirrors nimlab.loom.datatypes
const LOSSY = { 'json>text': 'warn', 'list>text': 'warn', 'text>number': 'warn', 'text>bool': 'warn', 'text>json': 'warn', 'text>list': 'warn', 'json>list': 'warn', 'number>bool': 'warn', 'bool>number': 'warn', 'json>number': 'warn', 'json>bool': 'warn', 'number>json': 'warn', 'bool>json': 'warn' };
function compat(s, t) {
  if (s === t || s === 'any' || t === 'any') return 'ok';
  if (t === 'text' && (s === 'number' || s === 'bool')) return 'ok';
  if (s === 'list' && t === 'json') return 'ok';
  if (s === 'vectors' && (t === 'list' || t === 'json')) return 'ok';
  return LOSSY[`${s}>${t}`] || 'error';
}
const KEYWORDS = new Set(['and', 'or', 'not', 'in', 'is', 'if', 'else', 'True', 'False', 'None', 'true', 'false', 'null']);
const dig = (o, path) => path.split('.').reduce((a, k) => (a && typeof a === 'object' ? a[k] : undefined), o);
const clone = (o) => JSON.parse(JSON.stringify(o));
const slug = (s) => (String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64).replace(/-+$/, '') || 'pipeline');
const lsGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch {} };
const nsvg = (html, cls = '') => h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.8', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', class: cls, 'aria-hidden': 'true', html });
const I = {
  undo: '<path d="M9 7 4.5 11.5 9 16"/><path d="M5 11.5h8a6 6 0 0 1 0 12" transform="translate(0 -4.5)"/>', redo: '<path d="m15 7 4.5 4.5L15 16"/><path d="M19 11.5h-8a6 6 0 0 0 0 12" transform="translate(0 -4.5)"/>',
  fit: '<path d="M4 9V5h4M20 9V5h-4M4 15v4h4M20 15v4h-4"/>', plus: '<path d="M12 5v14M5 12h14"/>', minus: '<path d="M5 12h14"/>', save: '<path d="M5 4h11l3 3v13H5Z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
  open: '<path d="M3.5 7.5a1.5 1.5 0 0 1 1.5-1.5h4l2 2.2h8a1.5 1.5 0 0 1 1.5 1.5V18a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 18Z"/>', run: '<path d="M7 5v14l12-7Z" fill="currentColor"/>', stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor"/>',
  export: '<path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19.5h14"/>', tray: '<path d="M4 6h16v12H4Z"/><path d="M4 14h5l1.5 2h3L15 14h5"/>', trash: '<path d="M4.5 7h15M9 7V4.5h6V7M6.5 7l1 13h9l1-13"/>', copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5"/>',
  x: '<path d="m6 6 12 12M18 6 6 18"/>', search: '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>',
};

// ---------- session: survives navigating away and back ----------
const SESSION = { pipeline: null, id: null, source: 'new', readonly: false, saved: '', view: null, hist: null, inputs: {} };
let cssReady = null;
function ensureCss() {
  return (cssReady ||= new Promise((res) => {
    if (document.querySelector('link[data-loom]')) return res();
    const l = h('link', { rel: 'stylesheet', href: 'css/screens/loom.css', 'data-loom': '1' });
    l.onload = l.onerror = () => res();
    document.head.append(l);
  }));
}

export async function mount(el, params = {}) {
  await ensureCss();
  const offs = [];
  const on = (t, ev, fn, o) => { t.addEventListener(ev, fn, o); offs.push(() => t.removeEventListener(ev, fn, o)); };

  // ---------------------------------------------------------------- state
  let types = {}, typeList = [];
  let P = SESSION.pipeline || null; // the pipeline document being edited (plain JSON)
  let view = SESSION.view || { x: 40, y: 30, z: 1 };
  const sel = new Set();
  let selEdge = null; // edge key
  let lint = { issues: [], ports: {}, ok: true };
  const RS = {}; // run state per node id
  let run = { status: 'idle', id: null, stop: null, info: null, outputs: null, log: [] };
  const NV = new Map(); // node id -> view refs
  const EV = new Map(); // edge key -> view refs
  const hist = SESSION.hist || { stack: [], i: -1, key: null, at: 0 };
  let raf = 0;
  const dirtyNodes = new Set();

  // ---------------------------------------------------------------- shell
  const ui = {};
  const btn = (label, ic, fn, cls = '', title = '') => h('button.btn.sm.lm-btn', { class: cls, title: title || label, 'aria-label': label, on: { click: fn } }, ic && nsvg(I[ic]), label && h('span', label));
  ui.name = h('input.lm-name', { type: 'text', 'aria-label': 'Pipeline name', spellcheck: 'false', placeholder: 'Untitled pipeline', on: { input: () => { P.title = ui.name.value; P.name = slug(ui.name.value) || 'untitled'; touch('rename'); } } });
  ui.src = h('span.lm-src.mono');
  ui.undo = btn('', 'undo', () => undo(), 'icon', 'Undo (⌘Z)'); ui.redo = btn('', 'redo', () => redo(), 'icon', 'Redo (⇧⌘Z)');
  ui.zoomOut = btn('', 'minus', () => zoomBy(1 / 1.2), 'icon', 'Zoom out (−)'); ui.zoomIn = btn('', 'plus', () => zoomBy(1.2), 'icon', 'Zoom in (+)');
  ui.zoom = h('button.lm-zoom.mono', { title: 'Reset zoom to 100%', on: { click: () => zoomTo(1) } }, '100%');
  ui.fit = btn('', 'fit', () => fit(true), 'icon', 'Fit to view (F)');
  ui.lintChip = h('button.lm-lint', { title: 'Show problems', on: { click: () => { selectNone(); renderInspector(); } } }, h('span.led'), h('span.lm-lint-t', 'Lint…'));
  ui.save = btn('Save', 'save', () => save(false), '', 'Save (⌘S)');
  ui.open = btn('Open', 'open', (e) => openMenu(e.currentTarget), '', 'Open a pipeline');
  ui.exp = btn('Export', 'export', (e) => exportMenu(e.currentTarget), '', 'Export as mermaid or a Python script');
  ui.trayBtn = btn('', 'tray', () => toggleTray(), 'icon', 'Outputs and log');
  ui.run = h('button.btn.sm.primary.lm-run', { title: 'Run (⌘↵)', on: { click: () => (run.status === 'running' ? stopRun() : startRun()) } }, nsvg(I.run), h('span', 'Run'));
  ui.status = h('span.lm-status.mono', { hidden: true });
  const bar = h('header.lm-bar',
    h('div.lm-bar-l', ui.open, h('div.lm-title', ui.name, ui.src), h('span.lm-sep'), ui.undo, ui.redo, h('span.lm-sep'), ui.zoomOut, ui.zoom, ui.zoomIn, ui.fit),
    h('div.lm-bar-r', ui.status, ui.lintChip, ui.trayBtn, ui.exp, ui.save, ui.run));

  // palette
  ui.pfilter = h('input.field.lm-psearch', { type: 'search', placeholder: 'Find a module…', 'aria-label': 'Filter modules', on: { input: () => renderPalette() } });
  ui.plist = h('div.lm-plist');
  const palette = h('aside.lm-palette', { 'aria-label': 'Module palette' }, h('div.lm-ph', h('span.label', 'Modules'), h('span.lm-hint', 'click or drag')), ui.pfilter, ui.plist);

  // canvas
  ui.svg = h('svg.lm-cables', { 'aria-hidden': 'true' });
  ui.gCables = h('g'); ui.gPulse = h('g.lm-pulses'); ui.gGhost = h('g');
  ui.svg.append(ui.gCables, ui.gPulse, ui.gGhost);
  ui.nodes = h('div.lm-nodes');
  ui.world = h('div.lm-world', ui.svg, ui.nodes);
  ui.rubber = h('div.lm-rubber', { hidden: true });
  ui.empty = h('div.lm-empty', { hidden: true }, h('b', 'Empty rack'), h('span', 'Drag a module from the left, or open a pipeline.'));
  ui.log = h('div.lm-log.mono'); ui.outs = h('div.lm-outs');
  ui.tray = h('section.lm-tray', { hidden: true, 'aria-label': 'Run outputs and log' },
    h('div.lm-tray-h', h('span.label', 'Outputs'), h('span.lm-rule'), h('button.btn.sm.ghost.icon', { 'aria-label': 'Close', on: { click: () => toggleTray(false) } }, nsvg(I.x))),
    h('div.lm-tray-b', ui.outs, h('div.lm-logwrap', h('span.label', 'Event log'), ui.log)));
  ui.canvas = h('div.lm-canvas', { tabindex: '0', 'aria-label': 'Patch bay: drag modules, drag from a jack to patch' }, ui.world, ui.rubber, ui.empty, ui.tray);
  ui.insp = h('aside.lm-insp', { 'aria-label': 'Inspector' });
  const root = h('div.lm', bar, palette, ui.canvas, ui.insp);
  el.append(root);

  // ---------------------------------------------------------------- model helpers
  const tOf = (n) => types[n.type];
  const nodeById = (id) => P.nodes.find((n) => n.id === id);
  const eKey = (e) => `${e.from}>${e.to}`;
  const split = (ref) => { const i = ref.indexOf('.'); return [ref.slice(0, i), ref.slice(i + 1)]; };
  function varsFrom(t, params) {
    const out = [];
    for (const src of t.input_vars || []) {
      let text = dig(params, src.path); if (typeof text !== 'string') continue;
      let names = [];
      if (src.kind === 'template') for (const m of text.matchAll(/\{\{\s*([A-Za-z_]\w*)(?:\.\w+)*\s*(?:\|\s*[A-Za-z_]\w*\s*)*\}\}/g)) names.push(m[1]);
      else { text = text.replace(/(["'])(?:\\.|(?!\1).)*\1/g, ' '); names = (text.match(/[A-Za-z_]\w*/g) || []).filter((w) => !KEYWORDS.has(w)); }
      for (const nm of names) if (!out.includes(nm) && !(t.input_exclude || []).includes(nm)) out.push(nm);
    }
    return out;
  }
  function portsOf(n) {
    const t = tOf(n); if (!t) { const lp = lint.ports[n.id]; return lp ? { ins: lp.inputs, outs: lp.outputs } : { ins: [], outs: [] }; }
    let ins = (t.inputs || []).map((p) => ({ ...p })), outs = (t.outputs || []).map((p) => ({ ...p }));
    const params = n.params || {};
    if (n.type === 'input') outs = [{ name: 'value', type: INPUT_T[params.type || 'string'] || 'text' }];
    if (n.type === 'map') {
      ins = [{ name: 'items', type: 'list' }];
      const st = types[params.step?.type || 'prompt'];
      if (st) {
        const itemPort = (st.inputs || []).find((p) => p.required !== false)?.name;
        for (const p of st.inputs || []) if (p.name !== itemPort && p.name !== 'item' && p.name !== 'index') ins.push({ ...p });
        for (const v of varsFrom(st, params.step?.params || {})) if (v !== 'item' && v !== 'index' && !ins.some((p) => p.name === v)) ins.push({ name: v, type: 'any' });
      }
    } else for (const v of varsFrom(t, params)) if (!ins.some((p) => p.name === v)) ins.push({ name: v, type: 'any' });
    return { ins, outs };
  }
  function layoutOf(n) { const pv = portsOf(n); return { ...pv, rows: Math.max(pv.ins.length, pv.outs.length, 1) }; }
  function jackPos(id, side, port) {
    const n = nodeById(id); if (!n) return null;
    const { ins, outs, rows } = layoutOf(n);
    const list = side === 'in' ? ins : outs; const i = list.findIndex((p) => p.name === port);
    const x = (n.pos?.x || 0) + (side === 'in' ? 0 : W);
    if (i < 0) return { x, y: (n.pos?.y || 0) + HEAD / 2, dangling: true };
    return { x, y: (n.pos?.y || 0) + HEAD + i * ROW + ROW / 2, dangling: false, type: list[i].type };
  }
  const portType = (id, side, port) => { const n = nodeById(id); if (!n) return 'any'; const l = portsOf(n)[side === 'in' ? 'ins' : 'outs'].find((p) => p.name === port); return l?.type || 'any'; };
  function uniqueId(base) { let i = 1; const ids = new Set(P.nodes.map((n) => n.id)); base = base.replace(/[^A-Za-z0-9_-]/g, '_'); while (ids.has(`${base}${i}`)) i++; return `${base}${i}`; }
  function reaches(from, to) { // does a path from -> to exist?
    const seen = new Set([from]); const q = [from];
    while (q.length) { const c = q.shift(); if (c === to) return true; for (const e of P.edges) { const [a] = split(e.from), [b] = split(e.to); if (a === c && !seen.has(b)) { seen.add(b); q.push(b); } } }
    return false;
  }
  const inputPortMulti = (id, port) => !!portsOf(nodeById(id)).ins.find((p) => p.name === port)?.multi;

  // ---------------------------------------------------------------- history
  const snap = () => JSON.stringify(P);
  function commit(label, coalesceKey) {
    const s = snap(), now = performance.now();
    if (coalesceKey && hist.key === coalesceKey && now - hist.at < 900 && hist.i === hist.stack.length - 1) hist.stack[hist.i] = s;
    else { hist.stack = hist.stack.slice(0, hist.i + 1); hist.stack.push(s); if (hist.stack.length > 100) hist.stack.shift(); hist.i = hist.stack.length - 1; }
    hist.key = coalesceKey || null; hist.at = now;
    afterChange();
  }
  function restore(s) {
    P = JSON.parse(s); SESSION.pipeline = P;
    for (const id of [...sel]) if (!nodeById(id)) sel.delete(id);
    if (selEdge && !P.edges.some((e) => eKey(e) === selEdge)) selEdge = null;
    renderAll(); renderInspector(); afterChange();
  }
  function undo() { if (hist.i > 0) { hist.i--; hist.key = null; restore(hist.stack[hist.i]); } }
  function redo() { if (hist.i < hist.stack.length - 1) { hist.i++; hist.key = null; restore(hist.stack[hist.i]); } }
  const touch = (key) => commit(key, key); // field edits coalesce by key
  function afterChange() {
    SESSION.pipeline = P; SESSION.hist = hist;
    ui.undo.disabled = hist.i <= 0; ui.redo.disabled = hist.i >= hist.stack.length - 1;
    const dirty = snap() !== SESSION.saved;
    ui.src.textContent = SESSION.readonly ? 'bundled · read-only: Save makes a copy' : SESSION.id ? (dirty ? 'unsaved changes' : 'saved') : 'new';
    ui.src.dataset.dirty = dirty ? '1' : '0';
    ui.empty.hidden = P.nodes.length > 0;
    lintSoon();
  }

  // ---------------------------------------------------------------- rendering: nodes
  function glyphOf(n) { return tOf(n)?.glyph || '?'; }
  function summary(n) {
    const p = n.params || {};
    switch (n.type) {
      case 'input': return p.default === undefined || p.default === '' ? `(${p.type || 'string'} input)` : String(typeof p.default === 'string' ? p.default : JSON.stringify(p.default));
      case 'prompt': case 'json': case 'template': return p.template || '';
      case 'branch': return `if ${p.condition || 'value'}`;
      case 'map': return `for each → ${p.step?.type || 'prompt'} ×${p.concurrency || 4}`;
      case 'regex': return `/${p.pattern || ''}/ ${p.mode || 'search'}`;
      case 'python_fn': return `${p.fn || '?'}(value)`;
      case 'http_get': return p.url || '';
      case 'council': return `council · ${p.strategy || 'moa'}`;
      case 'atlas_ask': return `atlas · ${p.collection || 'default'}`;
      case 'embed': return p.dedupe_threshold ? `dedupe ≥ ${p.dedupe_threshold}` : 'embeddings';
      case 'output': return `→ ${p.name || n.id}`;
      default: return tOf(n)?.description || '';
    }
  }
  function renderNode(n) {
    const old = NV.get(n.id); const { ins, outs, rows } = layoutOf(n); const t = tOf(n);
    const jacks = new Map();
    const rowsEl = [];
    for (let i = 0; i < rows; i++) {
      const pi = ins[i], po = outs[i];
      const mk = (side, p) => { const j = h('span.lm-jack', { class: side, 'data-side': side, 'data-port': p.name, 'data-type': p.type, style: { '--jc': tcol(p.type) }, title: `${p.name} · ${p.type}${p.multi ? ' · many cables' : ''}` }); jacks.set(`${side}:${p.name}`, j); return j; };
      rowsEl.push(h('div.lm-row',
        pi ? [mk('in', pi), h('span.lm-pn.in', pi.name)] : h('span'),
        po ? [h('span.lm-pn.out', po.name), mk('out', po)] : h('span')));
    }
    const led = h('span.lm-led', { title: 'No problems' });
    const lcdText = h('div.lm-lcd-t'), lcdMeta = h('div.lm-lcd-m');
    const el = h('div.lm-node', { 'data-id': n.id, 'data-type': n.type, style: { '--cat': `var(--c-${CAT[t?.category] || 'core'})`, height: nodeH(rows) + 'px' }, tabindex: '0', role: 'group', 'aria-label': `${t?.title || n.type} module ${n.label || n.id}` },
      h('div.lm-head', h('span.lm-glyph', glyphOf(n)), h('span.lm-ttl', n.label || n.id), h('span.lm-kind.mono', t?.title || n.type), led),
      h('div.lm-ports', rowsEl),
      h('div.lm-lcd', lcdText, lcdMeta),
      h('i.screw.tl'), h('i.screw.tr'));
    if (!t) el.classList.add('unknown');
    if (old) { old.el.replaceWith(el); } else ui.nodes.append(el);
    const v = { el, led, lcdText, lcdMeta, jacks, ins, outs };
    NV.set(n.id, v);
    place(n); if (EV.size) paintPlugs(); paintLed(n.id); paintLcd(n.id); el.classList.toggle('sel', sel.has(n.id));
    const st = RS[n.id]; if (st) el.dataset.run = st.status;
    return v;
  }
  function place(n) { const v = NV.get(n.id); if (v) v.el.style.transform = `translate(${n.pos.x}px, ${n.pos.y}px)`; }
  function paintLed(id) {
    const v = NV.get(id); if (!v) return;
    const issues = lint.issues.filter((i) => i.node === id);
    const err = issues.some((i) => i.severity === 'error'), warn = issues.length > 0;
    v.led.className = 'lm-led ' + (err ? 'bad' : warn ? 'warn' : 'ok');
    v.led.title = issues.length ? issues.map((i) => `${i.severity}: ${i.message}`).join('\n') : 'No problems';
    for (const [k, j] of v.jacks) { const bad = issues.some((i) => i.port && k.endsWith(':' + i.port) && k.startsWith('in:')); j.classList.toggle('flag', bad); }
    v.el.classList.toggle('flagged', warn);
  }
  function paintLcd(id) {
    const v = NV.get(id), n = nodeById(id); if (!v || !n) return;
    const st = RS[id];
    if (!st || st.status === 'idle') { v.lcdText.textContent = summary(n); v.lcdMeta.textContent = ''; v.lcdText.classList.add('idle'); return; }
    v.lcdText.classList.remove('idle');
    if (st.status === 'running') v.lcdText.textContent = (st.text ? st.text.slice(-260) : 'working') + '▌';
    else if (st.status === 'skipped') v.lcdText.textContent = `skipped · ${st.reason || ''}`;
    else if (st.status === 'error') v.lcdText.textContent = st.error || 'failed';
    else v.lcdText.textContent = st.preview || '(empty)';
    const bits = [];
    if (st.progress) bits.push(`${st.progress.done}/${st.progress.total}`);
    if (st.status === 'cached') bits.push('cached');
    if (st.latency != null && st.status !== 'running') bits.push(`${st.latency} ms`);
    if (st.tokens) bits.push(`${st.tokens} tok`);
    if (st.model) bits.push(shortModel(st.model));
    if (st.retry) bits.push(`retry ${st.retry}`);
    v.lcdMeta.textContent = bits.join(' · ');
  }
  function renderAll() {
    clear(ui.nodes); NV.clear();
    for (const n of P.nodes) renderNode(n);
    drawEdges(); paintSel(); applyView();
    ui.name.value = P.title || P.name || '';
    ui.empty.hidden = P.nodes.length > 0;
  }

  // ---------------------------------------------------------------- rendering: cables
  function cablePath(x1, y1, x2, y2) {
    const dx = Math.abs(x2 - x1), dy = Math.abs(y2 - y1);
    const sag = Math.min(110, 26 + dx * 0.14 + dy * 0.12), k = Math.max(46, dx * 0.5);
    return `M${x1} ${y1}C${x1 + k} ${y1 + sag} ${x2 - k} ${y2 + sag} ${x2} ${y2}`;
  }
  function edgeGeom(e) {
    const [a, ap] = split(e.from), [b, bp] = split(e.to);
    const p1 = jackPos(a, 'out', ap), p2 = jackPos(b, 'in', bp); if (!p1 || !p2) return null;
    return { d: cablePath(p1.x, p1.y, p2.x, p2.y), dangling: p1.dangling || p2.dangling, type: p1.type || p2.type || 'any', srcT: p1.type, dstT: p2.type };
  }
  function drawEdges() {
    for (const v of EV.values()) v.g.remove();
    EV.clear();
    for (const e of P.edges) {
      const geo = edgeGeom(e); if (!geo) continue;
      const shadow = h('path.lm-wire-shadow', { d: geo.d }), wire = h('path.lm-wire', { d: geo.d }), hit = h('path.lm-wire-hit', { d: geo.d });
      const g = h('g.lm-cable', { 'data-key': eKey(e) }, shadow, wire, hit);
      hit.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); selectEdge(eKey(e)); ui.canvas.focus({ preventScroll: true }); });
      hit.addEventListener('dblclick', (ev) => { ev.stopPropagation(); removeEdge(eKey(e)); });
      ui.gCables.append(g);
      const v = { e, g, wire, shadow, hit }; EV.set(eKey(e), v); styleEdge(v, geo);
    }
    paintPlugs();
  }
  function paintPlugs() {
    const used = new Set(); for (const e of P.edges) { used.add('out:' + e.from); used.add('in:' + e.to); }
    for (const [id, v] of NV) for (const [k, j] of v.jacks) { const [side, port] = [k.split(':')[0], k.slice(k.indexOf(':') + 1)]; j.classList.toggle('plugged', used.has(`${side}:${id}.${port}`)); }
  }
  function styleEdge(v, geo = edgeGeom(v.e)) {
    if (!geo) return;
    v.wire.setAttribute('d', geo.d); v.shadow.setAttribute('d', geo.d); v.hit.setAttribute('d', geo.d);
    v.g.style.setProperty('--wc', tcol(geo.type));
    const ck = geo.dangling ? 'error' : compat(geo.srcT || 'any', geo.dstT || 'any');
    v.g.classList.toggle('dangling', geo.dangling); v.g.classList.toggle('warn', ck === 'warn'); v.g.classList.toggle('bad', ck === 'error' && !geo.dangling);
    v.g.classList.toggle('sel', selEdge === eKey(v.e));
  }
  function rerouteNode(id) { for (const v of EV.values()) { const [a] = split(v.e.from), [b] = split(v.e.to); if (a === id || b === id) styleEdge(v); } }

  // ---------------------------------------------------------------- view
  function applyView() {
    ui.world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.z})`;
    ui.canvas.style.setProperty('--gs', `${24 * view.z}px`); ui.canvas.style.setProperty('--gx', `${view.x}px`); ui.canvas.style.setProperty('--gy', `${view.y}px`);
    ui.zoom.textContent = Math.round(view.z * 100) + '%'; SESSION.view = view;
  }
  const toWorld = (cx, cy) => { const r = ui.canvas.getBoundingClientRect(); return { x: (cx - r.left - view.x) / view.z, y: (cy - r.top - view.y) / view.z }; };
  function zoomAt(f, cx, cy) {
    const r = ui.canvas.getBoundingClientRect(); const px = cx - r.left, py = cy - r.top;
    const z = clamp(view.z * f, ZMIN, ZMAX); const k = z / view.z;
    view.x = px - (px - view.x) * k; view.y = py - (py - view.y) * k; view.z = z; applyView();
  }
  function zoomBy(f) { const r = ui.canvas.getBoundingClientRect(); zoomAt(f, r.left + r.width / 2, r.top + r.height / 2); }
  function zoomTo(z) { const r = ui.canvas.getBoundingClientRect(); zoomAt(z / view.z, r.left + r.width / 2, r.top + r.height / 2); }
  function boundsOf(ids) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of P.nodes) { if (ids && !ids.has(n.id)) continue; const hh = nodeH(layoutOf(n).rows); x0 = Math.min(x0, n.pos.x); y0 = Math.min(y0, n.pos.y); x1 = Math.max(x1, n.pos.x + W); y1 = Math.max(y1, n.pos.y + hh); }
    return x0 === Infinity ? null : { x0, y0, x1, y1 };
  }
  function fit(animate = false) {
    const b = boundsOf(); const r = ui.canvas.getBoundingClientRect();
    if (!b || r.width < 50) { view = { x: 40, y: 30, z: 1 }; return applyView(); }
    const pad = 56, bw = b.x1 - b.x0, bh = b.y1 - b.y0;
    // On a narrow canvas a whole-graph fit would be unreadably small: keep a legible floor and anchor to the left (pan for the rest).
    const z = clamp(Math.max(Math.min((r.width - pad * 2) / bw, (r.height - pad * 2) / bh), 0.4), ZMIN, 1.15);
    const next = { z, x: bw * z + pad * 2 > r.width ? pad / 2 - b.x0 * z : (r.width - bw * z) / 2 - b.x0 * z, y: (r.height - bh * z) / 2 - b.y0 * z };
    if (animate && !reduced()) { ui.world.classList.add('glide'); setTimeout(() => ui.world.classList.remove('glide'), 380); }
    view.x = next.x; view.y = next.y; view.z = next.z; applyView();
  }
  function centerOn(id) {
    const n = nodeById(id); if (!n) return; const r = ui.canvas.getBoundingClientRect();
    ui.world.classList.add('glide'); setTimeout(() => ui.world.classList.remove('glide'), 380);
    view.x = r.width / 2 - (n.pos.x + W / 2) * view.z; view.y = r.height / 2 - (n.pos.y + 60) * view.z; applyView();
  }

  // ---------------------------------------------------------------- selection
  function paintSel() { for (const [id, v] of NV) v.el.classList.toggle('sel', sel.has(id)); for (const v of EV.values()) v.g.classList.toggle('sel', selEdge === eKey(v.e)); }
  function select(ids, additive = false) {
    if (!additive) sel.clear();
    for (const id of ids) { if (additive && sel.has(id)) sel.delete(id); else sel.add(id); }
    selEdge = null; paintSel(); renderInspector();
  }
  function selectNone() { sel.clear(); selEdge = null; paintSel(); }
  function selectEdge(key) { sel.clear(); selEdge = key; paintSel(); renderInspector(); }

  // ---------------------------------------------------------------- graph edits
  function addNode(type, x, y, params) {
    const t = types[type]; if (!t) return null;
    const n = { id: uniqueId(type), type, params: params || clone(STARTERS[type] || {}), pos: { x: Math.round(x / 8) * 8, y: Math.round(y / 8) * 8 } };
    P.nodes.push(n); renderNode(n); ui.empty.hidden = true;
    select([n.id]); commit(`add ${type}`);
    return n;
  }
  function addAtCenter(type) {
    const r = ui.canvas.getBoundingClientRect(); const c = toWorld(r.left + r.width / 2, r.top + r.height / 2);
    return addNode(type, ...freeSpot(c.x - W / 2, c.y - 80));
  }
  function freeSpot(x, y) { // nudge down-right until the new module overlaps nothing
    const hit = (px, py) => P.nodes.some((n) => n.pos.x < px + W + 12 && n.pos.x + W + 12 > px && n.pos.y < py + 150 && n.pos.y + nodeH(layoutOf(n).rows) + 12 > py);
    for (let i = 0; i < 40 && hit(x, y); i++) { y += 40; if (i % 8 === 7) { x += W + 30; y -= 320; } }
    return [x, y];
  }
  function removeEdge(key, quiet) { const i = P.edges.findIndex((e) => eKey(e) === key); if (i < 0) return; P.edges.splice(i, 1); EV.get(key)?.g.remove(); EV.delete(key); if (selEdge === key) selEdge = null; refreshPortsAround(); if (!quiet) { commit('remove cable'); renderInspector(); } }
  function refreshPortsAround() { for (const n of P.nodes) paintLed(n.id); }
  function deleteSelection() {
    if (selEdge) { removeEdge(selEdge); return; }
    if (!sel.size) return;
    P.edges = P.edges.filter((e) => !sel.has(split(e.from)[0]) && !sel.has(split(e.to)[0]));
    P.nodes = P.nodes.filter((n) => !sel.has(n.id)); sel.clear();
    renderAll(); renderInspector(); commit('delete');
  }
  function duplicateSelection() {
    if (!sel.size) return; const map = {}; const created = [];
    for (const id of sel) { const n = clone(nodeById(id)); n.id = uniqueId(n.type); map[id] = n.id; n.pos.x += 40; n.pos.y += 40; P.nodes.push(n); created.push(n.id); }
    for (const e of [...P.edges]) { const [a, ap] = split(e.from), [b, bp] = split(e.to); if (map[a] && map[b]) P.edges.push({ from: `${map[a]}.${ap}`, to: `${map[b]}.${bp}` }); }
    renderAll(); select(created); commit('duplicate');
  }
  function connect(a, ap, b, bp) {
    if (a === b) return toast('A module cannot patch into itself', 'warn'), false;
    const ct = compat(portType(a, 'out', ap), portType(b, 'in', bp));
    if (ct === 'error') return toast(`Incompatible jacks: ${portType(a, 'out', ap)} → ${portType(b, 'in', bp)}`, 'warn'), false;
    if (reaches(b, a)) return toast('That cable would make a loop; pipelines must be acyclic', 'warn'), false;
    const key = `${a}.${ap}>${b}.${bp}`; if (P.edges.some((e) => eKey(e) === key)) return false;
    if (!inputPortMulti(b, bp)) for (const e of [...P.edges]) if (e.to === `${b}.${bp}`) removeEdge(eKey(e), true);
    P.edges.push({ from: `${a}.${ap}`, to: `${b}.${bp}` });
    drawEdges(); refreshPortsAround(); return true;
  }

  // ---------------------------------------------------------------- pointer interaction
  const isField = (t) => t && (t.closest?.('input,textarea,select,[contenteditable]') != null);
  let drag = null;
  function onNodeDown(ev) {
    if (ev.button !== 0) return;
    const nodeEl = ev.target.closest('.lm-node'); if (!nodeEl) return;
    const jack = ev.target.closest('.lm-jack');
    const id = nodeEl.dataset.id;
    if (jack) return startLink(ev, id, jack);
    ev.stopPropagation();
    if (ev.shiftKey || ev.metaKey || ev.ctrlKey) select([id], true);
    else if (!sel.has(id)) select([id]);
    ui.canvas.focus({ preventScroll: true });
    const start = { x: ev.clientX, y: ev.clientY };
    const items = [...sel].map((i) => { const n = nodeById(i); return n && { n, x: n.pos.x, y: n.pos.y }; }).filter(Boolean);
    drag = { kind: 'move', start, items, moved: false, pid: ev.pointerId, id };
    nodeEl.setPointerCapture?.(ev.pointerId);
  }
  function startLink(ev, id, jack) {
    ev.stopPropagation(); ev.preventDefault();
    const side = jack.dataset.side, port = jack.dataset.port;
    let from = { id, side, port };
    let detached = null;
    if (side === 'in') { // lifting a plugged cable out of an input jack re-patches it from its source
      const e = P.edges.find((x) => x.to === `${id}.${port}`);
      if (e) { const [a, ap] = split(e.from); detached = { ...e }; removeEdge(eKey(e), true); from = { id: a, side: 'out', port: ap }; }
    }
    const type = portType(from.id, from.side, from.port);
    const ghost = h('path.lm-ghost', { style: { '--wc': tcol(type) } }); ui.gGhost.append(ghost);
    ui.canvas.classList.add('linking'); ui.canvas.dataset.from = from.side;
    markTargets(from, true);
    drag = { kind: 'link', from, ghost, detached, pid: ev.pointerId, type };
    ui.canvas.setPointerCapture?.(ev.pointerId);
    moveLink(ev);
  }
  function markTargets(from, on) {
    for (const [id, v] of NV) for (const [k, j] of v.jacks) {
      const [side] = k.split(':');
      if (!on) { j.classList.remove('can', 'cant'); continue; }
      if (side === from.side || id === from.id) { j.classList.add('cant'); j.classList.remove('can'); continue; }
      const st = from.side === 'out' ? portType(from.id, 'out', from.port) : j.dataset.type, tt = from.side === 'out' ? j.dataset.type : portType(from.id, 'in', from.port);
      const ok = compat(st, tt) !== 'error'; j.classList.toggle('can', ok); j.classList.toggle('cant', !ok);
    }
  }
  function moveLink(ev) {
    const d = drag; if (!d || d.kind !== 'link') return;
    const p = toWorld(ev.clientX, ev.clientY); const a = jackPos(d.from.id, d.from.side, d.from.port) || p;
    const target = ev.target && document.elementFromPoint(ev.clientX, ev.clientY)?.closest?.('.lm-jack');
    let end = p;
    if (target && target.classList.contains('can')) { const tn = target.closest('.lm-node').dataset.id; const tp = jackPos(tn, target.dataset.side, target.dataset.port); if (tp) end = tp; }
    d.ghost.setAttribute('d', d.from.side === 'out' ? cablePath(a.x, a.y, end.x, end.y) : cablePath(end.x, end.y, a.x, a.y));
  }
  function endLink(ev) {
    const d = drag; drag = null; if (!d) return;
    d.ghost.remove(); markTargets(d.from, false); ui.canvas.classList.remove('linking'); delete ui.canvas.dataset.from;
    const under = document.elementFromPoint(ev.clientX, ev.clientY);
    let jack = under?.closest?.('.lm-jack'); const nodeEl = under?.closest?.('.lm-node');
    let ok = false;
    if (!jack && nodeEl && nodeEl.dataset.id !== d.from.id) jack = bestJack(nodeEl, d.from);
    if (jack && jack.dataset.side !== d.from.side) {
      const tid = jack.closest('.lm-node').dataset.id, port = jack.dataset.port;
      ok = d.from.side === 'out' ? connect(d.from.id, d.from.port, tid, port) : connect(tid, port, d.from.id, d.from.port);
    }
    if (ok) { commit('patch'); renderInspector(); }
    else if (d.detached) { drawEdges(); commit('unplug'); renderInspector(); }
  }
  function bestJack(nodeEl, from) {
    const want = from.side === 'out' ? 'in' : 'out'; const cands = [...nodeEl.querySelectorAll(`.lm-jack.${want}`)].filter((j) => j.classList.contains('can'));
    if (!cands.length) return null;
    const nid = nodeEl.dataset.id;
    const free = cands.find((j) => want === 'out' || !P.edges.some((e) => e.to === `${nid}.${j.dataset.port}`));
    return free || cands[0];
  }
  function onCanvasDown(ev) {
    if (ev.target.closest('.lm-tray')) return;
    if (ev.button === 1 || (ev.button === 0 && !ev.target.closest('.lm-node') && !ev.shiftKey)) {
      ev.preventDefault(); ui.canvas.focus({ preventScroll: true });
      drag = { kind: 'pan', sx: ev.clientX, sy: ev.clientY, vx: view.x, vy: view.y, moved: false, pid: ev.pointerId };
      ui.canvas.setPointerCapture?.(ev.pointerId); ui.canvas.classList.add('panning');
    } else if (ev.button === 0 && ev.shiftKey && !ev.target.closest('.lm-node')) {
      const r = ui.canvas.getBoundingClientRect();
      drag = { kind: 'rubber', sx: ev.clientX - r.left, sy: ev.clientY - r.top, pid: ev.pointerId }; ui.canvas.setPointerCapture?.(ev.pointerId);
    }
  }
  function onMove(ev) {
    if (!drag) return;
    if (drag.kind === 'move') {
      const dx = (ev.clientX - drag.start.x) / view.z, dy = (ev.clientY - drag.start.y) / view.z;
      if (!drag.moved && Math.hypot(dx, dy) * view.z < 3) return;
      drag.moved = true;
      for (const it of drag.items) { it.n.pos.x = Math.round(it.x + dx); it.n.pos.y = Math.round(it.y + dy); dirtyNodes.add(it.n.id); NV.get(it.n.id)?.el.classList.add('dragging'); }
      if (!raf) raf = requestAnimationFrame(flush);
    } else if (drag.kind === 'pan') {
      view.x = drag.vx + (ev.clientX - drag.sx); view.y = drag.vy + (ev.clientY - drag.sy); drag.moved = true; applyView();
    } else if (drag.kind === 'link') moveLink(ev);
    else if (drag.kind === 'rubber') {
      const r = ui.canvas.getBoundingClientRect(); const x = ev.clientX - r.left, y = ev.clientY - r.top;
      Object.assign(ui.rubber.style, { left: Math.min(x, drag.sx) + 'px', top: Math.min(y, drag.sy) + 'px', width: Math.abs(x - drag.sx) + 'px', height: Math.abs(y - drag.sy) + 'px' }); ui.rubber.hidden = false;
    }
  }
  function flush() { raf = 0; for (const id of dirtyNodes) { const n = nodeById(id); if (n) { place(n); rerouteNode(id); } } dirtyNodes.clear(); }
  function onUp(ev) {
    const d = drag; if (!d) return;
    if (d.kind === 'link') return endLink(ev);
    drag = null;
    if (d.kind === 'move') {
      for (const it of d.items) NV.get(it.n.id)?.el.classList.remove('dragging');
      if (d.moved) { flush(); commit('move'); } else if (!ev.shiftKey && !ev.metaKey && !ev.ctrlKey && sel.size > 1) select([d.id]);
    } else if (d.kind === 'pan') {
      ui.canvas.classList.remove('panning');
      if (!d.moved && !ev.target.closest('.lm-node')) { selectNone(); renderInspector(); }
    } else if (d.kind === 'rubber') {
      ui.rubber.hidden = true; const r = ui.canvas.getBoundingClientRect();
      const a = toWorld(Math.min(d.sx, ev.clientX - r.left) + r.left, Math.min(d.sy, ev.clientY - r.top) + r.top), b = toWorld(Math.max(d.sx, ev.clientX - r.left) + r.left, Math.max(d.sy, ev.clientY - r.top) + r.top);
      const hit = P.nodes.filter((n) => n.pos.x < b.x && n.pos.x + W > a.x && n.pos.y < b.y && n.pos.y + nodeH(layoutOf(n).rows) > a.y).map((n) => n.id);
      select(hit);
    }
  }
  function onWheel(ev) {
    if (ev.target.closest('.lm-tray')) return;
    ev.preventDefault();
    if (ev.ctrlKey || ev.metaKey) zoomAt(Math.exp(-ev.deltaY * (ev.deltaMode ? 0.05 : 0.01)), ev.clientX, ev.clientY);
    else { view.x -= ev.deltaX; view.y -= ev.deltaY; applyView(); }
  }
  on(ui.nodes, 'pointerdown', onNodeDown); on(ui.canvas, 'pointerdown', onCanvasDown);
  on(window, 'pointermove', onMove); on(window, 'pointerup', onUp); on(window, 'pointercancel', onUp);
  on(ui.canvas, 'wheel', onWheel, { passive: false });
  on(ui.canvas, 'dragover', (e) => { if (e.dataTransfer?.types.includes('text/x-loom-type')) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } });
  on(ui.canvas, 'drop', (e) => { const t = e.dataTransfer?.getData('text/x-loom-type'); if (!t) return; e.preventDefault(); const p = toWorld(e.clientX, e.clientY); addNode(t, p.x - W / 2, p.y - 20); });
  on(ui.nodes, 'keydown', (e) => { const nd = e.target.closest?.('.lm-node'); if (nd && e.target === nd && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select([nd.dataset.id], e.shiftKey); } });
  on(ui.nodes, 'dblclick', (e) => { const nd = e.target.closest('.lm-node'); if (nd && !e.target.closest('.lm-jack')) { select([nd.dataset.id]); ui.insp.querySelector('input,textarea,select')?.focus(); } });

  // ---------------------------------------------------------------- keyboard
  on(window, 'keydown', (e) => {
    if (!root.isConnected) return;
    const mod = e.metaKey || e.ctrlKey; const typing = isField(e.target);
    if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); save(false); return; }
    if (mod && e.key === 'Enter') { e.preventDefault(); if (run.status !== 'running') startRun(); return; }
    if (typing) return;
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); }
    else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
    else if (mod && e.key.toLowerCase() === 'a') { e.preventDefault(); select(P.nodes.map((n) => n.id)); }
    else if (mod && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicateSelection(); }
    else if (!mod && (e.key === 'Delete' || e.key === 'Backspace')) { e.preventDefault(); deleteSelection(); }
    else if (!mod && e.key === 'Escape') { if (drag?.kind === 'link') { drag.detached = null; endLink({ clientX: -1, clientY: -1 }); } selectNone(); renderInspector(); }
    else if (!mod && e.key.toLowerCase() === 'f') { fit(true); }
    else if (!mod && (e.key === '+' || e.key === '=')) zoomBy(1.2);
    else if (!mod && (e.key === '-' || e.key === '_')) zoomBy(1 / 1.2);
    else if (!mod && e.key.startsWith('Arrow') && sel.size) {
      e.preventDefault(); const s = e.shiftKey ? 40 : 8; const dx = e.key === 'ArrowLeft' ? -s : e.key === 'ArrowRight' ? s : 0, dy = e.key === 'ArrowUp' ? -s : e.key === 'ArrowDown' ? s : 0;
      for (const id of sel) { const n = nodeById(id); n.pos.x += dx; n.pos.y += dy; place(n); rerouteNode(id); } commit('nudge', 'nudge');
    }
  });

  // ---------------------------------------------------------------- palette
  function renderPalette() {
    const q = ui.pfilter.value.trim().toLowerCase(); clear(ui.plist);
    const groups = {};
    for (const t of typeList) if (!q || `${t.title} ${t.type} ${t.description}`.toLowerCase().includes(q)) (groups[t.category] ||= []).push(t);
    for (const cat of Object.keys(CAT)) {
      const items = groups[cat]; if (!items) continue;
      ui.plist.append(h('div.lm-pg', { style: { '--cat': `var(--c-${CAT[cat]})` } }, h('div.lm-pgh', h('i'), h('span.label', CAT_NAME[cat] || cat)),
        items.map((t) => h('button.lm-pi', { draggable: 'true', title: t.description, 'data-type': t.type,
          on: { click: () => addAtCenter(t.type), dragstart: (e) => { e.dataTransfer.setData('text/x-loom-type', t.type); e.dataTransfer.effectAllowed = 'copy'; } } },
          h('span.lm-pglyph', t.glyph), h('span.lm-pt', h('b', t.title), h('small', t.description))))));
    }
    if (!ui.plist.children.length) ui.plist.append(h('p.lm-none', 'No module matches.'));
  }

  // ---------------------------------------------------------------- inspector (schema-driven)
  function fieldRow(label, control, hint, required) {
    return h('label.lm-f', h('span.lm-fl', label, required && h('i.req', ' *')), control, hint && h('small.lm-fh', hint));
  }
  function schemaField(key, spec, value, set) {
    const label = spec.title || key; const hint = spec.description;
    const setv = (v) => set(v);
    if (spec.format === 'step') return stepField(key, spec, value ?? spec.default, set);
    if (spec.type === 'boolean') {
      const on = value ?? spec.default ?? false;
      const sw = h('button.switch', { type: 'button', role: 'switch', 'aria-checked': String(!!on), 'aria-label': label, on: { click: () => { const v = sw.getAttribute('aria-checked') !== 'true'; sw.setAttribute('aria-checked', String(v)); setv(v); } } });
      return h('div.lm-f.row', h('span.lm-fl', label), sw, hint && h('small.lm-fh', hint));
    }
    if (spec.enum) {
      const sel = h('select.field.sm', { 'aria-label': label, on: { change: () => setv(sel.value) } }, spec.enum.map((o) => h('option', { value: o, selected: (value ?? spec.default) === o }, o)));
      if (value === undefined && spec.default === undefined) sel.value = spec.enum[0];
      return fieldRow(label, sel, hint);
    }
    if (spec.type === 'number' || spec.type === 'integer') {
      const ranged = spec.minimum != null && spec.maximum != null && spec.maximum - spec.minimum <= 100;
      const cur = value ?? spec.default ?? 0;
      const num = h('input.field.sm.mono', { type: 'number', value: cur, min: spec.minimum, max: spec.maximum, step: spec.type === 'integer' ? 1 : 0.05, 'aria-label': label });
      const parse = () => { const v = num.value === '' ? undefined : Number(num.value); setv(v); };
      if (ranged) {
        const rng = h('input.lm-range', { type: 'range', min: spec.minimum, max: spec.maximum, step: spec.type === 'integer' ? 1 : 0.05, value: cur, 'aria-label': label + ' slider' });
        rng.addEventListener('input', () => { num.value = rng.value; parse(); }); num.addEventListener('input', () => { rng.value = num.value; parse(); });
        return fieldRow(label, h('div.lm-numrow', rng, num), hint);
      }
      num.addEventListener('input', parse); return fieldRow(label, num, hint);
    }
    if (spec.format === 'json' || spec.type === 'object' || (spec.type === 'array' && spec.format !== 'models')) {
      const ta = h('textarea.field.mono.lm-json', { rows: 5, spellcheck: 'false', 'aria-label': label }); ta.value = JSON.stringify(value ?? spec.default ?? (spec.type === 'array' ? [] : {}), null, 2);
      const err = h('small.lm-err', { hidden: true });
      ta.addEventListener('input', () => { try { const v = JSON.parse(ta.value || 'null'); ta.classList.remove('invalid'); err.hidden = true; setv(v); } catch (e) { ta.classList.add('invalid'); err.hidden = false; err.textContent = String(e.message).slice(0, 120); } });
      return fieldRow(label, [ta, err], hint);
    }
    if (spec.format === 'models') {
      const inp = h('input.field.sm.mono', { type: 'text', value: (value || []).join(', '), placeholder: 'default panel', 'aria-label': label, on: { input: () => setv(inp.value.split(',').map((s) => s.trim()).filter(Boolean)) } });
      return fieldRow(label, inp, hint);
    }
    if (spec.format === 'textarea') {
      const ta = h('textarea.field.lm-ta', { rows: 4, 'aria-label': label, spellcheck: 'false', placeholder: spec.default || '', on: { input: () => setv(ta.value) } }); ta.value = value ?? spec.default ?? '';
      return fieldRow(label, ta, hint, spec.__required);
    }
    const inp = h('input.field.sm', { type: 'text', value: value ?? '', placeholder: spec.default != null ? String(spec.default) : '', 'aria-label': label, spellcheck: 'false', class: spec.format === 'expr' || spec.format === 'code' ? 'mono' : '', on: { input: () => setv(inp.value) } });
    if (spec.format === 'model') { const dl = h('datalist', { id: 'lm-models' }, (store.models || []).map((m) => h('option', { value: m.id }))); inp.setAttribute('list', 'lm-models'); return fieldRow(label, [inp, dl], hint); }
    return fieldRow(label, inp, hint, spec.__required);
  }
  function paramsForm(t, params, setParam) {
    const props = t.params?.properties || {}; const req = new Set(t.params?.required || []);
    return Object.entries(props).map(([k, spec]) => schemaField(k, { ...spec, __required: req.has(k) }, params[k], (v) => setParam(k, v)));
  }
  function stepField(key, spec, value, set) {
    let step = clone(value && typeof value === 'object' ? value : spec.default || { type: 'prompt', params: {} });
    const box = h('div.lm-step');
    const choices = typeList.filter((t) => !['map', 'branch', 'merge', 'join', 'input', 'output'].includes(t.type));
    const sel = h('select.field.sm', { 'aria-label': 'Step type', on: { change: () => { step = { type: sel.value, params: clone(STARTERS[sel.value] || {}) }; build(); set(clone(step)); } } },
      choices.map((t) => h('option', { value: t.type, selected: t.type === step.type }, `${t.title} (${t.type})`)));
    function build() {
      clear(box); const t = types[step.type]; if (!t) return;
      box.append(...paramsForm(t, step.params || {}, (k, v) => { step.params = { ...(step.params || {}) }; if (v === undefined || v === '') delete step.params[k]; else step.params[k] = v; set(clone(step)); }));
    }
    build();
    return h('div.lm-f.step', h('span.lm-fl', 'Step run per item'), sel, box, spec.description && h('small.lm-fh', spec.description));
  }

  function setParam(n, key, value, label) {
    n.params = n.params || {};
    if (value === undefined || value === '' || (Array.isArray(value) && !value.length && key === 'models')) delete n.params[key]; else n.params[key] = value;
    const t = tOf(n); const dynamic = (t.input_vars || []).some((s) => s.path.split('.')[0] === key) || n.type === 'input' || n.type === 'map';
    if (dynamic) { renderNode(n); rerouteNode(n.id); refreshInspectorDefaults(n); } else paintLcd(n.id);
    paintLcd(n.id); commit(`edit ${n.id}.${key}`, `p:${n.id}.${key}`);
  }
  function refreshInspectorDefaults(n) { const box = ui.insp.querySelector('.lm-defaults'); if (box) box.replaceWith(defaultsBox(n)); }
  function defaultsBox(n) {
    const { ins } = portsOf(n); const box = h('div.lm-defaults');
    const free = ins.filter((p) => !P.edges.some((e) => e.to === `${n.id}.${p.name}`));
    if (!free.length) return box;
    box.append(h('div.lm-sec', h('span.label', 'Unwired inputs'), h('small', 'values used when nothing is patched in')));
    for (const p of free) {
      const cur = n.defaults?.[p.name]; const inp = h('input.field.sm', { type: 'text', value: cur === undefined ? '' : typeof cur === 'string' ? cur : JSON.stringify(cur), placeholder: p.required === false ? 'optional' : 'required', 'aria-label': `default for ${p.name}`, spellcheck: 'false' });
      inp.addEventListener('input', () => {
        n.defaults = n.defaults || {}; let v = inp.value;
        if (v === '') delete n.defaults[p.name]; else { if (p.type === 'number' && !isNaN(Number(v))) v = Number(v); else if (p.type === 'bool') v = v === 'true'; else if ((p.type === 'json' || p.type === 'list') && /^[[{]/.test(v)) { try { v = JSON.parse(v); } catch {} } n.defaults[p.name] = v; }
        if (!Object.keys(n.defaults).length) delete n.defaults; commit(`default ${n.id}.${p.name}`, `d:${n.id}.${p.name}`);
      });
      box.append(h('label.lm-f.row2', h('span.lm-jk', { style: { '--jc': tcol(p.type) } }), h('span.lm-fl.mono', p.name), inp));
    }
    return box;
  }
  function renderInspector() {
    clear(ui.insp);
    if (selEdge) return ui.insp.append(edgePanel());
    if (sel.size === 1) return ui.insp.append(nodePanel(nodeById([...sel][0])));
    if (sel.size > 1) return ui.insp.append(h('div.lm-ip', h('div.lm-ih', h('span.label', 'Selection')), h('p.lm-big', `${sel.size} modules`), h('div.lm-actions', btn('Duplicate', 'copy', duplicateSelection), btn('Delete', 'trash', deleteSelection, 'danger'))));
    ui.insp.append(pipelinePanel());
  }
  function edgePanel() {
    const v = EV.get(selEdge); const e = v?.e; if (!e) return h('div');
    const [a, ap] = split(e.from), [b, bp] = split(e.to); const ts = portType(a, 'out', ap), td = portType(b, 'in', bp);
    return h('div.lm-ip', h('div.lm-ih', h('span.label', 'Patch cable')),
      h('div.lm-cableinfo', h('span.lm-jk', { style: { '--jc': tcol(ts) } }), h('b.mono', `${a}.${ap}`), h('span', '→'), h('b.mono', `${b}.${bp}`)),
      h('p.lm-fh', `${ts} → ${td}${compat(ts, td) === 'warn' ? ' (converted at run time)' : ''}`),
      h('div.lm-actions', btn('Unplug', 'trash', () => removeEdge(selEdge), 'danger')));
  }
  function nodePanel(n) {
    const t = tOf(n); const st = RS[n.id];
    const head = h('div.lm-ih', { style: { '--cat': `var(--c-${CAT[t?.category] || 'core'})` } }, h('span.lm-glyph.big', glyphOf(n)), h('div', h('h3', t?.title || n.type), h('small', t?.description || 'Unknown module type')));
    const label = fieldRow('Label', h('input.field.sm', { type: 'text', value: n.label || '', placeholder: n.id, 'aria-label': 'Label', on: { input: (e) => { if (e.target.value) n.label = e.target.value; else delete n.label; NV.get(n.id).el.querySelector('.lm-ttl').textContent = n.label || n.id; commit('label', `label:${n.id}`); } } }), `id: ${n.id}`);
    const form = t ? paramsForm(t, n.params || {}, (k, v) => setParam(n, k, v)) : [h('p.lm-err', 'This module type is not registered on the server.')];
    const retries = fieldRow('Retries', h('input.field.sm.mono', { type: 'number', min: 0, max: 10, value: n.retries || 0, 'aria-label': 'Retries', on: { input: (e) => { const v = Math.max(0, Math.min(10, Number(e.target.value) || 0)); if (v) n.retries = v; else delete n.retries; commit('retries', `r:${n.id}`); } } }));
    const timeout = fieldRow('Timeout (s)', h('input.field.sm.mono', { type: 'number', min: 0, step: 1, value: n.timeout ?? '', placeholder: 'none', 'aria-label': 'Timeout', on: { input: (e) => { const v = Number(e.target.value); if (v > 0) n.timeout = v; else delete n.timeout; commit('timeout', `t:${n.id}`); } } }));
    const cache = h('select.field.sm', { 'aria-label': 'Cache', on: { change: (e) => { const v = e.target.value; if (v === 'default') delete n.cache; else n.cache = v === 'on'; commit('cache', `c:${n.id}`); } } }, ['default', 'on', 'off'].map((o) => h('option', { value: o, selected: (n.cache === undefined ? 'default' : n.cache ? 'on' : 'off') === o }, o === 'default' ? (t?.cacheable ? 'default (on)' : 'default (off)') : o)));
    const issues = lint.issues.filter((i) => i.node === n.id);
    return h('div.lm-ip', head, label, ...form, defaultsBox(n),
      h('details.lm-adv', h('summary.label', 'Reliability'), h('div.lm-grid2', retries, timeout), fieldRow('Result cache', cache)),
      issues.length ? h('div.lm-sec', h('span.label', 'Problems'), issues.map((i) => h('p.lm-issue', { class: i.severity }, h('span.led', { class: i.severity === 'error' ? 'bad' : 'warn' }), i.message))) : null,
      st && st.status !== 'idle' ? h('div.lm-sec', h('span.label', 'Last run'), h('p.lm-runinfo.mono', `${st.status}${st.latency != null ? ` · ${st.latency} ms` : ''}${st.tokens ? ` · ${st.tokens} tok` : ''}${st.model ? ` · ${shortModel(st.model)}` : ''}`),
        Object.entries(st.outputs || {}).map(([k, v]) => h('div.lm-outline', h('b.mono', k), h('pre.selectable', v))), st.error && h('p.lm-err', st.error)) : null,
      h('div.lm-actions', btn('Duplicate', 'copy', duplicateSelection), btn('Delete', 'trash', deleteSelection, 'danger')));
  }
  function inputNodes() { return P.nodes.filter((n) => n.type === 'input').map((n) => ({ n, name: n.params?.name || n.id, type: n.params?.type || 'string', required: !('default' in (n.params || {})) })); }
  function pipelinePanel() {
    const ins = inputNodes();
    const issues = lint.issues;
    const inputs = ins.length ? ins.map(({ n, name, type }) => {
      const cur = SESSION.inputs[name] ?? n.params?.default ?? '';
      const text = typeof cur === 'string' ? cur : JSON.stringify(cur);
      const mk = () => {
        if (type === 'boolean') { const sw = h('button.switch', { type: 'button', role: 'switch', 'aria-checked': String(cur === true || cur === 'true'), 'aria-label': name, on: { click: () => { const v = sw.getAttribute('aria-checked') !== 'true'; sw.setAttribute('aria-checked', String(v)); SESSION.inputs[name] = v; } } }); return sw; }
        const long = type === 'string' && (text.length > 60 || text.includes('\n')) || type === 'json' || type === 'list';
        const f = h(long ? 'textarea.field.lm-ta' : 'input.field.sm', { value: long ? undefined : text, rows: 4, 'aria-label': `input ${name}`, spellcheck: 'false', on: { input: () => { SESSION.inputs[name] = f.value; } } });
        if (long) f.value = text; return f;
      };
      return fieldRow(h('span', name, h('small.lm-ty', type)), mk(), n.params?.description);
    }) : [h('p.lm-fh', 'This pipeline has no input nodes.')];
    return h('div.lm-ip',
      h('div.lm-ih', h('span.label', 'Pipeline')),
      fieldRow('Title', h('input.field.sm', { type: 'text', value: P.title || '', 'aria-label': 'Title', on: { input: (e) => { P.title = e.target.value; ui.name.value = e.target.value; commit('title', 'title'); } } })),
      fieldRow('Description', h('textarea.field.lm-ta', { rows: 3, 'aria-label': 'Description', on: { input: (e) => { P.description = e.target.value; commit('desc', 'desc'); } } }, P.description || '')),
      h('div.lm-sec', h('span.label', 'Run inputs')), ...inputs,
      h('div.lm-sec', h('span.label', `Problems · ${issues.length}`)),
      issues.length ? issues.map((i) => h('button.lm-issue', { class: i.severity, on: { click: () => { if (i.node && nodeById(i.node)) { select([i.node]); centerOn(i.node); } } } }, h('span.led', { class: i.severity === 'error' ? 'bad' : 'warn' }), h('span', i.node ? [h('b.mono', i.node), ' ', i.message] : i.message))) : h('p.lm-fh', 'No problems found.'),
      h('div.lm-sec', h('span.label', 'Shortcuts')),
      h('dl.lm-keys', [['Drag jack → jack', 'patch'], ['Double-click cable', 'unplug'], ['Delete', 'remove'], ['⌘Z / ⇧⌘Z', 'undo / redo'], ['⌘S', 'save'], ['⌘↵', 'run'], ['F', 'fit'], ['⌘ scroll', 'zoom'], ['⇧ drag', 'select box']].map(([k, v]) => [h('dt', h('kbd', k)), h('dd', v)])));
  }

  // ---------------------------------------------------------------- lint
  const lintSoon = debounce(runLint, 300);
  let lintSeq = 0;
  async function runLint() {
    const my = ++lintSeq;
    try { const rep = await api.post('/loom/lint', { pipeline: exportPipeline() }); if (my !== lintSeq || !root.isConnected) return; lint = rep; }
    catch { return; }
    for (const n of P.nodes) paintLed(n.id);
    const e = lint.issues.filter((i) => i.severity === 'error').length, w = lint.issues.length - e;
    ui.lintChip.dataset.s = e ? 'bad' : w ? 'warn' : 'ok'; ui.lintChip.querySelector('.led').className = 'led ' + (e ? 'bad' : w ? 'warn' : 'on');
    ui.lintChip.querySelector('.lm-lint-t').textContent = e || w ? `${e ? e + ' error' + (e > 1 ? 's' : '') : ''}${e && w ? ' · ' : ''}${w ? w + ' warning' + (w > 1 ? 's' : '') : ''}` : 'Clean';
    if (!sel.size && !selEdge) renderInspector(); else if (sel.size === 1) { const f = ui.insp.querySelector('.lm-sec .lm-issue'); if (f || lint.issues.some((i) => i.node === [...sel][0])) renderInspectorKeepFocus(); }
  }
  function renderInspectorKeepFocus() { if (ui.insp.contains(document.activeElement) && isField(document.activeElement)) return; renderInspector(); }
  function exportPipeline() {
    const out = { version: P.version || 1, name: P.name || 'untitled', title: P.title || '', description: P.description || '', nodes: [], edges: P.edges.map((e) => ({ from: e.from, to: e.to })) };
    for (const n of P.nodes) {
      const o = { id: n.id, type: n.type, params: n.params || {}, pos: { x: n.pos.x, y: n.pos.y } };
      for (const k of ['label', 'defaults', 'retries', 'timeout', 'cache']) if (n[k] !== undefined) o[k] = n[k];
      out.nodes.push(o);
    }
    return out;
  }

  // ---------------------------------------------------------------- open / save / new / export
  function modal(content) {
    const scrim = h('div.scrim.lm-scrim', { on: { pointerdown: (e) => { if (e.target === scrim) close(); } } });
    const box = h('div.lm-modal', { role: 'dialog', 'aria-modal': 'true' }, content);
    scrim.append(box); document.body.append(scrim);
    const close = () => scrim.remove(); const esc = (e) => { if (e.key === 'Escape') { close(); removeEventListener('keydown', esc, true); } }; addEventListener('keydown', esc, true);
    return { close, box };
  }
  function askText(title, value, ok = 'Save') {
    return new Promise((res) => {
      const inp = h('input.field', { type: 'text', value, 'aria-label': title, spellcheck: 'false' });
      const m = modal(h('form', { on: { submit: (e) => { e.preventDefault(); m.close(); res(inp.value.trim() || null); } } }, h('h3', title), inp, h('div.lm-actions', h('button.btn.sm.ghost', { type: 'button', on: { click: () => { m.close(); res(null); } } }, 'Cancel'), h('button.btn.sm.primary', { type: 'submit' }, ok))));
      setTimeout(() => { inp.focus(); inp.select(); }, 30);
    });
  }
  function confirmBox(title, text, ok = 'Discard') {
    return new Promise((res) => { const m = modal(h('div', h('h3', title), h('p.lm-fh', text), h('div.lm-actions', h('button.btn.sm.ghost', { on: { click: () => { m.close(); res(false); } } }, 'Cancel'), h('button.btn.sm.danger', { on: { click: () => { m.close(); res(true); } } }, ok)))); });
  }
  const isDirty = () => snap() !== SESSION.saved;
  async function save(asCopy) {
    let id = SESSION.id; let name = P.title || P.name;
    if (!id || SESSION.readonly || asCopy) {
      const n = await askText(SESSION.readonly ? 'Save a copy as' : 'Save pipeline as', name && name !== 'untitled' ? name : 'My pipeline'); if (!n) return;
      id = slug(n); P.title = n; P.name = id; ui.name.value = n;
    }
    try {
      await api.request('PUT', `/loom/pipelines/${encodeURIComponent(id)}`, { pipeline: exportPipeline() });
      SESSION.id = id; SESSION.readonly = false; SESSION.source = 'user'; SESSION.saved = snap(); lsSet('nimlab.loom.last', id);
      afterChange(); toast(`Saved “${P.title || id}”`);
    } catch (e) { toast(e.message || 'Could not save', 'bad'); }
  }
  async function load(id, source) {
    try {
      const r = await api.get(`/loom/pipelines/${encodeURIComponent(id)}`);
      setPipeline(r.pipeline, { id: r.id, source: r.source, readonly: r.readonly });
      lsSet('nimlab.loom.last', r.id);
    } catch (e) { toast(e.message || 'Could not open that pipeline', 'bad'); }
  }
  function setPipeline(doc, info) {
    P = doc; P.edges = P.edges || []; layoutMissing(P);
    SESSION.pipeline = P; SESSION.id = info.id; SESSION.source = info.source; SESSION.readonly = !!info.readonly; SESSION.saved = snap();
    SESSION.inputs = {}; hist.stack = [snap()]; hist.i = 0; hist.key = null;
    sel.clear(); selEdge = null; for (const k of Object.keys(RS)) delete RS[k];
    renderAll(); fit(false); renderInspector(); afterChange(); resetRunUi();
  }
  function layoutMissing(doc) {
    const missing = doc.nodes.filter((n) => !n.pos); if (!missing.length) return;
    const depth = {}; const deps = {}; doc.nodes.forEach((n) => (deps[n.id] = []));
    doc.edges.forEach((e) => deps[split(e.to)[0]]?.push(split(e.from)[0]));
    const d = (id, seen = new Set()) => { if (depth[id] != null) return depth[id]; if (seen.has(id)) return 0; seen.add(id); return (depth[id] = Math.max(-1, ...deps[id].map((p) => d(p, seen))) + 1); };
    const rows = {};
    for (const n of doc.nodes) if (!n.pos) { const c = d(n.id); n.pos = { x: 40 + c * (W + 80), y: 40 + (rows[c] = (rows[c] ?? -1) + 1) * 220 }; }
  }
  async function newPipeline() {
    if (isDirty() && !(await confirmBox('Discard changes?', 'The current pipeline has unsaved changes.'))) return;
    const doc = { version: 1, name: 'untitled', title: '', description: '', nodes: [
      { id: 'question', type: 'input', params: { type: 'string', default: 'What is a patch bay?' }, pos: { x: 40, y: 80 } },
      { id: 'answer', type: 'prompt', params: { template: 'Answer in two sentences: {{question}}' }, pos: { x: 360, y: 60 } },
      { id: 'result', type: 'output', params: { name: 'answer' }, pos: { x: 680, y: 100 } }], edges: [{ from: 'question.value', to: 'answer.question' }, { from: 'answer.text', to: 'result.value' }] };
    setPipeline(doc, { id: null, source: 'new', readonly: false }); SESSION.saved = '';
    afterChange();
  }
  async function openMenu(anchor) {
    let items = []; try { items = await api.get('/loom/pipelines'); } catch (e) { return toast(e.message, 'bad'); }
    const m = popMenu(anchor, [
      { label: 'New pipeline', ic: 'plus', run: newPipeline },
      ...['bundled', 'user'].flatMap((src) => { const list = items.filter((i) => i.source === src); return list.length ? [{ head: src === 'bundled' ? 'Bundled examples' : 'Saved' }, ...list.map((i) => ({ label: i.title, sub: `${i.nodes} modules`, active: i.id === SESSION.id, run: async () => { if (isDirty() && !(await confirmBox('Discard changes?', 'The current pipeline has unsaved changes.'))) return; load(i.id, i.source); }, del: src === 'user' ? async () => { if (await confirmBox(`Delete “${i.title}”?`, 'This removes the saved file.', 'Delete')) { try { await api.del(`/loom/pipelines/${encodeURIComponent(i.id)}`); toast('Deleted'); if (SESSION.id === i.id) { SESSION.id = null; afterChange(); } } catch (e) { toast(e.message, 'bad'); } } } : null }))] : []; }),
    ]);
    return m;
  }
  function exportMenu(anchor) {
    const go = async (format) => { try { const r = await api.post('/loom/export', { pipeline: exportPipeline(), format }); await navigator.clipboard?.writeText(r.text); toast(`${format === 'python' ? 'Python script' : 'Mermaid diagram'} copied to the clipboard`); } catch (e) { toast(e.message || 'Export failed', 'bad'); } };
    popMenu(anchor, [{ label: 'Copy as mermaid', run: () => go('mermaid') }, { label: 'Copy as Python script', run: () => go('python') }, { label: 'Save as copy…', run: () => save(true) }, { label: 'Fit to view', run: () => fit(true) }]);
  }
  function popMenu(anchor, entries) {
    document.querySelector('.lm-menu')?.remove();
    const r = anchor.getBoundingClientRect();
    const menu = h('div.lm-menu', { role: 'menu', style: { left: Math.min(r.left, innerWidth - 300) + 'px', top: r.bottom + 6 + 'px' } });
    const close = () => { menu.remove(); removeEventListener('pointerdown', away, true); };
    const away = (e) => { if (!menu.contains(e.target)) close(); }; setTimeout(() => addEventListener('pointerdown', away, true), 0);
    for (const e of entries) {
      if (e.head) { menu.append(h('div.lm-mh.label', e.head)); continue; }
      menu.append(h('div.lm-mi', { class: e.active ? 'active' : '' }, h('button', { role: 'menuitem', on: { click: () => { close(); e.run(); } } }, e.ic && nsvg(I[e.ic]), h('span', e.label), e.sub && h('small.mono', e.sub)), e.del && h('button.lm-mdel', { 'aria-label': 'Delete ' + e.label, on: { click: (ev) => { ev.stopPropagation(); close(); e.del(); } } }, nsvg(I.trash))));
    }
    document.body.append(menu); offs.push(close);
    return menu;
  }

  // ---------------------------------------------------------------- running
  function resetRunUi() {
    for (const [id, v] of NV) { v.el.removeAttribute('data-run'); paintLcd(id); }
    for (const v of EV.values()) v.g.classList.remove('live', 'dead');
    ui.status.hidden = true; ui.run.classList.remove('stop'); ui.run.lastChild.textContent = 'Run'; ui.run.firstChild.replaceWith(nsvg(I.run));
  }
  function setRunning(on) {
    run.status = on ? 'running' : run.status; ui.run.classList.toggle('stop', on);
    ui.run.lastChild.textContent = on ? 'Stop' : 'Run'; ui.run.firstChild.replaceWith(nsvg(on ? I.stop : I.run));
    ui.canvas.classList.toggle('running', on);
  }
  function runInputs() {
    const out = {};
    for (const { n, name, type } of inputNodes()) { const v = SESSION.inputs[name]; if (v !== undefined && v !== '') out[name] = type === 'json' || type === 'list' ? (typeof v === 'string' ? v : JSON.stringify(v)) : v; }
    return out;
  }
  async function startRun() {
    if (run.status === 'running') return;
    const errs = lint.issues.filter((i) => i.severity === 'error');
    if (errs.length) { toast(`Fix ${errs.length} problem${errs.length > 1 ? 's' : ''} first: ${errs[0].message}`, 'warn', 6000); selectNone(); renderInspector(); if (errs[0].node) centerOn(errs[0].node); return; }
    for (const k of Object.keys(RS)) delete RS[k];
    resetRunUi(); run.log = []; ui.log.textContent = ''; clear(ui.outs); run.outputs = null;
    let res; try { res = await api.post('/loom/run', { pipeline: exportPipeline(), inputs: runInputs() }); } catch (e) { return toast(e.message || 'Could not start the run', 'bad', 6000); }
    run.id = res.run_id; run.status = 'running'; setRunning(true);
    ui.status.hidden = false; ui.status.dataset.s = 'run'; ui.status.textContent = 'RUNNING'; toggleTray(true, true);
    run.stop = api.stream(run.id, onEvent, () => { run.stop = null; });
  }
  function stopRun() { if (run.id) api.post(`/runs/${run.id}/cancel`).catch(() => {}); }
  const rs = (id) => (RS[id] ||= { status: 'idle', text: '', outputs: {} });
  const paintSoon = new Set(); let paintRaf = 0;
  function schedulePaint(id) { paintSoon.add(id); if (!paintRaf) paintRaf = requestAnimationFrame(() => { paintRaf = 0; for (const i of paintSoon) paintLcd(i); paintSoon.clear(); }); }
  function logLine(text, cls = '') { run.log.push(text); if (run.log.length > 400) run.log.shift(); const line = h('div', { class: cls }, text); ui.log.append(line); while (ui.log.childElementCount > 400) ui.log.firstChild.remove(); ui.log.scrollTop = ui.log.scrollHeight; }
  function onEvent(ev) {
    const d = ev.data || {}; const id = d.node; const v = id && NV.get(id);
    switch (ev.type) {
      case 'run.plan': for (const n of P.nodes) rs(n.id).status = 'queued'; for (const [i, nv] of NV) nv.el.dataset.run = 'queued'; logLine(`plan: ${d.nodes?.length} modules in ${d.levels?.length} stages`); break;
      case 'node.start': { const s = rs(id); Object.assign(s, { status: 'running', text: '', preview: '', error: null, retry: 0, t0: performance.now() }); if (v) v.el.dataset.run = 'running'; schedulePaint(id); logLine(`▶ ${id}`); break; }
      case 'node.token': { const s = rs(id); s.text = (s.text + d.text).slice(-600); schedulePaint(id); break; }
      case 'node.progress': { rs(id).progress = { done: d.done, total: d.total }; schedulePaint(id); break; }
      case 'node.retry': { const s = rs(id); s.retry = d.attempt; s.text = ''; schedulePaint(id); logLine(`↻ ${id} retry ${d.attempt}: ${d.error}`, 'warn'); break; }
      case 'node.end': {
        const s = rs(id); Object.assign(s, { status: d.cached ? 'cached' : 'done', preview: d.preview, outputs: d.outputs || {}, latency: d.latency_ms, tokens: d.tokens, model: d.model, progress: null });
        if (v) v.el.dataset.run = s.status; schedulePaint(id); logLine(`✓ ${id}  ${d.cached ? 'cached' : d.latency_ms + ' ms'}${d.tokens ? ' · ' + d.tokens + ' tok' : ''}`);
        for (const e of P.edges) { const [a, ap] = split(e.from); if (a !== id) continue; const ev2 = EV.get(eKey(e)); if (!ev2) continue; if ((d.emitted || []).includes(ap)) { ev2.g.classList.add('live'); ev2.g.classList.remove('dead'); pulse(ev2); } else { ev2.g.classList.add('dead'); ev2.g.classList.remove('live'); } }
        if (sel.size === 1 && sel.has(id)) renderInspector();
        break;
      }
      case 'node.skipped': { const s = rs(id); Object.assign(s, { status: 'skipped', reason: d.reason }); if (v) v.el.dataset.run = 'skipped'; schedulePaint(id); logLine(`· ${id} skipped`, 'dim'); for (const e of P.edges) if (split(e.from)[0] === id) EV.get(eKey(e))?.g.classList.add('dead'); break; }
      case 'node.error': { const s = rs(id); Object.assign(s, { status: 'error', error: d.error }); if (v) v.el.dataset.run = 'error'; schedulePaint(id); logLine(`✗ ${id}: ${d.error}`, 'bad'); break; }
      case 'run.done': finishRun(true, ev.data?.result); break;
      case 'run.failed': finishRun(false, ev.data); break;
    }
  }
  function finishRun(ok, data) {
    run.status = ok ? 'done' : 'failed'; setRunning(false);
    if (ok && data) {
      ui.status.dataset.s = 'ok'; ui.status.textContent = `DONE · ${data.elapsed_s}s · ${data.tokens} tok${data.cache_hits ? ' · ' + data.cache_hits + ' cached' : ''}`;
      run.outputs = data.outputs; clear(ui.outs);
      const entries = Object.entries(data.outputs || {});
      if (!entries.length) ui.outs.append(h('p.lm-fh', 'The run produced no outputs (every output sat behind a skipped branch).'));
      for (const [k, val] of entries) ui.outs.append(h('div.lm-out', h('div.lm-outh', h('b.mono', k), h('button.btn.sm.ghost.icon', { 'aria-label': `Copy ${k}`, on: { click: () => navigator.clipboard?.writeText(typeof val === 'string' ? val : JSON.stringify(val, null, 2)).then(() => toast('Copied')) } }, nsvg(I.copy))), h('pre.selectable', typeof val === 'string' ? val : JSON.stringify(val, null, 2))));
      logLine(`done in ${data.elapsed_s}s`);
    } else {
      const cancelled = data?.cancelled; ui.status.dataset.s = cancelled ? 'warn' : 'bad'; ui.status.textContent = cancelled ? 'CANCELLED' : 'FAILED';
      if (!cancelled) toast(data?.error || 'The run failed', 'bad', 7000); logLine(cancelled ? 'cancelled' : `failed: ${data?.error}`, 'bad');
      for (const [i, s] of Object.entries(RS)) if (s.status === 'running' || s.status === 'queued') { s.status = 'idle'; NV.get(i)?.el.removeAttribute('data-run'); schedulePaint(i); }
    }
    ui.canvas.classList.remove('running');
  }
  function pulse(ev) {
    if (reduced() || !ev.wire.getTotalLength) return;
    if (ui.gPulse.childElementCount > 40) return;
    const len = ev.wire.getTotalLength(); if (!len) return;
    const seg = Math.min(60, len * 0.35);
    const p = h('path.lm-pulse', { d: ev.wire.getAttribute('d'), style: { '--wc': ev.g.style.getPropertyValue('--wc') }, 'stroke-dasharray': `${seg} ${len + seg}` });
    ui.gPulse.append(p);
    const a = p.animate([{ strokeDashoffset: seg }, { strokeDashoffset: -len }], { duration: clamp(len * 2.4, 420, 1300), easing: 'cubic-bezier(.4,0,.2,1)' });
    a.onfinish = () => p.remove(); a.oncancel = () => p.remove();
  }
  function toggleTray(open, quiet) { const to = open ?? ui.tray.hidden; ui.tray.hidden = !to; ui.trayBtn.classList.toggle('on', to); }

  // ---------------------------------------------------------------- boot
  const offline = (msg) => { clear(ui.insp); root.classList.add('off'); ui.canvas.replaceChildren(h('div.empty', icon('bolt'), h('h3', 'Loom needs the Studio server'), h('p', msg))); };
  try {
    typeList = await api.get('/loom/node-types'); types = Object.fromEntries(typeList.map((t) => [t.type, t]));
  } catch (e) { offline(e.status === 404 ? 'The Loom routes are not mounted on this server (nimlab.studio.routes_loom).' : String(e.message || e)); return () => offs.forEach((f) => f()); }
  renderPalette();
  if (!P) {
    const want = params.pipeline || lsGet('nimlab.loom.last') || 'research-brief';
    try { const r = await api.get(`/loom/pipelines/${encodeURIComponent(want)}`); setPipeline(r.pipeline, { id: r.id, source: r.source, readonly: r.readonly }); }
    catch { try { const r = await api.get('/loom/pipelines/research-brief'); setPipeline(r.pipeline, { id: r.id, source: r.source, readonly: r.readonly }); } catch (e) { P = { version: 1, name: 'untitled', title: '', description: '', nodes: [], edges: [] }; SESSION.pipeline = P; hist.stack = [snap()]; hist.i = 0; renderAll(); } }
  } else { layoutMissing(P); renderAll(); if (!SESSION.view) fit(false); renderInspector(); }
  if (!hist.stack.length) { hist.stack = [snap()]; hist.i = 0; }
  const ro = new ResizeObserver(() => { if (!SESSION.view) fit(false); }); ro.observe(ui.canvas); offs.push(() => ro.disconnect());
  afterChange();
  runLint();
  return () => {
    run.stop?.(); SESSION.view = view; SESSION.pipeline = P; offs.forEach((f) => f()); cancelAnimationFrame(raf);
    document.querySelector('.lm-menu')?.remove();
  };
}
