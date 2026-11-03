// ⌘K — a patch-bay prompt: every row is a jack you plug into.
import { h, $, clear, esc, ago } from '../util.js';
import { icon } from '../icons.js';
import { describe } from '../descriptions.js';
import { store, go, toggleTheme, emit, APPS } from '../app.js';
import { pickGlobalRole, pickAppModel } from './models.js';

const ROLES = ['chat', 'reasoning', 'coder', 'vision', 'fast', 'embed', 'rerank', 'reward', 'safety'];

let open = null;
export function setPrefill(screen, text) { (store.prefill ||= {})[screen] = text; emit('prefill', { screen, text }); }

function items() {
  const A = (id) => `var(--c-${id})`;
  const list = [
    { t: 'Build…', d: describe('foundry').short, s: 'Foundry', kw: 'build', ic: 'forge', j: A('forge'), run: (q) => { setPrefill('foundry', q); go('foundry'); } },
    { t: 'Ask Atlas…', d: describe('atlas').short, s: 'Atlas', kw: 'ask', ic: 'atlas', j: A('atlas'), run: (q) => { setPrefill('atlas', q); go('atlas'); } },
    { t: 'Convene council…', d: describe('council').short, s: 'Council', kw: 'convene', ic: 'council', j: A('council'), run: (q) => { setPrefill('council', q); go('council'); } },
    { t: 'Route prompt…', d: describe('arena').short, s: 'Arena', kw: 'route', ic: 'route', j: A('arena'), run: (q) => { setPrefill('arena', q); go('arena'); } },
    { t: 'Analyze image…', d: describe('prism').short, s: 'Prism', kw: 'analyze', ic: 'prism', j: A('prism'), run: () => go('prism') },
    { t: 'Run safety scan…', d: describe('sentinel').short, s: 'Sentinel', kw: 'scan', ic: 'sentinel', j: A('sentinel'), run: () => go('sentinel') },
    { t: 'New pipeline…', d: describe('loom').short, s: 'Loom', kw: 'pipeline', ic: 'loom', j: A('loom'), run: () => go('loom') },
    { t: 'Generate dataset…', d: describe('distill').short, s: 'Distill', kw: 'generate', ic: 'distill', j: A('distill'), run: () => go('distill') },
    { t: 'Settings', s: '⌘,', ic: 'settings', j: 'var(--c-core)', run: () => import('../screens/settings.js').then((m) => m.openSettings()) },
    { t: 'Toggle theme', s: 'ink / paper', ic: 'sun', j: 'var(--c-core)', run: toggleTheme },
    ...APPS.map((a) => ({ t: `Go to ${a.name}`, d: describe(a.id)?.short, s: '⌘' + a.key, ic: a.icon, j: `var(--c-${a.app || 'core'})`, run: () => go(a.id) })),
    ...ROLES.map((r) => ({ t: `Switch model for ${r}…`, d: `Global ${r} model: pick one or return to Auto`, s: 'Models', kw: '', ic: 'bolt', j: 'var(--c-core)', run: () => setTimeout(() => pickGlobalRole(r), 0) })),
    ...APPS.filter((a) => a.app).map((a) => ({ t: `Switch ${a.name} model…`, d: `Models used by ${a.name} (this app only)`, s: 'Models', ic: a.icon, j: `var(--c-${a.app})`, run: () => { go(a.id); setTimeout(() => pickAppModel(a.id), 250); } })),
    { t: 'Replace API key…', s: 'Ignition', ic: 'key', j: 'var(--c-core)', run: () => import('../app.js').then((m) => m.maybeIgnite(true)) },
  ];
  for (const p of store.projects.slice(0, 8)) list.push({ t: `Project: ${p.name}`, s: `${p.files ?? '?'} files`, ic: 'folder', j: 'var(--c-forge)', run: () => go('foundry', { project: p.id }), group: 'Projects' });
  for (const r of store.runs.slice(0, 8)) list.push({ t: `${r.title || r.id}`, s: `${r.app} · ${r.status} · ${ago(r.started_at)}`, ic: r.app === 'forge' ? 'forge' : r.app, j: `var(--c-${r.app})`, run: () => go(r.app === 'forge' ? 'foundry' : r.app, { run: r.id }), group: 'Recent runs' });
  return list;
}
function fuzzy(q, text) {
  if (!q) return { score: 1, html: esc(text) };
  const t = text.toLowerCase(); let qi = 0, score = 0, last = -2, out = '';
  for (let i = 0; i < text.length; i++) {
    if (qi < q.length && t[i] === q[qi]) { score += (last === i - 1 ? 3 : 1) + (i === 0 || /\W/.test(text[i - 1]) ? 2 : 0); last = i; qi++; out += `<mark>${esc(text[i])}</mark>`; }
    else out += esc(text[i]);
  }
  return qi === q.length ? { score, html: out } : null;
}

export function openPalette() {
  if (open) return closePalette();
  const prev = document.activeElement;
  const all = items(); let sel = 0, shown = [];
  const input = h('input', { type: 'text', placeholder: 'Plug into…  build, ask, convene, route, analyze, scan, settings', 'aria-label': 'Command palette', autocomplete: 'off', spellcheck: 'false', role: 'combobox', 'aria-expanded': 'true', 'aria-controls': 'pal-list' });
  const listEl = h('div.palette-list#pal-list', { role: 'listbox' });
  const scrim = h('div.scrim', { on: { mousedown: closePalette } });
  const box = h('div.palette', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Command palette' },
    h('div.palette-in', icon('bolt'), input, h('kbd', 'esc')), listEl,
    h('div.palette-foot', h('span', '↑↓ navigate'), h('span', '↵ plug in'), h('span', 'type a verb + text to prefill')));
  document.body.append(scrim, box);

  function paint() {
    let q = input.value.trim().toLowerCase(), arg = '';
    const verb = all.find((a) => a.kw && q.startsWith(a.kw + ' '));
    if (verb) { arg = input.value.trim().slice(verb.kw.length + 1); shown = [{ ...verb, t: `${verb.t.replace('…', '')}: “${arg}”`, arg, html: esc(`${verb.t.replace('…', '')}: “${arg}”`) }]; }
    else shown = all.map((it) => { const f = fuzzy(q, it.t + (it.group ? '' : '')); return f && { ...it, html: f.html, score: f.score + (it.group ? 0 : 1) }; }).filter(Boolean).sort((a, b) => q ? b.score - a.score : 0).slice(0, 40);
    sel = Math.min(sel, Math.max(0, shown.length - 1));
    clear(listEl); let g = null;
    if (!shown.length) listEl.append(h('div.empty', h('p', 'No jacks match. Try “build”, “ask” or “settings”.')));
    shown.forEach((it, i) => {
      if (!q && it.group !== g) { g = it.group; if (g) listEl.append(h('div.pal-group.label', g)); }
      const row = h('button.pal-item', { role: 'option', id: 'pal-' + i, style: { '--j': it.j }, 'aria-selected': i === sel ? 'true' : 'false', on: { click: () => choose(i), mousemove: () => { if (sel !== i) { sel = i; mark(); } } } },
        h('span.jack', icon(it.ic)), h('span.t', { html: it.html }, ), it.d && h('span.d', it.d), h('span.s', it.s));
      listEl.append(row);
    });
    input.setAttribute('aria-activedescendant', 'pal-' + sel);
  }
  function mark() { [...listEl.querySelectorAll('.pal-item')].forEach((r, i) => { r.setAttribute('aria-selected', i === sel ? 'true' : 'false'); if (i === sel) r.scrollIntoView({ block: 'nearest' }); }); input.setAttribute('aria-activedescendant', 'pal-' + sel); }
  function choose(i) { const it = shown[i]; if (!it) return; closePalette(false); it.run(it.arg || ''); }
  input.addEventListener('input', () => { sel = 0; paint(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); sel = (sel + 1) % Math.max(1, shown.length); mark(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); sel = (sel - 1 + shown.length) % Math.max(1, shown.length); mark(); }
    else if (e.key === 'Enter') { e.preventDefault(); choose(sel); }
    else if (e.key === 'Escape') { e.preventDefault(); closePalette(); }
    else if (e.key === 'Tab') e.preventDefault();
  });
  open = { scrim, box, prev }; paint(); input.focus();
}
export function closePalette(restore = true) {
  if (!open) return; const { scrim, box, prev } = open; open = null;
  box.animate([{ opacity: 1 }, { opacity: 0, transform: 'translate(-50%,-8px) scale(.97)' }], { duration: 140 }).onfinish = () => box.remove();
  scrim.remove(); if (restore && prev?.focus) prev.focus();
}
