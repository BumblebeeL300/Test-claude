// Router, state store and shell. Screens are lazy ES modules exporting mount(el) -> unmount().
import * as api from './api.js';
import { h, $, clear, ago } from './util.js';
import { icon } from './icons.js';

// ---------- store ----------
export const store = { status: null, runs: [], projects: [], models: [], usage: null, offline: false, route: 'switchboard', params: {} };
const subs = {};
export const on = (ev, fn) => ((subs[ev] ||= new Set()).add(fn), () => subs[ev].delete(fn));
export const emit = (ev, d) => (subs[ev] || []).forEach((f) => f(d));
export { api };

export const APPS = [
  { id: 'switchboard', name: 'Switchboard', key: '1', cls: 'app-core', icon: 'switchboard', hint: 'Every module at a glance' },
  { id: 'foundry', name: 'Foundry', key: '2', cls: 'app-forge', icon: 'forge', app: 'forge', hint: 'Prompt it. Watch it get built.' },
  { id: 'council', name: 'Council', key: '3', cls: 'app-council', icon: 'council', app: 'council', hint: 'Many models, one verdict' },
  { id: 'atlas', name: 'Atlas', key: '4', cls: 'app-atlas', icon: 'atlas', app: 'atlas', hint: 'Ask your documents' },
  { id: 'arena', name: 'Arena', key: '5', cls: 'app-arena', icon: 'arena', app: 'arena', hint: 'Rank and route models' },
  { id: 'prism', name: 'Prism', key: '6', cls: 'app-prism', icon: 'prism', app: 'prism', hint: 'Put an image on the light table' },
  { id: 'sentinel', name: 'Sentinel', key: '7', cls: 'app-sentinel', icon: 'sentinel', app: 'sentinel', hint: 'Blast-test model safety' },
  { id: 'loom', name: 'Loom', key: '8', cls: 'app-loom', icon: 'loom', app: 'loom', hint: 'Patch models into pipelines' },
  { id: 'distill', name: 'Distill', key: '9', cls: 'app-distill', icon: 'distill', app: 'distill', hint: 'Synthetic datasets, refined' },
];
export const appMeta = (id) => APPS.find((a) => a.id === id || a.app === id);

// ---------- toasts ----------
export function toast(msg, kind = 'ok', ms = 3800) {
  const box = $('#toasts');
  const t = h('div.toast', { role: 'status' }, h('span.led', { class: kind === 'ok' ? 'on' : kind === 'bad' ? 'bad' : 'warn' }), h('span', msg));
  box.append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 260); }, ms);
}

// ---------- theme ----------
const lsGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} };
export { lsGet, lsSet };
export const effectiveTheme = () => document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
export function setTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; lsSet('nimlab.theme', t); emit('theme', effectiveTheme()); updateThemeBtn(); }
export const toggleTheme = () => setTheme(effectiveTheme() === 'light' ? 'dark' : 'light');
const initTheme = () => { const t = lsGet('nimlab.theme'); if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; };
function updateThemeBtn() { const b = $('#theme-btn'); if (b) { clear(b).append(icon(effectiveTheme() === 'light' ? 'moon' : 'sun')); b.title = 'Toggle theme'; } }

// ---------- data ----------
export async function refresh(which = ['status', 'runs', 'projects']) {
  const jobs = {
    status: () => api.get('/status').then((s) => { store.status = s; store.offline = false; emit('status', s); }),
    runs: () => api.get('/runs').then((r) => { store.runs = (Array.isArray(r) ? r : r.runs || []).sort((a, b) => (b.started_at || 0) - (a.started_at || 0)); emit('runs', store.runs); }),
    projects: () => api.get('/projects').then((p) => { store.projects = Array.isArray(p) ? p : p.projects || []; emit('projects', store.projects); }),
    models: () => api.get('/models').then((m) => { store.models = Array.isArray(m) ? m : m.models || []; emit('models', store.models); }),
    usage: () => api.get('/usage').then((u) => { store.usage = u; emit('usage', u); }),
  };
  const res = await Promise.allSettled(which.map((k) => jobs[k]()));
  if (which.includes('status') && res[which.indexOf('status')].status === 'rejected') { if (!store.offline) { store.offline = true; emit('offline', true); } return false; }
  if (store.offline) { store.offline = false; emit('offline', false); }
  return true;
}
export const isSim = () => store.status && (store.status.mode === 'mock' || !store.status.key?.set);
export const appStatus = (id) => (store.status?.apps || []).find((a) => a.id === id) || {};

// ---------- routing ----------
const screens = {};
let current = null, unmount = null, seq = 0;
export function go(route, params) { const q = params ? '?' + new URLSearchParams(params) : ''; location.hash = '#/' + route + q; }
export function parseHash() { const [r, q] = location.hash.replace(/^#\/?/, '').split('?'); return { route: APPS.find((a) => a.id === r) ? r : 'switchboard', params: Object.fromEntries(new URLSearchParams(q || '')) }; }

async function render() {
  const { route, params } = parseHash(); const my = ++seq;
  store.route = route; store.params = params;
  const meta = appMeta(route);
  $$rail().forEach((b) => b.dataset.id === route ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current'));
  $('#app').className = meta.cls; document.title = `${meta.name} — NIM Lab`;
  $('#crumb').replaceChildren(h('i'), meta.name);
  if (unmount) { try { unmount(); } catch (e) { console.error(e); } unmount = null; }
  const main = $('#main'); clear(main);
  if (store.offline) return showOffline(main);
  const el = h('div.screen.enter', { id: 'screen-' + route, 'data-screen': route, tabindex: '-1' });
  main.append(el);
  try {
    screens[route] ||= await import(`./screens/${route}.js`);
    if (my !== seq) return;
    unmount = (await screens[route].mount(el, params)) || null;
  } catch (e) { console.error(e); el.append(h('div.empty', icon('bolt'), h('h3', 'This module tripped a breaker'), h('p', String(e.message || e)))); }
  current = route;
}
const $$rail = () => [...document.querySelectorAll('.rail-btn[data-id]')];

function showOffline(main) {
  main.append(h('div.screen.enter', h('div.empty', { style: { minHeight: '100%', alignContent: 'center', '--accent': 'var(--bad)' } },
    h('svg', { viewBox: '0 0 120 40', width: 160, html: '<path d="M0 20h38l6-14 8 28 6-14h62" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 5" opacity=".7"/>' }),
    h('h3', 'Engine offline'), h('p', 'The Studio server isn’t answering. Check that `nimlab studio` is still running, then retry.'),
    h('div', { style: { display: 'flex', gap: '8px', marginTop: '8px' } },
      h('button.btn.primary', { on: { click: async (e) => { e.currentTarget.disabled = true; await refresh(); render(); } } }, icon('refresh'), 'Retry'),
      h('a.btn', { href: '?demo=1' }, 'Open demo data')))));
}

// ---------- shell ----------
function shell() {
  const bar = h('header.titlebar.pywebview-drag-region', { role: 'banner' },
    h('div.brand', h('svg', { viewBox: '0 0 32 32', html: '<circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" stroke-width="2" opacity=".35"/><path d="M3 16h7l3-8 4 16 3-8h9" fill="none" stroke="var(--accent)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>' }), 'NIM LAB'),
    h('div.crumb#crumb'), h('div.grow'),
    h('button.btn.sm.ghost.no-drag#palette-btn', { 'aria-label': 'Open command palette', on: { click: () => import('./components/palette.js').then((m) => m.openPalette()) } }, icon('search'), h('span.dim', 'Jump to…'), h('kbd', '⌘K')),
    h('button.badge-sim.no-drag#sim-badge', { hidden: true, title: 'Simulation mode: no key, deterministic mock models. Click to add a key.', on: { click: () => import('./screens/settings.js').then((m) => m.openSettings()) } }, h('span.led.warn'), 'SIMULATION'),
    h('div.sse#sse', { 'data-s': 'idle', title: 'Event stream' }, h('span.led'), h('span', 'IDLE')),
    h('button.btn.sm.icon.ghost#theme-btn', { 'aria-label': 'Toggle theme', on: { click: toggleTheme } }),
    h('button.btn.sm.icon.ghost', { 'aria-label': 'Settings', title: 'Settings ⌘,', on: { click: () => import('./screens/settings.js').then((m) => m.openSettings()) } }, icon('settings')));
  const rail = h('nav.rail', { 'aria-label': 'Modules' },
    APPS.map((a) => h('button.rail-btn', { class: a.cls, 'data-id': a.id, 'aria-label': `${a.name} (⌘${a.key})`, on: { click: () => go(a.id) } }, icon(a.icon), h('span.busy-dot', { hidden: true }), h('span.tip', a.name, h('kbd', '⌘' + a.key)))),
    h('div.sp'),
    h('button.rail-btn.app-core', { 'aria-label': 'Settings (⌘,)', on: { click: () => import('./screens/settings.js').then((m) => m.openSettings()) } }, icon('key'), h('span.tip', 'Key & settings', h('kbd', '⌘,'))));
  $('#app').append(bar, rail, h('main.main#main'), h('div#toasts', { 'aria-live': 'polite' }));
  updateThemeBtn();
  on('status', paintStatus); on('offline', paintStatus);
  api.onSse((s) => { const el = $('#sse'); el.dataset.s = s; el.lastChild.textContent = s === 'live' ? 'LIVE' : s === 'reconnecting' ? 'RECONNECTING' : 'IDLE'; });
}
function paintStatus() {
  const sim = $('#sim-badge');
  if (sim) {
    const st = store.status; sim.hidden = !st;
    if (st) {
      const simulated = isSim(); const masked = st.key?.masked || 'nvapi-••••';
      sim.classList.toggle('live', !simulated);
      sim.title = simulated ? 'Simulation mode: no key or --mock, deterministic models. Click to add a key.' : `Live: real NVIDIA NIM calls with ${masked}. Click for settings.`;
      sim.replaceChildren(h('span.led', { class: simulated ? 'warn' : 'on' }), simulated ? 'SIMULATION' : 'LIVE', ...(simulated ? [] : [h('span.badge-key.mono', masked)]));
      sim.dataset.mode = simulated ? 'sim' : 'live';
    }
  }
  const apps = store.status?.apps || [];
  $$rail().forEach((b) => { const a = appMeta(b.dataset.id); const busy = a?.app && apps.find((x) => x.id === a.app)?.status === 'busy'; const d = b.querySelector('.busy-dot'); if (d) d.hidden = !busy; });
}

function keys() {
  addEventListener('keydown', (e) => {
    const mod = e.metaKey || e.ctrlKey; if (!mod || e.altKey) return;
    if (e.key === 'k') { e.preventDefault(); import('./components/palette.js').then((m) => m.openPalette()); }
    else if (e.key === ',') { e.preventDefault(); import('./screens/settings.js').then((m) => m.openSettings()); }
    else if (/^[1-9]$/.test(e.key)) { e.preventDefault(); go(APPS[+e.key - 1].id); }
  });
  addEventListener('hashchange', render);
}

export async function maybeIgnite(force) {
  const k = store.status?.key; if (!force && (k?.set || lsGet('nimlab.sim') === '1')) return;
  const m = await import('./screens/ignition.js'); await m.showIgnition();
}

async function boot() {
  initTheme(); shell();
  if (api.DEMO) document.body.dataset.demo = '1';
  await refresh(['status', 'runs', 'projects', 'models', 'usage']);
  paintStatus();
  setInterval(() => refresh(['status', 'runs']).then(paintStatus), 5000);
  keys();
  if (!location.hash) location.hash = '#/switchboard'; else render();
  if (!store.offline) maybeIgnite();
  on('offline', () => render());
}
boot();
export { render as rerender, ago };
