// Model switching: the per-app "patch bay", the searchable model picker, global per-role rows
// (Settings -> Models) and the helpers the screens use to send overrides with a run.
//
// Precedence (backend): this request > this app's saved override > global role override
// (Settings) > NIMLAB_MODEL_<ROLE> env > nimlab.toml > live auto-pick > curated default.
import { h, clear, shortModel, debounce } from '../util.js';
import { icon } from '../icons.js';
import { store, api, toast, on, emit } from '../app.js';

const ROLE_ORDER = ['chat', 'reasoning', 'coder', 'vision', 'fast', 'embed', 'rerank', 'reward', 'safety'];
const SOURCE_LABEL = { request: 'this run', app: 'app override', studio: 'global override', env: 'env NIMLAB_MODEL_*', toml: 'nimlab.toml', auto: 'auto-substituted', default: 'default' };

// ---------- data ----------
let inflight = null;
export async function loadRoles(force = false) {
  if (store.modelRoles && !force) return store.modelRoles;
  inflight ||= api.get('/models/roles').then((r) => { store.modelRoles = r; emit('modelroles', r); return r; }).finally(() => { inflight = null; });
  return inflight;
}
const adopt = (payload) => { store.modelRoles = payload; emit('modelroles', payload); return payload; };
export const backendApp = (id) => (id === 'foundry' ? 'forge' : id);

/** {role: model} saved for this app (sent with a run so the request carries it explicitly). */
export async function modelOverrides(app) {
  try {
    const roles = await loadRoles();
    const out = {};
    for (const e of roles.apps?.[backendApp(app)]?.roles || []) if (e.override) out[e.role] = e.override;
    return out;
  } catch { return {}; }
}

// ---------- picker ----------
let openPick = null;
export function closePicker() { if (openPick) { const { el, off } = openPick; openPick = null; off(); el.remove(); } }

/**
 * Searchable dropdown of live models for `role`. `onPick(id|null)`: null = Auto / reset.
 * `anchor` positions it under an element; without one it is centered.
 */
export function openPicker({ role, title, current, autoLabel = 'Auto', anchor, onPick, resetLabel }) {
  closePicker();
  let all = false, items = [], note = '', sel = 0, needle = '';
  const search = h('input.field.sm', { type: 'search', placeholder: 'Search models…', 'aria-label': 'Search models', autocomplete: 'off', spellcheck: 'false', on: { input: debounce(() => { needle = search.value.trim().toLowerCase(); paint(); }, 60) } });
  const allBox = h('input', { type: 'checkbox', id: 'mp-all', on: { change: () => { all = allBox.checked; load(); } } });
  const list = h('div.mp-list', { role: 'listbox', 'aria-label': `Models for ${role}` });
  const noteEl = h('p.mp-note', { hidden: true });
  const el = h('div.mpick', { role: 'dialog', 'aria-label': title || `Choose a ${role} model`, 'data-role': role },
    h('div.mp-h', h('span.label', title || `${role} model`), h('button.btn.sm.icon.ghost', { 'aria-label': 'Close', on: { click: closePicker } }, icon('x'))),
    h('div.mp-tools', search, h('label.mp-all', allBox, 'All models')), list, noteEl);
  document.body.append(el);
  const place = () => {
    if (!anchor || !anchor.isConnected) { el.classList.add('centered'); return; }
    const r = anchor.getBoundingClientRect(), w = Math.min(380, innerWidth - 24);
    el.style.width = w + 'px';
    el.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left)) + 'px';
    const below = innerHeight - r.bottom, top = below > 300 ? r.bottom + 6 : Math.max(12, r.top - Math.min(420, r.top - 12) - 6);
    el.style.top = top + 'px'; el.style.maxHeight = Math.max(240, below > 300 ? below - 18 : r.top - 18) + 'px';
  };
  place();
  const down = (e) => { if (!el.contains(e.target) && !(anchor && anchor.contains(e.target))) closePicker(); };
  const key = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); closePicker(); anchor?.focus?.(); }
    else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, Math.min(rows().length - 1, sel + (e.key === 'ArrowDown' ? 1 : -1))); mark(); }
    else if (e.key === 'Enter' && document.activeElement !== allBox) { e.preventDefault(); rows()[sel]?.click(); }
  };
  document.addEventListener('mousedown', down, true); document.addEventListener('keydown', key, true);
  const off = () => { document.removeEventListener('mousedown', down, true); document.removeEventListener('keydown', key, true); };
  openPick = { el, off };
  const rows = () => [...list.querySelectorAll('.mp-row')];
  const mark = () => rows().forEach((r, i) => { r.classList.toggle('sel', i === sel); r.setAttribute('aria-selected', String(i === sel)); if (i === sel) r.scrollIntoView({ block: 'nearest' }); });

  function paint() {
    clear(list);
    const pick = (id) => async () => { closePicker(); try { await onPick(id); } catch (e) { toast(e.message || 'Could not change the model', 'bad'); } };
    list.append(h('button.mp-row.auto', { type: 'button', role: 'option', 'aria-selected': 'false', on: { click: pick(null) } },
      h('span.mp-name', resetLabel || autoLabel), h('span.mp-flags', current == null ? h('span.mp-tag.cur', 'current') : null)));
    const f = items.filter((m) => !needle || m.id.toLowerCase().includes(needle));
    for (const m of f) {
      list.append(h('button.mp-row', { type: 'button', role: 'option', 'aria-selected': 'false', 'data-id': m.id, title: m.id, on: { click: pick(m.id) } },
        h('span.mp-name.mono', m.id),
        h('span.mp-flags', m.id === current && h('span.mp-tag.cur', 'current'), m.preferred && h('span.mp-tag', 'preferred'), m.deprecated && h('span.mp-tag.warn', 'deprecated'), m.unavailable && h('span.mp-tag.bad', 'unavailable'),
          h('span.led', { class: m.live ? 'on' : '', title: m.live ? 'listed by the live catalog' : 'curated, not verified live' }))));
    }
    if (!f.length) list.append(h('p.mp-empty', items.length ? 'No models match.' : 'No models for this role yet.'));
    noteEl.hidden = !note; noteEl.textContent = note;
    sel = current ? Math.max(0, rows().findIndex((r) => r.dataset.id === current)) : 0; mark();
  }
  async function load() {
    try { const r = await api.get(`/models/options?role=${encodeURIComponent(role)}&all=${all ? 1 : 0}`); items = r.models || []; note = r.note || ''; }
    catch (e) { items = []; note = e.message || 'Could not load models'; }
    paint(); search.focus();
  }
  load();
  return el;
}

// ---------- the per-app patch bay ----------
const lastRun = {}; // backend app -> { picked, substitutions }
export function recordRun(app, info) { if (info) { lastRun[app] = info; emit('runmodels', app); } }
addEventListener('nimlab:run-done', (e) => {
  const { runId, info } = e.detail || {};
  const run = (store.runs || []).find((r) => r.id === runId);
  recordRun(backendApp(run?.app || store.route), info);
});

export function patchBay(appId) {
  const app = backendApp(appId);
  const root = h('div.patchbay', { role: 'group', 'aria-label': 'Models used by this app', 'data-app': app });
  const last = h('div.pb-last', { 'aria-live': 'polite', hidden: true });
  function paint() {
    const info = store.modelRoles?.apps?.[app]; clear(root);
    root.append(h('span.label.pb-label', 'Patch'));
    if (!info) { root.append(h('span.pb-wait', 'loading models…')); return; }
    for (const slot of info.roles) {
      const src = slot.override ? 'app' : slot.source;
      const chip = h('button.pb-chip', { type: 'button', 'data-role': slot.role, 'data-source': src, title: `${slot.label}: ${slot.model} (${SOURCE_LABEL[src] || src}). Click to change.`,
        'aria-label': `${slot.label} model: ${shortModel(slot.model)}. Change`, 'aria-haspopup': 'dialog',
        on: { click: (e) => chooser(slot, e.currentTarget) } },
      h('span.pb-role', slot.label), h('span.pb-model.mono', shortModel(slot.model)), h('i.pb-dot', { class: src }));
      root.append(chip);
    }
    if (info.overridden) root.append(h('button.pb-reset', { type: 'button', title: 'Reset every role of this app to Auto', on: { click: reset } }, icon('refresh'), 'Reset to Auto'));
    paintLast();
  }
  function paintLast() {
    const run = lastRun[app]; last.hidden = !run;
    if (!run) return;
    const subs = run.substitutions || [];
    const picked = [...new Map((run.picked || []).map((p) => [p.model, p])).values()];
    clear(last).append(h('span.label', 'Last run'), h('span.pb-via', 'via ', ...picked.flatMap((p, i) => [i ? ', ' : '', h('code.mono', { title: `${p.role} (${SOURCE_LABEL[p.source] || p.source})` }, p.model)])),
      ...subs.map((s) => h('span.pb-sub', { role: 'status' }, `auto-substituted ${shortModel(s.requested)} → ${shortModel(s.used)}`)));
  }
  async function chooser(slot, anchor) {
    openPicker({ role: slot.role, title: `${app} · ${slot.label}`, current: slot.override || null, anchor, autoLabel: `Auto (follow the global ${slot.role} setting)`, resetLabel: `Reset to Auto (${shortModel(slot.model)})`,
      onPick: async (id) => { adopt(await api.request('PUT', `/models/app/${app}`, { overrides: { [slot.role]: id } })); toast(id ? `${app} ${slot.label}: ${shortModel(id)}` : `${app} ${slot.label}: Auto`); } });
  }
  async function reset() { try { adopt(await api.del(`/models/app/${app}`)); toast(`${app}: models reset to Auto`); } catch (e) { toast(e.message, 'bad'); } }
  const offs = [on('modelroles', paint), on('runmodels', (a) => { if (a === app) paintLast(); })];
  const wrap = h('div.pb-wrap', root, last);
  wrap._dispose = () => offs.forEach((f) => f());
  loadRoles().then(paint).catch(() => paint());
  paint();
  return wrap;
}

// ---------- palette helpers ----------
export async function pickGlobalRole(role, anchor) {
  const roles = await loadRoles(true);
  const r = roles.roles.find((x) => x.role === role);
  openPicker({ role, title: `Global ${role} model`, current: r?.override || null, anchor, resetLabel: `Auto (${shortModel(r?.default || '')})`,
    onPick: async (id) => { adopt(await api.request('PUT', '/models/overrides', { overrides: { [role]: id } })); toast(id ? `${role}: ${shortModel(id)}` : `${role}: Auto`); } });
}
export async function pickAppModel(appId) {
  const app = backendApp(appId);
  const roles = await loadRoles(true);
  const slots = roles.apps?.[app]?.roles || [];
  if (slots.length === 1) return pickSlot(app, slots[0]);
  closePicker();
  const menu = h('div.mpick.centered', { role: 'dialog', 'aria-label': `Switch ${app} model` },
    h('div.mp-h', h('span.label', `Switch ${app} model`), h('button.btn.sm.icon.ghost', { 'aria-label': 'Close', on: { click: closePicker } }, icon('x'))),
    h('div.mp-list', slots.map((s) => h('button.mp-row', { type: 'button', on: { click: () => { closePicker(); pickSlot(app, s); } } }, h('span.mp-name', s.label), h('span.mp-flags', h('code.mono', shortModel(s.model)))))));
  document.body.append(menu);
  const off = () => { document.removeEventListener('keydown', k, true); };
  const k = (e) => { if (e.key === 'Escape') { e.stopPropagation(); closePicker(); } };
  document.addEventListener('keydown', k, true); openPick = { el: menu, off };
  menu.querySelector('.mp-row')?.focus();
}
function pickSlot(app, slot) {
  openPicker({ role: slot.role, title: `${app} · ${slot.label}`, current: slot.override || null, resetLabel: `Reset to Auto (${shortModel(slot.model)})`,
    onPick: async (id) => { adopt(await api.request('PUT', `/models/app/${app}`, { overrides: { [slot.role]: id } })); toast(id ? `${app} ${slot.label}: ${shortModel(id)}` : `${app} ${slot.label}: Auto`); } });
}

// ---------- Settings -> Models: one row per role ----------
export function rolePickers() {
  const box = h('div.rolepick'); const head = h('div.rp-head'); const rows = h('div.rp-rows');
  box.append(head, rows);
  function paint() {
    const p = store.modelRoles; clear(rows); clear(head);
    if (!p) { rows.append(h('p.fine', 'Loading…')); return; }
    head.append(h('span.fine', h('span.led', { class: p.live ? 'on' : 'warn' }),
      p.live ? `Live catalog: ${p.catalog.count} models (${p.catalog.source}, ${Math.round(p.catalog.age_s / 60)} min old)` : (p.note || 'Showing the curated catalog.')),
    h('button.btn.sm', { type: 'button', title: 'Refetch the live model list', on: { click: refreshLive } }, icon('refresh'), 'Refresh'));
    const order = [...p.roles].sort((a, b) => ROLE_ORDER.indexOf(a.role) - ROLE_ORDER.indexOf(b.role));
    for (const r of order) {
      const pinned = r.pinned_by;
      const btn = h('button.rp-btn', { type: 'button', 'data-role': r.role, 'aria-haspopup': 'dialog', 'aria-label': `${r.role} model: ${r.override ? shortModel(r.override) : 'Auto'}`,
        title: pinned ? `Also set by ${pinned === 'env' ? 'the NIMLAB_MODEL_' + r.role.toUpperCase() + ' environment variable' : 'nimlab.toml'}; that takes effect when no override is saved here.` : r.model,
        on: { click: (e) => pickGlobalRole(r.role, e.currentTarget) } },
      h('span.mono.rp-val', r.override ? r.override : 'Auto'), h('span.rp-auto', r.override ? '' : `→ ${shortModel(r.model)}`), icon('chevron'));
      rows.append(h('div.rp-row', { 'data-role': r.role }, h('span.label.rp-role', r.role), btn,
        h('span.rp-src', { class: r.source === 'auto' ? 'warn' : '' }, SOURCE_LABEL[r.source] || r.source),
        r.override && h('button.btn.sm.ghost', { type: 'button', 'aria-label': `Reset ${r.role} to Auto`, on: { click: async () => { adopt(await api.request('PUT', '/models/overrides', { overrides: { [r.role]: null } })); } } }, 'Auto')));
    }
  }
  async function refreshLive() { try { adopt(await api.post('/models/refresh')); toast('Model list refreshed'); } catch (e) { toast(e.message, 'bad'); } }
  const off = on('modelroles', paint);
  box._dispose = off;
  loadRoles(true).then(paint).catch(() => paint()); paint();
  return box;
}
