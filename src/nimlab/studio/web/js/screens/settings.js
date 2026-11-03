// Settings sheet (⌘,): key management, engine, appearance, model catalog, about.
import { h, clear, shortModel, fmtNum, debounce } from '../util.js';
import { icon } from '../icons.js';
import { DESCRIPTIONS } from '../descriptions.js';
import { store, api, toast, refresh, setTheme, effectiveTheme, on, lsGet, maybeIgnite } from '../app.js';

let open = null;
export function mount() {}
export function openSettings() {
  if (open) return closeSettings();
  const prev = document.activeElement; const offs = [];
  const scrim = h('div.scrim', { on: { mousedown: closeSettings } });
  const body = h('div.set-body');
  const sheet = h('div.sheet.settings', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Settings' },
    h('header.set-h', h('div', h('div.label', 'Settings'), h('h2', 'Engine room')), h('button.btn.sm.icon.ghost', { 'aria-label': 'Close settings', on: { click: closeSettings } }, icon('x'))), body);
  document.body.append(scrim, sheet);
  const esc = (e) => { if (e.key === 'Escape') { e.stopPropagation(); closeSettings(); } }; document.addEventListener('keydown', esc, true);
  open = { scrim, sheet, esc, prev, offs };

  const appsBox = h('div.set-sec'), keyBox = h('div.set-sec'), engine = h('div.set-sec'), look = h('div.set-sec'), models = h('div.set-sec'), about = h('div.set-sec');
  body.append(keyBox, engine, look, appsBox, models, about);
  appsBox.append(h('div.label', 'Apps'), h('div.set-apps', Object.entries(DESCRIPTIONS).map(([id, d]) => h('div.set-app', { class: 'app-' + (id === 'foundry' ? 'forge' : id) }, icon(id === 'foundry' ? 'forge' : id), h('b', d.name), h('span', d.line)))));

  // ---- key ----
  const probe = h('div.probe.mono', { 'aria-live': 'polite' });
  function paintKey() {
    const k = store.status?.key || {}; clear(keyBox);
    const rep = h('form.rep', { hidden: true, on: { submit: async (e) => { e.preventDefault(); const v = inp.value.trim(); if (!v) return; save.disabled = true; try { const r = await api.post('/key', { api_key: v }); if (!r.ok) throw new Error(r.error || 'rejected'); toast(r.verified === false ? 'Saved, but couldn’t reach NVIDIA (unverified)' : `Key replaced · ${r.model_count ?? '?'} models online`, r.verified === false ? 'warn' : 'ok'); await refresh(['status', 'models']); paintKey(); paintModels(); } catch (er) { toast(er.message, 'bad'); save.disabled = false; } } } });
    const inp = h('input.field.mono', { type: 'password', placeholder: 'nvapi-…', 'aria-label': 'New API key', autocomplete: 'off', spellcheck: 'false' }); const save = h('button.btn.primary.sm', { type: 'submit' }, 'Save to Keychain');
    rep.append(inp, save, h('button.btn.sm.ghost', { type: 'button', on: { click: () => (rep.hidden = true) } }, 'Cancel'));
    const confirm = h('div.confirm', { hidden: true }, h('span', 'Remove the key from the Keychain? The Studio drops to simulation.'), h('button.btn.sm.danger', { on: { click: async () => { try { await api.del('/key'); toast('Key removed. Simulation mode is on.', 'warn'); await refresh(['status']); paintKey(); } catch (e) { toast(e.message, 'bad'); } } } }, 'Remove'), h('button.btn.sm.ghost', { on: { click: () => (confirm.hidden = true) } }, 'Keep'));
    keyBox.append(h('div.label', 'NVIDIA API key'),
      h('div.keyrow', h('span.led', { class: k.set ? 'on' : 'warn' }), h('code.mono', k.set ? k.masked || 'nvapi-••••' : 'No key set'), k.set && h('span.chip.mono', { style: { pointerEvents: 'none' } }, k.source || 'keychain'),
        h('span.grow'), h('button.btn.sm', { on: { click: test } }, icon('wifi'), 'Test'), h('button.btn.sm', { on: { click: () => { rep.hidden = !rep.hidden; if (!rep.hidden) inp.focus(); } } }, icon('refresh'), k.set ? 'Replace' : 'Add key'), k.set && h('button.btn.sm.danger', { on: { click: () => (confirm.hidden = !confirm.hidden) } }, icon('trash'), 'Remove')),
      probe, rep, confirm, h('p.fine', icon('key'), k.source === 'file' ? 'Stored in a private file in your Studio folder (the Keychain is used on macOS). Get a free key at ' : 'Stored in the macOS Keychain. Get a free key at ', h('a', { href: 'https://build.nvidia.com', target: '_blank', rel: 'noopener' }, 'build.nvidia.com'), '.'));
  }
  async function test() { probe.replaceChildren(h('span.led.busy'), ' probing…'); try { const r = await api.post('/key/test', {}); const ok = r.ok !== false && r.verified !== false; probe.replaceChildren(h('span.led', { class: ok ? 'on' : 'bad' }), ok ? ` signal ok · ${r.model_count ?? '?'} models${r.latency_ms ? ` · ${Math.round(r.latency_ms)} ms` : ''}` : ` ${r.error || 'couldn’t reach NVIDIA'}`); } catch (e) { probe.replaceChildren(h('span.led.bad'), ' ' + e.message); } }

  // ---- engine ----
  function paintEngine() {
    const s = store.status || {}; const live = s.mode === 'live'; clear(engine);
    engine.append(h('div.label', 'Engine'), h('div.kv', h('span', 'Mode'), h('span.mode-pill', { class: live ? 'live' : 'sim' }, h('span.led', { class: live ? 'on' : 'warn' }), live ? 'LIVE' : 'SIMULATION'), h('small', live ? 'Real NVIDIA NIM calls.' : 'Deterministic mock models; retrieval, grading, Elo and sandbox are still real.')),
      h('div.kv', h('span', 'Base URL'), h('code.mono', s.base_url || '—'), h('button.btn.sm.ghost.icon', { 'aria-label': 'Copy base URL', on: { click: () => { navigator.clipboard?.writeText(s.base_url || ''); toast('Copied'); } } }, icon('copy'))),
      h('div.kv', h('span', 'Network'), h('span.mono', { class: { ok: 'okc', blocked: 'badc' }[s.network] }, ({ ok: 'reachable', blocked: 'blocked from here', unknown: 'not probed yet' })[s.network] || s.network || '—')));
  }
  // ---- look ----
  function paintLook() {
    const cur = lsGet('nimlab.theme') || 'auto'; clear(look);
    look.append(h('div.label', 'Appearance'), h('div.seg', { role: 'radiogroup', 'aria-label': 'Theme' }, [['auto', 'Auto'], ['dark', 'Ink'], ['light', 'Paper']].map(([v, n]) => h('button', { role: 'radio', 'aria-checked': String(cur === v), on: { click: () => { setTheme(v === 'auto' ? null : v); paintLook(); } } }, n))), h('p.fine', 'Ink is the dark bench; Paper is the daylight print. Auto follows macOS.'));
  }
  // ---- models ----
  let role = 'all', needle = '';
  const list = h('div.mlist'), chips = h('div.chips');
  function paintModels() {
    const all = store.models; clear(chips); clear(list);
    const roles = ['all', ...new Set(all.flatMap((m) => m.roles || []))];
    roles.forEach((r) => chips.append(h('button.chip.mono', { 'aria-pressed': String(role === r), on: { click: () => { role = r; paintModels(); } } }, r)));
    const f = all.filter((m) => (role === 'all' || (m.roles || []).includes(role)) && (!needle || m.id.toLowerCase().includes(needle)));
    if (!f.length) list.append(h('p.fine', all.length ? 'No models match.' : 'The catalog is empty. Connect a key or try again.'));
    f.forEach((m) => list.append(h('div.mrow', h('code.mono', m.id), h('span.rl', (m.roles || []).join(' · ')), h('span.tier', { title: `cost tier ${m.cost_tier}` }, Array.from({ length: 5 }, (_, i) => h('i', { class: i < (m.cost_tier || 0) ? 'on' : '' }))), h('span.num.dim', m.context_window ? fmtNum(m.context_window) + ' ctx' : ''))));
  }
  const search = h('input.field.sm', { type: 'search', placeholder: 'Filter models…', 'aria-label': 'Filter models', on: { input: debounce(() => { needle = search.value.trim().toLowerCase(); paintModels(); }, 80) } });
  models.append(h('div.label', 'Model catalog'), h('div.mhead', search, chips), list);
  about.append(h('div.label', 'About'), h('div.kv', h('span', 'NIM Lab Studio'), h('code.mono', 'v' + (store.status?.version || '?'))),
    h('div.keys', [['⌘K', 'Command palette'], ['⌘1–9', 'Switch module'], ['⌘,', 'Settings'], ['⌘↵', 'Run / send']].map(([k, t]) => h('span', h('kbd', k), t))),
    h('p.fine', 'Eight apps on NVIDIA NIM: Council, Atlas, Forge, Arena, Prism, Sentinel, Loom, Distill. Everything here runs on your machine.'));
  paintKey(); paintEngine(); paintLook();
  if (!store.models.length) refresh(['models']).then(paintModels); paintModels();
  offs.push(on('status', () => { paintEngine(); }));
  sheet.querySelector('button')?.focus();
}
export function closeSettings() {
  if (!open) return; const { scrim, sheet, esc, prev, offs } = open; open = null; offs.forEach((f) => f()); document.removeEventListener('keydown', esc, true);
  sheet.animate([{ opacity: 1 }, { opacity: 0, transform: 'translate(-50%,-48%) scale(.96)' }], { duration: 160 }).onfinish = () => sheet.remove(); scrim.remove(); prev?.focus?.();
}
