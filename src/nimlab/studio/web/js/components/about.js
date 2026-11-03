// "Spec plate": a compact silkscreened label at the top of each app, saying what the app does.
import { h } from '../util.js';
import { icon } from '../icons.js';
import { describe } from '../descriptions.js';

const KEY = (id) => `nimlab.plate.${id}`;
const read = (id) => { try { return localStorage.getItem(KEY(id)) === 'collapsed'; } catch { return false; } };
const write = (id, collapsed) => { try { localStorage.setItem(KEY(id), collapsed ? 'collapsed' : 'open'); } catch {} };

export function specPlate(app) {
  const d = describe(app.id); if (!d) return null;
  let collapsed = read(app.id);
  const btn = h('button.sp-toggle', { type: 'button', 'aria-controls': `sp-body-${app.id}`, on: { click: () => set(!collapsed) } }, icon('chevron'));
  const body = h('div.sp-body', { id: `sp-body-${app.id}` },
    h('ol.sp-steps', d.steps.map((s, i) => h('li', h('b.num', String(i + 1)), s))),
    h('div.sp-roles', { 'aria-label': 'Model roles used' }, h('span.label', 'Models'), d.roles.map((r) => h('span.chip.mono.sp-chip', { title: `Uses ${r.toLowerCase()} models` }, r))));
  const plate = h('section.spec-plate', { 'aria-label': `About ${d.name}` },
    h('i.screw.tl'), h('i.screw.bl'),
    h('span.sp-glyph', { 'aria-hidden': 'true' }, icon(app.icon)),
    h('div.sp-main', h('div.sp-head', h('span.label', d.name), h('span.sp-short', d.short), h('span.sp-line', d.line)), body),
    btn);
  function set(c) {
    collapsed = c; write(app.id, c);
    plate.classList.toggle('collapsed', c);
    btn.setAttribute('aria-expanded', String(!c));
    btn.setAttribute('aria-label', c ? `Show what ${d.name} does` : `Hide the ${d.name} description`);
    btn.title = btn.getAttribute('aria-label');
    body.hidden = c;
  }
  set(collapsed);
  return plate;
}
