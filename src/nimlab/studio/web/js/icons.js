// Inline SVG icon set: 24px grid, 1.7 stroke, round caps. One consistent hand.
import { h } from './util.js';
const P = {
  switchboard: '<rect x="3.5" y="4" width="17" height="6" rx="1.5"/><rect x="3.5" y="14" width="17" height="6" rx="1.5"/><circle cx="7.5" cy="7" r=".6" fill="currentColor"/><circle cx="7.5" cy="17" r=".6" fill="currentColor"/><path d="M12 7h5M12 17h5"/>',
  forge: '<path d="M4 9h11a4 4 0 0 0 4-4H9.5C7 5 5 6.5 4 9Z"/><path d="M8 9v3l-2 2v3h10v-3l-2-2V9"/><path d="M5 20h14"/>',
  council: '<circle cx="12" cy="12" r="3.2"/><circle cx="12" cy="3.8" r="1.6"/><circle cx="19.8" cy="9.6" r="1.6"/><circle cx="16.8" cy="19" r="1.6"/><circle cx="7.2" cy="19" r="1.6"/><circle cx="4.2" cy="9.6" r="1.6"/>',
  atlas: '<circle cx="12" cy="12" r="8.5"/><path d="m15.5 8.5-2 5-5 2 2-5 5-2Z"/>',
  arena: '<path d="M5 20V11M10 20V5M15 20v-7M20 20V8"/><path d="M3 20.5h19"/>',
  prism: '<path d="M12 3.5 21 19.5H3Z"/><path d="M2.5 13.5 10 11.8"/><path d="m12.6 10.6 8.9-2.6M13.2 13l8.3-.6M13.6 15.4l7.4 2.4" opacity=".75"/>',
  sentinel: '<path d="M12 3 4.5 6v5.5c0 4.6 3.1 8 7.5 9.5 4.4-1.5 7.5-4.9 7.5-9.5V6Z"/><path d="m9 12 2.2 2.2L15.5 10"/>',
  loom: '<circle cx="4.5" cy="8" r="1.8"/><circle cx="19.5" cy="16" r="1.8"/><path d="M6.3 8C12 8 12 16 17.7 16"/><circle cx="4.5" cy="16" r="1.8"/><circle cx="19.5" cy="8" r="1.8"/><path d="M6.3 16C12 16 12 8 17.7 8"/>',
  distill: '<path d="M4.5 4.5h15L13.5 12v6.5l-3 2V12Z"/><path d="M7.5 8h9" opacity=".7"/>',
  settings: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>',
  paste: '<rect x="6" y="5" width="12" height="16" rx="2"/><path d="M9 5V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1"/><path d="M9.5 12h5M9.5 15.5h3"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>', x: '<path d="m6 6 12 12M18 6 6 18"/>',
  play: '<path d="M7 5v14l12-7Z"/>', stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2H10l2 2.5h6.5a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2Z"/>',
  file: '<path d="M6 3.5h8l4.5 4.5V19a1.5 1.5 0 0 1-1.5 1.5H6A1.5 1.5 0 0 1 4.5 19V5A1.5 1.5 0 0 1 6 3.5Z"/><path d="M14 3.5V8h4.5"/>',
  diff: '<path d="M8 4v10M3.5 9h9M3.5 19h9M16 6h5M18.5 18v-8"/><path d="M16 15l2.5 3 2.5-3" transform="translate(0 0)" opacity="0"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.3-4.3L4 8.5M4 4v4.5h4.5M4 13a8 8 0 0 0 14.3 4.3L20 15.5M20 20v-4.5h-4.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="m11 12 8.5-8.5M16 7l2.5 2.5M14 9l2 2"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4.5 15v3.5a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5V15"/>',
  bolt: '<path d="M13 2.5 5 13.5h6l-1 8 8-11h-6Z"/>',
  finder: '<rect x="3.5" y="4.5" width="17" height="15" rx="3"/><path d="M12 4.5v15M8 10v1.5M16 10v1.5M8 15c2.5 2 5.5 2 8 0"/>',
  iterate: '<path d="M4 12a8 8 0 0 1 13.5-5.8L20 8.5M20 4v4.5h-4.5"/><path d="M12 8v4l2.5 1.5"/>',
  chevron: '<path d="m9 6 6 6-6 6"/>', plus: '<path d="M12 5v14M5 12h14"/>',
  terminal: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="m7.5 10 3 2-3 2M13 15h4"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="2.8"/>',
  doc: '<path d="M6 3.5h9l3.5 3.5v13.5H6Z"/><path d="M9 11h6M9 14.5h6M9 18h3"/>',
  trash: '<path d="M4.5 7h15M9 7V4.5h6V7M6.5 7l1 13h9l1-13"/>',
  wifi: '<path d="M3 9.5a13 13 0 0 1 18 0M6 13a8.5 8.5 0 0 1 12 0M9 16.5a4 4 0 0 1 6 0"/><circle cx="12" cy="19.5" r=".8" fill="currentColor"/>',
  route: '<circle cx="5.5" cy="6" r="2"/><circle cx="18.5" cy="18" r="2"/><path d="M7.5 6H15a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h7.5"/>',
};
export function icon(name, cls = '') {
  const s = h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.7', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true', class: cls });
  s.innerHTML = P[name] || P.bolt;
  return s;
}
export const iconHTML = (name) => icon(name).outerHTML;
export const APP_ICON = { forge: 'forge', council: 'council', atlas: 'atlas', arena: 'arena', prism: 'prism', sentinel: 'sentinel', loom: 'loom', distill: 'distill', switchboard: 'switchboard' };
