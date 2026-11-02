// Tiny DOM + formatting helpers shared by every screen.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** hyperscript: h('div.cls#id', {attrs, on:{click}}, ...children) */
export function h(sig, props, ...kids) {
  if (props != null && (typeof props !== 'object' || props.nodeType || Array.isArray(props))) { kids.unshift(props); props = null; }
  const m = sig.match(/^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i) || [];
  const svgTags = /^(svg|path|g|circle|line|rect|text|defs|filter|polyline|polygon|ellipse)$/;
  const tag = m[1] || 'div';
  const el = svgTags.test(tag) ? document.createElementNS('http://www.w3.org/2000/svg', tag) : document.createElement(tag);
  (m[2] || '').replace(/([.#])([\w-]+)/g, (_, k, v) => { if (k === '.') el.classList.add(v); else el.id = v; });
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'on') for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === 'style' && typeof v === 'object') for (const [p, val] of Object.entries(v)) p.startsWith('--') ? el.style.setProperty(p, val) : (el.style[p] = val);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'class') v.split(' ').filter(Boolean).forEach((c) => el.classList.add(c));
    else el.setAttribute(k, v === true ? '' : v);
  }
  const add = (k) => { if (k == null || k === false) return; if (Array.isArray(k)) return k.forEach(add); el.append(k.nodeType ? k : document.createTextNode(String(k))); };
  kids.forEach(add);
  return el;
}
export function clear(el) { while (el.firstChild) el.firstChild.remove(); return el; }
export const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function ago(t) {
  if (!t) return '—';
  const ms = typeof t === 'number' ? (t < 1e12 ? t * 1000 : t) : Date.parse(t);
  if (!ms) return '—';
  const s = Math.max(0, (Date.now() - ms) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return Math.round(s / 60) + 'm ago';
  if (s < 86400) return Math.round(s / 3600) + 'h ago';
  return Math.round(s / 86400) + 'd ago';
}
export const fmtNum = (n) => (n == null || isNaN(n)) ? '—' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e4 ? (n / 1e3).toFixed(0) + 'k' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(Math.round(n));
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const shortModel = (id) => String(id || '').split('/').pop();
export const appVar = (id) => `var(--c-${id === 'forge' || id === 'council' || id === 'atlas' || id === 'arena' ? id : 'core'})`;

/** Springy entrance for freshly inserted elements. */
export function pop(el, opts = {}) {
  if (reduced() || !el.animate) return el;
  el.animate([{ opacity: 0, transform: `translateY(${opts.y ?? -18}px) scale(${opts.s ?? .92})` }, { opacity: 1, transform: 'none' }],
    { duration: opts.d ?? 520, easing: 'cubic-bezier(.34,1.56,.64,1)', delay: opts.delay ?? 0, fill: 'backwards' });
  return el;
}
export function debounce(fn, ms = 120) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
export function autogrow(ta, max = 360) { const f = () => { ta.style.height = 'auto'; ta.style.height = Math.min(max, ta.scrollHeight) + 'px'; }; ta.addEventListener('input', f); requestAnimationFrame(f); return f; }
