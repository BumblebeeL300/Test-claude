// Ignition: first-launch key entry. Dial spins up, the scope line locks, "N models online".
import { h, $, sleep, reduced, clamp } from '../util.js';
import { icon } from '../icons.js';
import { api, store, refresh, lsSet, toast, go } from '../app.js';

const RX = /^nvapi-[A-Za-z0-9_\-]{20,}$/;
let live = null;

function scope(canvas, st) {
  const ctx = canvas.getContext('2d'); let raf, t = 0, W, H, dpr;
  const size = () => { dpr = devicePixelRatio || 1; W = canvas.clientWidth; H = canvas.clientHeight; canvas.width = W * dpr; canvas.height = H * dpr; };
  size(); const ro = new ResizeObserver(size); ro.observe(canvas);
  const cols = () => { const cs = getComputedStyle(document.documentElement); return ['--c-council', '--c-atlas', '--c-forge', '--c-arena'].map((v) => cs.getPropertyValue(v).trim()); };
  let amp = 0.12, chaos = 0.3, lock = 0;
  function frame() {
    t += st.mode === 'spin' ? 0.09 : 0.02;
    const target = { idle: [0.12, 0.35], spin: [0.5, 1], lock: [0.2, 0], done: [0.16, 0] }[st.mode];
    amp += (target[0] - amp) * 0.06; chaos += (target[1] - chaos) * 0.06; lock += ((st.mode === 'lock' || st.mode === 'done' ? 1 : 0) - lock) * 0.05;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const C = cols(); ctx.lineWidth = 1.6; ctx.lineCap = 'round';
    const mid = H * 0.5;
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--hair').trim(); ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo(0, H * i / 8); ctx.lineTo(W, H * i / 8); ctx.stroke(); }
    for (let i = 1; i < 16; i++) { ctx.beginPath(); ctx.moveTo(W * i / 16, 0); ctx.lineTo(W * i / 16, H); ctx.stroke(); }
    for (let k = 0; k < 4; k++) {
      ctx.beginPath(); ctx.strokeStyle = C[k]; ctx.globalAlpha = (0.55 - lock * 0.35 * (k > 0)) * (k === 2 ? 1 : 0.8); ctx.lineWidth = k === 2 && lock > .5 ? 2.4 : 1.5;
      ctx.shadowColor = C[k]; ctx.shadowBlur = 12;
      for (let x = 0; x <= W; x += 4) {
        const u = x / W, env = Math.sin(Math.PI * u) ** 0.6;
        const clean = Math.sin(u * 14 + t * (1 + k * .0)) ;
        const messy = Math.sin(u * (9 + k * 5) + t * (1 + k * .4)) * .6 + Math.sin(u * (31 + k * 7) - t * 2.1) * .4 * chaos + (Math.sin(u * 77 + t * 5 + k) * .25 * chaos);
        const y = mid + (messy * (1 - lock) + clean * lock * (k === 2 ? 1 : .25)) * amp * H * env * (k === 2 ? 1 : .8 + k * .08);
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke(); ctx.shadowBlur = 0;
    }
    ctx.globalAlpha = 1; raf = requestAnimationFrame(frame);
  }
  if (reduced()) { st.mode = 'lock'; frame(); cancelAnimationFrame(raf); } else frame();
  return () => { cancelAnimationFrame(raf); ro.disconnect(); };
}

function dialSVG() {
  const R = 86, ticks = [];
  for (let i = 0; i < 60; i++) { const a = (i / 60) * Math.PI * 2, long = i % 5 === 0; ticks.push(`<line x1="${100 + Math.sin(a) * (R - (long ? 10 : 5))}" y1="${100 - Math.cos(a) * (R - (long ? 10 : 5))}" x2="${100 + Math.sin(a) * R}" y2="${100 - Math.cos(a) * R}" stroke="currentColor" stroke-width="${long ? 2 : 1}" opacity="${long ? .8 : .35}"/>`); }
  const arc = (a0, a1, c) => { const p = (a) => [100 + Math.sin(a) * 66, 100 - Math.cos(a) * 66]; const [x0, y0] = p(a0), [x1, y1] = p(a1); return `<path d="M${x0} ${y0} A66 66 0 0 1 ${x1} ${y1}" stroke="var(--c-${c})" stroke-width="5" stroke-linecap="round" fill="none" class="arc" data-app="${c}"/>`; };
  const q = Math.PI / 2, g = 0.18;
  return `<svg viewBox="0 0 200 200" aria-hidden="true"><circle cx="100" cy="100" r="96" fill="var(--panel)" stroke="var(--hair-2)"/>${ticks.join('')}${arc(g, q - g, 'council')}${arc(q + g, 2 * q - g, 'atlas')}${arc(2 * q + g, 3 * q - g, 'forge')}${arc(3 * q + g, 4 * q - g, 'arena')}
  <g class="needle"><path d="M100 100 L100 30" stroke="var(--text)" stroke-width="3" stroke-linecap="round"/><circle cx="100" cy="30" r="4" fill="var(--accent)"/></g><circle cx="100" cy="100" r="11" fill="var(--raised)" stroke="var(--hair-2)" stroke-width="2"/><circle class="hubdot" cx="100" cy="100" r="4" fill="var(--faint)"/></svg>`;
}

export function showIgnition() {
  if (live) return live.promise;
  const st = { mode: 'idle' };
  const canvas = h('canvas.ig-scope', { 'aria-hidden': 'true' });
  const dial = h('div.ig-dial', { html: dialSVG() });
  const input = h('input.ig-input', { type: 'password', placeholder: 'nvapi-…', 'aria-label': 'NVIDIA API key', autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off' });
  const reveal = h('button.btn.sm.icon.ghost', { type: 'button', 'aria-label': 'Show key', on: { click: () => { const p = input.type === 'password'; input.type = p ? 'text' : 'password'; reveal.setAttribute('aria-label', p ? 'Hide key' : 'Show key'); } } }, icon('eye'));
  const paste = h('button.btn.sm', { type: 'button', on: { click: async () => { try { input.value = (await navigator.clipboard.readText()).trim(); input.dispatchEvent(new Event('input')); input.focus(); } catch { toast('Clipboard blocked: paste with ⌘V instead', 'warn'); input.focus(); } } } }, icon('paste'), 'Paste');
  const hint = h('div.ig-hint', { 'aria-live': 'polite' });
  const go1 = h('button.btn.primary.ig-go', { type: 'submit', disabled: true }, icon('bolt'), 'Ignite');
  const result = h('div.ig-result', { 'aria-live': 'polite' });
  const form = h('form.ig-form', { on: { submit: (e) => { e.preventDefault(); ignite(); } } },
    h('label.label', { for: 'ig-key' }, 'NVIDIA API key'),
    h('div.ig-field', input, reveal, paste), hint,
    h('div.ig-actions', go1, h('button.btn.ghost', { type: 'button', on: { click: simulate } }, 'Enter simulation mode')),
    h('p.ig-fine', icon('key'), h('span', 'Stored in the macOS Keychain, never in a file. Get a free key at ', h('a', { href: 'https://build.nvidia.com', target: '_blank', rel: 'noopener' }, 'build.nvidia.com'), '.')));
  input.id = 'ig-key';
  const root = h('section.ignition', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Ignition: add your NVIDIA API key' },
    canvas, h('div.ig-drag.pywebview-drag-region'),
    h('div.ig-stack', dial, h('h1.wordmark', { 'aria-label': 'NIM LAB' }, h('span', { 'data-t': 'NIM' }, 'NIM'), h('span.sp'), h('span', { 'data-t': 'LAB' }, 'LAB')),
      h('p.ig-sub', 'A bench for NVIDIA models. Four instruments, one signal path.'), form, result));
  document.body.append(root);
  const stopScope = scope(canvas, st);

  function validate() {
    const v = input.value.trim(); let msg, cls = '';
    if (!v) msg = 'waiting for signal'; else if (!v.startsWith('nvapi-')) { msg = 'keys begin with nvapi-'; cls = 'warn'; }
    else if (!RX.test(v)) { msg = `keep going · ${v.length} chars`; cls = 'warn'; } else { msg = 'format looks right'; cls = 'ok'; }
    hint.className = 'ig-hint ' + cls; hint.replaceChildren(h('span.led', { class: cls === 'ok' ? 'on' : cls === 'warn' ? 'warn' : '' }), msg);
    go1.disabled = cls !== 'ok';
  }
  input.addEventListener('input', validate); validate();

  const needle = () => dial.querySelector('.needle');
  const arcs = () => [...dial.querySelectorAll('.arc')];
  arcs().forEach((a) => (a.style.opacity = .28));

  let busy = false;
  async function ignite() {
    if (busy || go1.disabled) return; busy = true; result.replaceChildren(); go1.disabled = true; input.disabled = true;
    st.mode = 'spin'; hint.replaceChildren(h('span.led.busy'), 'spinning up…');
    const spin = needle().animate([{ transform: 'rotate(0)' }, { transform: 'rotate(720deg)' }], { duration: 900, iterations: Infinity, easing: 'linear' });
    needle().style.transformOrigin = '100px 100px';
    let res, err;
    try { [res] = await Promise.all([api.post('/key', { api_key: input.value.trim() }), sleep(1300)]); } catch (e) { err = e; }
    spin.cancel();
    if (err || !res?.ok) {
      st.mode = 'idle'; input.disabled = false; busy = false; go1.disabled = false;
      hint.className = 'ig-hint bad'; hint.replaceChildren(h('span.led.bad'), (err?.message || res?.error || 'The key was rejected'));
      root.querySelector('.ig-field').animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-8px)' }, { transform: 'translateX(8px)' }, { transform: 'translateX(-4px)' }, { transform: 'none' }], { duration: 380 });
      return;
    }
    // lock
    st.mode = 'lock';
    needle().animate([{ transform: 'rotate(-40deg)' }, { transform: 'rotate(8deg)' }, { transform: 'rotate(0)' }], { duration: 700, easing: 'cubic-bezier(.34,1.56,.64,1)', fill: 'forwards' });
    for (const a of arcs()) { a.style.transition = 'opacity .3s, filter .3s'; a.style.opacity = 1; a.style.filter = `drop-shadow(0 0 6px var(--c-${a.dataset.app}))`; await sleep(140); }
    dial.querySelector('.hubdot').setAttribute('fill', 'var(--ok)');
    st.mode = 'done';
    const n = res.model_count ?? 0, verified = res.verified !== false;
    hint.className = 'ig-hint ok'; hint.replaceChildren(h('span.led.on'), 'signal locked');
    const big = h('b.num', '0');
    result.replaceChildren(verified ? h('div.ig-online', big, h('span', ' models online')) : h('div.ig-online.soft', h('span', 'Saved. Couldn’t reach NVIDIA from here (unverified).')),
      h('p.ig-fine', verified ? 'Key verified and filed in your Keychain.' : 'The key is stored; calls will be attempted on demand.'));
    if (verified) { const t0 = performance.now(); const tick = (now) => { const p = clamp((now - t0) / 900, 0, 1); big.textContent = Math.round(n * (1 - (1 - p) ** 3)); if (p < 1) requestAnimationFrame(tick); }; requestAnimationFrame(tick); }
    await refresh(['status', 'models']);
    await sleep(verified ? 1700 : 2400); finish();
  }
  function simulate() { lsSet('nimlab.sim', '1'); toast('Simulation mode: deterministic mock models, no key needed', 'warn'); finish(); }
  function finish() {
    const r = live; live = null; stopScope();
    root.animate([{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'scale(1.04)' }], { duration: reduced() ? 1 : 420, easing: 'ease-in' }).onfinish = () => root.remove();
    refresh(); r.resolve();
  }
  root.addEventListener('keydown', (e) => { if (e.key === 'Escape' && store.status?.key?.set) finish(); });
  let resolve; const promise = new Promise((r) => (resolve = r));
  live = { promise, resolve }; requestAnimationFrame(() => input.focus());
  return promise;
}
export function mount() {}
