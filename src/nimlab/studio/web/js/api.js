// fetch + SSE client. Reads ?t= once, keeps it in sessionStorage, sends X-Studio-Token.
const qs = new URLSearchParams(location.search);
const ss = { get: (k) => { try { return sessionStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { sessionStorage.setItem(k, v); } catch {} } };
if (qs.get('t')) ss.set('nimlab.t', qs.get('t'));
if (qs.get('demo')) ss.set('nimlab.demo', qs.get('demo'));
export const token = ss.get('nimlab.t') || '';
export const DEMO = ss.get('nimlab.demo') === '1';
let demoMod = null;
const loadDemo = async () => (demoMod ||= await import('./demo.js'));

export class ApiError extends Error { constructor(msg, status, body) { super(msg); this.status = status; this.body = body; } }

export async function request(method, path, body, opts = {}) {
  if (DEMO) return (await loadDemo()).request(method, path, body);
  const headers = { 'X-Studio-Token': token };
  let payload;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
  let res;
  try { res = await fetch('/api' + path, { method, headers, body: payload, signal: opts.signal }); }
  catch (e) { throw new ApiError('Engine unreachable', 0, null); }
  const text = await res.text();
  let data = text; try { data = text ? JSON.parse(text) : null; } catch {}
  if (!res.ok) throw new ApiError((data && (data.detail?.message || data.detail || data.error || data.message)) || res.statusText, res.status, data);
  return data;
}
export const get = (p, o) => request('GET', p, undefined, o);
export const post = (p, b, o) => request('POST', p, b ?? {}, o);
export const del = (p) => request('DELETE', p);

// ---- SSE ----
const listeners = new Set(); let active = 0, reconnecting = 0;
export const onSse = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const notify = () => listeners.forEach((f) => f(active === 0 ? 'idle' : reconnecting > 0 ? 'reconnecting' : 'live'));
const TYPES = ['run.start','run.done','run.failed','phase.start','phase.end','plan','tool.call','tool.result','file.write','test.run','test.result','review','log','usage','stage.start','stage.end','model.start','model.end','consensus','token','sources'];
export const TERMINAL = new Set(['run.done', 'run.failed']);

/** Subscribe to a run's events. Past events are replayed by the server; we dedupe by seq. */
export function stream(runId, onEvent, onEnd) {
  if (DEMO) { let closed = false; let off = () => {}; loadDemo().then((d) => { if (!closed) off = d.stream(runId, onEvent) || off; }); active++; notify(); return () => { if (!closed) { closed = true; active--; notify(); } off(); }; }
  const seen = new Set(); let done = false, errored = false;
  const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events?t=${encodeURIComponent(token)}`);
  active++; notify();
  const finish = () => { if (done) return; done = true; es.close(); active--; if (errored) reconnecting--; notify(); onEnd && onEnd(); };
  const handle = (m) => {
    if (errored) { errored = false; reconnecting--; notify(); }
    let ev; try { ev = JSON.parse(m.data); } catch { return; }
    if (ev.type == null && m.type && m.type !== 'message') ev.type = m.type;
    const key = ev.seq ?? `${ev.type}:${ev.ts}`; if (seen.has(key)) return; seen.add(key);
    try { onEvent(ev); } catch (e) { console.error('event handler', e, ev); }
    const info = (ev.data?.result || ev.result)?.model_info;
    if (ev.type === 'run.done' && info) window.dispatchEvent(new CustomEvent('nimlab:run-done', { detail: { runId, info } }));
    if (TERMINAL.has(ev.type)) finish();
  };
  es.onmessage = handle;
  TYPES.forEach((t) => es.addEventListener(t, handle));
  es.onerror = () => { if (done) return; if (!errored) { errored = true; reconnecting++; notify(); } if (es.readyState === EventSource.CLOSED) finish(); };
  return finish;
}
