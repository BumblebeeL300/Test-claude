// Right-hand drawer used by the code viewer and the diff viewer.
import { h, $, sleep } from '../util.js';
import { icon } from '../icons.js';
import { codeBlock, diffBlock, langOf } from '../highlight.js';
import { api, toast } from '../app.js';

let cur = null;
export function closeDrawer() {
  if (!cur) return; const d = cur; cur = null; d.el.classList.add('out'); setTimeout(() => d.el.remove(), 240); document.removeEventListener('keydown', d.key); d.prev?.focus?.();
}
export function openDrawer(title, body, actions = []) {
  closeDrawer();
  const el = h('aside.drawer', { role: 'dialog', 'aria-label': title },
    h('div.drawer-h', h('div.ttl', title), actions, h('button.btn.sm.icon.ghost', { 'aria-label': 'Close', on: { click: closeDrawer } }, icon('x'))),
    h('div.drawer-b', body));
  const key = (e) => { if (e.key === 'Escape') closeDrawer(); };
  document.addEventListener('keydown', key);
  cur = { el, key, prev: document.activeElement };
  document.body.append(el); el.querySelector('button')?.focus();
  return el;
}

export async function openFile(projectId, path, hint) {
  const body = h('div', h('div.skeleton', { style: { height: '300px', margin: '16px' } }));
  const copyBtn = h('button.btn.sm', icon('copy'), 'Copy');
  openDrawer(path, body, [copyBtn]);
  try {
    const r = await api.get(`/projects/${encodeURIComponent(projectId)}/file?path=${encodeURIComponent(path)}`);
    const text = typeof r === 'string' ? r : r.content ?? r.text ?? r.body ?? '';
    body.replaceChildren(codeBlock(text, langOf(path, r.language || hint)));
    copyBtn.onclick = () => { navigator.clipboard?.writeText(text); toast('Copied ' + path.split('/').pop()); };
  } catch (e) { body.replaceChildren(h('div.empty', icon('file'), h('h3', 'Can’t open this file'), h('p', e.message))); }
}

export function diffView(text) {
  const wrap = h('div');
  const files = String(text || '').split(/^(?=diff --git )/m).filter((s) => s.trim());
  if (!files.length) return h('div.empty', icon('diff'), h('h3', 'No changes to show'), h('p', 'The working tree matches the last snapshot.'));
  for (const f of files) {
    const name = (f.match(/^diff --git a\/(.+?) b\//) || [])[1] || (f.match(/^\+\+\+ b\/(.+)$/m) || [])[1] || 'changes';
    const add = (f.match(/^\+(?!\+\+)/gm) || []).length, del = (f.match(/^-(?!--)/gm) || []).length;
    wrap.append(h('div.diff-file', h('span', name), h('span.a', '+' + add), h('span.d', '−' + del)), diffBlock(f));
  }
  return wrap;
}
export async function openDiff(projectId) {
  const body = h('div', h('div.skeleton', { style: { height: '300px', margin: '16px' } }));
  openDrawer('Diff · ' + projectId, body);
  try {
    const r = await api.get(`/projects/${encodeURIComponent(projectId)}/diff`);
    const text = typeof r === 'string' ? r : r.diff ?? r.patch ?? (r.files ? r.files.map((f) => f.diff || f.patch).join('\n') : '');
    body.replaceChildren(diffView(text));
  } catch (e) { body.replaceChildren(h('div.empty', icon('diff'), h('h3', 'No diff available'), h('p', e.message))); }
}
