// Foundry: an intent line that turns into an assembly line. Events drive every panel.
import { h, $, clear, sleep, ago, pop, reduced, autogrow, fmtNum, shortModel, clamp } from '../util.js';
import { icon } from '../icons.js';
import { store, on, go, api, toast, refresh, appStatus } from '../app.js';
import { openFile, openDiff } from '../components/drawer.js';
import { langOf, LANG_COLOR, codeBlock } from '../highlight.js';

const STATIONS = [
  { p: 'explore', n: 'EXPLORE' }, { p: 'plan', n: 'PLAN' }, { p: 'implement', n: 'BUILD' }, { p: 'verify', n: 'TEST' },
  { p: 'debug', n: 'DEBUG' }, { p: 'review', n: 'REVIEW' }, { p: 'report', n: 'SHIP' },
];
const IDEAS = ['A CLI task tracker with tests', 'A token-bucket rate limiter library', 'A Markdown to HTML converter', 'A tiny URL router with path params', 'A CSV to JSON command-line tool'];
const msOf = (ts) => (typeof ts === 'number' ? (ts < 1e12 ? ts * 1000 : ts) : Date.parse(ts)) || Date.now();

// ---- session survives navigation; the view is rebuilt from events ----
const F = { runId: null, project: null, prompt: '', events: [], status: 'idle', stop: null, ui: null, maxIters: 3 };
function attach(runId, prompt, project) {
  F.stop?.(); Object.assign(F, { runId, prompt: prompt ?? F.prompt, project: project ?? F.project, events: [], status: 'running' });
  F.stop = api.stream(runId, (ev) => { F.events.push(ev); if (ev.type === 'run.done') F.status = 'done'; if (ev.type === 'run.failed') F.status = 'failed'; F.ui?.apply(ev, false); if (ev.type === 'run.done' || ev.type === 'run.failed') refresh(['status', 'runs', 'projects']); }, () => { F.stop = null; });
}

export function mount(el, params) {
  const offs = [];
  let ui = null;
  const ctx = { sel: null, project: F.project, viewProject: null };

  // ===== left rail (projects) =====
  const projList = h('div.pj-list', { role: 'listbox', 'aria-label': 'Projects' });
  const side = h('aside.fd-side', h('div.fd-side-h', h('span.label', 'Projects'), h('button.btn.sm.icon.ghost', { 'aria-label': 'New project', title: 'New project', on: { click: () => { selectProject(null); intent.focus(); } } }, icon('plus'))), projList);
  function paintProjects() {
    clear(projList);
    projList.append(h('button.pj', { role: 'option', 'aria-selected': ctx.project == null, on: { click: () => selectProject(null) } }, h('span.pj-ic', icon('plus')), h('span.pj-t', h('b', 'New project'), h('small', 'auto-named'))));
    if (!store.projects.length) projList.append(h('p.pj-empty', 'Nothing forged yet. Your builds will be shelved here.'));
    for (const p of store.projects) projList.append(h('button.pj', { role: 'option', 'aria-selected': ctx.project === p.id, on: { click: () => selectProject(p.id) } },
      h('span.led', { class: p.last_status === 'done' || p.last_status === 'success' ? 'on' : p.last_status === 'failed' ? 'bad' : '' }),
      h('span.pj-t', h('b', p.name), h('small', `${p.files ?? '?'} files · ${ago(p.updated_at)}`))));
  }
  function selectProject(id) {
    if (F.status === 'running') return toast('A build is in progress. Cancel it first to switch projects.', 'warn');
    ctx.project = id; F.project = id; projSel.value = id || ''; paintProjects();
    if (F.status !== 'idle') resetToIdle(true);
    paintProjectView();
  }

  // ===== intent =====
  const intent = h('textarea.intent', { rows: 2, placeholder: 'Describe something to build…', 'aria-label': 'What should Forge build?', spellcheck: 'false', on: { keydown: (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); start(); } } } });
  const grow = autogrow(intent, 240); intent.addEventListener('transitionend', (e) => { if (e.propertyName === 'font-size') grow(); });
  const projSel = h('select.field.sm', { 'aria-label': 'Project', on: { change: () => selectProject(projSel.value || null) } });
  const iters = h('select.field.sm', { 'aria-label': 'Debug budget', on: { change: () => (F.maxIters = +iters.value) } }, [2, 3, 5].map((n) => h('option', { value: n, selected: n === F.maxIters }, `${n} debug rounds`)));
  const startBtn = h('button.btn.primary.big', { on: { click: start } }, icon('play'), 'Start build', h('kbd', '⌘↵'));
  const cancelBtn = h('button.btn.sm', { on: { click: cancel } }, icon('stop'), 'Cancel');
  const newBtn = h('button.btn.sm', { on: { click: () => { selectProject(null); intent.value = ''; grow(); intent.focus(); } } }, icon('plus'), 'New build');
  const clock = h('span.num.clock', '00:00'), tokNum = h('b.num', '0'), tokBar = h('div.tokbar', Array.from({ length: 20 }, () => h('i')));
  const ticketMeta = h('div.ticket-meta', h('span.led.busy'), h('span.label.st', 'BUILDING'), clock, h('span.sep'), h('span.label', 'TOKENS'), tokNum, tokBar, h('span.grow'), cancelBtn, newBtn);
  const intentBox = h('div.intent-box', h('div.bp-corner.tl'), h('div.bp-corner.tr'), h('div.bp-corner.bl'), h('div.bp-corner.br'),
    h('div.bp-tag.mono', 'INTENT · REV A'), intent);
  const chips = h('div.chips', IDEAS.map((t) => h('button.chip', { on: { click: () => { intent.value = t; grow(); intent.focus(); } } }, t)));
  const controls = h('div.intent-ctl', h('label.ctl', h('span.label', 'Project'), projSel), h('label.ctl', h('span.label', 'Budget'), iters), h('span.grow'), startBtn);
  const projView = h('div.pv');
  const hero = h('section.hero', intentBox, h('div.hero-rest', chips, controls, h('div.preview-h', h('span.label', 'The line, once you press start'), h('i')), h('div.preview', STATIONS.map((s) => h('span', s.n)))));

  // ===== assembly line =====
  const track = h('div.track', h('i.fill'));
  const carriage = h('div.carriage', { 'aria-hidden': 'true' }, h('i'));
  const stEls = {};
  const stations = h('div.stations', { role: 'list' }, track, carriage, STATIONS.map((s, i) => {
    const pin = h('span.pin'), metric = h('span.metric', '—');
    const b = h('button.station', { role: 'listitem', 'data-p': s.p, 'data-s': 'idle', 'aria-expanded': 'false', on: { click: () => selectStation(s.p) } },
      pin, h('span.sn.num', String(i + 1).padStart(2, '0')), h('span.name', s.n), metric);
    stEls[s.p] = { b, pin, metric };
    return b;
  }));
  const detail = h('div.st-detail', { hidden: true, 'aria-live': 'polite' });
  const line = h('section.line', stations, detail);

  // ===== panels =====
  const planList = h('ol.bp-list'); const planCard = panel('Plan · blueprint', planList, 'plan');
  const fileGrid = h('div.tiles'); const fileCount = h('span.num.dim', '0'); const filesCard = panel('Files', fileGrid, 'files', fileCount);
  const testGrid = h('div.lights'); const testSum = h('span.num.sum', '—'); const testsCard = panel('Tests', testGrid, 'tests', testSum);
  const stampBox = h('div.stamp-box'); const reviewCard = panel('Review', stampBox, 'review');
  const logBody = h('div.log.selectable'); const logCount = h('span.num.dim', '0');
  const logToggle = h('button.log-h', { 'aria-expanded': 'false', on: { click: () => { const o = logToggle.getAttribute('aria-expanded') !== 'true'; logToggle.setAttribute('aria-expanded', o); logWrap.classList.toggle('open', o); } } }, icon('terminal'), h('span.label', 'Telemetry'), logCount, h('span.grow'), icon('chevron', 'chev'));
  const logWrap = h('section.panel.logp', logToggle, logBody);
  function panel(title, body, cls, extra) { return h('section.panel.pn.pn-' + cls, h('i.screw.tl'), h('i.screw.tr'), h('div.panel-h', h('span.label', title), extra), body); }
  const shipped = h('div.shipped', { hidden: true });
  const panels = h('div.panels', planCard, filesCard, testsCard, reviewCard);
  const run = h('div.run', { hidden: true }, ticketMeta, line, shipped, panels, logWrap);

  const stage = h('div.fd-stage', hero, projView, run);
  const root = h('div.fd', side, stage); el.append(root);

  // ===== view model + apply =====
  const vm = () => ({ phase: {}, plan: [], planDone: new Set(), files: new Map(), tests: new Map(), tokens: 0, calls: 0, tools: 0, review: null, logs: [], byPhase: {}, t0: null, result: null, debugRuns: 0, verifyRuns: 0, curPhase: null });
  let M = vm(); let replaying = false; let timer = null;

  function describe(ev) {
    const d = ev.data || {};
    switch (ev.type) {
      case 'tool.call': return { k: 'call', t: d.summary ? String(d.summary).slice(0, 140) : `${d.name || d.tool || 'tool'}(${d.args ? Object.entries(d.args).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ').slice(0, 90) : ''})` };
      case 'tool.result': { const extra = d.preview ?? d.result ?? d.output; return { k: d.ok === false ? 'warn' : 'res', t: `↳ ${d.ok === false ? 'error' : 'ok'}${extra ? ': ' + String(extra).slice(0, 140) : d.chars != null ? ` · ${d.chars} chars` : ''}` }; }
      case 'log': return { k: d.level || 'info', t: d.message ?? d.text ?? JSON.stringify(d) };
      case 'test.run': return { k: 'call', t: `$ ${d.command || 'pytest'}${d.total ? ` (${d.total} tests)` : ''}` };
      case 'test.result': return { k: d.failed ? 'warn' : 'ok', t: d.summary || `${d.passed ?? 0} passed, ${d.failed ?? 0} failed` };
      case 'file.write': return { k: 'file', t: `${d.action || 'wrote'} ${d.path} (${d.lines ?? '?'} lines)` };
      case 'plan': return { k: 'info', t: `plan: ${(d.steps || []).length} steps` };
      case 'review': return { k: /approv/.test(d.verdict) ? 'ok' : 'warn', t: d.message || `review verdict: ${d.verdict}` };
      case 'phase.start': return { k: 'phase', t: `▶ ${ev.phase}` };
      case 'phase.end': return { k: 'phase', t: `■ ${ev.phase} done${d.seconds != null ? ` (${d.seconds}s)` : ''}` };
      case 'run.failed': return { k: 'bad', t: `failed: ${d.error || d.message || 'unknown error'}` };
      default: return null;
    }
  }

  function apply(ev, isReplay) {
    replaying = isReplay; const d = ev.data || {};
    if (M.t0 == null) M.t0 = msOf(ev.ts);
    const line_ = describe(ev);
    if (line_) { line_.ts = msOf(ev.ts) - M.t0; if (ev.phase) (M.byPhase[ev.phase] ||= []).push(line_); if (ev.type !== 'phase.start' && ev.type !== 'phase.end' || true) addLog(line_); }
    switch (ev.type) {
      case 'phase.start': {
        const p = ev.phase; if (!stEls[p]) break; M.curPhase = p;
        setStation(p, 'active'); if (p === 'verify') M.verifyRuns++; if (p === 'debug') M.debugRuns++;
        if (p === 'review' || p === 'report') if (!M.phase.debug) setStation('debug', 'skipped');
        moveCarriage(p, isReplay); metrics(); break;
      }
      case 'phase.end': if (stEls[ev.phase]) { setStation(ev.phase, M.failedPhase === ev.phase ? 'failed' : 'done'); if (ev.phase === 'implement') M.plan.forEach((s) => tick(s.id, true)); } break;
      case 'plan': paintPlan(d.steps || [], d.goal); metrics(); break;
      case 'tool.call': M.tools++; metrics(); break;
      case 'file.write': addFile(d); tickFor(d.path); metrics(); break;
      case 'test.result': paintTests(d); metrics(); break;
      case 'review': paintReview(d); metrics(); break;
      case 'usage': { const t = d.total_tokens ?? d.tokens ?? ((d.prompt_tokens || d.tokens_in || 0) + (d.completion_tokens || d.tokens_out || 0)); M.tokens += t; M.calls += d.calls ?? 1; paintTokens(); break; }
      case 'run.done': finish(true, d); break;
      case 'run.failed': finish(false, d); break;
    }
    if (ctx.sel) paintDetail();
  }
  function setStation(p, s) { M.phase[p] = s; const e = stEls[p]; e.b.dataset.s = s; e.pin.dataset.s = s; }
  function metrics() {
    const m = (p, t) => (stEls[p].metric.textContent = t);
    if (M.phase.explore) m('explore', `${M.tools} tool calls`); if (M.plan.length) m('plan', `${M.plan.length} steps`);
    if (M.files.size) m('implement', `${M.files.size} files`);
    if (M.tests.size) { const ok = [...M.tests.values()].filter((o) => o === 'passed').length; m('verify', `${ok}/${M.tests.size} green${M.verifyRuns > 1 ? ` · run ${M.verifyRuns}` : ''}`); }
    if (M.phase.debug === 'skipped') m('debug', 'not needed'); else if (M.debugRuns) m('debug', `${M.debugRuns} fix${M.debugRuns > 1 ? 'es' : ''}`);
    if (M.review) m('review', String(M.review.verdict || 'done'));
    if (M.phase.report === 'done') m('report', 'shipped');
  }

  // carriage
  function moveCarriage(p, instant) {
    const pin = stEls[p].pin, sr = stations.getBoundingClientRect(), pr = pin.getBoundingClientRect(), vert = line.dataset.vert === '1';
    const x = pr.left - sr.left + pr.width / 2, y = pr.top - sr.top + pr.height / 2;
    carriage.style.transition = instant || reduced() ? 'none' : ''; carriage.style.transform = `translate(${x}px, ${y}px)`; carriage.classList.add('on');
    const fill = track.firstChild, t0 = STATIONS.findIndex((s) => s.p === p);
    const first = stEls.explore.pin.getBoundingClientRect(); fill.style.transition = carriage.style.transition;
    if (vert) { fill.style.width = ''; fill.style.height = Math.max(0, pr.top - first.top) + 'px'; } else { fill.style.height = ''; fill.style.width = Math.max(0, pr.left - first.left) + 'px'; }
    carriage.dataset.at = p;
  }
  const ro = new ResizeObserver(() => { line.dataset.vert = line.clientWidth < 880 ? '1' : '0'; if (carriage.dataset.at) moveCarriage(carriage.dataset.at, true); }); ro.observe(line); offs.push(() => ro.disconnect());

  function selectStation(p) { ctx.sel = ctx.sel === p ? null : p; STATIONS.forEach((s) => stEls[s.p].b.setAttribute('aria-expanded', ctx.sel === s.p)); paintDetail(); }
  function paintDetail() {
    detail.hidden = !ctx.sel; if (!ctx.sel) return; const L = M.byPhase[ctx.sel] || [];
    const st = STATIONS.find((s) => s.p === ctx.sel);
    detail.replaceChildren(h('div.det-h', h('span.label', st.n + ' · ' + (M.phase[ctx.sel] || 'waiting')), h('button.btn.sm.icon.ghost', { 'aria-label': 'Close detail', on: { click: () => selectStation(ctx.sel) } }, icon('x'))),
      L.length ? L.filter((l) => l.k !== 'phase').slice(-12).map(logRow) : h('p.dim', 'Nothing reported from this station yet.'));
  }
  const logRow = (l) => h('div.lr', { class: l.k }, h('span.ts', fmtT(l.ts)), h('span.tx', l.t));
  const fmtT = (ms) => `+${String(Math.floor(ms / 60000)).padStart(2, '0')}:${((ms % 60000) / 1000).toFixed(1).padStart(4, '0')}`;
  function addLog(l) { logBody.append(logRow(l)); logCount.textContent = logBody.childElementCount; if (logWrap.classList.contains('open')) logBody.scrollTop = logBody.scrollHeight; }

  // plan
  function paintPlan(steps, goal) {
    M.plan = steps; clear(planList); if (goal) planList.append(h('li.bp.goal', h('span.label', 'Goal'), h('p', goal)));
    steps.forEach((s, i) => { const li = h('li.bp', { 'data-id': s.id }, h('span.bp-n', h('svg', { viewBox: '0 0 24 24', html: '<path d="m5 12.5 4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" pathLength="1"/>' }), h('b.num', s.id ?? i + 1)), h('div', h('p', s.description || s.title || s.text), h('div.bp-f', (s.files || []).map((f) => h('code', f))))); planList.append(isReplayNow() ? li : pop(li, { delay: i * 90, y: 8, s: .97 })); });
    if (!steps.length) planList.append(h('li.dim', 'No steps reported.'));
  }
  const isReplayNow = () => replaying;
  function tick(id, silent) { const li = planList.querySelector(`[data-id="${CSS.escape(String(id))}"]`); if (li && !li.classList.contains('done')) { li.classList.add('done'); } }
  function tickFor(path) { for (const s of M.plan) if ((s.files || []).includes(path)) tick(s.id); }

  // files
  function addFile(d) {
    const path = d.path || d.file; if (!path) return; const prev = M.files.get(path);
    const rec = { path, lines: d.lines ?? prev?.lines, lang: langOf(path, d.language), rev: (prev?.rev || 0) + 1 }; M.files.set(path, rec);
    const base = path.split('/').pop(), dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/') + 1) : '';
    const tile = h('button.tile', { 'data-path': path, style: { '--lc': LANG_COLOR[rec.lang] || 'var(--dim)' }, title: path, on: { click: () => openFile(projId(), path, rec.lang) } },
      h('span.tl-top', h('span.ext', rec.lang === 'txt' ? 'file' : rec.lang), rec.rev > 1 && h('span.rev', 'rev ' + rec.rev)), h('span.tl-name', base), h('span.tl-dir', dir || './'), h('span.tl-n.num', `${rec.lines ?? '?'} lines`), h('i.thud'));
    const old = fileGrid.querySelector(`[data-path="${CSS.escape(path)}"]`);
    if (old) { old.replaceWith(tile); if (!replaying) tile.animate([{ boxShadow: '0 0 0 0 var(--lc)' }, { boxShadow: '0 0 0 10px transparent' }], { duration: 700 }); }
    else { fileGrid.append(tile); if (!replaying && !reduced()) tile.animate([{ transform: 'translateY(-70px) rotate(-4deg) scale(1.05)', opacity: 0 }, { transform: 'translateY(0) rotate(1deg)', opacity: 1, offset: .55 }, { transform: 'translateY(-6px)', offset: .75 }, { transform: 'none' }], { duration: 650, easing: 'cubic-bezier(.3,.9,.4,1)' }); }
    fileCount.textContent = M.files.size;
  }
  const projId = () => F.project || ctx.project || F.result_project;

  // tests
  function paintTests(d) {
    let list = d.results || d.tests;
    if (!list && d.names) list = d.names.map((name) => ({ name, outcome: (d.failing || []).includes(name) ? 'failed' : 'passed' }));
    (list || []).forEach((t) => {
      const raw = t.name || t.id || '?', st = t.outcome || t.status || '', out = /pass/i.test(st) ? 'passed' : /skip|xfail/i.test(st) ? 'skipped' : 'failed'; const was = M.tests.get(raw); M.tests.set(raw, out);
      let cell = testGrid.querySelector(`[data-n="${CSS.escape(raw)}"]`);
      const nice = raw.split('::').pop().replace(/^test_/, '').replace(/_/g, ' ');
      if (!cell) { cell = h('div.lt', { 'data-n': raw, title: raw }, h('span.bulb'), h('span.ln', nice)); testGrid.append(isReplayNow() ? cell : pop(cell, { delay: testGrid.childElementCount * 40, y: 6, s: .9 })); }
      cell.dataset.o = out;
      if (was === 'failed' && out === 'passed' && !replaying) { cell.classList.remove('flip'); void cell.offsetWidth; cell.classList.add('flip'); }
    });
    const ok = [...M.tests.values()].filter((o) => o === 'passed').length; testSum.textContent = `${ok}/${M.tests.size}`; testSum.dataset.ok = ok === M.tests.size ? '1' : '0';
  }
  // review
  function paintReview(d) {
    M.review = d; const v = String(d.verdict || '').toLowerCase(), good = /approv|pass|ship|lgtm/.test(v);
    const stamp = h('div.stamp', { class: good ? 'good' : 'bad', role: 'img', 'aria-label': 'Review verdict: ' + v }, h('span', good ? 'APPROVED' : String(d.verdict || 'CHANGES').toUpperCase().replace('_', ' ')), h('small', good ? 'NIM LAB · FORGE REVIEW' : 'SEE NOTES'));
    const issueList = Array.isArray(d.issues) ? d.issues : typeof d.issues === 'number' && d.issues ? [{ severity: d.blocking ? 'high' : 'low', message: d.message || `${d.issues} issue(s) noted` }] : (d.message ? [{ severity: 'info', message: d.message }] : []);
    const issues = issueList.map((i) => h('li', h('span.sev', { class: i.severity || 'info' }, i.severity || 'note'), h('span', typeof i === 'string' ? i : i.message || i.description || '')));
    stampBox.replaceChildren(stamp, issues.length ? h('ul.issues', issues) : null);
    if (!replaying && !reduced()) stamp.animate([{ transform: 'scale(2.6) rotate(-18deg)', opacity: 0 }, { transform: 'scale(.94) rotate(-6deg)', opacity: 1, offset: .6 }, { transform: 'scale(1) rotate(-5deg)' }], { duration: 600, easing: 'cubic-bezier(.3,1,.4,1)' });
  }
  function paintTokens() { tokNum.textContent = fmtNum(M.tokens); const lit = Math.min(20, Math.round((M.tokens / 40000) * 20)); [...tokBar.children].forEach((b, i) => b.classList.toggle('on', i < lit)); }

  // finish
  function finish(ok, d) {
    clearInterval(timer); ticketMeta.classList.add('over'); ticketMeta.querySelector('.led').className = 'led ' + (ok ? 'on' : 'bad'); ticketMeta.querySelector('.st').textContent = !ok ? 'STOPPED' : (d.result || d).ok === false ? 'UNFINISHED' : 'SHIPPED';
    cancelBtn.hidden = true; newBtn.hidden = false; carriage.classList.toggle('done', ok);
    if (ok) { STATIONS.forEach((s) => { if (M.phase[s.p] === 'active') setStation(s.p, 'done'); }); setStation('report', 'done'); metrics(); } else if (M.curPhase) setStation(M.curPhase, 'failed');
    const r = d.result || d; M.result = r; const pid = r.project && typeof r.project === 'object' ? r.project.id : r.project; if (pid) F.project = F.project || pid; F.result_project = pid || F.project;
    const partial = ok && (r.ok === false || /fail|incomplete|stuck/.test(r.status || ''));
    shipped.hidden = false; clear(shipped);
    const word = !ok ? 'STOPPED' : partial ? 'UNFINISHED' : 'SHIPPED';
    const letters = h('div.ship-word', { 'aria-label': word }, [...word].map((c, i) => h('span', { style: { '--i': i } }, c)));
    shipped.className = 'shipped ' + (!ok ? 'bad' : partial ? 'warn' : 'ok');
    shipped.append(letters, h('div.ship-body', h('p.ship-sum', ok ? String(r.summary || 'Build complete.').split(/\n\s*\n/)[0] : (d.error || d.message || r.error || 'The run ended before finishing.')),
      h('div.ship-act', ok && h('button.btn.primary', { on: { click: () => openDiff(projId()) } }, icon('diff'), 'View diff'), ok && h('button.btn', { on: { click: reveal } }, icon('finder'), 'Reveal in Finder'), h('button.btn', { on: { click: iterate } }, icon('iterate'), ok ? 'Iterate' : 'Retry with edits'))));
    if (!replaying && !reduced()) { shipped.animate([{ opacity: 0, transform: 'translateY(16px) scale(.97)' }, { opacity: 1, transform: 'none' }], { duration: 600, easing: 'cubic-bezier(.34,1.56,.64,1)' }); if (ok) { shipped.classList.add('flash'); setTimeout(() => shipped.classList.remove('flash'), 1400); } }
    if (ok && !partial) toast('Shipped. Open the diff or reveal it in Finder.'); else if (partial) toast('Forge stopped with failing tests. Review the diff or iterate.', 'warn');
    refresh(['projects']).then(paintProjects);
  }
  async function reveal() {
    const id = projId(); const p = store.projects.find((x) => x.id === id);
    try { if (window.pywebview?.api?.open_path && p?.path) { await window.pywebview.api.open_path(p.path); return; } await api.post(`/projects/${encodeURIComponent(id)}/reveal`); toast('Revealed in Finder'); } catch (e) { toast('Couldn’t reveal: ' + e.message, 'bad'); }
  }
  function iterate() { const p = F.prompt; resetToIdle(true); intent.value = (p ? p + '\n\n' : '') + 'Iterate: '; grow(); intent.focus(); intent.setSelectionRange(intent.value.length, intent.value.length); }

  // ===== lifecycle =====
  function setBuilding(on) { stage.classList.toggle('building', on); run.hidden = !on; hero.querySelector('.hero-rest').hidden = on; projView.hidden = on; intent.readOnly = on; requestAnimationFrame(() => { grow(); }); setTimeout(grow, 650); }
  function reset() {
    M = vm(); clearInterval(timer); [fileGrid, testGrid, planList, logBody, stampBox, shipped].forEach(clear); shipped.hidden = true; fileCount.textContent = '0'; logCount.textContent = '0'; testSum.textContent = '—'; tokNum.textContent = '0'; paintTokens();
    STATIONS.forEach((s) => { setStation(s.p, 'idle'); stEls[s.p].metric.textContent = '—'; }); carriage.classList.remove('on', 'done'); delete carriage.dataset.at; track.firstChild.style.width = track.firstChild.style.height = '0';
    ctx.sel = null; detail.hidden = true; ticketMeta.classList.remove('over'); ticketMeta.querySelector('.led').className = 'led busy'; ticketMeta.querySelector('.st').textContent = 'BUILDING'; cancelBtn.hidden = false; newBtn.hidden = true; clock.textContent = '00:00';
  }
  function resetToIdle(keepPrompt) { F.stop?.(); F.stop = null; F.status = 'idle'; F.runId = null; F.events = []; reset(); setBuilding(false); if (!keepPrompt) { intent.value = ''; grow(); } paintProjectView(); }
  function startClock(t0) { clearInterval(timer); const f = () => { const s = Math.floor((Date.now() - t0) / 1000); clock.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; }; f(); timer = setInterval(f, 500); }

  async function start() {
    const prompt = intent.value.trim();
    if (!prompt) { toast('Describe what to build first', 'warn'); intent.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-8px)' }, { transform: 'translateX(8px)' }, { transform: 'none' }], { duration: 300 }); intent.focus(); return; }
    startBtn.disabled = true;
    try {
      const body = { prompt, max_iters: F.maxIters }; if (ctx.project) body.project = ctx.project;
      const r = await api.post('/forge/build', body);
      reset(); F.prompt = prompt; F.project = r.project || ctx.project; ctx.project = F.project; setBuilding(true); startClock(Date.now());
      F.ui = ui; attach(r.run_id, prompt, F.project); refresh(['status', 'runs']);
    } catch (e) { toast('Couldn’t start: ' + e.message, 'bad'); } finally { startBtn.disabled = false; }
  }
  async function cancel() { if (!F.runId) return; try { await api.post(`/runs/${encodeURIComponent(F.runId)}/cancel`); toast('Cancelling…', 'warn'); } catch (e) { toast(e.message, 'bad'); } }

  // project view (idle)
  async function paintProjectView() {
    clear(projView); const id = ctx.project; const my = (ctx.viewProject = id); if (!id) { projView.hidden = true; return; }
    projView.hidden = false; const p = store.projects.find((x) => x.id === id) || { name: id };
    const tree = h('div.tiles', h('div.skeleton', { style: { height: '86px' } }), h('div.skeleton', { style: { height: '86px' } }));
    const rep = h('div.report.selectable', h('div.skeleton', { style: { height: '90px' } }));
    projView.append(h('div.pv-h', h('div', h('span.label', 'Project'), h('h2', p.name), h('code.dim', p.path || '')), h('span.grow'),
      h('button.btn.sm', { on: { click: () => openDiff(id) } }, icon('diff'), 'Diff'), h('button.btn.sm', { on: { click: () => { F.project = id; F.result_project = id; reveal(); } } }, icon('finder'), 'Reveal')),
      h('div.pv-cols', h('section.panel.pn', h('div.panel-h', h('span.label', 'Files')), tree), h('section.panel.pn', h('div.panel-h', h('span.label', 'Last report')), rep)));
    try {
      let t = await api.get(`/projects/${encodeURIComponent(id)}/tree`); if (my !== ctx.viewProject) return;
      t = Array.isArray(t) ? t : t.files || t.tree || t.entries || []; const files = t.map((f) => (typeof f === 'string' ? { path: f } : f)).filter((f) => f.type !== 'dir' && f.type !== 'directory');
      clear(tree); if (!files.length) tree.append(h('p.dim', 'Empty project.'));
      files.forEach((f, i) => { const lang = langOf(f.path, f.language); tree.append(pop(h('button.tile', { style: { '--lc': LANG_COLOR[lang] }, on: { click: () => openFile(id, f.path, lang) } }, h('span.tl-top', h('span.ext', lang === 'txt' ? 'file' : lang)), h('span.tl-name', f.path.split('/').pop()), h('span.tl-dir', f.path.includes('/') ? f.path.slice(0, f.path.lastIndexOf('/') + 1) : './'), h('span.tl-n.num', f.lines != null ? `${f.lines} lines` : f.size != null ? `${f.size} B` : '')), { delay: i * 40, y: 8, s: .96 })); });
      const rp = files.find((f) => /report\.md$/i.test(f.path)) || files.find((f) => /readme\.md$/i.test(f.path));
      clear(rep); if (rp) { const r = await api.get(`/projects/${encodeURIComponent(id)}/file?path=${encodeURIComponent(rp.path)}`); rep.append(codeBlock(typeof r === 'string' ? r : r.content ?? '', 'md')); } else rep.append(h('p.dim', 'No report written for this project yet.'));
    } catch (e) { clear(tree); tree.append(h('p.dim', 'Couldn’t read this project: ' + e.message)); clear(rep); }
  }

  function fill() {
    clear(projSel); projSel.append(h('option', { value: '' }, 'New project (auto-named)'), store.projects.map((p) => h('option', { value: p.id, selected: p.id === ctx.project }, p.name))); projSel.value = ctx.project || '';
  }
  ui = { apply }; F.ui = ui;
  offs.push(on('projects', () => { paintProjects(); fill(); }), on('prefill', (d) => { if (d.screen === 'foundry' && F.status !== 'running') { if (F.status !== 'idle') resetToIdle(true); intent.value = d.text || intent.value; if (store.prefill) delete store.prefill.foundry; grow(); intent.focus(); } }));
  paintProjects(); fill();

  // prefill / resume
  const pre = store.prefill?.foundry; if (pre != null) { if (F.status !== 'running') { F.runId = null; F.events = []; F.status = 'idle'; } intent.value = pre; delete store.prefill.foundry; }
  if (params.project) { ctx.project = params.project; F.project = params.project; fill(); paintProjects(); }
  if (params.run && params.run !== F.runId) { const r = store.runs.find((x) => x.id === params.run); F.stop?.(); F.runId = params.run; intent.value = r?.title || ''; reset(); setBuilding(true); startClock(Date.now()); F.events = []; F.status = 'running'; attach(params.run, r?.title || '', params.project || F.project); }
  else if (F.runId && F.events.length) { intent.value = F.prompt; reset(); setBuilding(true); const evs = F.events.slice(); evs.forEach((ev) => apply(ev, true)); if (F.status === 'running') startClock(M.t0 || Date.now()); }
  else if (F.runId && F.status === 'running') { intent.value = F.prompt; reset(); setBuilding(true); startClock(Date.now()); }
  else { setBuilding(false); paintProjectView(); }
  requestAnimationFrame(() => { grow(); if (!stage.classList.contains('building')) intent.focus({ preventScroll: true }); });
  return () => { clearInterval(timer); offs.forEach((f) => f && f()); if (F.ui === ui) F.ui = null; };
}
