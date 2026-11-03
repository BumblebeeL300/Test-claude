// A deliberately tiny highlighter (py / js / json / md / sh). Returns an array of HTML line strings.
import { esc } from './util.js';
const KW = {
  py: 'def class return if elif else for while in not and or is import from as with try except finally raise pass break continue lambda yield async await None True False self assert global del',
  js: 'const let var function return if else for while of in new class extends import from export default async await try catch finally throw switch case break continue typeof this null undefined true false',
};
const LANG = { py: 'py', python: 'py', js: 'js', mjs: 'js', ts: 'js', javascript: 'js', typescript: 'js', json: 'json', md: 'md', markdown: 'md', sh: 'sh', toml: 'sh', yml: 'sh', yaml: 'sh', html: 'html', css: 'css' };
export const langOf = (path = '', hint = '') => LANG[(hint || '').toLowerCase()] || LANG[(path.split('.').pop() || '').toLowerCase()] || 'txt';
export const LANG_COLOR = { py: 'var(--c-atlas)', js: 'var(--c-warn, #FFC83D)', json: 'var(--c-arena)', md: 'var(--c-council)', sh: 'var(--c-forge)', html: 'var(--c-arena)', css: 'var(--c-core)', txt: 'var(--dim)' };

function tint(line, lang) {
  if (lang === 'md') {
    if (/^\s{0,3}#{1,6}\s/.test(line)) return `<span class="tk-h">${esc(line)}</span>`;
    return esc(line).replace(/`([^`]+)`/g, '<span class="tk-s">`$1`</span>').replace(/^(\s*[-*]|\d+\.)\s/, '<span class="tk-k">$1</span> ');
  }
  const kw = new RegExp(`^(?:${(KW[lang] || '').split(' ').join('|')})$`);
  const cm = lang === 'py' || lang === 'sh' ? '#' : '//';
  let out = '', i = 0;
  while (i < line.length) {
    const c = line[i], rest = line.slice(i);
    if (rest.startsWith(cm) && (lang !== 'json')) { out += `<span class="tk-c">${esc(rest)}</span>`; break; }
    if (c === '"' || c === "'" || c === '`') {
      let j = i + 1; while (j < line.length && line[j] !== c) j += line[j] === '\\' ? 2 : 1;
      const s = line.slice(i, j + 1);
      const isKey = lang === 'json' && /^\s*:/.test(line.slice(j + 1));
      out += `<span class="${isKey ? 'tk-f' : 'tk-s'}">${esc(s)}</span>`; i = j + 1; continue;
    }
    const num = rest.match(/^\d+(\.\d+)?/);
    if (num && !/\w/.test(line[i - 1] || '')) { out += `<span class="tk-n">${num[0]}</span>`; i += num[0].length; continue; }
    const w = rest.match(/^[A-Za-z_]\w*/);
    if (w) {
      const word = w[0]; const after = line[i + word.length];
      let cls = kw.test(word) ? 'tk-k' : after === '(' ? 'tk-f' : '';
      if (lang === 'json' && /^(true|false|null)$/.test(word)) cls = 'tk-k';
      out += cls ? `<span class="${cls}">${word}</span>` : esc(word); i += word.length; continue;
    }
    if (lang === 'py' && c === '@') { out += `<span class="tk-p">${esc(rest.match(/^@[\w.]*/)[0])}</span>`; i += rest.match(/^@[\w.]*/)[0].length; continue; }
    out += esc(c); i++;
  }
  return out;
}
export function highlightLines(src, lang) { return String(src ?? '').replace(/\n$/, '').split('\n').map((l) => tint(l, lang)); }
export function codeBlock(src, lang) {
  const pre = document.createElement('pre'); pre.className = 'code selectable';
  pre.innerHTML = highlightLines(src, lang).map((l) => `<span class="ln">${l || ' '}</span>`).join('');
  return pre;
}
export function diffBlock(text) {
  const pre = document.createElement('pre'); pre.className = 'code selectable'; pre.style.counterReset = 'none';
  pre.innerHTML = String(text || '').split('\n').map((l) => {
    const cls = l.startsWith('+++') || l.startsWith('---') || l.startsWith('diff ') || l.startsWith('index ') ? 'meta' : l.startsWith('@@') ? 'hunk' : l.startsWith('+') ? 'add' : l.startsWith('-') ? 'del' : '';
    return `<span class="ln diff-line ${cls}" style="padding-left:16px">${esc(l) || ' '}</span>`;
  }).join('');
  // diff has its own gutter; hide line counter
  pre.querySelectorAll('.ln').forEach((n) => n.style.setProperty('--x', '0'));
  pre.classList.add('nolines');
  return pre;
}
