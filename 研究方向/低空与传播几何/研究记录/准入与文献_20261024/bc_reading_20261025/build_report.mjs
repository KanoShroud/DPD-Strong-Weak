// Build the same local report URL from reviewed source; no network requests.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'file:///C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const out = path.dirname(root);
let md = fs.readFileSync(path.join(root, '综合结论.md'), 'utf8');
md = md.replace(/\]\(([^)]+)\)/g, (full, href) => {
  if (/^(https?:|#)/.test(href)) return full;
  return `](${path.posix.normalize(`bc_reading_20261025/${href}`)})`;
});
fs.writeFileSync(path.join(out, '准入报告.md'), md, 'utf8');
const sections = ['done', 'path', 'papers', 'decision', 'method', 'next', 'classification', 'unknown', 'records'];
let content = marked.parse(md);
content = content.replace(/<h1>.*?<\/h1>/s, '');
let index = 0;
content = content.replace(/<h2>(.*?)<\/h2>/g, (_, text) => `<h2 id="${sections[index++]}">${text}</h2>`);
content = content.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, '</table></div>');
// Paper-by-paper evidence is available in compact, accessible disclosure cards.
content = content.replace(/<h3>(.*?)<\/h3>([\s\S]*?)(?=<h3>|<h2(?:\s|>)|$)/g,
  (_, title, body) => `<details class="paper"><summary>${title}</summary><div class="paper-body">${body}</div></details>`);
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'source_manifest.json'), 'utf8'));
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const sources = manifest.map(ref => {
  const pdf = ref.items.flatMap(x => x.attachments)[0];
  const href = pdf ? new URL(`file:///${pdf.path.replaceAll('\\', '/')}`).href : 'bc_reading_20261025/C3_kreissig_itg10.pdf';
  return `<li><b>${ref.id}</b> <a href="${escape(href)}">${escape(ref.title)}</a> <span>${pdf ? pdf.pages : 4} 页</span></li>`;
}).join('\n');
content += `<details class="source-list"><summary>打开19篇本地原文（含C3作者版）</summary><ul>${sources}</ul></details>`;
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>低空盲多源定位｜十九篇全文后的研究收敛</title>
<style>
:root{--ink:#193b43;--muted:#5c7175;--teal:#14746b;--paper:#f5f5ef;--line:#d7e2dd;--soft:#e8f1eb}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--teal);text-underline-offset:4px;overflow-wrap:anywhere}a:hover{color:#0e4944}a:focus-visible,summary:focus-visible{outline:3px solid #b07e32;outline-offset:4px}.wrap{max-width:1120px;width:calc(100% - 64px);margin:auto}header{padding:32px 0 26px}header .eyebrow{color:var(--teal);font-size:13px;letter-spacing:.12em}h1{font-size:clamp(30px,4vw,48px);line-height:1.3;max-width:900px;margin:20px 0}header p{max-width:830px;font-size:18px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0 8px}.stat{background:#fff;border:1px solid var(--line);border-radius:14px;padding:19px 23px}.stat b{font-size:28px;display:block}.stat span{font-size:13px;color:var(--muted)}nav{position:sticky;top:0;z-index:2;background:#f5f5eff5;border-block:1px solid var(--line);backdrop-filter:blur(8px)}nav .wrap{display:flex;gap:22px;overflow-x:auto;white-space:nowrap;padding-block:13px}nav a{text-decoration:none;font-size:14px;color:var(--ink)}main{padding-bottom:36px}main>h2{font-size:27px;line-height:1.45;margin:52px 0 18px;padding-top:12px;border-top:2px solid var(--teal)}p{margin:14px 0}main>p:first-child{color:var(--muted);font-size:13px}main>p:nth-child(2){font-size:20px;background:var(--soft);padding:22px 26px;border-radius:14px}blockquote{margin:20px 0;padding:18px 25px;background:var(--soft);border-left:4px solid var(--teal);border-radius:0 12px 12px 0}blockquote p{margin:0;font-size:19px}strong{font-weight:650}ol,ul{padding-left:26px}li+li{margin-top:12px}main>ol>li{padding:5px 0}table{width:100%;border-collapse:collapse;font-size:14px;background:white}th,td{text-align:left;vertical-align:top;padding:15px 18px;border-bottom:1px solid var(--line);min-width:160px}th{background:var(--soft)}tr:last-child td{border:0}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:13px;margin:18px 0}details{background:#fff;border:1px solid var(--line);border-radius:12px;margin:12px 0}summary{padding:18px 22px;cursor:pointer;font-weight:650;line-height:1.6}summary:hover{background:#f0f6f1;border-radius:12px}.paper-body{padding:0 23px 18px;border-top:1px solid var(--line)}.paper-body p:first-child{font-size:15px}.paper-body p{margin:16px 0 0}.source-list ul{padding:0 30px 20px 45px;font-size:14px}.source-list span{color:var(--muted);white-space:nowrap}pre{overflow-x:auto;max-width:100%;padding:22px;border:1px solid var(--line);border-radius:13px;background:white;font:14px/1.9 Consolas,"Microsoft YaHei",monospace}code{overflow-wrap:anywhere;font-size:13px}footer{border-top:1px solid var(--line);padding:26px 0 40px;color:var(--muted);font-size:13px}.pill{display:inline-block;border-radius:6px;background:var(--soft);color:var(--teal);font-size:12px;padding:4px 9px;margin-right:7px}.pill.amber{background:#f6ecd7;color:#855c28}#motion{scroll-margin-top:90px}
@media(max-width:680px){.wrap{width:calc(100% - 32px)}header{padding-top:23px}.stats{gap:8px}.stat{padding:14px 11px}.stat b{font-size:23px}.stat span{font-size:12px}nav .wrap{gap:19px}main>h2{font-size:23px;margin-top:38px}main>p:nth-child(2),blockquote{padding:17px 19px;font-size:18px}summary{padding:16px}th,td{padding:13px;min-width:145px}pre{font-size:12px;padding:15px}}
@media print{nav{position:static}details:not([open])>.paper-body{display:block}summary{list-style:none}.wrap{width:100%}.stats{break-inside:avoid}body{background:white}h2{break-after:avoid}}
</style></head><body>
<header class="wrap"><div class="eyebrow">RESEARCH ADMISSION · 2026.10.25</div><h1>低空盲多源定位<br>十九篇全文后的研究收敛</h1>
<p>从“哪些站可靠”，收敛到“哪些源—路径解释可信”。逐篇依据、先验边界与下一步判定放在同一份报告里。</p>
<span class="pill">第二、三批精读完成</span><span class="pill">主题分类已核验</span><span class="pill amber">条件机制准入</span>
<div class="stats"><div class="stat"><b>10篇</b><span>本轮137页全文精读与关键页核查</span></div><div class="stat"><b>19篇</b><span>前三批累计284页，不含缺失独立补充</span></div><div class="stat"><b>0次</b><span>新定位实验；不提前宣布方法成立</span></div></div></header>
<nav aria-label="报告导航"><div class="wrap"><a href="#path">审查路径</a><a href="#papers">逐篇结论</a><a href="#decision">研究目标</a><a href="#method">解决思路</a><a href="#next">后续计划</a><a href="#classification">Zotero分类</a><a href="#unknown">未确定事项</a><a href="#records">原始记录</a></div></nav>
<main class="wrap">${content}</main>
<footer class="wrap">源文件、证据记录和执行建议均可从本页打开。阅读状态只记在文档中。论文、定位器、实验配置和原始结果没有改动。<br>旧链接兼容：<a id="motion" href="#decision">三维与运动</a> · <a id="structure-record" href="#papers">结构近邻</a> · <a id="integrity" href="#unknown">来源完整性</a></footer>
</body></html>`;
fs.writeFileSync(path.join(out, '准入报告.html'), html, 'utf8');
console.log(JSON.stringify({sections:index,papers:manifest.length,html_bytes:Buffer.byteLength(html)}));
