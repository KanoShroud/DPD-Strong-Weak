import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
const root=path.dirname(fileURLToPath(import.meta.url));
const {marked}=await import(pathToFileURL('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js'));
const escape=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const documents=[
 ['structure-record','2026-10-25补读 · 两篇结构近邻及准入修订','两篇结构近邻精读与准入修订.md'],
 ['full-record','完整准入报告 · 决策、最小模型与停止条件','准入报告.md'],
 ['ma-record','Ma2020全文精读 · 公式、算法、三维与运动','Ma2020精读与运动建模判断.md'],
 ['links-record','链路近邻核验 · 逐源受损、多径与隐藏先验','link_nearest_audit.md'],
 ['backup-record','备选方向核验 · 间歇发射与通信受限','backup_directions_audit.md'],
 ['execution-record','执行与检索记录 · 做过什么、没做什么','执行与检索记录.md'],
 ['scope-record','用户确认的研究边界与准入协议','研究边界与准入审查.md'],
 ['first-record','第一批模型核对 · 保留历史阅读范围','首批模型核对记录.md'],
];
const hashes={};
const panels=documents.map(([id,title,file])=>{
 const bytes=fs.readFileSync(path.join(root,file)),text=bytes.toString('utf8');hashes[file]=crypto.createHash('sha256').update(bytes).digest('hex');
 // Keep source equations readable without requiring a CDN math renderer.
 const math=[];const prepared=text.replace(/\\\[([\s\S]*?)\\\]/g,(_,v)=>{math.push(v.trim());return `\n\nMATHBLOCK${math.length-1}END\n\n`;});
 let body=marked.parse(prepared,{gfm:true});
 body=body.replace(/<p>MATHBLOCK(\d+)END<\/p>/g,(_,i)=>`<pre class="math" aria-label="公式的LaTeX表示">${escape(math[Number(i)])}</pre>`);
 return `<details id="${id}"><summary>${title}</summary><div class="detail reading"><p class="small"><a href="${file}">打开可编辑Markdown原记录</a></p>${body}</div></details>`;
}).join('\n');
const audit=JSON.parse(fs.readFileSync(path.join(root,'source_audit.json'),'utf8'));
const additions=JSON.parse(fs.readFileSync(path.join(root,'followup_20261025/sources.json'),'utf8'));
for(const r of additions){
 const digest=crypto.createHash('sha256').update(fs.readFileSync(r.path)).digest('hex');
 if(digest!==r.sha256)throw new Error('New PDF identity changed');
 if(!audit.records.some(x=>x.sha256===digest))audit.records.push({path:r.path,sha256:digest,
  registries:['followup_20261025/sources.json'],structural_verdict:r.structural_verdict,pages:r.total_pages});
}
audit.unique_local_pdfs=audit.records.length;
fs.writeFileSync(path.join(root,'source_audit_20261025.json'),JSON.stringify(audit,null,2),'utf8');
let html=fs.readFileSync(path.join(root,'report_template.html'),'utf8')
 .replace('@@DOCUMENTS@@',panels).replace('@@PDFCOUNT@@',String(audit.unique_local_pdfs))
 .replace('@@PASSCOUNT@@',String(audit.records.filter(r=>r.structural_verdict==='PASS').length))
 .replace('@@BUILDNOTE@@','更新：2026-10-25。已补读TMC/TWC两篇全文；历史检索与本次补读范围分别保留。没有新增定位实验。');
if(/@@[A-Z]+@@/.test(html))throw new Error('Unfilled template token');
fs.writeFileSync(path.join(root,'准入报告.html'),html,'utf8');
fs.writeFileSync(path.join(root,'report_manifest.json'),JSON.stringify({inputs:hashes,html_sha256:crypto.createHash('sha256').update(html).digest('hex'),scope:'Literature admission only; no localization experiment'},null,2),'utf8');
console.log(JSON.stringify({documents:documents.length,local_pdfs:audit.unique_local_pdfs,html_bytes:Buffer.byteLength(html)}));
