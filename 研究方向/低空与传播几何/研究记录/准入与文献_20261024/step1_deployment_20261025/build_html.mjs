import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {marked} from 'file:///C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js';
const root=path.dirname(fileURLToPath(import.meta.url));
const parent=path.dirname(root), main=path.join(parent,'准入报告.html');
let previous=fs.readFileSync(main,'utf8');
const archive=path.join(parent,'准入报告_部署核验前.html');
if(!fs.existsSync(archive)) fs.copyFileSync(main,archive);
const css=previous.match(/<style>([\s\S]*?)<\/style>/)[1];
const extra=`main{padding-top:16px}main>p:nth-child(2){font-size:14px;background:none;padding:0}h1{font-size:clamp(27px,4vw,38px)}.top{padding-top:25px}.intro{padding:20px;border-left:4px solid #177e80;background:#e6efea;border-radius:0 12px 12px 0}.table-wrap{overflow:auto}td{min-width:120px}summary{cursor:pointer;font-weight:650;padding:18px 0}details{border-top:1px solid #d4dfdc;margin-top:24px}code{overflow-wrap:anywhere}footer{padding:24px 0 50px;color:#627575}`;
for(const [name,title] of [['部署与多径歧义报告','部署与多径歧义核验'],['推导与文献对应','几何推导与文献依据']]){
  let body=marked.parse(fs.readFileSync(path.join(root,name+'.md'),'utf8'));
  body=body.replaceAll('<table>','<div class="table-wrap"><table>').replaceAll('</table>','</table></div>');
  // Link the mathematical note to its rendered companion while preserving raw sources elsewhere.
  body=body.replace('href="推导与文献对应.md"','href="推导与文献对应.html"');
  const intro=name==='部署与多径歧义报告'?'已知哪四站直达：合理部署能解决一类双解。<br>不知道哪四站直达：错误路径组合仍可能对应另一个位置。':'从距离差方程与论文原文出发，区分几何双解、路径误关联和评分假峰。';
  const html=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title}</title><style>${css}\n${extra}</style></head><body><div class="wrap top"><a href="../准入报告.html">← 文献准入总报告</a> · <a href="部署与多径歧义报告.html">本轮执行报告</a><p class="intro">${intro}</p></div><main class="wrap">${body}</main><footer class="wrap">2026-10-25 · 本页说明模型与机制核验，不是性能排行榜。原始结果和失败尝试均保留。</footer></body></html>`;
  fs.writeFileSync(path.join(root,name+'.html'),html,'utf8');
}
previous=previous.replace(/<section id="step1-latest"[\s\S]*?<\/section>/,'');
const banner=`<section id="step1-latest" style="padding:22px 26px;margin:24px 0;background:#e6efea;border-left:4px solid #177e80;border-radius:0 12px 12px 0"><h2 style="margin:0 0 12px;font-size:24px">最新：部署与多径路径歧义核验完成</h2><p>同平面部署＋上方空间在非退化、正确直达集合条件下排除镜像双解；但未知直达集合仍有不同位置产生相同IQ的反例。反例的短绕行与路径强弱条件已公开，不能解释为真实场景失败率。</p><p><a href="step1_deployment_20261025/部署与多径歧义报告.html"><strong>查看执行过程、路径假位置与原文依据 →</strong></a> · <a href="step1_four_los_20261025/四直达核验报告.html">上一轮四直达</a></p><p style="font-size:13px;color:#526d66">以下是文献准入历史报告；最新边界以上述链接为准。尚未启动正式定位性能比较。</p></section>`;
previous=previous.replace('<main class="wrap">',`<main class="wrap">${banner}`);
fs.writeFileSync(main,previous,'utf8');
console.log('Updated deployment report, mathematical note and admission banner.');
