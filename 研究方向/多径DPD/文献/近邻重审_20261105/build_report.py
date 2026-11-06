"""Offline reader for the bibliography matrix and migration proposal."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def esc(s):
    return html.escape(str(s), quote=True)


def main():
    rows = json.loads((ROOT / '文献矩阵.json').read_text(encoding='utf-8'))
    stats = json.loads((ROOT / 'matrix_summary.json').read_text(encoding='utf-8'))
    body = []
    for r in rows:
        official = 'https://doi.org/' + r['doi'] if r['doi'] else r['url']
        links = f'<a href="{esc(official)}" target="_blank" rel="noopener">正式书目 / DOI ↗</a>'
        if r.get('fulltext_url'):
            links += f' · <a href="{esc(r["fulltext_url"])}" target="_blank" rel="noopener">公开全文线索 ↗</a>'
        need = '0' if r['local_pdf'] else '1'
        body.append(f'''<article class="paper" data-grade="{r['grade']}" data-need="{need}">
<div class="paperhead"><span class="tag">{r['id']} · {esc(r['grade_name'].split('｜')[0])}</span><span class="state">{esc(r['library_status'])}</span></div>
<h3>{esc(r['title'])}</h3><p class="muted">{esc(r['venue'])} · {esc(r['year'])} · {esc(r.get('priority',''))}</p>
<p>{esc(r['information_conditions'])}</p><p><b>借鉴点：</b>{esc(r['idea'])}</p>
<details><summary>阅读范围与库内记录</summary><p>{esc(r['evidence_scope'])}</p><p>Zotero键：{esc(', '.join(r['zotero_keys']) or '未匹配')}</p><p>{esc(r['action'])}</p></details>
<p class="links">{links}</p></article>''')
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双源盲多径DPD · 文献重审与整理</title>
<style>
:root{--ink:#193544;--muted:#657782;--green:#006c68;--paper:#fff;--bg:#f2f5f5;--line:#d9e3e5}*{box-sizing:border-box}body{margin:0;font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;color:var(--ink);background:var(--bg)}main{max-width:1120px;margin:auto;padding:40px 24px 70px}a{color:var(--green);text-underline-offset:4px}h1{font-size:36px;line-height:1.35;letter-spacing:-.6px;margin:12px 0}h2{font-size:25px;margin:32px 0 16px}h3{font-size:18px;line-height:1.5;margin:12px 0}.eyebrow{color:var(--green);font-size:13px;letter-spacing:1.5px}.lead{font-size:19px;max-width:900px}.muted,.state{color:var(--muted);font-size:14px}.notice{padding:18px 22px;border-left:4px solid #bd8e40;background:#fff8e9;border-radius:8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.card,.panel,.paper{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px}.card strong{display:block;font-size:32px;color:var(--green)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.paperhead{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.tag{font-size:12px;background:#e6f2ef;padding:2px 9px;border-radius:20px;color:var(--green)}nav{display:flex;flex-wrap:wrap;gap:20px;margin:22px 0}.toolbar{display:flex;flex-wrap:wrap;gap:10px;background:#e8eeee;padding:15px;border-radius:12px;margin-bottom:14px}input,select,button{font:inherit;padding:9px 12px;border:1px solid #b9cecf;border-radius:7px;background:white;color:var(--ink);min-width:0}input{flex:1;min-width:180px}button{cursor:pointer}.paper p{margin:10px 0}.links{font-size:14px}details{margin-top:12px}summary{cursor:pointer;color:var(--green)}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;background:#f5f8f8;padding:16px;border-radius:9px;font-size:14px}li{margin:6px 0}.listlinks{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0}footer{border-top:1px solid var(--line);padding-top:20px;margin-top:35px;font-size:13px;color:var(--muted)}.paper[hidden]{display:none}@media(max-width:700px){main{padding:24px 14px 40px}h1{font-size:27px}.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.card,.panel,.paper{padding:17px}.lead{font-size:17px}.state{font-size:12px}input{width:100%}}
</style></head><body><main><div class="eyebrow">2026.11.05 · LITERATURE & REPRODUCIBILITY</div><h1>先把近邻与信息条件核清，<br>再决定怎样改算法。</h1><p class="lead">Xia原文已回读；文献分级库已建立；历史目录已盘点。新定位实验和文件搬移均未启动。</p>
<nav><a href="#xia">Xia核对</a><a href="#papers">分级文献 / 下载</a><a href="#archive">整理方案</a><a href="#next">下一步</a></nav>
<div class="cards"><div class="card"><strong>@@TOTAL@@</strong>候选书目条目</div><div class="card"><strong>@@NEW@@</strong>相对旧46表新增</div><div class="card"><strong>@@PDF@@</strong>库内PDF可打开</div><div class="card"><strong>@@NEED@@</strong>待补库 / 全文</div></div>
<div class="notice"><b>完成边界：</b>不是68篇全部精读完毕。Xia2021四页已逐项重读并视觉核对，其余逐项注明历史阅读、当前定向核读或摘要。检索扩展了覆盖，不保证绝无遗漏，也未证明创新风险已消除。</div>
<section id="xia"><h2>Xia：没有发现一项被漏掉的“万能开关”</h2><div class="grid"><div class="panel"><h3>原文确实写了什么</h3><p>五个同步站、参考站原点；BPSK，1MHz采样、200kbps符号率、2000点；1000初始粒子；五反射表内总功率比约0.254。</p><p>图2与图3(a)纵轴是10log10(RMSE)，主要统计第一个目标。完整双源RMSE和缺失输出要另外报告。</p></div><div class="panel"><h3>哪些仍不能假装知道</h3><p>绝对损耗与SNR怎样联用、成形与分数延迟、复相位统计、G计数和空簇输出等仍有未明示项。聚类严格“增量&lt;ε”与ε=0的组合也需公开解释。</p><p>旧实现属于公开信息重建；后来的大粒子数尺度适配，不等于已经复现作者原曲线。</p></div></div><p><a href="Xia原文逐项复核.md">查看全文逐项复核：正文、公式、图表与前作 →</a></p></section>
<section id="papers"><h2>分级文献与下载清单</h2><p>四级是场景距离，阅读优先级另列。一级也不等于全部同信息可比；有些四级理论论文必须优先读。</p><p class="muted">Zotero：多源定位研究｜文献导航 → 03 双源盲多径DPD｜近邻分级。复用37条现有记录，不改旧分类，不添加阅读状态。</p>
<div class="listlinks"><a href="分级文献矩阵.md">完整Markdown矩阵</a><a href="待导入与全文缺口.md">33项补全文清单（分三批）</a><a href="search_expansion/扩展检索记录与判读.md">检索路径与限制</a><a href="文献矩阵.json">结构化JSON</a></div>
<div class="toolbar"><input id="query" placeholder="搜索题名、先验、方法或年份" aria-label="搜索文献"><select id="grade" aria-label="分级"><option value="">全部级别</option><option value="A">一级 · 核心场景</option><option value="B">二级 · 关联与盲分离</option><option value="C">三级 · 不同信息条件</option><option value="D">四级 · 选模与理论</option></select><label><input id="missing" type="checkbox" style="width:auto;min-width:0"> 只看待补全文</label></div><p id="counter" class="muted"></p><div class="grid" id="catalog">@@PAPERS@@</div></section>
<section id="archive"><h2>文件整理：先解开依赖，再搬移证据</h2><div class="grid"><div class="panel"><h3>不是一个普通的输出文件夹</h3><p>旧research_admission_20261027有14,582文件、约10.54GiB；1,826个Python文件含大量冻结快照。7个一键入口硬编码旧树，当前源码也跨轮导入。</p><p>直接按日期剪切会破坏导入、账本身份和断点续跑。相同内容的快照也不能直接去重删除。</p></div><div class="panel"><h3>建议的分离</h3><pre>统一模型代码/research/mp_dpd/  可执行源码
运行入口/多径DPD/            一键入口
docs/mp_dpd/                 文献、时间线、阶段记录
outputs_e2e/mp_dpd/          数据、运行、冻结证据</pre><p>先建时间线与映射 → 抽离当前必需代码 → 固定输入等价核验 → 整包复制与哈希验收 → 切换入口。旧账本不改，清退另行审批。</p></div></div><p><a href="archive_plan/集中整理与解耦方案.md">查看详细搬移方案、依赖实例、验收与回滚 →</a></p></section>
<section id="next"><h2>下一步明确停在哪里</h2><ol><li>优先补齐第一批12项，特别是Xia信道来源、源径关联、SAGE自适应阶数和盲解混条件。</li><li>核心近邻全文齐备后，逐篇按统一模板核对模型、先验、理论、完整流程和仿真，再形成最终综述与建议。</li><li>随后完成有限Xia锚定与公平比较合同，不按结果选择实现约定，不为让自家方法胜出而改基线。</li><li>目录搬移方案等待审批。当前0/1/2反射和其他场景设定不变，不启动480/780/3120或640条实验。</li></ol><a href="阅读与研究推进安排.md">完整状态与逐篇精读模板 →</a></section>
<footer>来源：本轮Zotero只读快照、原文PDF、公开出版社/作者资料、历史证据与目录盘点。由AI辅助整理并核对，获取全文、机器解析、语义精读和实验验证分开记录。<br><a href="执行协议.md">执行协议</a> · <a href="audit_v2.json">交付与分类核查</a></footer>
<script>const papers=[...document.querySelectorAll('.paper')];function filter(){const q=document.querySelector('#query').value.toLowerCase(),g=document.querySelector('#grade').value,need=document.querySelector('#missing').checked;let n=0;papers.forEach(p=>{const show=(!g||p.dataset.grade===g)&&(!need||p.dataset.need==='1')&&p.textContent.toLowerCase().includes(q);p.hidden=!show;n+=show});document.querySelector('#counter').textContent=`显示 ${n} / ${papers.length} 篇`;}document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',filter));filter();</script></main></body></html>'''
    for key, value in {'TOTAL': stats['total'], 'NEW': stats['new_to_old_matrix'], 'PDF': stats['local_pdf'], 'NEED': stats['need_fulltext'], 'PAPERS': ''.join(body)}.items():
        doc = doc.replace('@@' + key + '@@', str(value))
    (ROOT / '文献重审与整理报告.html').write_text(doc, encoding='utf-8')


if __name__ == '__main__':
    main()
