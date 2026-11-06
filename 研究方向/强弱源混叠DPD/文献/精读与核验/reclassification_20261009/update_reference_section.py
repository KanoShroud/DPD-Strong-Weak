from pathlib import Path
import json,re
root=Path('F:/SourceCount_DPD_E2E');out=root/'outputs_e2e/literature/reclassification_20261009'
c=json.loads((out/'catalog_final.json').read_text(encoding='utf-8'))
mapping=json.loads((out/'number_mapping.json').read_text(encoding='utf-8'))
old=(out/'端到端多源定位_文献综述与方案建议.md.before').read_text(encoding='utf-8')
def renumber(s):
 s=re.sub(r'R(\d{2,3})[–—－-]R?(\d{2,3})(?!\d)',lambda m:'、'.join(f'R{i:02}' for i in range(int(m[1]),int(m[2])+1)),s)
 return re.sub(r'(?<![A-Za-z0-9_-])R\d{2,3}(?!\d)',lambda m:mapping.get(m[0],m[0]),s)
def cell(s,limit=None):
 s=re.sub(r'\s+',' ',s).replace('|','／').strip()
 if limit and len(s)>limit:s=s[:limit].rstrip('，；。')+'…'
 return s
def entry(r,short=False):
 title=cell(r['标题']);link=r.get('线上链接','');label=f'[{title}]({link})' if link else title
 why=cell(r.get('与本研究的关系',r.get('一句话概括','')),130 if not short else 85)
 evidence='待全文核验' if r['旧编号'].startswith('L') or r['旧编号']=='R66' else '见CSV精读记录'
 if r['旧编号']=='R57':evidence='作者公开稿；库内无PDF'
 if r['旧编号'].startswith('N'):evidence='全文已读'
 return f"| {r['编号']} | {label} | {renumber(why)} | {evidence} |"
parts=['## 2. 文献表与阅读顺序','',
 '共116项，按核心18篇、条件补充14篇、历史归档84项连续编号。编号采用三位数字R001—R116；新旧编号和Zotero条目对应见[编号对照](参考文献编号对照.csv)。历史实验与阶段文档保留原编号，查阅时使用该对照。精读记录和迁移依据见[文献精读结果](文献精读结果.csv)。','',
 '### 2.1 01 核心文献：R001—R018','',
 '用于当前问题定位、关键方法设计和主要对照选择。核心不等于全部需要复现。','',
 '| 编号 | 论文标题与入口 | 当前用途 | 阅读证据 |','| --- | --- | --- | --- |']
parts += [entry(r) for r in c['rows'] if r['分类']=='01 核心文献']
parts += ['', '### 2.2 02 条件补充：R019—R032','',
 '用于基础引文、特定先验、备选求解及后续扩展；满足相应条件后启用。','',
 '| 编号 | 论文标题与入口 | 当前用途 | 阅读证据 |','| --- | --- | --- | --- |']
parts += [entry(r,True) for r in c['rows'] if r['分类']=='02 条件补充']
parts += ['', '### 2.3 03 历史归档：R033—R116','',
 '归档不表示论文质量低，也不表示它不影响未来主张。以下只保留题录，完整阅读记录保存在CSV；原10条延伸线索已纳入编号，但未读全文的状态保持不变。','']
for group in ['01 传统方法与变体','02 学习与联合任务','03 特殊观测与场景','04 检测关联与基础理论']:
 rs=[r for r in c['rows'] if r['归档子类']==group]
 parts += [f'<details><summary>{group}（{len(rs)}项，{rs[0]["编号"]}—{rs[-1]["编号"]}）</summary>','', '| 编号 | 论文标题与入口 |','| --- | --- |']
 for r in rs:
  title=cell(r['标题']);url=r.get('线上链接','');label=f'[{title}]({url})' if url else title
  parts.append(f'| {r["编号"]} | {label} |')
 parts += ['', '</details>', '']
start=old.index('## 2. 文献表与阅读顺序');end=old.index('## 3. 已有方法的能力')
new=renumber(old[:start])+'\n'.join(parts)+'\n'+renumber(old[end:])
a=new.index('编号表共101项');b=new.index('\n\n',a)
new=new[:a]+'编号表共116项（含1项专利），其中核心18篇、条件补充14篇、历史归档84项。原10条延伸线索已纳入编号，未完成全文评估的状态保持不变。R026仍仅有摘要；R005可核对作者公开稿，但当前Zotero条目没有PDF。阅读版本、正文与补充材料的取得情况分别记录。'+new[b:]
(out/'reference_section_validation.json').write_text(json.dumps({'count':len(c['rows']),'mapped_citations':len(mapping),'research_chapters_substantively_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')
(root/'端到端多源定位_文献综述与方案建议.md').write_text(new,encoding='utf-8')
print('Reference chapter reclassified; citation IDs elsewhere mechanically mapped. Research chapters not substantively rewritten.')
