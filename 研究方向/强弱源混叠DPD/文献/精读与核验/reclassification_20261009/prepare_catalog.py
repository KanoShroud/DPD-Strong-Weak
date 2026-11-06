from pathlib import Path
import csv, json, re

ROOT = Path('F:/SourceCount_DPD_E2E')
OUT = ROOT / 'outputs_e2e/literature/reclassification_20261009'
c = json.loads((OUT/'catalog_draft.json').read_text(encoding='utf-8'))
library = {x['key']:x['data'] for x in json.loads((OUT/'library_before.json').read_text(encoding='utf-8'))}
learning = {12,13,14,15,16,17,18,21,22,25,26,37,38,39,40,41,42,43,49,56,67,71,84}
special = {3,6,19,23,24,29,30,36,47,48,54,62,68,69,70,76,80,82,83,85,90,97,99,100}
background = {5,9,10,11,20,28,31,45,46,50,55}
groups = ['01 传统方法与变体','02 学习与联合任务','03 特殊观测与场景','04 检测关联与基础理论']
extra_groups = {'N1':0,'N2':0,'N5':0,'L01':2,'L02':2,'L03':3,'L04':3,'L05':3,'L06':0,'L07':1,'L08':1,'L09':2,'L10':1}
new_notes = {
 'N1':('多阵列少快拍下的IAA功率谱定位。','各站独立拟合后累加功率；跨站时延相位在功率中抵消。','原式(12)、(21)—(24)在各向同性M=1且无位置幅度先验时不保留位置辨识信息；需重建跨站联合模型。','全文；Signal Processing 161:203–213，§4式(12)、(21)—(24)，§5。'),
 'N2':('用黎曼距离拟合协方差，提高多阵列DPD精度。','建模忽略跨站时延，位置由阵列方向响应承载。','原模型M=1时A(p)不再含位置；单源准则的双峰实验不能替代多源联合求解。','全文；ICASSP 2024:8521–8525，§2式(1)—(6)，§4，§5。'),
 'N3':('用路径损耗上下界约束未知链路，改善单源DPD。','MM优化每个位置下的增益；位置仍按网格搜索。','保留多频时延，单天线适配有基础；需验证传播先验误差，不能视为位置搜索加速。','全文；IEEE SPL 32:1241–1245，§II式(8)—(12)，§III，§IV。'),
 'N4':('用位置约束时延更新，再由时延更新位置，折中精度和成本。','未知波形、标量接收站；含静态TDOA分支，单源近似EM和多初值。','可借鉴几何约束候选与闭式位置更新；多源需要保留多个相关峰并核验归属，原局部置信区不能直接保证弱源保留。','全文；DSP 96:102600，§2—3，§4.1式(24)—(34)，§4.2.2，§5.1—5.2。'),
 'N5':('用潜在距离向量构造EM，把位置更新化为闭式计算。','标量站、单源、已知发射波形；估计位置、发射时刻和复增益。','原算法没有估计未知混合源波形；与现保留的宽带多源EM和非合作单源方法相比新增用途有限。','全文；Signal Processing 133:32–39，§2式(3)—(7)，§3式(25)，§5表1；RMSE截去3%尾部。'),
}
for r in c['rows']:
 old=r['旧编号']; data=library[r['ZoteroKey']]
 if old in c['core']:r['分类']='01 核心文献';r['归档子类']='';rank=c['core'].index(old)
 elif old in c['supplement']:r['分类']='02 条件补充';r['归档子类']='';rank=c['supplement'].index(old)
 else:
  n=int(old[1:]) if old.startswith('R') else None
  gi=extra_groups.get(old,1 if n in learning else 2 if n in special else 3 if n in background else 0)
  r['分类']='03 历史归档';r['归档子类']=groups[gi];rank=gi*1000+(n if n is not None else 200+int(old[1:]))
 r['_rank']=rank
 r['阅读等级']=r['分类']
 r['DOI']=data.get('DOI','')
 r['线上链接']='https://doi.org/'+r['DOI'] if r['DOI'] else data.get('url','')
 if old.startswith(('N','L')):
  creators=data.get('creators',[])
  authors='、'.join(z.get('lastName',z.get('name','')) for z in creators[:3])
  r['作者年份']=f"{authors}；{data.get('date','')}；{data.get('publicationTitle',data.get('proceedingsTitle',''))}"
  if old.startswith('N'):
   summary,method,limit,evidence=new_notes[old]
   r.update({'一句话概括':summary,'核心方法':method,'主要局限':limit,'关键障碍与依据':limit,'证据定位':evidence+' DOI:'+r['DOI']+'；Zotero '+r['ZoteroKey'],'论文结论与支持边界':limit,'与本研究的关系':summary+' '+limit,'迁移判断':'可借鉴，需按原模型条件适配' if old in ['N3','N4'] else '归档；不列为当前直接对照'})
  else:
   r.update({'一句话概括':data.get('abstractNote','') or '未完成全文评估。','证据定位':'Zotero书目和现有摘要；未完成全文评估。','迁移判断':'待全文核验；当前归档','与本研究的关系':'延伸线索；不作为当前缺口已被排除的证据。'})
c['rows'].sort(key=lambda r:(r['分类'],r['_rank']))
mapping={}
for i,r in enumerate(c['rows'],1):r['编号']=f'R{i:03}';r['序号']=str(i);mapping[r['旧编号']]=r['编号']
def renumber(text):
 def expand(m):
  a,b=int(m.group(1)),int(m.group(2))
  if a<=b:return '、'.join(f'R{x:02}' for x in range(a,b+1))
  return m.group(0)
 text=re.sub(r'R(\d{2,3})[–—－-]R(\d{2,3})(?!\d)',expand,text)
 return re.sub(r'(?<![A-Za-z0-9_-])R\d{2,3}(?!\d)',lambda m:mapping.get(m.group(),m.group()),text)
original_fields=list(csv.DictReader((OUT/'文献精读结果.csv.before').open(encoding='utf-8-sig',newline='')).fieldnames)
fields=['序号','编号','旧编号','分类','归档子类','ZoteroKey','DOI','线上链接']+[k for k in original_fields if k not in ['序号','编号']]
matrix=[fields]
for r in c['rows']:
 values=[]
 for f in fields:
  v=str(r.get(f,''))
  if f not in ['序号','编号','旧编号','DOI','线上链接','ZoteroKey']:v=renumber(v)
  values.append(v)
 matrix.append(values)
(OUT/'csv_matrix.json').write_text(json.dumps(matrix,ensure_ascii=False),encoding='utf-8')
(OUT/'catalog_final.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'number_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
print('Count',len(c['rows']))
for group in ['01 核心文献','02 条件补充']+groups:
 rs=[r for r in c['rows'] if group in [r['分类'],r['归档子类']]]
 print(group,len(rs),rs[0]['编号'] if rs else '',rs[-1]['编号'] if rs else '')
for r in c['rows'][:32]:print(r['编号'],r['旧编号'],r['ZoteroKey'])
