from pathlib import Path
import csv,json,re,urllib.request

ROOT=Path('F:/SourceCount_DPD_E2E')
OUT=ROOT/'outputs_e2e/literature/reclassification_20261009'
c=json.loads((OUT/'catalog_final.json').read_text(encoding='utf-8'))
def fetch(kind):
 items=[]
 for start in range(0,5000,100):
  with urllib.request.urlopen(f'http://127.0.0.1:23119/api/users/0/{kind}?format=json&limit=100&start={start}') as r:batch=json.load(r)
  items+=batch
  if len(batch)<100:return items
 raise RuntimeError('Pagination incomplete')
after_items=fetch('items');after_cols=fetch('collections')
(OUT/'library_after.json').write_text(json.dumps(after_items,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'collections_after.json').write_text(json.dumps(after_cols,ensure_ascii=False,indent=2),encoding='utf-8')
before_items=json.loads((OUT/'library_before.json').read_text(encoding='utf-8'))
before_cols=json.loads((OUT/'collections_before.json').read_text(encoding='utf-8'))
def scope(cols):
 keys={'PFQ2G77F'}
 for _ in range(5):keys|={x['key'] for x in cols if x['data'].get('parentCollection') in keys}
 return keys
oldscope=scope(before_cols);newscope=scope(after_cols)
before={x['key']:x['data'] for x in before_items};after={x['key']:x['data'] for x in after_items}
assert before.keys()==after.keys(),'Library item set changed'
for key,data in before.items():
 assert set(data.get('collections',[]))-oldscope==set(after[key].get('collections',[]))-newscope,('External collection membership changed',key)
 for field in set(data)|set(after[key]):
  if field not in ['version','dateModified','collections','tags']:assert data.get(field)==after[key].get(field),(key,field)
 assert {x['tag'] for x in data.get('tags',[])} <= {x['tag'] for x in after[key].get('tags',[])},('Original tags lost',key)
names={x['key']:x['data']['name'] for x in after_cols}
leaves={'WVVMS2NG','EAIQDTIB','B8WVY34X','P9YSV683','K5KTMCUE','WAV2VCSC'}
counts={k:0 for k in leaves}
for row in c['rows']:
 data=after[row['ZoteroKey']];own=set(data['collections'])&newscope
 assert len(own)==1 and own<=leaves,(row['编号'],own)
 key=next(iter(own));counts[key]+=1
 assert names[key]==(row['归档子类'] or row['分类'])
 assert 'DPD综述:'+row['编号'] in {x['tag'] for x in data['tags']}
top=[x['data']['name'] for x in after_cols if x['data'].get('parentCollection')=='PFQ2G77F']
assert sorted(top)==['01 核心文献','02 条件补充','03 历史归档']
for key in ['WVVMS2NG','EAIQDTIB']:assert not any(x['data'].get('parentCollection')==key for x in after_cols)
with (ROOT/'文献精读结果.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==116
assert [r['编号'] for r in rows]==[f'R{i:03}' for i in range(1,117)]
assert len({r['旧编号'] for r in rows})==len({r['ZoteroKey'] for r in rows})==116
for r,t in zip(rows,c['rows']):
 for field in ['编号','旧编号','标题','分类','归档子类','ZoteroKey','DOI']:assert r[field]==t[field]
with (ROOT/'参考文献编号对照.csv').open(encoding='utf-8-sig',newline='') as f:mrows=list(csv.DictReader(f))
assert len(mrows)==116
for r,m in zip(rows,mrows):assert r['编号']==m['编号'] and r['旧编号']==m['旧编号'] and r['ZoteroKey']==m['ZoteroKey']
with (ROOT/'文献精读结果_详版.csv').open(encoding='utf-8-sig',newline='') as f:detail=list(csv.DictReader(f))
assert len(detail)==37
for filename in ['文献精读结果.csv','文献精读结果_详版.csv']:
 assert (ROOT/filename).read_bytes().startswith(b'\xef\xbb\xbf')==(OUT/(filename+'.before')).read_bytes().startswith(b'\xef\xbb\xbf'),'UTF-8 BOM changed'
byold={r['旧编号']:r for r in rows}
for r in detail:assert r['编号']==byold[r['旧编号']]['编号'] and r['标题']==byold[r['旧编号']]['标题']
md=(ROOT/'端到端多源定位_文献综述与方案建议.md').read_text(encoding='utf-8')
listed=re.findall(r'^\| (R\d{3}) \|',md,re.M)
assert listed==[r['编号'] for r in rows],('Reference table order',listed)
assert set(re.findall(r'R\d{3}(?!\d)',md))<=set(listed)
notes=json.loads((OUT/'review_notes.json').read_text(encoding='utf-8'))
assert len(notes)==32 and {r['old'] for r in notes}=={r['旧编号'] for r in rows[:32]}
report={'status':'PASS','references':116,'core':18,'supplement':14,'archive':84,'archive_subcategories':{names[k]:v for k,v in counts.items() if k not in ['WVVMS2NG','EAIQDTIB']},'original_library_items_preserved':len(before),'external_collections_unchanged':True,'metadata_and_existing_tags_preserved':True,'historical_csv_rows':37,'review_records':32,'review_scope':'Recheck models, methods, experiments and migration against prior reading records','abstract_only':'R026','author_manuscript_online':'R005','formal_experiments_run':False}
(OUT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
