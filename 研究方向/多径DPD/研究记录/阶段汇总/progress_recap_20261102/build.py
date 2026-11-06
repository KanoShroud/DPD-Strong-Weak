"""Read saved evidence; produce a standalone reader, without new experiments."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
files={
 'repair':'unknown_order_20261030/feasible_repair_review_20261030_221307_859401/result.json',
 'unseen':'unknown_order_20261030/feasible_unseen_review_20261030_221713_265293/result.json',
 'risk':'risk_feasibility_20261030/run_20261030_222922_024770/result.json',
 'selection':'risk_feasibility_20261030/selection_20261030_223207_927491/result.json',
}
data={k:json.loads((BASE/p).read_text(encoding='utf-8')) for k,p in files.items()}
assert all(d['all_checks_pass'] for d in data.values())
labels={'L0':'无反射','L1':'每链路1条反射','L2':'每链路2条反射','L4':'每链路4条反射','MIX':'不同链路0/1/2/4条混合','HIGH':'4条，总反射功率超过直达','L3':'未见的每链路3条反射'}
rows={s['condition']:s for d in (data['repair'],data['unseen']) for s in d['summaries']}
table=''
bars=''
for c in ('L0','L1','L2','L4','MIX','HIGH','L3'):
    r=rows[c];a,b=r['los_rmse'],r['final_rmse']
    basis='新IQ、独立种子' if c=='L3' else ('1条重跑＋1条等价复用' if c=='HIGH' else '2条等价复用')
    table+=f'<tr><th scope="row">{labels[c]}</th><td>{a:.3f}</td><td>{b:.3f}</td><td>{"、".join(map(str,r["orders"]))}</td><td>{basis}</td></tr>'
    bars+=f'''<div class="bar-group"><div class="bar-label">{labels[c]}</div><div class="bar-pair" aria-label="{labels[c]}，LOS {a:.3f}米，多径参照{b:.3f}米"><div class="bar-line"><span class="bar los" style="width:{a/1.5*100:.3f}%"></span><span>{a:.3f}</span></div><div class="bar-line"><span class="bar mp" style="width:{b/1.5*100:.3f}%"></span><span>{b:.3f}</span></div></div></div>'''
text=(HERE/'template.html').read_text(encoding='utf-8').replace('__TABLE__',table).replace('__BARS__',bars)
assert '__TABLE__' not in text and '__BARS__' not in text
(HERE/'最近几轮研究进展.html').write_text(text,encoding='utf-8')
manifest={'page_date':'2026-11-02','evidence_through':'2026-10-30','new_localization_runs':0,
          'sources':{k:{'path':str(BASE/p),'sha256':hashlib.sha256((BASE/p).read_bytes()).hexdigest()} for k,p in files.items()}}
(HERE/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('已生成',HERE/'最近几轮研究进展.html')
