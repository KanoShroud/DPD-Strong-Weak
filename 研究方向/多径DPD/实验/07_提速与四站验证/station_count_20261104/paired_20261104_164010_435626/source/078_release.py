"""Saved evidence replay and source-identity freeze, not a new performance run."""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

HERE=Path(__file__).resolve().parent
PILOT=HERE/'pilot_20261104_160749_988104'
PAIR=HERE/'paired_20261104_160907_648761'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks={};sources={}
    for folder in (PILOT,PAIR):
        data=read(folder/'results.json')
        checks[f'{folder.name}/complete']=data['complete']
        checks[f'{folder.name}/identity']=read(folder/'postflight.json')['unchanged']
        for path,digest in read(folder/'identity.json')['hashes'].items():
            checks[f'{folder.name}/{path}']=sha(path)==digest
            if Path(path).suffix=='.py':sources[path]=digest
        for p in folder.glob('*_audit.json'):
            checks.update({f'{folder.name}/{p.name}/{k}':bool(v) for k,v in read(p).items()})
        checks[f'{folder.name}/position_fidelity']=all(p['fidelity'] for p in data['pairs'])
    data=read(PAIR/'results.json');rows=data['rows'];summary=[]
    for cond in ('L0','L2','L3'):
        for method in ('HYBRID','GATED'):
            rr=[r for r in rows if r['case'].startswith(cond+'_') and r['method']==method]
            summary.append(dict(condition=cond,method=method,n=len(rr),
                RMSE=float(np.sqrt(np.mean([e**2 for r in rr for e in r['errors_m']]))),
                mean_s=float(np.mean([r['seconds'] for r in rr])),max_s=max(r['seconds'] for r in rr),
                under3=sum(r['seconds']<=3 for r in rr),under5=sum(r['seconds']<=5 for r in rr)))
        old,new=summary[-2:]
        checks[cond+'/RMSE_fidelity']=new['RMSE']<=old['RMSE']+.01
    checks['all_normal']=all(r['normal'] for r in rows)
    checks['all_supports_same']=all(p['same_links'] for p in data['pairs'])
    assert all(checks.values())
    result=dict(summary=summary,max_position_delta_m=max(p['position_delta_max_m'] for p in data['pairs']),
                checks=checks,check_count=len(checks),all_pass=True,
                elapsed_algorithm_s=sum(r['seconds'] for r in rows)+sum(r['seconds'] for r in read(PILOT/'results.json')['rows']))
    (HERE/'配对汇总与审计.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    freeze=dict(version='MP_DPD_GATED_V1',status='NUMERICALLY_FROZEN_NOT_ALL_CASES_WITHIN_5S',
        evidence=str(PAIR),sources=sources,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),blas_threads=2,pg_tolerance=1e-7,time_based_stop=False,
        scope='same original IQ, one geometry, L0/L2/L3; no new geometry or high reflection power validation',
        formal_large_batch_released=False)
    (HERE/'冻结清单.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,max_position_delta_m=result['max_position_delta_m'],
                          checks=len(checks),source_files=len(sources)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
