"""Complete-IQ paired regression, unchanged original estimator and auditable gate."""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gated_solver',HERE/'solver.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
fc,ref,rg=g.fc,g.ref,g.rg


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['pilot','paired'],required=True)
    ap.add_argument('--tags',nargs='+')
    args=ap.parse_args()
    tags=args.tags or (['L0_R00','L2_R00','L3_R00','L3_R05'] if args.mode=='pilot' else ['L0_R01']+[f'{c}_R{i:02d}' for i in range(1,8) for c in ['L2','L3']])
    out=HERE/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    print('结果目录',out,flush=True)
    hashes=rg.preflight(out,tags)
    contract=fc.load(rg.FORMAL/'contract.json')['contract']
    extra=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))
    extra += [Path(contract['conditions'][c]['bank']) for c in sorted(set(t.split('_')[0] for t in tags))]
    extra += list(g.NAT.glob('*.py'))
    hashes.update({str(p.resolve(strict=True)):ref.sha(p) for p in extra})
    snap=out/'source';snap.mkdir()
    for i,p in enumerate(hashes):
        if Path(p).suffix in ('.py','.md'):shutil.copy2(p,snap/f'{i:03d}_{Path(p).name}')
    fc.write(out/'identity.json',dict(hashes=hashes,command=sys.argv))
    rows=[];pairs=[]
    try:
        for i,tag in enumerate(tags):
            iq=rg.inputs(tag);results={}
            for method in (['HYBRID','GATED'] if i%2==0 else ['GATED','HYBRID']):
                print('START',tag,method,flush=True)
                result=g.estimate(iq,method)
                fc.write(out/f'{tag}_{method}.json',result)
                checks=g.h.runner.audit(result,iq)
                f,y,n,_=fc.cal.spectral(iq)
                rr,jj=ref.evaluate(np.asarray(result['final']['x']),f,y,result['paths'])
                x=np.asarray(result['final']['x']);pg=float(np.max(abs(x-ref.project(x-jj.T@rr,n))))
                checks['final_pg']=bool(pg<=1e-7)
                checks['normal']=result['normal']
                fc.write(out/f'{tag}_{method}_audit.json',checks)
                # Truth is consumed by this evaluator only after the IQ-only estimator returns.
                with np.load(contract['conditions'][tag.split('_')[0]]['bank'],allow_pickle=False) as z:truth=z['p']
                errors=np.asarray(ref.error(result['final']['p'],truth))
                row=dict(case=tag,method=method,seconds=result['seconds'],RMSE=float(np.sqrt(np.mean(errors**2))),
                    errors_m=errors,K=len(result['paths']),PG=pg,normal=result['normal'],stop=result['stop'],
                    frontend=result['frontend_seconds'],fit_seconds=sum(v['fit']['seconds'] for v in result['history'] if 'fit' in v),
                    search_seconds=sum(v['search']['seconds'] for v in result['history']),
                    gate_returns=sum(v['fit'].get('gate_returns',0) for v in result['history'] if 'fit' in v),
                    audit=all(checks.values()))
                rows.append(row);results[method]=result
                fc.write(out/'results.json',dict(rows=rows,pairs=pairs,complete=False))
                print(row,flush=True)
                assert all(checks.values()),checks
            old,new=results['HYBRID'],results['GATED']
            difference=np.asarray(ref.error(new['final']['p'],old['final']['p']))
            pair=dict(case=tag,position_delta_max_m=float(difference.max()),same_K=len(new['paths'])==len(old['paths']),
                      same_links=new['paths']==old['paths'],J_delta=new['final']['cost']-old['final']['cost'],
                      seconds_old=old['seconds'],seconds_new=new['seconds'],fidelity=bool(difference.max()<=.01))
            pairs.append(pair);fc.write(out/'results.json',dict(rows=rows,pairs=pairs,complete=False))
            assert pair['fidelity'],pair
        fc.write(out/'results.json',dict(rows=rows,pairs=pairs,complete=True))
    finally:
        same=all(ref.sha(p)==v for p,v in hashes.items())
        fc.write(out/'postflight.json',dict(unchanged=same))
        assert same


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
