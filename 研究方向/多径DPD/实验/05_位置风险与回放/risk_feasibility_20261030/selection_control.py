"""Independent evaluation of a selected fixed linear estimator: math only."""
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'unknown_order_20261030'))
from reference import sha,write  # noqa: E402


def main():
    prior=Path(json.loads((HERE/'latest.json').read_text(encoding='utf-8'))['path'])
    d=json.loads((prior/'result.json').read_text(encoding='utf-8'));assert d['all_checks_pass']
    old=json.loads((prior/'preflight.json').read_text(encoding='utf-8'))['hashes']
    assert all(sha(p)==h for p,h in old.items())
    files=[Path(__file__),HERE/'独立评价控制协议.md',prior/'result.json']
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    out=HERE/('selection_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    (out/'selection_control.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json',dict(hashes=hashes,training_seed=2026103032,evaluation_seed=2026103033))
    rng=np.random.default_rng(2026103032);test=np.random.default_rng(2026103033)
    q,_=np.linalg.qr(rng.normal(size=(12,12)));a=q[:,:4];h=rng.normal(size=(4,4))*.3
    sigma=.2;rows=[];checks={}
    for spacing in (.2,1.):
        c=a@h+spacing*q[:,4:8];g=np.column_stack((a,c));t=np.linalg.pinv(g)[:4]
        models=[np.linalg.pinv(g[:,:4+k])[:4] for k in (0,2,4)]
        for amp in (0.,.25,1.):
            theta=np.r_[[.2,-.1,.3,-.4],amp*np.array([1.,-.8,.6,-.4])];mu=g@theta
            train=mu[:,None]+sigma*rng.normal(size=(12,20000));held=mu[:,None]+sigma*test.normal(size=(12,20000))
            outputs=[]
            for y in (train,held):
                scores=[];losses=[]
                for l in models:
                    scores.append(np.sum(((l-t)@y)**2,axis=0)+2*sigma**2*np.sum(l*t)-sigma**2*np.sum(t*t))
                    losses.append(np.sum((l@y-theta[:4,None])**2,axis=0))
                outputs.append((np.array(scores),np.array(losses)))
            chosen=np.argmin(outputs[0][0],axis=0);idx=np.arange(20000)
            e=outputs[1][0][chosen,idx]-outputs[1][1][chosen,idx];se=float(e.std(ddof=1)/np.sqrt(len(e)))
            key=f'{spacing}/{amp}'
            checks[key]=abs(float(e.mean()))<=6*se+1e-10
            original=next(r for r in d['linear'] if r['spacing']==spacing and r['amplitude']==amp)
            checks[key+'/training_reproduced']=abs(float(outputs[0][0][chosen,idx].mean())-original['selected_score'])<1e-12
            rows.append(dict(spacing=spacing,amplitude=amp,train_score=float(outputs[0][0][chosen,idx].mean()),
                train_loss=float(outputs[0][1][chosen,idx].mean()),held_score=float(outputs[1][0][chosen,idx].mean()),
                held_loss=float(outputs[1][1][chosen,idx].mean()),held_paired_error=float(e.mean()),held_paired_se=se))
    checks['identities']=all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in old.items())
    write(out/'result.json',dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),rf_algorithm=False))
    write(HERE/'selection_latest.json',dict(path=str(out)))
    print(json.dumps(rows,ensure_ascii=False,indent=2));print('完成',len(checks),all(checks.values()),out)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
