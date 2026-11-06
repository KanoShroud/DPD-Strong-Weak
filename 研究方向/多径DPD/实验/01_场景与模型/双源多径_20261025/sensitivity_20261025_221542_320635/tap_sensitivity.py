"""Oracle-only noise/window attribution; no additional practical output."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiment import ROOT, preprocess, write, sha
from model import C, STATIONS, pack, Model
from two_path import TwoPath


def main():
    parser=argparse.ArgumentParser();parser.add_argument('ground');args=parser.parse_args()
    folder=Path(args.ground).resolve(strict=True)
    out=ROOT/('sensitivity_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'two_path.py',ROOT/'model.py',ROOT/'experiment.py',ROOT/'噪声与窗口归因合同.md']
    identity={str(p):sha(p) for p in files}
    for p in files:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json',{'identity':identity,'input':str(folder),'oracle_only':True,'threads':2})
    rows=[];started=time.perf_counter()
    for file in sorted(folder.glob('input_*.npz')):
        identity[str(file)]=sha(file)
        with np.load(file) as data:
            clean=(data['los']+data['echo']).sum(axis=0);noise=data['noise'];truth=data['truth'];gain=data['direct_gain']
            direct_clean=data['los'].sum(axis=0)
        y,f=preprocess(clean,True);ny,_=preprocess(noise,True)
        distance=np.linalg.norm(truth[:,None,:]-STATIONS[None,:,:],axis=-1)
        virtual=truth.copy();virtual[:,2]*=-1
        reflected=np.linalg.norm(virtual[:,None,:]-STATIONS[None,:,:],axis=-1)
        base=pack(truth,gain/gain[:,:1]);rho=.7*distance/reflected;delay=(reflected-distance)/C
        x=np.r_[base,rho.ravel(),np.full(10,np.pi),(delay*1e6).ravel()]
        a=np.empty((len(f),5,2),complex)
        for q in range(2):
            for m in range(5):
                a[:,m,q]=gain[q,m]/gain[q,0]*np.exp(-2j*np.pi*f*(distance[q,m]-distance[q,0])/C)*(1-rho[q,m]*np.exp(-2j*np.pi*f*delay[q,m]))
        coeff=np.stack([np.linalg.lstsq(aa,yy,rcond=None)[0] for aa,yy in zip(a,y.T)])
        matched=np.einsum('fmk,fk->fm',a,coeff).T
        mismatch=float(np.linalg.norm(y-matched)/np.linalg.norm(y))
        # Effective local position sensitivity after eliminating gain/tap directions.
        def sensitivity(j):
            nuisance=j[:,6:]
            u,s,_=np.linalg.svd(nuisance,full_matrices=False)
            u=u[:,s>s[0]*1e-10]
            residual=j[:,:6]-u@(u.T@j[:,:6])
            return np.linalg.svd(residual,compute_uv=False).tolist()
        _,jm=TwoPath(matched,f,2).residual_jac(x)
        yd,_=preprocess(direct_clean,True)
        _,jd=Model(yd,f).residual_jac(base,2)
        from model import evaluate
        experiments=[]
        for label,obs in [('actual_noiseless',y),('matched_noiseless',matched),('matched_same_noise',matched+ny)]:
            model=TwoPath(obs,f,2);initial,_=model.residual_jac(x)
            fit=model.fit(x)
            experiments.append({'label':label,'initial_loss':float(initial@initial),'fit':fit,'evaluation':evaluate(fit['positions'],truth)})
            print(json.dumps({'input':file.name,'label':label,'initial_loss':float(initial@initial),'rmse':experiments[-1]['evaluation']['rmse_m'],'stop':fit['message']}),flush=True)
        row={'input':str(file),'input_sha256':sha(file),'relative_window_model_residual':mismatch,
             'position_singular_values_LOS':sensitivity(jd),'position_singular_values_two_path':sensitivity(jm),
             'sensitivity_units':'normalized_residual_per_1000m_parameter_after_nuisance_projection_not_CRB',
             'experiments':experiments}
        rows.append(row);write(out/(file.stem+'.json'),row)
    write(out/'result.json',{'identity':identity,'input_source_unchanged':all(sha(Path(p))==h for p,h in identity.items()),
                            'seconds':time.perf_counter()-started,'oracle_only':True,'rows':rows})
    print('COMPLETED '+str(out),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
