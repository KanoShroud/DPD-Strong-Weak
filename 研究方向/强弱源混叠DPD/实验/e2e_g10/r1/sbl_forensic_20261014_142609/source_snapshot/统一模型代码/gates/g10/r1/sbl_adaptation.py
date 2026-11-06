"""Focused SBL cause checks; saved IQ, fixed evidence, no truth in optimizers."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize
from ..physics import atom
from .physics import Model
from .sbl import statistics
from .runtime import fresh, preflight, load_scene, read, write, sha, finish_audit, Ledger
from .scenes import observation, digest
from .config import BASE


def evidence_problem(m, points, gamma, ids, covariance, scale=1000.):
    """Exactly the same native objective; vectorized active coordinates."""
    old=np.asarray(points)[ids].copy()
    active=gamma[ids]
    a=m.response(old)
    fixed=covariance-(a*active[None,None])@a.mH

    def fun(flat):
        m.check()
        p=torch.tensor((np.asarray(flat)*scale).reshape(-1,2),dtype=torch.float64,
                       device=m.device,requires_grad=True)
        a=atom(p,m.f,m.stations)
        c=fixed+(a*active[None,None])@a.mH
        chol=torch.linalg.cholesky(c)
        invx=torch.cholesky_solve(m.x[...,None],chol).squeeze(-1)
        v=(2*torch.log(chol.diagonal(dim1=1,dim2=2).real).sum(-1)
           +(m.x.conj()*invx).sum(-1).real).mean()
        g=torch.autograd.grad(v,p)[0]*scale
        m.counts['frequency_response_points']+=len(p)*m.n
        return float(v.detach().cpu()),g.detach().cpu().numpy().ravel()
    return old,fun


def forensic():
    out=fresh('sbl_forensic')
    preflight(out,dict(stage='SBL_FIXED_NATIVE_OBJECTIVE_FORENSIC',truth_in_optimizer=False))
    clock=Ledger(out); rows=[]
    for clean,folder in [(True,BASE/'native_probe_20261014_135430'),
                         (False,BASE/'native_probe_20261014_140311')]:
        r=read(folder/'rows/3_SBL.json'); saved=read(folder/'maps/3_SBL.json')
        assert sha(folder/'rows/3_SBL.json')==read(folder/'final_audit_report.json')['rows']['3_SBL.json']
        scene=load_scene(folder,'3_0_0')
        iq=scene['signal'][1].copy() if clean else observation(scene,'weak')[0]
        assert digest(iq)==r['input_sha']
        m=Model(iq,clock.check); pp=np.asarray(saved['points'])
        gamma=torch.tensor(saved['gamma'],dtype=torch.float64,device='cuda')
        stats=statistics(m,pp,gamma,256,True)
        covariance=torch.cat(stats['covs']); rank=np.argsort(-np.asarray(saved['gamma']))
        assert abs(stats['evidence']-r['result']['events'][-1]['native_objective'])<1e-8
        for count in (1,2,4,8):
            ids=rank[:count]
            old,fun=evidence_problem(m,pp,gamma,ids,covariance)
            initial=old.ravel()/1000
            v,g=fun(initial)
            # A directional finite difference checks the actual coordinate objective.
            direction=np.arange(1,initial.size+1,dtype=float); direction/=np.linalg.norm(direction)
            eps=1e-6
            fd=(fun(initial+eps*direction)[0]-fun(initial-eps*direction)[0])/(2*eps)
            assert np.isclose(fd,g@direction,rtol=2e-4,atol=2e-6),(fd,g@direction)
            for radius in (20.,100.):
                bounds=[(max(-1000.,p-radius)/1000,min(1000.,p+radius)/1000) for p in old.ravel()]
                start=time.perf_counter()
                res=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                             options=dict(maxiter=100,ftol=1e-12,gtol=1e-8,maxls=30))
                pos=(res.x*1000).reshape(-1,2)
                row=dict(clean=clean,active_count=count,radius_m=radius,initial_evidence=v,
                         final_evidence=float(res.fun),positions=pos.tolist(),old_positions=old.tolist(),
                         active_gamma=gamma[ids].cpu().tolist(),iterations=int(res.nit),
                         optimizer_success=bool(res.success),message=str(res.message),
                         error_of_highest_gamma_m=float(np.linalg.norm(pos[0]-np.array(r['truth'])[0])),
                         derivative_analytic=float(g@direction),derivative_fd=float(fd),
                         seconds=time.perf_counter()-start,input_sha=r['input_sha'])
                rows.append(row); write(out/'report.json',dict(rows=rows))
                print({k:row[k] for k in ('clean','active_count','radius_m','final_evidence','error_of_highest_gamma_m')},flush=True)
        del m,covariance,stats,gamma
        torch.cuda.empty_cache()
    clock.flush();finish_audit(out)
    print('OUT='+str(out),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['forensic'])
    args=p.parse_args()
    if args.stage=='forensic':forensic()


if __name__=='__main__':main()
