"""Finite-IQ vs diagonal frequency model, explicitly oracle diagnostic only."""
from pathlib import Path
from datetime import datetime
import json
import time

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, STATIONS, C, B, sha, write, signature
from structural_probe import response
from position_reduction import position_jac, channel

IQ=ROOT/'finite_iq_clues_20261029_204430_528859'
INIT=ROOT/'robust_geometry_20261029_210612_452075'
CHANNEL=ROOT/'joint_structure_20261029_132004_058171'
PROTOCOL=ROOT/'后端模型衔接协议.md'
FS=50e6
N=8192


def vector(g,d):
    los=g[:,:,0]/g[:1,:,0]
    ratio=g[:,:,1:]/g[:,:,:1]
    return np.r_[los[1:].real.ravel(),los[1:].imag.ravel(),ratio.real.ravel(),
                 ratio.imag.ravel(),np.diff(d,axis=2).ravel()]


def project(p,v,f,y):
    g,d=channel(v)
    h,_=response(p,d,g,f)
    coef=np.einsum('fqm,fm->fq',np.linalg.pinv(h,rcond=1e-12),y)
    fitted=np.einsum('fmq,fq->fm',h,coef)
    return y-fitted,fitted,h


def fit(p,v,f,y):
    # Fixed true channel ONLY for a named oracle diagnostic. No blind claim.
    cache=[None,None,None]
    norm=np.linalg.norm(y)
    def state(x):
        if cache[0] is None or not np.array_equal(cache[0],x):
            pp=x.reshape(2,2)*(C/B)
            r,_,_=project(pp,v,f,y)
            r=np.sqrt(2)/norm*np.r_[r.real.ravel(),r.imag.ravel()]
            j=position_jac(v,pp,f,y)
            cache[:]=[x.copy(),r,j]
        return cache[1],cache[2]
    start=time.perf_counter()
    opt=least_squares(lambda x:state(x)[0],p.ravel()/(C/B),jac=lambda x:state(x)[1],
         bounds=(-3000/(C/B),3000/(C/B)),method='trf',x_scale='jac',
         max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-7)
    return dict(p=opt.x.reshape(2,2)*(C/B),cost=opt.cost,nfev=opt.nfev,optimality=opt.optimality,
                status=opt.status,message=opt.message,seconds=time.perf_counter()-start)


def main():
    files=[Path(__file__),PROTOCOL,ROOT/'geometry_association_gate.py',ROOT/'structural_probe.py',
           ROOT/'position_reduction.py',ROOT/'solver_reliability.py',IQ/'result.json',INIT/'result.json']
    for layout in [0,1]:
        for rep in [0,5]:
            files += [IQ/f'L{layout}_R{rep}_components.npz',CHANNEL/f'L{layout}_R{rep}_P2.npz']
            for path in ['LOS','MP']:
                files += [INIT/f'L{layout}_R{rep}_{path}_AB_snr6_CC_ABS.json']
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    assert all(Path(p).is_relative_to(ROOT) for p in hashes)
    old=json.loads((IQ/'protocol.json').read_text(encoding='utf-8'))['hashes']
    for file in files:
        if str(file.resolve()) in old:
            assert sha(file)==old[str(file.resolve())]
    out=ROOT/('backend_contract_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for file in files[:6]:
        (out/'source'/file.name).write_bytes(file.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,oracle=True,not_a_blind_algorithm=True,
       gain='true constant complex channel, held fixed only for the oracle diagnostic',
       waveform='unknown and profiled per frequency',fft_normalization='ortho',window='rectangular original record'))
    freq=np.fft.fftfreq(N,1/FS)
    mask=abs(freq)<=B/2
    f=freq[mask]
    rows=[]
    checks={}
    begin=time.perf_counter()
    print('OUTPUT',out,flush=True)
    for layout in [0,1]:
        for rep in [0,5]:
            data=dict(np.load(IQ/f'L{layout}_R{rep}_components.npz'))
            truth=data['truth']
            known=dict(np.load(CHANNEL/f'L{layout}_R{rep}_P2.npz'))
            assert np.array_equal(truth,known['p'])
            noise=np.fft.fft(data['noise'],axis=1,norm='ortho')[:,mask].T
            for path in ['LOS','MP']:
                tag=f'L{layout}_R{rep}_{path}'
                gain=known['g'].copy()
                if path=='LOS':
                    gain[:,:,1:]=0
                v=vector(gain,known['d'])
                clean_iq=(data['los']+(data['reflected'] if path=='MP' else 0)).sum(axis=0)
                y=np.fft.fft(clean_iq,axis=1,norm='ortho')[:,mask].T
                initrow=json.loads((INIT/(tag+'_AB_snr6_CC_ABS.json')).read_text(encoding='utf-8'))
                checks[tag+'/identity']=signature(clean_iq+data['noise'])==initrow['iq_sha']
                pp=np.array(initrow['methods']['GEO_ALL']['p'])
                ii,jj=linear_sum_assignment(np.sum((pp[:,None]-truth[None])**2,axis=2))
                p=np.empty_like(pp)
                p[jj]=pp[ii]  # Explicit oracle label alignment, never a blind start rule.
                r,consistent,h=project(truth,v,f,y)
                rn,_,_=project(truth,v,f,noise)
                direct=np.array([yy-hh@np.linalg.lstsq(hh,yy,rcond=1e-12)[0] for hh,yy in zip(h,y)])
                checks[tag+'/direct_lstsq']=bool(np.allclose(direct,r,atol=1e-10,rtol=1e-9))
                rc,_,_=project(truth,v,f,consistent)
                checks[tag+'/consistent_zero']=bool(np.linalg.norm(rc)/np.linalg.norm(consistent)<1e-12)
                results={}
                for kind,base in [('finite_iq',y),('model_consistent_control',consistent)]:
                    for label,nn in [('clean',0),('snr6',noise)]:
                        obs=base+nn
                        if not rows and kind=='finite_iq' and label=='clean':
                            step=1e-5
                            scale=C/B
                            def res(x):
                                rr,_,_=project(x.reshape(2,2)*scale,v,f,obs)
                                return np.sqrt(2)/np.linalg.norm(obs)*np.r_[rr.real.ravel(),rr.imag.ravel()]
                            x=p.ravel()/scale
                            fd=np.column_stack([(res(x+np.eye(4)[k]*step)-res(x-np.eye(4)[k]*step))/(2*step) for k in range(4)])
                            analytic=position_jac(v,p,f,obs)
                            checks['position_jacobian']=bool(np.linalg.norm(fd-analytic)/np.linalg.norm(fd)<1e-6)
                        fitout=fit(p,v,f,obs)
                        fitout['errors_m']=np.linalg.norm(fitout['p']-truth,axis=1)
                        results[kind+'/'+label]=fitout
                rec=dict(tag=tag,oracle=True,initial_error_m=np.linalg.norm(p-truth,axis=1),
                   relative_model_residual=float(np.linalg.norm(r)/np.linalg.norm(y)),
                   model_to_noise_residual_energy=float(np.linalg.norm(r)**2/np.linalg.norm(rn)**2),
                   observed_projected_noise_energy=float(np.linalg.norm(rn)**2),
                   expected_projected_noise_energy=float(data['variance']*len(f)*3),
                   frequency_count=len(f),results=results)
                rows.append(rec)
                write(out/(tag+'.json'),rec)
                np.savez_compressed(out/(tag+'.npz'),f=f,y=y,noise=noise,model_consistent=consistent,
                                    initial=p,truth=truth,true_channel_vector=v)
                assert all(checks.values()),checks
                print(tag,'model_residual',round(rec['relative_model_residual'],4),
                      'ratio_noise',round(rec['model_to_noise_residual_energy'],3),
                      'finite_noisy_error',np.round(results['finite_iq/snr6']['errors_m'],3),flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-begin,
                                not_a_blind_localization_comparison=True))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
