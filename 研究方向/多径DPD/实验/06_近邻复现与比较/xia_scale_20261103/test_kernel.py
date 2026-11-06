"""Correctness and engineering timing only, with frozen IQ and no new location run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from kernel import CrossSpectrumKernel

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE.parent/'xia_pairing/recheck'))
import xia_checked as xia  # noqa: E402
import xia_reference as original  # noqa: E402

FS=50e6
N=8192
B=12.5e6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_summary(direct,fast):
    error=abs(direct-fast)
    scale=float(abs(direct).max())
    mask=abs(direct)>max(scale*1e-10,np.finfo(float).tiny)
    return dict(absolute_max=float(error.max()),
                normalized_absolute=float(error.max()/scale),
                nonzero_relative_max=float((error[mask]/abs(direct[mask])).max()),
                nonzero_relative_floor_fraction=1e-10)


def check_decode(points,direct,fast):
    a=xia.decode(points,direct,2,400.)
    b=xia.decode(points,fast,2,400.)
    for key in ('status','positions','retained','modal_bin','histogram','trace','first_position'):
        assert a.get(key)==b.get(key),(key,a,b)
    assert np.array_equal(xia.threshold(direct)['keep'],xia.threshold(fast)['keep'])
    return dict(status=a['status'],same_all_selected_positions=True,
                same_keep_histogram_cluster_trace=True)


def main(output=None):
    if output is not None:
        output=Path(output).resolve()
        if not output.is_relative_to(HERE) or output.exists():
            raise ValueError('Output must be a new file inside the kernel task directory')
    rng=np.random.default_rng(2026110319)
    f0=np.fft.fftfreq(N,1/FS)
    f=f0[abs(f0)<=B/2]
    stations=rng.uniform(-3000,3000,(5,2))
    points=rng.uniform(-3000,3000,(1000,2))
    y=rng.normal(size=(5,len(f)))+1j*rng.normal(size=(5,len(f)))
    # Avoid artificial near-zero synthetic reference, without changing real IQ.
    y[0]=np.exp(1j*rng.uniform(-np.pi,np.pi,len(f)))
    k=CrossSpectrumKernel(y,f,N,FS)
    direct,components=original.score(points,stations,y,f,N)
    fast=k.scores(points,stations)
    ce=error_summary(components,k.components(points,stations))
    se=error_summary(direct,fast)
    assert ce['normalized_absolute']<2e-11 and se['normalized_absolute']<5e-11
    checks={'synthetic_components':ce,'synthetic_score':se,
            'synthetic_decode':check_decode(points,direct,fast)}
    for label,pp,ss in [('translated',points+5e5,stations+5e5),
                        ('far_points',points*1e6,stations),
                        ('other_stations',points,rng.uniform(-1e6,1e6,(5,2)))]:
        dc=original.score(pp,ss,y,f,N)[0]
        fc=k.scores(pp,ss)
        err=error_summary(dc,fc)
        assert err['normalized_absolute']<5e-9,(label,err)
        checks[label]=err
    delays=rng.uniform(-100*k.period,100*k.period,(33,5))
    cross=original.cross_spectrum(y)
    dc=np.sum(cross[None]*np.exp(2j*np.pi*delays[:,:,None]*f[None,None]),axis=-1)/N
    err=error_summary(dc,k.components_from_delays(delays))
    assert err['normalized_absolute']<2e-9,err
    checks['arbitrary_signed_100_period_delays']=err
    ny=y.copy(); ny[0,0]=0
    try:
        CrossSpectrumKernel(ny,f,N,FS)
    except ValueError as exc:
        assert 'REFERENCE_SPECTRAL_ZERO' in str(exc)
        checks['zero_reference_rejected']=True
    else:
        raise AssertionError('Zero reference silently accepted')
    formal=BASE/'risk_calibration_20261102/formal_20261102_181736_622970'
    native=BASE/'complete_neighbor_20261103/native_20261103_164443_255193'
    contract=json.loads((formal/'contract.json').read_text(encoding='utf-8'))['payload']['contract']
    # Station coordinates are public contract parameters in the unchanged module.
    sys.path.insert(0,str(BASE))
    from geometry_association_gate import STATIONS
    paths=[HERE/'kernel.py',Path(__file__),Path(original.__file__),Path(xia.__file__),formal/'contract.json']
    paths += [formal/'cases'/f'{condition}_R00'/'input.npz' for condition in ('L0','L2','L3','HIGH')]
    paths += [native/f'{condition}_R00'/'decision_before_truth.json' for condition in ('L0','L2','L3','HIGH')]
    hashes={str(p):sha(p) for p in paths}
    real=[]
    kernel=None
    yy=None
    for condition in ('L0','L2','L3','HIGH'):
        with np.load(formal/'cases'/f'{condition}_R00'/'input.npz',allow_pickle=False) as z:
            assert set(z.files)=={'iq'}
            iq=z['iq']
        yy=np.fft.fft(iq,axis=1,norm='ortho')[:,abs(f0)<=B/2]
        kernel=CrossSpectrumKernel(yy,f,N,FS)
        seed=20261029223+10000*20
        pts=np.random.default_rng(seed).uniform(-3000,3000,(1000,2))
        ds=xia.scores(pts,STATIONS,yy,f,N)
        fs=kernel.scores(pts,STATIONS)
        err=error_summary(ds,fs)
        assert err['normalized_absolute']<5e-11 and err['nonzero_relative_max']<2e-8,err
        decoded=check_decode(pts,ds,fs)
        saved=json.loads((native/f'{condition}_R00'/'decision_before_truth.json').read_text(encoding='utf-8'))['payload']
        saved_rounds=[]
        for step in saved['output']['steps']:
            sp=np.asarray(step['points'])
            sd=xia.scores(sp,STATIONS,yy,f,N)
            sf=kernel.scores(sp,STATIONS)
            serr=error_summary(sd,sf)
            assert serr['normalized_absolute']<5e-11
            assert np.allclose(sd,step['scores'],rtol=2e-12,atol=0)
            saved_rounds.append(dict(round=step['round'],error=serr,
                decode=check_decode(sp,sd,sf)))
        # Small components use independent direct sums; no true coordinates loaded.
        _,dcomp=original.score(pts[:37],STATIONS,yy,f,N)
        comp=error_summary(dcomp,kernel.components(pts[:37],STATIONS))
        assert comp['normalized_absolute']<2e-11
        real.append(dict(condition=condition,score_error=err,components_error=comp,
                         decode=decoded,saved_rounds=saved_rounds,kernel=kernel.metadata()))
    assert contract['conditions']  # Existence only; no banks/truth are opened.
    benchmark_points=rng.uniform(-3000,3000,(8192,2))
    t=time.perf_counter(); ds=xia.scores(benchmark_points,STATIONS,yy,f,N); direct_s=time.perf_counter()-t
    t=time.perf_counter(); fs=kernel.scores(benchmark_points,STATIONS); fast_s=time.perf_counter()-t
    err=error_summary(ds,fs)
    assert err['normalized_absolute']<5e-11 and err['nonzero_relative_max']<2e-8
    big=rng.uniform(-3000,3000,(1000000,2))
    t=time.perf_counter(); big_scores=kernel.scores(big,STATIONS); million_s=time.perf_counter()-t
    assert big_scores.shape==(1000000,) and np.isfinite(big_scores).all()
    # Independent spot comparison does not generate/select a location output.
    idx=np.arange(0,1000000,10000)
    spot=xia.scores(big[idx],STATIONS,yy,f,N)
    spot_err=error_summary(spot,big_scores[idx])
    assert spot_err['normalized_absolute']<5e-11
    assert all(sha(p)==h for p,h in hashes.items())
    report=dict(passed=True,checks=checks,real_records=real,
        benchmark=dict(points8192=err,direct_seconds=direct_s,fast_seconds=fast_s,
            preparation_seconds=kernel.preparation_seconds,
            million_fast_seconds=million_s,million_spot_error=spot_err),
        hashes=hashes,hashes_unchanged=True,truth_loaded=False,new_localization_runs=0,
        python=sys.version,numpy=np.__version__,threads=2)
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if output is not None:
        with output.open('x',encoding='utf-8') as stream:
            stream.write(text+'\n')
    print(text,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.output)
