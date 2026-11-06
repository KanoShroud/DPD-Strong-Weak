"""Public finite Fourier extension capacity; explicit oracle geometry diagnostic."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

from output_json import dumps

ROOT=Path(__file__).resolve().parent
FRONT=ROOT/'frontend_20261025_204005_991746'
C=299792458.


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def public_matrix(paths,t,bandwidth,factor):
    period=factor*len(t)*(t[1]-t[0])
    k=int(np.floor(bandwidth/2*period+1e-9))
    freq=np.arange(-k,k+1)/period
    V=np.exp(2j*np.pi*freq[:,None]*t[None,:])
    H=np.zeros((2,5,len(freq)),complex)
    for q,m,length,gain in paths:
        H[q,m]+=gain*np.exp(-2j*np.pi*freq*length/C)
    A=np.concatenate([(H[q,:,:,None]*V[None,:,:]).transpose(0,2,1).reshape(5*len(t),len(freq)) for q in range(2)],axis=1)
    return A,freq


def main():
    start=time.perf_counter()
    prior=json.loads((FRONT/'result.json').read_text(encoding='utf-8'))
    if not prior['all_checks_pass'] or not all(sha(Path(p))==h for p,h in prior['identity'].items()):
        raise RuntimeError('Completed frontend identity failed')
    if not all(sha(FRONT/p)==h for p,h in prior['public_input_sha256'].items()):
        raise RuntimeError('Public input identity failed')
    geometry=next(Path(p) for p in prior['identity'] if 'finite_scene' in p and p.endswith('result.json'))
    data=json.loads(geometry.read_text(encoding='utf-8'))
    public=json.loads((FRONT/'public.json').read_text(encoding='utf-8'))
    with np.load(FRONT/'observations.npz') as d:
        y=d['mixture']
    t=np.arange(y.shape[1])/public['fs_hz']
    paths=[]
    for q,sid in enumerate([9,12]):
        for r in data['records']:
            if r['scene']=='two_AB' and r['source_id']==sid:
                for p in r['paths']:
                    length=p['length_m']
                    paths.append((q,r['station_id'],length,(1 if p['kind']=='direct' else -.5)/length))
    out=ROOT/('representation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'output_json.py',ROOT/'公共表示检查合同.md',FRONT/'result.json',FRONT/'public.json',FRONT/'observations.npz',geometry]
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(dumps({'sha256':identity,'numpy':np.__version__,'threads':2},indent=2),encoding='utf-8')
    for p in files[:3]:
        (out/p.name).write_bytes(p.read_bytes())
    results=[]
    for factor in [2,3]:
        tick=time.perf_counter()
        A,freq=public_matrix(paths,t,public['bandwidth_hz'],factor)
        U,s,Vh=svd(A,full_matrices=False,check_finite=False,lapack_driver='gesdd')
        yvector=y.ravel()
        projected=U.conj().T@yvector
        entries=[]
        for cutoff in [1e-8,1e-10,1e-12]:
            rank=int(np.sum(s>s[0]*cutoff))
            coefficients=Vh[:rank].conj().T@(projected[:rank]/s[:rank])
            fitted=A@coefficients
            error=float(np.linalg.norm(fitted-yvector)/np.linalg.norm(yvector))
            projection=U[:,:rank]@projected[:rank]
            projection_error=float(np.linalg.norm(fitted-projection)/max(np.linalg.norm(projection),1e-30))
            entries.append({'cutoff':cutoff,'rank':rank,'relative_time_residual':error,'coefficient_norm':float(np.linalg.norm(coefficients)),
                            'svd_replay_agreement':projection_error})
        result={'period_factor':factor,'frequencies_per_source':len(freq),'matrix_shape':list(A.shape),
                'singular_values':s.tolist(),'settings':entries,'seconds':time.perf_counter()-tick}
        results.append(result)
        print(dumps({k:v for k,v in result.items() if k!='singular_values'}),flush=True)
    historical=ROOT/'frontend_20261025_202954_590401/blind_output.json'
    checks={'all_inputs_unchanged':all(sha(Path(p))==h for p,h in identity.items()),
            'frontend_output_unchanged':sha(historical)==sha(FRONT/'blind_output.json'),
            'svd_replay':all(e['svd_replay_agreement']<1e-5 for r in results for e in r['settings'])}
    capacity=any(e['relative_time_residual']<1e-6 and e['cutoff']>=1e-10 for r in results for e in r['settings'])
    final={'identity':identity,'scope':'ORACLE_GEOMETRY_AND_GAINS_PUBLIC_FREQUENCY_BASIS_CAPACITY_ONLY',
           'results':results,'checks':checks,'all_checks_pass':all(checks.values()),'capacity_gate_pass':capacity,'seconds':time.perf_counter()-start}
    (out/'result.json').write_text(dumps(final,indent=2),encoding='utf-8')
    print(dumps({'out':str(out),'checks':checks,'capacity_gate_pass':capacity,'seconds':final['seconds']}))
    if not final['all_checks_pass']:
        raise RuntimeError('Numerical or identity verification failed')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
