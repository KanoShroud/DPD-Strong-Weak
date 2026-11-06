"""Export public observations, run isolated initializer, evaluate only afterwards."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from frontend import C, solve, unit_tests
from output_json import dumps

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'reduced_oracle/iq_20261025_195523_357517/result.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    begin=time.perf_counter()
    old=json.loads(PRIOR.read_text(encoding='utf-8'))
    if not old['all_checks_pass'] or not all(sha(Path(p))==h for p,h in old['identity'].items()):
        raise RuntimeError('Frozen evidence mismatch')
    geometry=next(Path(p) for p in old['identity'] if 'finite_scene' in p and p.endswith('result.json'))
    iq_path=next(Path(p) for p in old['identity'] if p.endswith('.npz'))
    data=json.loads(geometry.read_text(encoding='utf-8'))
    with np.load(iq_path) as iq:
        mixture=iq['mixture'];dt=float(iq['t_s'][1]-iq['t_s'][0])
    out=ROOT/('frontend_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'frontend.py',ROOT/'output_json.py',ROOT/'执行合同.md',PRIOR,geometry,iq_path]
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(json.dumps({'sha256':identity,'numpy':np.__version__,'threads':2},indent=2),encoding='utf-8')
    for p in files[:4]:
        (out/p.name).write_bytes(p.read_bytes())
    public={'stations_m':data['stations_m'],'fs_hz':1/dt,'bandwidth_hz':12.5e6,'passband_hz':7e6,
            'K':2,'minimum_LOS_per_source':3,'maximum_vertical_reflectors':2,
            'bounds_m':[[-1000.,-1000.,20.],[1000.,1000.,150.]]}
    (out/'public.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    np.savez_compressed(out/'observations.npz',mixture=mixture)
    tests={k:bool(v) for k,v in unit_tests().items()}
    if not all(tests.values()):
        raise RuntimeError(tests)
    started=time.perf_counter()
    output=solve(mixture,public)
    elapsed=time.perf_counter()-started
    (out/'blind_output.json').write_text(dumps(output,indent=2),encoding='utf-8')
    # Truth first used here, never passed into solve or used to select candidates.
    truth=np.array(data['positions_m'])[[9,12]]
    evaluation={}
    for mode,method in output.items():
        edges=[]
        for q,sid in enumerate([9,12]):
            los=[r for r in data['records'] if r['scene']=='two_AB' and r['source_id']==sid and any(p['kind']=='direct' for p in r['paths'])]
            for ii,a in enumerate(los):
                for b in los[ii+1:]:
                    i,j=sorted([a['station_id'],b['station_id']])
                    delta=np.linalg.norm(truth[q]-np.array(public['stations_m'])[i])-np.linalg.norm(truth[q]-np.array(public['stations_m'])[j])
                    pair=method['pairs'][f'{i}_{j}']
                    errors={key:min((abs(p['range_difference_m']-delta) for p in pair[key]),default=None) for key in ['peaks','all_peaks']}
                    edges.append({'source':q,'stations':[i,j],'true_range_difference_m':float(delta),'nearest_error_m':errors})
        versions={}
        for denom,version in method['versions'].items():
            positions=np.array([p['position_m'] for p in version['candidate_points']])
            nearest=[]
            for q in range(2):
                if not len(positions):
                    nearest.append(None);continue
                errors=np.linalg.norm(positions-truth[q],axis=1);idx=int(np.argmin(errors))
                nearest.append({'error_m':float(errors[idx]),'rank':idx+1,'candidate':version['candidate_points'][idx]})
            versions[denom]={'graphs':len(version['graphs']),'graphs3':sum(len(g['nodes'])==3 for g in version['graphs']),
                             'graphs4':sum(len(g['nodes'])==4 for g in version['graphs']),
                             'candidate_points':len(positions),'nearest':nearest,
                             'LOS_edges_within_tolerance':sum(e['nearest_error_m']['peaks'] is not None and e['nearest_error_m']['peaks']<=C/(int(denom)*public['bandwidth_hz']) for e in edges)}
        evaluation[mode]={'LOS_edge_diagnostics':edges,'versions':versions}
    checks={**tests,'input_unchanged':all(sha(Path(p))==h for p,h in identity.items())}
    result={'identity':identity,'public_input_sha256':{'public.json':sha(out/'public.json'),'observations.npz':sha(out/'observations.npz')},
            'checks':checks,'all_checks_pass':all(checks.values()),'evaluation':evaluation,
            'blind_seconds':elapsed,'total_seconds':time.perf_counter()-begin,
            'scope':'RF_GRAPH_CANDIDATE_PROTOTYPE_NOT_FULL_DATEMM_NOT_COMPLETE_DPD'}
    (out/'result.json').write_text(dumps(result,indent=2),encoding='utf-8')
    print(dumps({'out':str(out),'checks':checks,'seconds':elapsed,'evaluation':evaluation}))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
