"""Final saved-result audit and local source-sharing attribution."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
RUNS=['support_20261025_195234_243209','iq_20261025_195523_357517','stress_20261025_195753_625322']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    results=[json.loads((ROOT/run/'result.json').read_text(encoding='utf-8')) for run in RUNS]
    identity={str(ROOT/run/'result.json'):sha(ROOT/run/'result.json') for run in RUNS}
    checks={'all_saved_checks':all(r['all_checks_pass'] for r in results),
            'all_inputs_unchanged':all(sha(Path(p))==h for r in results for p,h in r['identity'].items())}
    first=results[0]
    checks['all_B_supports_have_min_three']=all(b['LOS_count']>=3 for b in first['B_branches'])
    checks['three_LOS_B_not_skipped']=sum(b['LOS_count']==3 for b in first['B_branches'])==24
    checks['all_finite_B_roots_followed']=len(first['A_assignments'])==136*sum(len(b['roots']) for b in first['B_branches'])
    checks['no_unresolved']=not first['unresolved']
    geometry=next(Path(p) for p in results[1]['identity'] if 'finite_scene' in p and p.endswith('result.json'))
    data=json.loads(geometry.read_text(encoding='utf-8'))
    stations=np.array(data['stations_m'])/1000
    positions=np.array(data['positions_m'])[[9,12]]/1000
    rows=[]
    for q,sid in enumerate([9,12]):
        for r in data['records']:
            if r['scene']=='two_AB' and r['source_id']==sid:
                for p in r['paths']:
                    rows.append((q,r['station_id'],-1 if p['kind']=='direct' else {'A':0,'B':1}[p['panel']]))
    x=np.r_[positions.ravel(),0.,-.25,np.pi/2,-.55,0.,0.]
    def lengths(v):
        values=[]
        for q,m,label in rows:
            p=v[3*q:3*q+3]
            if label>=0:
                theta,b=v[6+2*label:8+2*label]
                n=np.array([np.cos(theta),np.sin(theta),0.])
                p=p-2*(p@n-b)*n
            values.append(np.linalg.norm(p-stations[m])-v[10+q])
        return np.array(values)
    h=2e-6
    J=np.column_stack([(lengths(x+np.eye(12)[i]*h)-lengths(x-np.eye(12)[i]*h))/(2*h) for i in range(12)])
    attribution=[]
    for sources in [[0],[1],[0,1]]:
        ids=[i for i,r in enumerate(rows) if r[0] in sources]
        posids=[3*q+j for q in sources for j in range(3)]
        nuisanceids=[6,7,8,9]+[10+q for q in sources]
        P=J[np.ix_(ids,posids)];N=J[np.ix_(ids,nuisanceids)]
        U,s,_=np.linalg.svd(N,full_matrices=False);nr=int(np.sum(s>s[0]*1e-8))
        Z=P-U[:,:nr]@(U[:,:nr].T@P)
        values=np.linalg.svd(Z,compute_uv=False)
        attribution.append({'sources':sources,'position_rank':int(np.sum(values>1e-8)),
                            'nuisance_rank':nr,'position_singular_values':values.tolist()})
    checks['path_and_IQ_local_ranks_agree']=[a['position_rank'] for a in attribution]==[3,3,6]
    artifact={'identity':identity,'audit_sha256':sha(Path(__file__)),'checks':checks,'all_checks_pass':all(checks.values()),
              'independent_path_attribution':attribution,
              'interpretation':'A alone already locally identifiable using shared walls; cross-source necessity NOT established.'}
    (ROOT/'independent_audit.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(artifact,ensure_ascii=False))
    if not artifact['all_checks_pass']:
        raise RuntimeError('Final audit failed')


if __name__=='__main__':
    main()
