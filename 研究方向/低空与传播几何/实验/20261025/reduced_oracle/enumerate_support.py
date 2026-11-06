"""Remove B four-LOS hint; enumerate exact source-labelled paths, not IQ."""
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'unknown_environment'
PRIOR = OLD/'run_20261025_193552_712855/result.json'
spec = importlib.util.spec_from_file_location('previous_check', OLD/'check_environment.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
SCALE = 1000.
TOL = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labelings(observations, symmetry=False):
    choices = [[v for v in itertools.permutations([-1,0,1],len(g)) if -1 not in v[1:]] for g in observations]
    for combo in itertools.product(*choices):
        labels = tuple(k for g in combo for k in g)
        if labels.count(-1)<3:
            continue
        reflected = [k for k in labels if k>=0]
        if symmetry and reflected and reflected[0]!=0:
            continue
        yield labels


def virtual_roots(stations, events, obs, labels):
    """Exact algebraic finite roots for >=3-observation virtual groups."""
    stations = stations/SCALE
    obs = obs/SCALE
    groups = []
    linear = []
    polynomials = []
    for label in [-1,0,1]:
        ids = [i for i,k in enumerate(labels) if k==label]
        if len(ids)<3:
            groups.append({'ids':ids,'small':True})
            continue
        anchors = stations[[events[i][1] for i in ids]]
        values = obs[ids]
        matrix = 2*(anchors[1:,:2]-anchors[0,:2])
        rhs = (anchors[1:,:2]**2).sum(axis=1)-(anchors[0,:2]**2).sum()-(values[1:]**2-values[0]**2)
        dt = 2*(values[1:]-values[0])
        if np.linalg.matrix_rank(matrix,tol=1e-10)<2:
            return [],'unresolved_collinear_group'
        uv = np.linalg.lstsq(matrix,np.column_stack([rhs,-dt]),rcond=1e-12)[0]
        u,v = uv[:,0],uv[:,1]
        residual0,residual1 = matrix@u-rhs,matrix@v+dt
        linear.extend(zip(residual0,residual1))
        a = u-anchors[0,:2]
        poly = np.array([1-v@v,2*(values[0]-a@v),values[0]**2-a@a])
        polynomials.append(poly)
        groups.append({'ids':ids,'small':False,'u':u,'v':v,'poly':poly})
    informative = [(a,b) for a,b in linear if abs(b)>1e-10]
    if informative:
        a,b = max(informative,key=lambda pair:abs(pair[1]))
        origins = [-a/b]
    else:
        differences = [p-polynomials[0] for p in polynomials[1:]]
        differences = [d for d in differences if np.max(abs(d))>1e-10]
        if not differences:
            return [],'unresolved_origin_family'
        polynomial = max(differences,key=lambda d:np.linalg.norm(d))
        roots = np.roots(np.trim_zeros(np.where(abs(polynomial)<1e-12,0.,polynomial),'f'))
        origins = [float(r.real) for r in roots if abs(r.imag)<1e-9]
    output=[]
    unresolved=False
    for origin in origins:
        if max((abs(a+b*origin) for a,b in linear),default=0)>1e-9:
            continue
        heights2=[float(np.polyval(g['poly'],origin)) for g in groups if not g['small']]
        if min(heights2)<=0 or np.ptp(heights2)>1e-9 or np.min(obs+origin)<=0:
            continue
        height = stations[0,2]+np.sqrt(heights2[0])
        possibilities=[]
        reject=False
        for g in groups:
            ids = g['ids']
            if g['small']:
                if len(ids)<2:
                    unresolved=True;reject=True;break
                answers,status=previous.image_positions(stations[[events[i][1] for i in ids]]*SCALE,
                                                       (obs[ids]+origin)*SCALE,height*SCALE)
                if status!='accepted':
                    reject=True;break
                possibilities.append([p/SCALE for p in answers])
            else:
                point=np.r_[g['u']+g['v']*origin,height]
                if np.max(abs(np.linalg.norm(stations[[events[i][1] for i in ids]]-point,axis=1)-(obs[ids]+origin)))>TOL/SCALE:
                    reject=True;break
                possibilities.append([point])
        if reject:
            continue
        for points in itertools.product(*possibilities):
            p,im0,im1=np.array(points)*SCALE
            planes=[previous.plane_from_image(p,im) for im in [im0,im1]]
            if any(plane is None for plane in planes):
                continue
            output.append({'position':p,'planes':planes,'origin_m':origin*SCALE})
    return output, 'unresolved_small_group' if unresolved else 'finite_checked'


def main():
    start=time.perf_counter()
    old=json.loads(PRIOR.read_text(encoding='utf-8'))
    if not old['all_checks_pass'] or not all(sha(Path(p))==h for p,h in old['identity'].items()):
        raise RuntimeError('Frozen prior identity mismatch')
    data_path=next(Path(p) for p in old['identity'] if p.endswith('result.json'))
    data=json.loads(data_path.read_text(encoding='utf-8'))
    out=ROOT/('support_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'执行合同.md',PRIOR,OLD/'check_environment.py',data_path]
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(json.dumps({'sha256':identity,'python':platform.python_version(),'numpy':np.__version__},indent=2),encoding='utf-8')
    for p in files[:2]:
        (out/p.name).write_bytes(p.read_bytes())
    stations=np.array(data['stations_m'])
    observations=old['observations_relative_length_m']
    events=[[(q,m,j) for m,g in enumerate(observations[q]) for j,_ in enumerate(g)] for q in range(2)]
    obs=[np.array([observations[q][m][j] for _,m,j in events[q]]) for q in range(2)]
    branches=[]; candidates=[]; assignments=[]
    unresolved=[]; total_a=0
    for b_labels in labelings(observations[1],True):
        roots,status=virtual_roots(stations,events[1],obs[1],b_labels)
        branch={'labels':b_labels,'LOS_count':b_labels.count(-1),'status':status,'roots':[]}
        if status.startswith('unresolved'):
            unresolved.append({'stage':'B','labels':b_labels,'status':status})
        for root in roots:
            pB,planes=root['position'],root['planes']
            branch['roots'].append({'position_m':pB.tolist(),'origin_m':root['origin_m'],
                                    'planes':[{'normal':n.tolist(),'offset_m':b} for n,b in planes]})
            for a_labels in labelings(observations[0]):
                total_a+=1
                anchors=np.array([stations[m] if lab==-1 else previous.mirror(stations[m],planes[lab])
                                  for (_,m,_),lab in zip(events[0],a_labels)])
                answer=previous.above_plane_position(anchors,obs[0])
                assignments.append({'B_branch':len(branches),'B_root':len(branch['roots'])-1,'A_labels':a_labels,'status':answer['status']})
                if answer['status']=='rank_deficient':
                    unresolved.append({'stage':'A','B_labels':b_labels,'A_labels':a_labels,'status':'rank_deficient'})
                if answer['status']!='accepted':
                    continue
                positions=np.array([answer['position'],pB])
                witness=previous.finite_witness(positions,stations,events[0]+events[1],a_labels+b_labels,planes,observations)
                candidates.append({'positions_m':positions.tolist(),'A_labels':a_labels,'B_labels':b_labels,
                                   'length_residual_m':answer['residual_m'],'witness':witness})
        branches.append(branch)
    truth=np.array(data['positions_m'])[[9,12]]
    for cand in candidates:
        cand['posthoc_error_m']=np.linalg.norm(np.array(cand['positions_m'])-truth,axis=1).tolist()
    checks={'frozen_inputs':all(sha(Path(p))==h for p,h in identity.items()),
            'positive_control':any(max(c['posthoc_error_m'])<TOL and c['witness']['status']=='full_first_order_paths_match' for c in candidates)}
    result={'information':'SOURCE_LABELLED_EXACT_PATHS_B_LOS_UNKNOWN_MINIMUM_THREE_PER_SOURCE',
            'identity':identity,'B_branches':branches,'A_assignments':assignments,'unresolved':unresolved,'candidates':candidates,
            'checks':checks,'all_checks_pass':all(checks.values()),'wall_seconds':time.perf_counter()-start}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'B_branches':len(branches),'B_three_LOS':sum(b['LOS_count']==3 for b in branches),
                      'B_roots':sum(len(b['roots']) for b in branches),'A_assignments':total_a,'unresolved':len(unresolved),
                      'candidates':len(candidates),'errors':[c['posthoc_error_m'] for c in candidates],
                      'witnesses':[c['witness']['status'] for c in candidates],'checks':checks,'seconds':result['wall_seconds']}))
    if not result['all_checks_pass']:
        raise RuntimeError('Positive control failed')


if __name__=='__main__':
    main()
