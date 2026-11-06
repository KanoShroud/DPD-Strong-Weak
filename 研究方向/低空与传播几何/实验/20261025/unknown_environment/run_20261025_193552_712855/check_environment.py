"""Conditional unknown-reflector identifiability, not a blind IQ estimator."""
import hashlib
import importlib.util
import itertools
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent/'finite_scene'
RUN = PREV/'run_20261025_190128_421395'
spec = importlib.util.spec_from_file_location('finite_scene',PREV/'finite_scene.py')
geo = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = geo
spec.loader.exec_module(geo)
SCALE = 1000.
TOL = 1e-6


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def above_plane_position(anchors, observations):
    """All anchors at equal z; return upper solution and original residual."""
    assert np.ptp(anchors[:,2])<1e-7
    delta = anchors[1:,:2]-anchors[0,:2]
    rd = observations[1:]-observations[0]
    matrix = np.column_stack([2*delta,2*rd])
    rhs = np.sum(delta**2,axis=1)-rd**2
    sol,_,rank,_ = np.linalg.lstsq(matrix,rhs,rcond=1e-12)
    if rank<3:
        return {'status':'rank_deficient'}
    h2 = sol[2]**2-sol[:2]@sol[:2]
    if sol[2]<=0 or h2<=0:
        return {'status':'no_upper_real_root'}
    p = np.r_[anchors[0,:2]+sol[:2],anchors[0,2]+np.sqrt(h2)]
    ranges = np.linalg.norm(p-anchors,axis=1)
    residual = float(np.max(abs((ranges[1:]-ranges[0])-rd)))
    return {'status':'accepted' if residual<TOL else 'inconsistent',
            'position':p,'offset':float(ranges[0]-observations[0]),'residual_m':residual}


def image_positions(stations, lengths, height):
    radii2 = lengths**2-(height-stations[:,2])**2
    if np.any(radii2<0):
        return [],'negative_radius'
    xy = stations[:,:2]
    if len(lengths)==1:
        return [],'continuous_family'
    if len(lengths)==2:
        v = xy[1]-xy[0]
        d = np.linalg.norm(v)
        if d<1e-10:
            return [],'continuous_family'
        e = v/d
        a = (radii2[0]-radii2[1]+d*d)/(2*d)
        h2 = radii2[0]-a*a
        if h2<-1e-6:
            return [],'inconsistent'
        normal = np.array([-e[1],e[0]])
        roots = [xy[0]+a*e+sign*np.sqrt(max(0,h2))*normal for sign in [1,-1]]
    else:
        matrix = 2*(xy[1:]-xy[0])
        rhs = np.sum(xy[1:]**2,axis=1)-xy[0]@xy[0]+radii2[0]-radii2[1:]
        sol,_,rank,_ = np.linalg.lstsq(matrix,rhs,rcond=1e-12)
        if rank<2:
            return [],'continuous_family'
        roots = [sol]
    answers = []
    for x in roots:
        p = np.r_[x,height]
        if np.max(abs(np.linalg.norm(stations-p,axis=1)-lengths))<TOL:
            answers.append(p)
    return answers, 'accepted' if answers else 'inconsistent'


def plane_from_image(p,image):
    diff = p-image
    if np.linalg.norm(diff)<1e-7:
        return None
    n = diff/np.linalg.norm(diff)
    return n,float(n@((p+image)/2))


def mirror(point,plane):
    n,b = plane
    return point-2*(point@n-b)*n


def jacobian_diagnosis(data,rows):
    stations = np.array(data['stations_m'])/SCALE
    positions = np.array(data['positions_m'])[[9,12]]/SCALE
    x = np.r_[positions.ravel(),0.,-.25,np.pi/2,-.55,0.,0.]
    # The unknown per-source origin absorbs any common arrival length.
    def model(v,derivative=False):
        p = v[:6].reshape(2,3)
        out,jac = [],[]
        for q,m,kind,label in rows:
            J = np.zeros(12)
            if kind=='direct':
                image = p[q]; transform = np.eye(3)
            else:
                k = 0 if label=='A' else 1
                theta,b = v[6+2*k:8+2*k]
                n = np.array([np.cos(theta),np.sin(theta),0.])
                dn = np.array([-np.sin(theta),np.cos(theta),0.])
                image = p[q]-2*(p[q]@n-b)*n
                transform = np.eye(3)-2*np.outer(n,n)
            L = np.linalg.norm(image-stations[m])
            u = (image-stations[m])/L
            J[3*q:3*q+3] = u@transform
            if kind!='direct':
                J[6+2*k] = u@(-2*((p[q]@dn)*n+(p[q]@n-b)*dn))
                J[7+2*k] = u@(2*n)
            J[10+q] = -1
            out.append(L-v[10+q]);jac.append(J)
        return np.array(jac) if derivative else np.array(out)
    J = model(x,True)
    step = 1e-6
    fd = np.column_stack([(model(x+np.eye(12)[i]*step)-model(x-np.eye(12)[i]*step))/(2*step) for i in range(12)])
    def profile(pos,nuisance):
        U,s,_ = np.linalg.svd(nuisance,full_matrices=False)
        rank = int(np.sum(s>s[0]*1e-8)) if len(s) and s[0]>0 else 0
        remaining = pos-U[:,:rank]@(U[:,:rank].T@pos)
        values = np.linalg.svd(remaining,compute_uv=False)
        return {'position_rank':int(np.sum(values>values[0]*1e-8)),
                'singular_values':values.tolist(),'nuisance_rank':rank}
    ids = [i for i,r in enumerate(rows) if r[2]=='direct']
    reflected = [i for i,r in enumerate(rows) if r[2]!='direct']
    independent = np.column_stack([J[:,10:],np.eye(len(rows))[:,reflected]])
    return {'normalized_parameters':'position/1000, plane_offset/1000, azimuth_rad, origin_length/1000',
            'full_jacobian_singular_values':np.linalg.svd(J,compute_uv=False).tolist(),
            'analytic_vs_fd_max_error':float(np.max(abs(J-fd))),
            'direct_only':profile(J[ids,:6],J[ids,10:]),
            'independent_reflection_delays':profile(J[:,:6],independent),
            'shared_unknown_planes':profile(J[:,:6],J[:,6:])}


def finite_witness(positions,stations,events,labels,planes,observations):
    """Construct sufficient rectangles, not an exhaustive extent search."""
    needed = [[],[]]
    for (q,m,_),label in zip(events,labels):
        if label==-1:
            continue
        p,r = positions[q],stations[m]
        n,b = planes[label]
        if (p@n-b)*(r@n-b)<=1e-10:
            return {'status':'reflection_wrong_side'}
        image = mirror(p,planes[label])
        t = (b-image@n)/((r-image)@n)
        if not 0<t<1:
            return {'status':'reflection_invalid_fraction'}
        point = image+t*(r-image)
        if point[2]<=0:
            return {'status':'below_ground_reflection'}
        needed[label].append(point)
    choices = []
    for q in range(2):
        for m in range(5):
            if any(e[0]==q and e[1]==m and lab==-1 for e,lab in zip(events,labels)):
                continue
            candidates = []
            p,r = positions[q],stations[m]
            for k,(n,b) in enumerate(planes):
                divisor = (r-p)@n
                if abs(divisor)<1e-10:
                    continue
                t = (b-p@n)/divisor
                point = p+t*(r-p)
                if 0<t<1 and point[2]>0:
                    candidates.append((k,point))
            if not candidates:
                return {'status':'missing_LOS_cannot_be_blocked'}
            choices.append(candidates)
    tried = 0
    for combination in itertools.product(*choices):
        tried += 1
        corners = [[v.copy() for v in group] for group in needed]
        for k,point in combination:
            corners[k].append(point)
        panels = []
        for k,points in enumerate(corners):
            if not points:
                break
            n,b = planes[k]
            tangent = np.cross([0.,0.,1.],n)
            points = np.array(points)
            lo,hi = np.min(points@tangent)-1e-4,np.max(points@tangent)+1e-4
            height = np.max(points[:,2])+1e-4
            center = b*n+(lo+hi)/2*tangent+np.array([0.,0.,height/2])
            panels.append(geo.Panel(str(k),center,n,hi-lo,height))
        if len(panels)!=2:
            continue
        good,maxerror = True,0.
        regenerated = []
        for q in range(2):
            actual = [geo.trace(positions[q],s,panels) for s in stations]
            obs = observations[q]
            if any(a['degenerate'] or len(a['paths'])!=len(o) for a,o in zip(actual,obs)):
                good=False;break
            deltas = np.concatenate([np.sort([p['length_m'] for p in a['paths']])-np.array(o)
                                     for a,o in zip(actual,obs) if len(o)])
            error = float(np.max(abs(deltas-np.mean(deltas))))
            maxerror=max(maxerror,error)
            if error>TOL:
                good=False;break
            regenerated.append(actual)
        if good:
            return {'status':'full_first_order_paths_match','tried_rectangles':tried,
                    'panels':[p.as_dict() for p in panels],'max_relative_length_error_m':maxerror,
                    'regenerated':regenerated}
    return {'status':'no_rectangle_witness_found','tried_rectangles':tried}


def main():
    start = time.perf_counter()
    prior = json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    if not prior['all_checks_pass'] or not all(sha(Path(p))==h for p,h in prior['identity'].items()):
        raise RuntimeError('Frozen scene input identity failed')
    out=ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'执行合同.md',PREV/'finite_scene.py',RUN/'result.json',RUN/'two_AB_iq.npz']
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(json.dumps({'sha256':identity,'python':platform.python_version(),'numpy':np.__version__},ensure_ascii=False,indent=2),encoding='utf-8')
    for p in files[:2]:
        (out/p.name).write_bytes(p.read_bytes())
    stations=np.array(prior['stations_m'])
    observations, rows=[],[]
    for q,sid in enumerate([9,12]):
        records=sorted([r for r in prior['records'] if r['scene']=='two_AB' and r['source_id']==sid],key=lambda r:r['station_id'])
        lengths=[sorted(p['length_m'] for p in r['paths']) for r in records]
        origin=lengths[0][0]
        observations.append([[L-origin for L in group] for group in lengths])
        rows.extend((q,r['station_id'],p['kind'],p['panel']) for r in records for p in r['paths'])
    local=jacobian_diagnosis(prior,rows)
    nonempty=[m for m,o in enumerate(observations[1]) if o]
    B=above_plane_position(stations[nonempty],np.array([observations[1][m][0] for m in nonempty]))
    if B['status']!='accepted':
        raise RuntimeError('Oracle B direct reconstruction failed')
    pB=B['position']
    b_ref=[(m,j,obs+B['offset']) for m,group in enumerate(observations[1]) for j,obs in enumerate(group) if j>0]
    partitions,plane_sets=[],[]
    for bits in itertools.product([0,1],repeat=len(b_ref)-1):
        assignment=(0,)+bits
        if any(len({assignment[i] for i,e in enumerate(b_ref) if e[0]==m})!=sum(e[0]==m for e in b_ref) for m in range(5)):
            continue
        entry={'assignment':assignment,'groups':[],'status':None}
        groups=[]
        for k in [0,1]:
            entries=[e for e,a in zip(b_ref,assignment) if a==k]
            images,status=image_positions(stations[[e[0] for e in entries]],np.array([e[2] for e in entries]),pB[2])
            groups.append((images,status));entry['groups'].append({'size':len(entries),'status':status,'images':[x.tolist() for x in images]})
        if any(status in ['inconsistent','negative_radius'] for _,status in groups):
            entry['status']='rejected_circle_constraints'
        elif any(status=='continuous_family' for _,status in groups):
            entry['status']='unresolved_continuous_family'
        else:
            entry['status']='finite_plane_candidates'
            for images in itertools.product(*(g[0] for g in groups)):
                planes=[plane_from_image(pB,image) for image in images]
                if any(p is None for p in planes):
                    continue
                plane_sets.append({'planes':planes,'b_assignment':assignment,'partition':len(partitions)})
        partitions.append(entry)
    candidates=[]; tried=0;rank_deficient=0; statuses={}
    Aevents=[(0,m,j) for m,group in enumerate(observations[0]) for j,_ in enumerate(group)]
    Aobs=np.array([observations[0][m][j] for _,m,j in Aevents])
    for plane_entry in plane_sets:
        planes=plane_entry['planes']
        perstation=[]
        for group in observations[0]:
            perstation.append([a for a in itertools.permutations([-1,0,1],len(group)) if -1 not in a[1:]])
        for combo in itertools.product(*perstation):
            assignment=tuple(k for group in combo for k in group)
            if assignment.count(-1)<3:
                continue
            tried+=1
            anchors=np.array([stations[m] if lab==-1 else mirror(stations[m],planes[lab]) for (_,m,_),lab in zip(Aevents,assignment)])
            answer=above_plane_position(anchors,Aobs)
            statuses[answer['status']]=statuses.get(answer['status'],0)+1
            if answer['status']=='rank_deficient':
                rank_deficient+=1
            if answer['status']!='accepted':
                continue
            events=list(Aevents);labels=list(assignment)
            for m,group in enumerate(observations[1]):
                for j,_ in enumerate(group):
                    events.append((1,m,j))
                    labels.append(-1 if j==0 else plane_entry['b_assignment'][next(i for i,e in enumerate(b_ref) if e[:2]==(m,j))])
            positions=np.array([answer['position'],pB])
            witness=finite_witness(positions,stations,events,labels,planes,observations)
            candidates.append({'positions_m':positions.tolist(),'A_labels':assignment,
                               'B_labels':plane_entry['b_assignment'],'planes':[{'normal':n.tolist(),'offset_m':b} for n,b in planes],
                               'length_residual_m':answer['residual_m'],'finite_witness':witness})
    truth=np.array(prior['positions_m'])[[9,12]]
    for candidate in candidates:
        candidate['posthoc_position_error_m']=np.linalg.norm(np.array(candidate['positions_m'])-truth,axis=1).tolist()
    checks={'analytic_gradient':local['analytic_vs_fd_max_error']<1e-7,
            'positive_truth_candidate_present':any(max(c['posthoc_position_error_m'])<1e-4 and c['finite_witness']['status']=='full_first_order_paths_match' for c in candidates),
            'input_unchanged':all(sha(Path(p))==h for p,h in identity.items())}
    result={'identity':identity,'information':'ORACLE_SOURCE_PATHS_AND_B_FOUR_LOS_NOT_BLIND_IQ',
            'observations_relative_length_m':observations,'local_diagnosis':local,
            'B_reconstruction':{'position_m':pB.tolist(),'relative_length_residual_m':B['residual_m']},
            'partitions':partitions,'plane_sets_count':len(plane_sets),'A_assignments_tested':tried,
            'A_status_counts':statuses,'rank_deficient_A':rank_deficient,'candidates':candidates,
            'unresolved_B_partitions':sum(p['status']=='unresolved_continuous_family' for p in partitions),
            'checks':{k:bool(v) for k,v in checks.items()},'all_checks_pass':bool(all(checks.values())),
            'wall_seconds':time.perf_counter()-start}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'ranks':{k:local[k]['position_rank'] for k in ['direct_only','independent_reflection_delays','shared_unknown_planes']},
     'partitions':len(partitions),'planes':len(plane_sets),'assignments':tried,'candidates':len(candidates),
     'witness_statuses':[c['finite_witness']['status'] for c in candidates], 'unresolved':result['unresolved_B_partitions'],
     'errors':[c['posthoc_position_error_m'] for c in candidates],'checks':checks,'seconds':result['wall_seconds']},ensure_ascii=False))
    if not result['all_checks_pass']:
        raise RuntimeError('Diagnostic positive control failed')


if __name__=='__main__':
    main()
