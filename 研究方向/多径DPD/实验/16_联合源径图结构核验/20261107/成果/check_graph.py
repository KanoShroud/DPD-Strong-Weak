"""Ideal unlabeled path-pair interface only; not a blind-IQ estimator."""
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PAIRS=list(combinations(range(4),2))


def associate(edges):
    # edges contains ONLY lag and complex coefficient, no source or path labels.
    found=[]
    for star in product(*(range(len(edges[(0,m)])) for m in (1,2,3))):
        t=np.r_[0.,[edges[(0,m)][star[m-1]][0] for m in (1,2,3)]]
        ids={(0,m):star[m-1] for m in (1,2,3)}
        valid=True
        for a,b in [(1,2),(1,3),(2,3)]:
            errors=np.array([abs(v[0]-(t[b]-t[a])) for v in edges[a,b]])
            k=int(np.argmin(errors))
            if errors[k]>1e-14:valid=False;break
            ids[a,b]=k
        if not valid:continue
        c={pair:edges[pair][ids[pair]][1] for pair in PAIRS}
        for a,b,d in combinations(range(4),3):
            z=c[a,b]*c[b,d]*c[a,d].conjugate()
            if abs(z/abs(z)-1)>1e-8:valid=False;break
        if not valid:continue
        amplitude=np.array([abs(c[0,1]*c[2,3]),abs(c[0,2]*c[1,3]),abs(c[0,3]*c[1,2])])
        if np.max(abs(np.log(amplitude/amplitude[0])))>1e-8:continue
        found.append(dict(edges=[(PAIRS.index(pair),ids[pair]) for pair in PAIRS],
            relative_arrival=t,score=float(sum(np.log(abs(c[pair])) for pair in PAIRS))))
    parent=list(range(len(found)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    owners={}
    for i,clique in enumerate(found):
        for edge in clique['edges']:
            if edge in owners:parent[root(i)]=root(owners[edge])
            else:owners[edge]=i
    groups={}
    for i in range(len(found)):groups.setdefault(root(i),[]).append(i)
    selected=[max(ids,key=lambda i:found[i]['score']) for ids in groups.values()]
    return dict(cliques=found,groups=list(groups.values()),selected=selected)


def main():
    out=HERE.parent/'运行数据/ideal30';out.mkdir(parents=True,exist_ok=False)
    identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'执行合同.md']}
    stations=np.array([[-3000.,-3000.],[3000.,-3000.],[3000.,3000.],[-3000.,3000.]])
    positions=np.array([[-900.,800.],[1200.,-600.]])
    propagation=np.linalg.norm(stations[:,None]-positions[None],axis=2)/299792458.
    rows=[];start=time.perf_counter()
    for reflections in (0,1,2):
        for repeat in range(10):
            rng=np.random.default_rng(np.random.SeedSequence([2026110718,reflections,repeat]))
            length=reflections+1
            delays=np.zeros((4,2,length))
            delays[:,:,1:]=rng.uniform(.5e-6,10e-6,size=(4,2,reflections))
            gains=(.7+rng.random((4,2,1)))*np.exp(1j*rng.uniform(-np.pi,np.pi,(4,2,length)))
            gains[:,:,1:]*=rng.uniform(.2,.9,size=(4,2,reflections))
            arrival=propagation[:,:,None]+delays
            edges={};labels={}
            for a,b in PAIRS:
                vals=[];truth=[]
                for q in range(2):
                    for j,k in product(range(length),repeat=2):
                        vals.append((arrival[b,q,k]-arrival[a,q,j],gains[a,q,j].conjugate()*gains[b,q,k]))
                        truth.append((q,j,k))
                order=rng.permutation(len(vals))
                edges[a,b]=[vals[k] for k in order];labels[a,b]=[truth[k] for k in order]
            answer=associate(edges)
            # Source/path labels accessed only after unlabeled decision.
            selected_labels=[]
            for i in answer['selected']:
                ll=[labels[PAIRS[a]][b] for a,b in answer['cliques'][i]['edges']]
                selected_labels.append(ll)
            all_direct=len(selected_labels)==2 and all(all(j==k==0 for q,j,k in z) for z in selected_labels)
            source_set={z[0][0] for z in selected_labels}
            assert all_direct and source_set=={0,1}
            assert len(answer['cliques'])==2*length**4
            row=dict(reflections=reflections,repeat=repeat,cliques=len(answer['cliques']),
                     group_count=len(answer['groups']),group_sizes=sorted(map(len,answer['groups'])),
                     correct_direct_groups=all_direct,scope='IDEAL_INTERFACE_NOT_IQ_LOCALIZATION')
            rows.append(row)
            def default(x):
                if isinstance(x,np.ndarray):return x.tolist()
                if isinstance(x,np.generic):return x.item()
                raise TypeError(type(x).__name__)
            (out/f'L{reflections}_{repeat}.json').write_text(json.dumps(answer,default=default),encoding='utf-8')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in identity.items())
    result=dict(count=30,rows=rows,seconds=time.perf_counter()-start,source_hashes=identity,
        scope='IDEAL_INTERFACE_NOT_IQ_LOCALIZATION',new_IQ_estimator=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('30/30 ideal interface checks passed; no blind-IQ or localization claim',result['seconds'])


if __name__=='__main__':main()
