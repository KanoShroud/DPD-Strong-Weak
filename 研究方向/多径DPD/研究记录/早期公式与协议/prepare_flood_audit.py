"""Pin authors' public MATLAB OT implementation and prepare truth-free checks."""
from datetime import datetime
import hashlib
import json

import numpy as np
import requests
from scipy.io import savemat

from geometry_association_gate import ROOT, PAIRS, C, B, sha, write
from robust_geometry import predicted

REPO='https://api.github.com/repos/gabrielleflood/omt-based-toda-localization'
SOURCE=ROOT/'independent_backend_20261029_222111_026721'


def main():
    commit=requests.get(REPO+'/commits/main',timeout=30)
    commit.raise_for_status()
    identity=commit.json()['sha']
    tree=requests.get(REPO+'/git/trees/'+identity+'?recursive=1',timeout=30)
    tree.raise_for_status()
    out=ROOT/('flood_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'author').mkdir()
    source_hash={}
    for x in tree.json()['tree']:
        name=x['path']
        if x['type']=='blob' and (name.startswith('utils/omt/') or name in ['README.md','simple_experiment.m']):
            url=f'https://raw.githubusercontent.com/gabrielleflood/omt-based-toda-localization/{identity}/{name}'
            response=requests.get(url,timeout=30)
            response.raise_for_status()
            dest=out/'author'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(response.content)
            source_hash[name]=dict(url=url,sha256=hashlib.sha256(response.content).hexdigest())
    write(out/'repository.json',dict(commit=identity,tree=tree.json(),files=source_hash))
    hashes={str((ROOT/n).resolve(strict=True)):sha(ROOT/n) for n in [
       'prepare_flood_audit.py','run_flood_audit.m','近邻共同后端核验协议.md','robust_geometry.py','geometry_association_gate.py']}
    for name in ['近邻共同后端核验协议.md','run_flood_audit.m','prepare_flood_audit.py']:
        (out/name).write_bytes((ROOT/name).read_bytes())
    files=[]
    # A well-conditioned tiny problem checks the solver independently of localisation.
    tiny=np.array([[0.,4.,8.],[.01,4.,8.],[4.,0.,8.],[4.,.01,8.],[8.,8.,0.],[8.,8.,.01]])
    savemat(out/'synthetic.mat',dict(C=tiny,trash=np.array(3.),R=np.array(3.),epsilon=np.array(1e-7),eta=np.array(1.)))
    files.append('synthetic')
    for path in ['LOS','MP']:
        tag='L20_R200_'+path
        rowpath=SOURCE/(tag+'.json'); poolpath=SOURCE/(tag+'_geometry.npz')
        hashes[str(rowpath)]=sha(rowpath); hashes[str(poolpath)]=sha(poolpath)
        row=json.loads(rowpath.read_text(encoding='utf-8'))
        p=np.load(poolpath)['initial_candidates']
        lags=np.array([sorted(np.array(row['peaks'][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
        pred=predicted(p)[0]
        costs=((pred[:,:,None]-lags[None])/(C/B))**2
        matrix=costs.reshape(len(p),-1).T
        savemat(out/(tag+'.mat'),dict(C=matrix,trash=np.array(np.percentile(matrix,95,method='hazen')),
             trash_last_row=np.array(np.percentile(matrix[-1],95,method='hazen')),R=np.array(5.),
             epsilon=np.array(1e-7),eta=np.array(1.),p=p,lags=lags))
        files.append(tag)
    write(out/'protocol.json',dict(commit=identity,hashes=hashes,files=files,scale_m=C/B,truth_in_input=False))
    print('OUTPUT',out,flush=True)


if __name__=='__main__':
    main()
