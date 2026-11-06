"""Ground reflection, power-only controls and separately labelled oracle checks."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiment import ROOT, FS, B, LAYOUTS, generate, preprocess, write, sha, tests
from model import C, STATIONS, Model, solve, evaluate


def ground(layout,seed,n):
    los,_,noise,meta=generate(layout,seed,n,True)
    rng=np.random.default_rng(seed+10000*layout)
    full=4*n;f=np.fft.fftfreq(full,1/FS);mask=abs(f)<=B/2
    spectrum=np.zeros((2,full),complex)
    spectrum[:,mask]=rng.normal(size=(2,mask.sum()))+1j*rng.normal(size=(2,mask.sum()))
    spectrum/=np.sqrt(np.sum(abs(spectrum)**2,axis=1,keepdims=True))
    phase=np.exp(1j*rng.uniform(-np.pi,np.pi,5))
    p=LAYOUTS[layout]
    mirrored=p.copy();mirrored[:,2]*=-1
    length=np.linalg.norm(mirrored[:,None,:]-STATIONS[None,:,:],axis=-1)
    echo=[];independent=[]
    for q in range(2):
        rows=[];direct=[]
        for m in range(5):
            sf=spectrum[q]*np.exp(-2j*np.pi*f*length[q,m]/C)*(-.7*500/length[q,m])*phase[m]
            rows.append((np.fft.ifft(sf)*full)[n:2*n])
            dd=np.linalg.norm(p[q]-STATIONS[m])
            sf=spectrum[q]*np.exp(-2j*np.pi*f*dd/C)*(500/dd)*phase[m]
            direct.append((np.fft.ifft(sf)*full)[n:2*n])
        echo.append(rows);independent.append(direct)
    if not np.array_equal(los,np.array(independent)):
        raise RuntimeError('Ground scene changed direct signal')
    echo=np.array(echo)
    mixed=los+echo
    scale=np.sqrt(np.mean(abs(mixed)**2,axis=-1)/np.mean(abs(los)**2,axis=-1))
    power_only=los*scale[:,:,None]
    if not np.allclose(np.mean(abs(mixed)**2,axis=-1),np.mean(abs(power_only)**2,axis=-1),rtol=1e-12,atol=1e-12):
        raise RuntimeError('Power match failed')
    meta['excess_delay_times_B']=(length-meta['distance'])/C*B
    meta['received_power_change_db']=20*np.log10(scale)
    return los,echo,power_only,noise,meta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['ground','oracle'])
    parser.add_argument('--inputs',nargs='*',default=[])
    args=parser.parse_args()
    out=ROOT/(args.stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    files=[ROOT/'model.py',ROOT/'experiment.py',Path(__file__),ROOT/'执行合同.md',ROOT/'初始化修订_v2.md',ROOT/'求解修订_v3.md',ROOT/'后续条件诊断.md']
    identity={str(p):sha(p) for p in files}
    for p in files:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json',{'identity':identity,'threads':2,'inputs':args.inputs})
    test=tests();write(out/'tests.json',test)
    if not test['all_checks_pass']:
        raise RuntimeError(test)
    rows=[];started=time.perf_counter()
    if args.stage=='ground':
        for layout in range(2):
            seed=200;n=4096
            los,echo,power,noise,meta=ground(layout,seed,n)
            truth=meta.pop('truth');gain=meta.pop('direct_gain')
            file=out/f'input_{layout}_{seed}.npz'
            np.savez_compressed(file,los=los,echo=echo,power_only=power,noise=noise,truth=truth,direct_gain=gain)
            digest=sha(file)
            for label,ids in [('A',[0]),('B',[1]),('dual',[0,1])]:
                for mode,components in [('ground',los+echo),('power_only',power)]:
                    y,f=preprocess(components[ids].sum(axis=0)+noise,True)
                    output=solve(y,f,len(ids))
                    row={'stage':'ground','layout':layout,'seed':seed,'condition':label+'_'+mode,
                         'input_sha256':digest,'generator_metadata':meta,'output':output,
                         'evaluation':evaluate(output['selected']['positions'],truth[ids])}
                    rows.append(row);write(out/f'{layout}_{seed}_{label}_{mode}.json',row)
                    print(json.dumps({'case':[layout,label,mode],'evaluation':row['evaluation'],'seconds':output['seconds']}),flush=True)
            if sha(file)!=digest:
                raise RuntimeError('Input changed')
    else:
        for directory in args.inputs:
            folder=Path(directory).resolve(strict=True)
            if ROOT.resolve() not in folder.parents:
                raise RuntimeError('Input outside current experiment')
            for source in sorted(folder.glob('*_dual_*.json')):
                prior=json.loads(source.read_text(encoding='utf-8'))
                layout,seed=prior['layout'],prior['seed']
                file=folder/f'input_{layout}_{seed}.npz'
                if sha(file)!=prior['input_sha256']:
                    raise RuntimeError('Frozen IQ identity mismatch')
                identity[str(source)]=sha(source);identity[str(file)]=sha(file)
                with np.load(file) as data:
                    if prior['condition']=='dual_power_only':
                        continue
                    iq=data['los'].sum(axis=0)+data['noise']
                    if prior['condition']!='dual_los':
                        iq+=data['echo'].sum(axis=0)
                    truth=data['truth'];g=data['direct_gain'];g=g/g[:,:1]
                y,f=preprocess(iq,prior['stage']!='periodic')
                model=Model(y,f)
                fit=model.fit(truth,g)
                row={'prior':str(source),'condition':prior['condition'],'layout':layout,'seed':seed,
                     'oracle':'true_positions_and_direct_gains_initialization_only',
                     'fit':fit,'evaluation':evaluate(fit['positions'],truth),
                     'blind_evaluation':prior['evaluation'],'blind_loss':prior['output']['selected']['loss']}
                rows.append(row);write(out/f'{folder.name}_{layout}_{seed}_{prior["condition"]}.json',row)
                print(json.dumps({'case':[folder.name,layout,seed,prior['condition']],
                      'oracle_rmse':row['evaluation']['rmse_m'],'blind_rmse':prior['evaluation']['rmse_m'],
                      'loss_delta':fit['loss']-row['blind_loss']}),flush=True)
    write(out/'result.json',{'stage':args.stage,'count':len(rows),'seconds':time.perf_counter()-started,
                           'identity':identity,'input_source_unchanged':all(sha(Path(p))==h for p,h in identity.items()),
                           'numerical_tests_pass':test['all_checks_pass'],'rows':rows})
    print('COMPLETED '+str(out),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
