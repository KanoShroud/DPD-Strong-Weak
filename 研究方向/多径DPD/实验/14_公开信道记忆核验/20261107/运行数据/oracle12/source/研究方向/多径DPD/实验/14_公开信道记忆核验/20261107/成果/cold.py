"""Same capped model, frozen oracle pair indices; not deployable performance."""
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from run import HERE, ROOT, BASE, OLD, gate, engine_for


def main():
    paired=HERE.parent/'运行数据/paired18'
    done=gate.read(paired/'completion.json');assert done['count']==54 and done['identity_verified']
    assert all(gate.sha(Path(p))==h for p,h in {**done['source_hashes'],**done['input_hashes']}.items())
    out=HERE.parent/'运行数据/oracle12';out.mkdir(parents=True,exist_ok=False)
    identity={str(p):gate.sha(p) for p in list((BASE/'代码/mp_dpd').rglob('*.py'))+
              [Path(__file__),HERE/'run.py',HERE/'重建控制合同.md']}
    for p in identity:gate.copy_checked(Path(p),out/'source'/Path(p).relative_to(ROOT))
    input_hashes={}
    src=OLD/'运行数据/cold12/rows.json';gate.copy_checked(src,out/'old_oracle_rows.json')
    input_hashes[str(out/'old_oracle_rows.json')]=gate.sha(src)
    old_common={r['tag']:r for r in gate.read(OLD/'运行数据/common18/rows.json')}
    rows=[];start=time.perf_counter()
    for previous in gate.read(out/'old_oracle_rows.json'):
        tag=previous['tag'];local=out/'inputs'/tag/'iq.npz'
        src=paired/'inputs'/tag/'iq.npz'
        gate.copy_checked(src,local);input_hashes[str(local)]=gate.sha(src)
        with np.load(local,allow_pickle=False) as z:iq=z['iq']
        e=engine_for('CAP_COUNTED');ids=np.asarray(previous['pair'],int)
        token="pairs = [selection['first'], selection['second']] if two_regions else [fc.engine.joint_select(geom['reward'])[0]]"
        e.estimate_function=gate.api.runtime().ad.clone(e.estimate_function,
            [(token,'pairs = [diagnostic_pair]')],{'diagnostic_pair':ids.copy()})
        result=e.mp(iq);gate.cfg.write(out/f'{tag}_decision.json',result)
        assert max(result['final']['x'][e.D+2::3],default=0)<=20e-6*e.ref.B+1e-7
        bank=Path(old_common[tag]['truth_evaluation_file']);assert gate.sha(bank)==old_common[tag]['truth_file_sha']
        dst=out/'evaluation'/bank.name
        if not dst.exists():gate.copy_checked(bank,dst)
        input_hashes[str(dst)]=gate.sha(dst)
        with np.load(dst,allow_pickle=False) as z:truth=z['p']
        errors=gate.errors(result['final']['p'],truth)
        rows.append(dict(tag=tag,condition=previous['condition'],pair=ids,errors_m=errors,
            rmse=float(np.sqrt(np.mean(errors**2))),normal=result['normal'],stop=result['stop'],
            cost=result['final']['cost'],seconds=result['seconds'],K=len(result['paths']),
            scope='ORACLE_PAIR_NOT_BLIND'))
        gate.cfg.write(out/'rows.json',rows)
        print(tag,round(rows[-1]['rmse'],4),result['stop'],flush=True)
    assert len(rows)==12
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**input_hashes}.items())
    gate.cfg.write(out/'completion.json',dict(count=12,seconds=time.perf_counter()-start,
        source_hashes=identity,input_hashes=input_hashes,identity_verified=True,scope='ORACLE_PAIR_NOT_BLIND'))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
