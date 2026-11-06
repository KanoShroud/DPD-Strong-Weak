"""Same-model control: observe original solver vs check its existing terminal gate."""
from datetime import datetime
import inspect
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import attribute as a

d, s, fc, ref, rg = a.d, a.s, a.fc, a.ref, a.rg
HERE = Path(__file__).resolve().parent


class Accepted(Exception):
    def __init__(self, x, info):
        self.x, self.info = x, info


class Replay:
    def __init__(self, f, y, paths, n, x0, policy):
        self.f, self.y, self.paths, self.n = f, y, paths, n
        self.x0, self.policy = x0, policy
        self.calls, self.gate_calls = 0, 0
        self.points = []
        self.start = time.perf_counter()
        self.initial_cost = s.objective(ref.project(x0, n), f, y, paths)[0]
        scope = {**d.h.scope, 'minimize':self.minimize, 'objective':self.objective}
        base = types.FunctionType(fc.fit_base.__code__, scope, argdefs=fc.fit_base.__defaults__)
        scope['fit_base'] = base
        self.fit = types.FunctionType(d.h.fit.__code__, {**d.h.fit.__globals__, 'base_fit':base}, argdefs=d.h.fit.__defaults__)

    def objective(self, *args):
        self.calls += 1
        return s.objective(*args)

    def minimize(self, fun, x0, *args, **kwargs):
        unpack = inspect.getclosurevars(fun).nonlocals['unpack']

        def callback(z):
            x = unpack(z)
            item = dict(x=x.copy(), calls=self.calls, elapsed=time.perf_counter()-self.start)
            self.points.append(item)
            if self.policy != 'CHECK_ORIGINAL_GATE':
                return
            feasible = ref.project(x, self.n)
            cost, grad = s.objective(feasible, self.f, self.y, self.paths)
            self.gate_calls += 1
            pg = float(np.max(abs(feasible-ref.project(feasible-grad, self.n))))
            if np.isfinite(cost) and cost <= self.initial_cost+1e-10 and pg <= 1e-7:
                raise Accepted(feasible, dict(cost=cost, PG=pg, projection=float(np.linalg.norm(x-feasible))))

        assert 'callback' not in kwargs
        kwargs['callback'] = callback
        return minimize(fun, x0, *args, **kwargs)

    def run(self):
        try:
            x, state = self.fit(self.x0, self.f, self.y, self.paths, self.n)
            result = dict(x=x, original_state=state, early=False)
        except Accepted as end:
            x = end.x
            result = dict(x=x, original_gate=end.info, early=True)
        result.update(seconds=time.perf_counter()-self.start, callbacks=len(self.points),
                      solver_calls=self.calls, extra_gate_calls=self.gate_calls, points=self.points)
        return result


def curvature(x, f, y, paths):
    _, j = s.evaluate(x, f, y, paths)
    scale = np.maximum(np.linalg.norm(j, axis=0), 1e-15)
    _, sv, vh = np.linalg.svd(np.linalg.qr(j/scale, mode='reduced')[1], full_matrices=False)
    v = vh[-1]
    order = np.argsort(abs(v))[-10:][::-1]
    return dict(scaled_condition=float(sv[0]/sv[-1]), singular_max=float(sv[0]),
                singular_min=float(sv[-1]), weakest_indices=order, weakest_loading=v[order],
                position_loading=float(np.linalg.norm(v[:4])))


def main():
    out = HERE/('replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('结果目录', out, flush=True)
    hashes=rg.preflight(out, ['L3_R01', 'L3_R05'])
    extra=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[a.TRACE]
    extra += [a.FINAL/f'L3_R{i:02d}_HYBRID.json' for i in range(1,8)]
    hashes.update({str(p.resolve(strict=True)):ref.sha(p) for p in extra})
    snap=out/'source';snap.mkdir()
    for i,p in enumerate(hashes):
        if Path(p).suffix in ('.py','.md'):
            shutil.copy2(p,snap/f'{i:03d}_{Path(p).name}')
    fc.write(out/'identity.json',dict(hashes=hashes,command=sys.argv))
    start=time.perf_counter()
    try:
        f,y,n,_=fc.cal.spectral(rg.inputs('L3_R05'))
        record=a.read(a.TRACE)['records'][-1]
        paths=[tuple(p) for p in record['paths']]
        x0=np.array(record['points'][0]['x'])
        target=np.array(record['terminal']['x'])
        rows,checks=[],{}
        # AB / BA, no parameter or seed changes between repeats.
        for repeat, order in enumerate([['ORIGINAL','CHECK_ORIGINAL_GATE'],['CHECK_ORIGINAL_GATE','ORIGINAL']]):
            for policy in order:
                print('START',repeat,policy,flush=True)
                res=Replay(f,y,paths,n,x0,policy).run()
                xx=np.array(res['x']);rr,jj=ref.evaluate(xx,f,y,paths)
                pg=float(np.max(abs(xx-ref.project(xx-jj.T@rr,n))))
                key=f'{repeat}/{policy}'
                checks[key+'/PG']=pg<=1e-7
                checks[key+'/feasible']=bool(np.linalg.norm(xx-ref.project(xx,n))<1e-7)
                if policy=='ORIGINAL':checks[key+'/identical']=bool(np.array_equal(xx,target))
                else:
                    _,finish=d.h.finish(xx,dict(calls=0,iterations=300,seconds=0),x0,f,y,paths,n)
                    checks[key+'/original_finish']=bool(finish['stationary'])
                row={k:res[k] for k in ('seconds','callbacks','solver_calls','extra_gate_calls','early')}
                row.update(repeat=repeat,policy=policy,PG=pg,cost=float(.5*rr@rr),position_difference_m=d.max_position(xx-target))
                fc.write(out/f'{repeat}_{policy}.json',res)
                rows.append(row);fc.write(out/'results.json',dict(rows=rows,checks=checks,complete=False))
                print(row,flush=True)
                assert all(checks.values())
        metrics=[]
        for rep in range(1,8):
            tag=f'L3_R{rep:02d}'
            old=a.read(a.FINAL/(tag+'_HYBRID.json'))
            ff,yy,_,_=fc.cal.spectral(rg.inputs(tag))
            met=curvature(np.array(old['final']['x']),ff,yy,[tuple(p) for p in old['paths']])
            met.update(case=tag,K=len(old['paths']))
            metrics.append(met)
        fc.write(out/'curvature.json',metrics)
        # Explain the transition using observed estimates, not physical truth.
        pair=[]
        for model,rec in enumerate(a.read(a.TRACE)['records']):
            if len(rec['paths'])<=26:continue
            for pos in (0,len(rec['points'])-1):
                x=np.array(rec['points'][pos]['x']);aa=x[20:].reshape(-1,3)
                rho=aa[:,0]+1j*aa[:,1]
                pair.append(dict(model=model,index=pos,K=len(rec['paths']),link=rec['paths'][18],
                    delays_B=aa[[18,26],2],amplitudes=abs(rho[[18,26]]),
                    relative_phase_deg=float(np.angle(rho[18]/rho[26],deg=True)),
                    sum_amplitude=float(abs(rho[18]+rho[26]))))
        fc.write(out/'pair_evolution.json',pair)
        fc.write(out/'results.json',dict(rows=rows,checks=checks,complete=True))
    finally:
        same=all(ref.sha(p)==v for p,v in hashes.items())
        fc.write(out/'postflight.json',dict(unchanged=same,seconds=time.perf_counter()-start))
        assert same


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
