"""Frozen single-case timing and original acceptance-gate attribution."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'position_stop_20261104'))
import diagnose as d  # noqa: E402

s, fc, ref, rg = d.s, d.fc, d.ref, d.rg
FINAL = d.NAT/'followup_20261104_140420_955153'
TRACE = BASE/'position_stop_20261104/trace_20261104_144858_651707/L3_R05_trace.json'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def properties(x, paths, f):
    a = np.asarray(x)[20:].reshape(-1, 3)
    rho = a[:, 0]+1j*a[:, 1]
    pairs = []
    for i in range(len(paths)):
        for j in range(i):
            if tuple(paths[i]) != tuple(paths[j]):
                continue
            corr = abs(np.mean(np.exp(-2j*np.pi*f*(a[i, 2]-a[j, 2])/ref.B)))
            pairs.append(dict(i=i, j=j, link=paths[i], delta_B=float(abs(a[i, 2]-a[j, 2])),
                              coherence=float(corr), amp_i=float(abs(rho[i])), amp_j=float(abs(rho[j]))))
    return dict(active_indices=np.flatnonzero(abs(rho) >= 1-1e-6), amplitudes=abs(rho),
                most_coherent=sorted(pairs, key=lambda p: -p['coherence'])[:6])


def offline(out):
    totals, stages = [], []
    for rep in range(1, 8):
        tag = f'L3_R{rep:02d}'
        old = read(FINAL/(tag+'_HYBRID.json'))
        totals.append(dict(case=tag, seconds=old['seconds'], frontend=old['frontend_seconds'],
            search=sum(a['search']['seconds'] for a in old['history']),
            fit=sum(a['fit']['seconds'] for a in old['history'] if 'fit' in a),
            K=len(old['paths']), PG=old['final']['projected_gradient']))
        f, _, _, _ = fc.cal.spectral(rg.inputs(tag))
        for model, a in enumerate(old['history']):
            if 'fit' not in a:
                continue
            fit = a['fit']
            props = properties(fit['x'], old['paths'][:a['K']], f)
            stages.append(dict(case=tag, model=model, K=a['K'], seconds=fit['seconds'],
                iterations=fit['iterations'], calls=fit['calls'], PG=fit['projected_gradient'],
                message=fit.get('message'), switch=fit.get('hybrid_switch'),
                properties=props))
    fc.write(out/'case_stage_comparison.json', dict(totals=totals, stages=stages))
    f, y, n, _ = fc.cal.spectral(rg.inputs('L3_R05'))
    trace = read(TRACE)
    data, summary, checks = [], [], {}
    for model, record in enumerate(trace['records']):
        paths = [tuple(p) for p in record['paths']]
        init = ref.project(np.array(record['points'][0]['x']), n)
        initial_cost = s.objective(init, f, y, paths)[0]
        terminal = np.array(record['terminal']['x'])
        rows = []
        for index, point in enumerate(record['points']):
            x = np.array(point['x'])
            z = ref.project(x, n)
            raw_cost, raw_g = s.objective(x, f, y, paths)
            cost, g = s.objective(z, f, y, paths)
            mapping = z-ref.project(z-g, n)
            pg = float(np.max(abs(mapping)))
            raw_pg = float(np.max(abs(x-ref.project(x-raw_g, n))))
            accept = bool(np.isfinite(z).all() and np.isfinite(cost) and cost <= initial_cost+1e-10 and pg <= 1e-7)
            rows.append(dict(model=model, K=len(paths), index=index, phase=point['phase'],
                elapsed=point['elapsed'], raw_cost=raw_cost, cost=cost, raw_PG=raw_pg, PG=pg,
                projection_change=float(np.linalg.norm(z-x)), accept=accept,
                position_to_terminal_m=d.max_position(z-terminal),
                gain_to_terminal=float(np.linalg.norm(z[4:20]-terminal[4:20])),
                channel_to_terminal=float(np.linalg.norm(z[20:]-terminal[20:])),
                PG_index=int(np.argmax(abs(mapping))), raw_PG_index=int(np.argmax(abs(x-ref.project(x-raw_g, n))))))
        first = next((r for r in rows if r['accept']), None)
        if first:
            px = np.array(record['points'][first['index']]['x'])
            z = ref.project(px, n)
            r, j = ref.evaluate(z, f, y, paths)
            pg = float(np.max(abs(z-ref.project(z-j.T@r, n))))
            checks[f'{model}/original_pg'] = abs(pg-first['PG']) < 1e-9 and pg <= 1e-7
            checks[f'{model}/original_cost'] = abs(.5*r@r-first['cost']) < 1e-10
            _, state = d.h.finish(px, dict(calls=0, iterations=300, seconds=0), init, f, y, paths, n)
            checks[f'{model}/unchanged_finish'] = bool(state['stationary'])
        summary.append(dict(model=model, K=len(paths), points=len(rows), first_acceptable=first,
            final=rows[-1], acceptable_points=sum(r['accept'] for r in rows),
            terminal_properties=properties(terminal, paths, f)))
        data += rows
        fc.write(out/'trajectory.json', dict(rows=data, summary=summary, checks=checks))
        print('MODEL', len(paths), 'first accepted', first, 'total points', len(rows), flush=True)
    assert all(checks.values())
    fc.write(out/'offline_complete.json', dict(complete=True, checks=checks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['offline'], default='offline')
    args = ap.parse_args()
    out = HERE/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('结果目录', out, flush=True)
    hashes = rg.preflight(out, [f'L3_R{i:02d}' for i in range(1, 8)])
    files = list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[TRACE]
    files += [FINAL/f'L3_R{i:02d}_HYBRID.json' for i in range(1, 8)]
    hashes.update({str(p.resolve(strict=True)):ref.sha(p) for p in files})
    snap=out/'source'; snap.mkdir()
    for i, p in enumerate(hashes):
        if Path(p).suffix in ('.py', '.md'):
            shutil.copy2(p, snap/f'{i:03d}_{Path(p).name}')
    fc.write(out/'identity.json', dict(hashes=hashes, command=sys.argv))
    start=time.perf_counter()
    try:
        offline(out)
    finally:
        same=all(ref.sha(p)==value for p,value in hashes.items())
        fc.write(out/'postflight.json', dict(unchanged=same, seconds=time.perf_counter()-start))
        assert same


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
