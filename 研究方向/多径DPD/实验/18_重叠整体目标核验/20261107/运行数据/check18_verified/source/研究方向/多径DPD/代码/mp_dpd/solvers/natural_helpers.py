# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Auditable exact-kernel and full-IQ solver gates; no time-based termination."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
import 研究方向.多径DPD.代码.mp_dpd.solvers.natural_stop as s

fc, rg, ref, HERE = s.fc, s.rg, s.ref, s.HERE


def audit(result, iq):
    f, y, n, _ = fc.cal.spectral(iq)
    checks = {}
    count = 0
    pp = result['paths']
    for state, paths in [(result['los'], [])]+[(h['fit'], pp[:h['K']]) for h in result['history'] if 'fit' in h]:
        x = np.array(state['x'])
        r, j = ref.evaluate(x, f, y, paths)
        pg = float(np.max(abs(x-ref.project(x-j.T@r, n))))
        checks[f'{count}/cost'] = bool(abs(.5*r@r-state['cost']) < 1e-10)
        checks[f'{count}/PG'] = bool(abs(pg-state['projected_gradient']) < 1e-9)
        checks[f'{count}/feasible'] = bool(np.linalg.norm(x-ref.project(x, n)) < 1e-7)
        count += 1
    checks['monotone'] = all(h.get('decrease', True) for h in result['history'])
    return checks


def kernel(out, tags):
    checks, rows = {}, []
    for tag in tags:
        iq = rg.inputs(tag)
        f, y, n, sigma = fc.cal.spectral(iq)
        old = fc.load(rg.FORMAL/'cases'/tag/'solve.json')
        x, pp = np.array(old['final']['x']), [tuple(p) for p in old['paths']]
        x[:4] += [.01, -.02, .015, .01]
        r, j = ref.evaluate(x, f, y, pp)
        rr, jj = s.evaluate(x, f, y, pp)
        cost, g = s.objective(x, f, y, pp)
        checks[tag+'/residual'] = bool(np.allclose(r, rr, rtol=1e-9, atol=1e-11))
        checks[tag+'/jacobian'] = bool(np.allclose(j, jj, rtol=1e-7, atol=1e-10))
        checks[tag+'/gradient'] = bool(np.allclose(g, j.T@r, rtol=1e-7, atol=1e-10))
        checks[tag+'/cost'] = bool(abs(cost-.5*r@r) < 1e-11)
        direction = np.random.default_rng(441).normal(size=len(x)); direction /= np.linalg.norm(direction)
        eps = 1e-5
        fd = (s.objective(x+eps*direction, f, y, pp)[0]-s.objective(x-eps*direction, f, y, pp)[0])/(2*eps)
        checks[tag+'/finite_difference'] = bool(np.isclose(fd, g@direction, rtol=2e-4, atol=2e-8))
        use = abs(np.fft.fftfreq(n, 1/ref.FS)) <= ref.B/2
        _, b1, m1 = fc.search(x, pp, f, y, n, use, sigma, 1)
        _, b2, m2 = s.search(x, pp, f, y, n, use, sigma, 1)
        _, b3, m3 = s.fast_search(x, pp, f, y, n, use, sigma, 1)
        checks[tag+'/search_rank'] = m1['rank'] == m2['rank']
        checks[tag+'/search_valid'] = bool(np.array_equal(b1[:, 3], b2[:, 3]))
        ids = b1[:, 3] == 1
        checks[tag+'/search_scores'] = bool(np.allclose(b1[ids, 4], b2[ids, 4], rtol=1e-5, atol=1e-5))
        checks[tag+'/QR_rank'] = m1['rank'] == m3['rank']
        checks[tag+'/QR_valid'] = bool(np.array_equal(b1[:, 3], b3[:, 3]))
        checks[tag+'/QR_scores'] = bool(np.allclose(b1[ids, 4], b3[ids, 4], rtol=1e-5, atol=1e-5))
        times = {}
        for name, fun in [('previous_objective', fc.objective), ('guarded_objective', s.objective), ('previous_jacobian', fc.evaluate), ('guarded_jacobian', s.evaluate)]:
            tt = []
            for _ in range(5):
                start = time.perf_counter(); fun(x, f, y, pp); tt.append(time.perf_counter()-start)
            times[name] = float(np.median(tt))
        times.update(search_original=m1['seconds'], search_guarded=m2['seconds'], search_QR=m3['seconds'])
        rows.append(dict(case=tag, times=times))
        print(tag, times, flush=True)
    rng = np.random.default_rng(42)
    h = rng.normal(size=(30, 5, 2))+1j*rng.normal(size=(30, 5, 2))
    h[10:, :, 1] = h[10:, :, 0]+1e-8*h[10:, :, 1]
    h[20:, :, 1] = h[20:, :, 0]
    checks['ill_conditioned_fallback'] = bool(np.allclose(s.GuardedLinalg().pinv(h), np.linalg.pinv(h, rcond=1e-12)))
    fc.write(out/'kernel.json', dict(checks=checks, rows=rows, all_pass=all(checks.values())))
    assert all(checks.values()), [k for k, v in checks.items() if not v]


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
