"""Bounded native single-source diagnostics; saved separately from formal evidence."""
import argparse
from datetime import datetime
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .. import methods as old
from ..physics import Model, Spectrum, grid, sbl_posterior
from ..runtime import preflight, write, sha, verify, resource_check
from ..metrics import evaluate
from .config import BASE, ROOT
from .scenes import make, observation, geometry_audit


def identity():
    paths = list((ROOT/'统一模型代码/gates/g10/r1').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def spectra(scene, out):
    iq, truth, count = observation(scene, 'weak')
    rows = []
    pp = grid(20.)
    for segments in (4, 8, 16, 32):
        for method in ('MUSIC', 'MVDR'):
            start = time.perf_counter()
            m = Model(iq, resource_check)
            spec = Spectrum(m, method, count, segments)
            noise = segments/m.n
            weights = (spec.cov.diagonal(dim1=1, dim2=2).real.sum(1)-4*noise).clamp_min(0)
            for weighted in (False, True) if method == 'MUSIC' else (False,):
                def score(points):
                    vals = []
                    with torch.no_grad():
                        for lo in range(0, len(points), 64):
                            a = m.response(points[lo:lo+64], spec.freq)
                            den = (a.conj()*(spec.matrix@a)).sum(1).real.clamp_min(1e-15)
                            if method == 'MUSIC':
                                den = ((weights[:, None]*den).sum(0)/weights.sum()
                                       if weighted else den.mean(0))
                                v = 1/den
                            else:
                                v = (1/den).mean(0)
                            vals.extend(v.cpu().tolist())
                    return np.asarray(vals)
                ss = score(pp)
                ids = old.peaks(pp, ss, 5)
                options = []
                for i in ids:
                    z = minimize(lambda p: -score(np.asarray(p)[None])[0]/ss[i], pp[i],
                                 method='Powell', bounds=[(max(-1000, v-20), min(1000, v+20)) for v in pp[i]],
                                 options=dict(maxiter=20, xtol=.01, ftol=1e-9))
                    options.append((float(score(z.x[None])[0]), z.x))
                best = max(options, key=lambda x: x[0])[1]
                row = dict(method=method, segments=segments, weighted=weighted,
                           initial=pp[int(np.argmax(ss))].tolist(), prediction=best.tolist(),
                           metrics=evaluate(truth, best[None]), seconds=time.perf_counter()-start,
                           truth_score=float(score(truth)[0]), maximum_score=float(ss.max()),
                           noise_weight_fraction=float(weights[abs(spec.freq)>6.25e6].sum().cpu()/weights.sum().cpu()))
                rows.append(row)
                print(row, flush=True)
                write(out/'spectrum_rows.json', rows)
    return rows


def sbl_probe(scene, out):
    iq, truth, _ = observation(scene, 'weak')
    m = Model(iq, resource_check)
    # Historical 50 m for cause isolation only; NOT the R1 common-grid comparison.
    pp = grid(50.)
    q = len(pp)
    gamma = torch.full((q,), float(m.energy.cpu())/m.n/q, device='cuda', dtype=torch.float64)
    rows = []
    start = time.perf_counter()
    for iteration in range(301):
        second, power, numerator, denominator = [torch.zeros_like(gamma) for _ in range(4)]
        evidence = 0.
        for lo in range(0, m.n, 512):
            resource_check()
            a = m.response(pp, m.f[lo:lo+512])
            mu, diagonal, cov, nll = sbl_posterior(a, m.x[lo:lo+512], gamma, 1/m.n)
            second += (abs(mu)**2+diagonal).sum(0)
            power += (abs(mu)**2).sum(0)
            numerator += (abs(mu/gamma[None])**2).sum(0)
            denominator += ((gamma[None]-diagonal)/gamma[None]**2).sum(0)
            evidence += float(nll.sum().cpu())/m.n
        index = int(power.argmax().cpu())
        if iteration in (0, 1, 2, 5, 10, 20, 50, 100, 200, 300):
            row = dict(iteration=iteration, objective=evidence, prediction=pp[index].tolist(),
                       metrics=evaluate(truth, pp[index:index+1]), seconds=time.perf_counter()-start,
                       gamma_max=float(gamma.max().cpu()), grid=50., update='mm')
            rows.append(row)
            print(row, flush=True)
            write(out/'sbl_rows.json', rows)
        # Positive-covariance majorization update; checks of exact evidence follow in tests.
        gamma = (gamma*torch.sqrt(numerator/denominator.clamp_min(1e-30))).clamp_min(1e-20)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=('spectra', 'sbl'))
    ap.add_argument('--group', type=int, default=0)
    args = ap.parse_args()
    out = BASE/('diagnose_'+args.part+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    ids = identity()
    preflight(out, dict(stage='R1_NATIVE_ADAPTATION_DIAGNOSTIC', part=args.part, r1_sources=ids,
                        geometry=geometry_audit(), note='Not final performance; truth only diagnoses'))
    scene = make(args.group, 0)
    write(out/'input_identity.json', scene['meta'])
    print('OUT='+str(out), flush=True)
    rows = spectra(scene, out) if args.part == 'spectra' else sbl_probe(scene, out)
    verify(ids)
    write(out/'audit.json', dict(completed=True, rows=len(rows), sources=ids,
                                meaning='Diagnostic evidence; not a benchmark qualification'))


if __name__ == '__main__':
    main()
