"""Explain one positive score / uphill amplitude ray using exact saved state."""
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose import HERE, cfg, ad, stable, data


def main():
    tag = 'IN_103_R00_L3_P150'
    src = HERE/'single_full_v2'
    stable.inside.rr.check_hashes(cfg.read(src/'identity.json'))
    old = cfg.read(src/(tag+'.json'))['result']
    which = len(old['history'])-1
    previous = old['history'][which-1]['fit']
    x = np.asarray(previous['x'])
    paths = old['paths']
    iq = data(tag)
    e = stable.Engine()
    f, y, n, sigma = ad.b.fc.cal.spectral(iq)
    use = abs(np.fft.fftfreq(n, 1/cfg.FS)) <= cfg.B/2
    pick, _, meta = e.search(x, paths, f, y, n, use, sigma, which+1)
    pp = paths+[(pick['m'], pick['q'])]
    seed = e.ref.project(np.r_[x, pick['rho'], pick['delay']], n)
    zero = seed.copy(); zero[-3:-1] = 0
    amplitude_ray = seed-zero
    cost, gradient = e.objective(zero, f, y, pp)
    raw, tangent, _ = e.evaluate(x, f, y, paths, True)
    raw2, aug_tangent, _ = e.evaluate(zero, f, y, pp, True)
    assert np.allclose(raw, raw2)
    b = aug_tangent[:, -3:-1]
    compensation = -np.linalg.pinv(tangent, rcond=1e-10)@(b@amplitude_ray[-3:-1])
    joint_ray = amplitude_ray.copy(); joint_ray[:len(x)] = compensation
    disks = []
    for k in range(e.D, len(x), 3):
        rho = x[k:k+2]
        if np.linalg.norm(rho) > 1-1e-6:
            disks.append(dict(path_index=(k-e.D)//3, norm=float(np.linalg.norm(rho)),
                outward_derivative=float(2*rho@compensation[k:k+2])))
    probes = []
    for alpha in [1e-4, 1e-6, 1e-8]:
        free = zero+alpha*joint_ray
        feasible = e.ref.project(free, n)
        probes.append(dict(alpha=alpha,
            free_delta_per_alpha=(e.objective(free, f, y, pp)[0]-cost)/alpha,
            projected_delta_per_alpha=(e.objective(feasible, f, y, pp)[0]-cost)/alpha,
            projected_directional_derivative=float(gradient@(feasible-zero)/alpha),
            infeasibility=float(np.linalg.norm(free-feasible))))
    result = dict(tag=tag, evidence=pick['evidence'], threshold=meta['threshold'],
        old_pg=previous['projected_gradient'], old_gradient_max=float(np.max(abs(gradient[:len(x)]))),
        amplitude_only_slope=float(gradient@amplitude_ray),
        compensated_joint_slope=float(gradient@joint_ray), active_disks=disks, probes=probes,
        source_sha=cfg.sha(__file__), state_sha=cfg.sha(src/(tag+'.json')),
        scope='Local diagnostic, no changed score and no new estimator; not proof all score failures share this cause')
    cfg.write(HERE/'boundary_score_audit.json', result)
    print(result)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
