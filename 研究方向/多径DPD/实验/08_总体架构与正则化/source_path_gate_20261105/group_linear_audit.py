"""Saved-state joint-versus-marginal linear increment information; no localization."""
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose import HERE, cfg, ad, stable, data


def main():
    source = HERE/'increment_control_v1'
    stable.inside.rr.check_hashes(cfg.read(source/'identity.json'))
    rows = []
    for path in sorted(source.glob('*_start.json')):
        tag = path.name.removesuffix('_start.json')
        d = cfg.read(path)
        iq = data(tag)
        assert cfg.digest(iq) == d['iq_sha']
        e = stable.Engine()
        f, y, n, sigma = ad.b.fc.cal.spectral(iq)
        x = np.asarray(d['previous']['x'])
        residual, tangent, info = e.evaluate(x, f, y, d['paths'], True)
        h, inverse, waveform, base = info
        u, singular, _ = np.linalg.svd(tangent, full_matrices=False)
        u = u[:, singular > singular[0]*1e-10]
        rp = residual-u@(u.T@residual)
        blocks = []
        individual = []
        for choice in d['choices']:
            m, q, delay = map(int, choice[:3])
            bc = np.zeros((len(f), e.M, 2), complex)
            atom = base[:, m, q]*waveform[:, q]*np.exp(-2j*np.pi*f*delay/cfg.FS)
            bc[:, m, 0], bc[:, m, 1] = atom, 1j*atom
            bc -= h@(inverse@bc)
            b = e.ref.cols(bc)
            z = b-u@(u.T@b)
            blocks.append(z)
            fitted = z@(np.linalg.pinv(z, rcond=1e-12)@rp)
            individual.append(float(fitted@fitted))
            assert np.isclose(individual[-1]/sigma, choice[4], rtol=2e-5, atol=1e-5)
        matrix = np.column_stack(blocks)
        uu, ss, _ = np.linalg.svd(matrix, full_matrices=False)
        uu = uu[:, ss > ss[0]*1e-12]
        joint = float(np.sum((uu.T@rp)**2))
        bases = [np.linalg.qr(b, mode='reduced')[0] for b in blocks]
        correlation = max(float(np.linalg.svd(a.T@b, compute_uv=False)[0])
            for i, a in enumerate(bases) for b in bases[i+1:])
        rows.append(dict(tag=tag, components=len(blocks), marginal_sum=float(sum(individual)),
            joint_linear_reduction=joint, marginal_sum_over_joint=float(sum(individual)/joint),
            max_canonical_correlation=correlation, no_new_estimation=True))
    cfg.write(HERE/'group_linear_audit.json', dict(rows=rows, source_sha=cfg.sha(__file__),
        scope='Local linear diagnostic only; not a proof of the unique cause of nonlinear failure'))
    for row in rows:
        print(row, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
