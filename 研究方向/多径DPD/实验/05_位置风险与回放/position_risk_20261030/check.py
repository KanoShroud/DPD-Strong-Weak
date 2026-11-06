"""Position-focused LOCAL risk checks, not an adaptive DPD estimator."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from structural_probe import STATIONS, B, C  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    def native(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=native,
                           allow_nan=False), encoding='utf-8')


def info(a, b):
    """Both inputs already residualized against common nuisance directions."""
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    if s[-1] < s[0]*1e-10:
        return None
    l0 = np.linalg.pinv(a, rcond=1e-10)
    br = b-u@(u.T@b)
    sb = np.linalg.svd(br, compute_uv=False)
    if sb[-1] <= np.linalg.norm(b)*1e-10:
        return None
    ca = np.linalg.inv(br.T@br)
    d = l0@b
    la = ca@br.T
    l1 = l0-d@la
    penalty = d@ca@d.T
    return l0, l1, la, d, ca, penalty


def linear_checks():
    rng = np.random.default_rng(2026103011)
    q = np.linalg.qr(rng.normal(size=(40, 40)))[0]
    a = q[:, :4]@np.diag([1., .8, .6, .4])
    nuisance = q[:, 4:9]
    noise = rng.normal(size=(40, 20000))
    checks, rows = {}, []
    for coupling in (0., .5, .95):
        b = coupling*q[:, :2]+np.sqrt(1-coupling**2)*q[:, 9:11]
        l0, l1, la, d, ca, penalty = info(a, b)
        direct = np.linalg.pinv(np.column_stack([a, nuisance, b]))[:4]
        checks[f'{coupling}/full_nuisance_LS'] = np.allclose(l1, direct, atol=1e-10)
        checks[f'{coupling}/variance_identity'] = np.allclose(l1@l1.T-l0@l0.T, penalty, atol=1e-10)
        for amplitude in (0., .5, 3.):
            alpha = np.array([amplitude, -.4*amplitude])
            y = b@alpha[:, None]+noise
            p0, p1 = l0@y, l1@y
            ahat = la@y
            losses = np.sum(p0*p0-p1*p1, axis=0)
            risk_est = np.sum((d@ahat)**2, axis=0)-2*np.trace(penalty)
            risk0 = np.sum((d@alpha)**2)+np.sum(l0*l0)
            risk1 = np.sum(l1*l1)
            theoretical = float(risk0-risk1)
            mse0, mse1 = np.sum(p0*p0, axis=0), np.sum(p1*p1, axis=0)
            for label, samples, expected in [('mse0', mse0, risk0), ('mse1', mse1, risk1),
                                               ('risk_difference', risk_est, theoretical)]:
                se = float(np.std(samples, ddof=1)/np.sqrt(len(samples)))
                checks[f'{coupling}/{amplitude}/{label}'] = abs(float(np.mean(samples))-expected) <= 6*se+1e-10
            rows.append(dict(coupling=coupling, amplitude=amplitude,
                             theoretical_mse_without=float(risk0), theoretical_mse_with=float(risk1),
                             empirical_mse_without=float(mse0.mean()), empirical_mse_with=float(mse1.mean()),
                             theoretical_improvement=theoretical, estimated_improvement_mean=float(risk_est.mean()),
                             empirical_improvement=float(losses.mean()),
                             expected_residual_reduction=float(alpha@(b-a@(l0@b)).T@(b-a@(l0@b))@alpha+2),
                             monte_carlo_n=len(losses)))
    checks['exact_alias_is_rejected'] = info(a, a[:, :2]) is None
    # Fixed bank: it is NOT a bank estimated/selected from these same noises.
    tests = []
    for _ in range(32):
        b = rng.normal(size=(40, 2))
        b -= nuisance@(nuisance.T@b)
        l0, l1, la, d, ca, penalty = info(a, b)
        ahat = la@noise
        tests.append(np.einsum('in,ij,jn->n', ahat, np.linalg.inv(ca), ahat))
    scores = np.asarray(tests)
    raw = float(np.mean(np.max(scores, axis=0) > chi2.isf(.05, 2)))
    corrected = float(np.mean(np.max(scores, axis=0) > chi2.isf(.05/32, 2)))
    checks['fixed_bank_family_control'] = corrected <= .05+6*np.sqrt(.05*.95/20000)
    return checks, rows, dict(candidates=32, alpha=.05, uncorrected_family_false_positive=raw,
                             bonferroni_family_false_positive=corrected,
                             scope='Fixed linear Gaussian design only; no continuous delay search or data-dependent tangent guarantee.')


def real_columns(z):
    return np.sqrt(2)*np.concatenate([z.real.reshape(-1, z.shape[-1]), z.imag.reshape(-1, z.shape[-1])])


def rf_directions(record, inp):
    saved = json.loads(record.read_text(encoding='utf-8'))
    p, v = np.asarray(saved['p']), np.asarray(saved['v'])
    g = np.ones((5, 2), complex)
    g[1:] = (v[:8]+1j*v[8:16]).reshape(4, 2)
    with np.load(inp, allow_pickle=False) as data:
        signal = data['los'].copy()
        if saved['observation'] == 'MP':
            signal += data['reflected']
        iq = signal.sum(axis=0)+data['noise']
    # No true position, gain or path delay is read from the generating file.
    frequency = np.fft.fftfreq(iq.shape[1], 1/50e6)
    use = abs(frequency) <= B/2
    f = frequency[use]
    y = np.fft.fft(iq, axis=1, norm='ortho')[:, use].T
    vector = p[None]-STATIONS[:, None]
    dist = np.linalg.norm(vector, axis=2)
    unit = vector/dist[:, :, None]
    phase = np.exp(-2j*np.pi*f[:, None, None]*dist[None]/C)
    h = phase*g[None]
    inv = np.linalg.pinv(h, rcond=1e-12)
    s = np.einsum('fqm,fm->fq', inv, y)

    def pw(z):
        return z-np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, z))

    deriv = np.zeros((len(f), 5, 20), complex)
    for source in range(2):
        mean = h[:, :, source]*s[:, source, None]
        for xy in range(2):
            deriv[:, :, 2*source+xy] = mean*(-2j*np.pi*f[:, None]/C)*unit[None, :, source, xy]
        for m in range(1, 5):
            k = 4+(m-1)*2+source
            deriv[:, m, k] = phase[:, m, source]*s[:, source]
            deriv[:, m, k+8] = 1j*deriv[:, m, k]
    jr = real_columns(pw(deriv))
    un, sn, _ = np.linalg.svd(jr[:, 4:], full_matrices=False)
    un = un[:, sn > sn[0]*1e-10]

    def pn(z):
        return z-un@(un.T@z)

    a = pn(jr[:, :4])
    rows = []
    for m in range(5):
        for source in range(2):
            for delay in (1., 2., 4., 8., 16.):
                b = np.zeros((len(f), 5, 2), complex)
                b[:, m, 0] = h[:, m, source]*s[:, source]*np.exp(-2j*np.pi*f*delay/B)
                b[:, m, 1] = 1j*b[:, m, 0]
                br = pn(real_columns(pw(b)))
                terms = info(a, br)
                row = dict(tag=saved['tag'], station=m, source=source, normalized_probe_delay=delay,
                           local_regular=terms is not None)
                if terms is not None:
                    l0, l1, la, d, ca, penalty = terms
                    bias_per_unit = d@np.array([1., 0.])
                    row.update(position_response_energy=float(np.sum(bias_per_unit**2)),
                               same_source_energy=float(np.sum(bias_per_unit[2*source:2*source+2]**2)),
                               other_source_energy=float(np.sum(bias_per_unit[2*(1-source):2*(1-source)+2]**2)),
                               variance_penalty_per_unit_noise=float(np.trace(penalty)),
                               residual_visibility=float(np.linalg.norm(br-a@(l0@br))**2),
                               variance_identity_relative=float(np.linalg.norm(l1@l1.T-l0@l0.T-penalty)/max(np.linalg.norm(penalty), 1e-30)))
                rows.append(row)
    return rows


def main():
    clock = time.perf_counter()
    archive = BASE/'contribution_check_20261030'
    pointer = archive/'task2_full_latest.json'
    old = Path(json.loads(pointer.read_text(encoding='utf-8'))['path']).resolve(strict=True)
    protocol = json.loads((old/'protocol.json').read_text(encoding='utf-8'))
    selected = []
    for tag in ('L20_R200', 'L40_R300'):
        input_candidates = [Path(k) for k in protocol['hashes'] if Path(k).name == tag+'_input.npz']
        assert len(input_candidates) == 1
        for mode in ('LOS', 'MP'):
            selected.append((old/(tag+'_'+mode+'_BLOCK.json'), input_candidates[0]))
    inputs = [pointer, old/'protocol.json', HERE/'执行协议.md', Path(__file__), BASE/'structural_probe.py']
    inputs += [v for pair in selected for v in pair]
    hashes = {str(p.resolve(strict=True)): sha(p) for p in inputs}
    for p, digest in hashes.items():
        assert Path(p).is_relative_to(BASE.resolve())
        if p in protocol['hashes']:
            assert digest == protocol['hashes'][p]
    out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'check.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json', dict(hashes=hashes, python=platform.python_version(), numpy=np.__version__,
                                   kind='FIXED_LOCAL_RISK_NOT_LOCALIZATION', seed=2026103011))
    checks, linear, multiple = linear_checks()
    write(out/'linear.json', linear)
    rf = []
    for record, inp in selected:
        rf.extend(rf_directions(record, inp))
        print('完成局部方向：', record.stem, flush=True)
    write(out/'rf_directions.json', rf)
    regular = [r for r in rf if r['local_regular']]
    checks['rf_variance_identity'] = all(r['variance_identity_relative'] < 1e-7 for r in regular)
    checks['source_identity_unchanged'] = all(sha(p) == h for p, h in hashes.items())
    write(out/'result.json', dict(checks=checks, passed=sum(bool(v) for v in checks.values()), total=len(checks),
                                 all_pass=all(checks.values()), multiple_testing=multiple,
                                 rf_count=len(rf), rf_regular=len(regular),
                                 max_cross_source_energy_fraction=max(r['other_source_energy']/max(r['position_response_energy'],1e-30) for r in regular),
                                 wall_seconds=time.perf_counter()-clock, new_localization_solves=0))
    write(HERE/'latest.json', dict(path=str(out)))
    print('结果：', out, '检查', sum(bool(v) for v in checks.values()), '/', len(checks))
    assert all(checks.values()), 'Failures retained'


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
