"""Information-contract audit only; never estimates source positions."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = BASE.parents[2].resolve()
ARCHIVE = BASE / 'contribution_check_20261030'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    def native(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              default=native, allow_nan=False), encoding='utf-8')


def project_relative_gain(z):
    """Euclidean projection onto product of CLOSED complex unit discs.

    Absolute LOS gains must be nonzero for relative coordinates to exist.
    No constraint is imposed on the sum of reflected path powers.
    Strict individual dominance is checked separately, not enforced by an
    arbitrary power margin. This utility does not select a path order.
    """
    z = np.asarray(z, dtype=np.complex128)
    return z / np.maximum(1., np.abs(z))


def project_gap_closure(gaps):
    """Nonnegative increment closure, no generator separation or upper bound.

    Zero increments are boundary/duplicate parameters, NOT positive-delay
    physical paths. They must be reported, not silently accepted as strict.
    """
    return np.maximum(np.asarray(gaps, dtype=float), 0.)


def contract_tests():
    rng = np.random.default_rng(2026103007)
    checks = {}
    for count in (0, 1, 2, 4):
        for trial in range(25):
            shape = (5, 2, count)
            z = 3 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
            p = project_relative_gain(z)
            w = project_relative_gain(rng.normal(size=shape) + 1j*rng.normal(size=shape))
            tag = f'capacity{count}/trial{trial}'
            checks[tag+'/shape'] = p.shape == shape
            checks[tag+'/feasible'] = np.all(np.abs(p) <= 1.+1e-14)
            checks[tag+'/idempotent'] = np.allclose(project_relative_gain(p), p, atol=1e-14, rtol=0)
            checks[tag+'/variational'] = np.real(np.vdot(z-p, w-p)) <= 1e-10
            checks[tag+'/convex'] = np.all(abs(.37*p+.63*w) <= 1.+1e-14)
            checks[tag+'/interior_unchanged'] = np.allclose(project_relative_gain(.7*w), .7*w, atol=1e-14, rtol=0)
            gap = rng.normal(size=shape)*10
            gp = project_gap_closure(gap)
            gw = rng.uniform(0., 100., size=shape)
            checks[tag+'/gap_feasible'] = np.all(gp >= 0.)
            checks[tag+'/gap_projection'] = np.sum((gap-gp)*(gw-gp)) <= 1e-10
            checks[tag+'/gap_idempotent'] = np.array_equal(project_gap_closure(gp), gp)
    z = np.array([.8, .8], dtype=complex)
    boundary = dict(los_power=1., individual_reflection_powers=abs(z)**2,
                    total_reflection_power=float(np.sum(abs(z)**2)),
                    individual_strict_allowed=bool(np.all(abs(z) < 1)),
                    old_total_strict_allowed=bool(np.sum(abs(z)**2) < 1))
    checks['total_can_exceed_LOS'] = boundary['individual_strict_allowed'] and not boundary['old_total_strict_allowed']
    checks['strict_vs_closure'] = bool(abs(project_relative_gain(np.array([2.]))[0]) == 1)
    checks['positive_gap_not_strict_at_boundary'] = bool(project_gap_closure([-1.])[0] == 0)
    # Coordinate convention in frozen v: 16 LOS entries, then real/imag rho.
    v = np.arange(76, dtype=float)
    rho = v[16:36].reshape(5, 2, 2) + 1j*v[36:56].reshape(5, 2, 2)
    projected = v.copy()
    pr = project_relative_gain(rho)
    projected[16:36], projected[36:56] = pr.real.ravel(), pr.imag.ravel()
    checks['frozen_coordinate_mapping'] = np.array_equal(projected[:16], v[:16]) and np.array_equal(projected[56:], v[56:])
    checks['closed_real_coordinate_discs'] = np.all(projected[16:36]**2+projected[36:56]**2 <= 1.+1e-14)
    return checks, boundary


def representation_tests():
    bandwidth, fs, n = 12.5e6, 50e6, 8192
    f = np.fft.fftfreq(n, d=1/fs)
    f = f[abs(f) <= bandwidth/2]
    delay = 5/bandwidth
    merged = 1.4*np.exp(-2j*np.pi*f*delay)
    rows = []
    for normalized_gap in (0., 1e-2, 1e-3, 1e-4):
        dt = normalized_gap/bandwidth
        split = .7*np.exp(-2j*np.pi*f*(delay-dt/2)) + .7*np.exp(-2j*np.pi*f*(delay+dt/2))
        rows.append(dict(normalized_gap=normalized_gap,
                         relative_response_error=float(np.linalg.norm(split-merged)/np.linalg.norm(merged)),
                         both_individually_weaker=True,
                         distinct_positive_delays=bool(dt > 0 and delay-dt/2 > 0)))
    shifted = 1.4*np.exp(-2j*np.pi*f*(delay+n/fs))
    alias_error = float(np.linalg.norm(shifted-merged)/np.linalg.norm(merged))
    checks = dict(exact_duplicate_response=rows[0]['relative_response_error'] < 1e-14,
                  decreasing_nonzero_gap_error=rows[1]['relative_response_error'] > rows[2]['relative_response_error'] > rows[3]['relative_response_error'],
                  fourier_periodic_alias=alias_error < 1e-10)
    return checks, dict(rows=rows, alias_period_s=n/fs, alias_error=alias_error,
                        scope='Algebraic frequency-response diagnostic, NOT position nonidentifiability or finite nonperiodic IQ equivalence.',
                        warning='Individual physical path dominance is valid; finite-band unresolved components need not have a unique parameter decomposition.')


def archive_rows(paths):
    rows = []
    for path in paths:
        obj = json.loads(path.read_text(encoding='utf-8'))
        v = np.asarray(obj['v'])
        rho = v[16:36].reshape(10, 2) + 1j*v[36:56].reshape(10, 2)
        rows.append(dict(file=str(path), method=obj['method'], observation=obj['observation'],
                         phase=obj['phase'], tag=obj['tag'],
                         max_individual_ratio_power=float(np.max(abs(rho)**2)),
                         max_sum_ratio_power=float(np.max(np.sum(abs(rho)**2, axis=1))),
                         min_gap=float(np.min(v[56:])), max_gap=float(np.max(v[56:])),
                         sum_boundary_1e6=bool(np.min(1-np.sum(abs(rho)**2, axis=1)) <= 1e-6),
                         lower_gap_boundary_1e6=bool(np.min(v[56:]-3) <= 1e-6),
                         upper_gap_boundary_1e6=bool(np.min(20-v[56:]) <= 1e-6),
                         stop_reason=obj['reason'],
                         errors_m=obj['errors_m']))
    summary = []
    for observation in ('LOS', 'MP'):
        for method in ('JOINT', 'BLOCK'):
            subset = [r for r in rows if r['observation'] == observation and r['method'] == method]
            summary.append(dict(method=method, observation=observation, count=len(subset),
                                total_power_boundary=sum(r['sum_boundary_1e6'] for r in subset),
                                lower_gap_boundary=sum(r['lower_gap_boundary_1e6'] for r in subset),
                                upper_gap_boundary=sum(r['upper_gap_boundary_1e6'] for r in subset),
                                max_sum_ratio_power=max(r['max_sum_ratio_power'] for r in subset),
                                min_gap=min(r['min_gap'] for r in subset), max_gap=max(r['max_gap'] for r in subset)))
    return rows, summary


def main():
    start = time.perf_counter()
    pointer = ARCHIVE/'task2_full_latest.json'
    latest = json.loads(pointer.read_text(encoding='utf-8'))
    # Resolve through the saved pointer, never silently select newest folders.
    if isinstance(latest, str):
        result_dir = Path(latest)
    else:
        result_dir = Path(latest.get('path', latest.get('directory', latest.get('out', ''))))
    if not result_dir.is_absolute():
        result_dir = PROJECT/result_dir
    result_dir = result_dir.resolve(strict=True)
    assert result_dir.is_relative_to(ARCHIVE.resolve()), result_dir
    records = sorted(result_dir.glob('L*_R*_*_*.json'))
    assert len(records) == 96, len(records)
    old_code = [BASE/name for name in ('boundary_update.py', 'blind_initialization.py', 'position_reduction.py', 'solver_reliability.py')]
    inputs = [pointer, result_dir/'protocol.json', *records, *old_code, HERE/'执行协议.md', Path(__file__), *sorted((HERE/'sources').glob('*'))]
    hashes = {str(p.resolve(strict=True)): sha(p) for p in inputs}
    assert all(Path(p).is_relative_to(PROJECT) for p in hashes)
    old_protocol = json.loads((result_dir/'protocol.json').read_text(encoding='utf-8'))
    known_identity = {p: sha(p) == digest for p, digest in old_protocol['hashes'].items() if Path(p).is_file()}
    assert all(known_identity.values()), 'Frozen input/source identity drift'
    out = HERE/('audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'contract_audit.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json', dict(hashes=hashes, original_protocol_identity=known_identity,
                                  python=platform.python_version(), numpy=np.__version__,
                                  kind='CONTRACT_AND_SAVED_RESULT_AUDIT_NOT_NEW_LOCALIZATION'))
    checks, boundary = contract_tests()
    rc, representations = representation_tests()
    checks.update({'representation/'+k: v for k, v in rc.items()})
    rows, summary = archive_rows(records)
    identity = {p: sha(p) == digest for p, digest in hashes.items()}
    checks['input_and_source_unchanged'] = all(identity.values())
    write(out/'archive_rows.json', rows)
    write(out/'result.json', dict(checks=checks, passed=sum(bool(v) for v in checks.values()),
                               total=len(checks), all_pass=all(checks.values()),
                               boundary_case=boundary, representation=representations,
                               archive_summary=summary, identity=identity,
                               wall_seconds=time.perf_counter()-start,
                               new_localization_solves=0, task3_started=False))
    write(HERE/'latest_audit.json', dict(path=str(out)))
    print('单条直达占优核查：', sum(bool(v) for v in checks.values()), '/', len(checks))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('输出：', out)
    assert all(checks.values()), 'Check failure retained in result.json'


if __name__ == '__main__':
    main()
