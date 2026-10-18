"""Semantic and numerical tests; CPU-only mode never creates a CUDA context."""
import os
import subprocess
import sys
import traceback
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from ..metrics import evaluate
from ..physics import fit
from ..r1.scenes import make as make_scene
from ..r2.band.physics import Model as ReferenceModel
from .solvers import Model, initial_starts, canonical_positions, projected_gradient
from .config import PREP_METHODS, SHORT_TASKS, FORMAL_TASKS, jobs
from .runtime import (reference, source_ids, verify, new_layouts, require_exclusive,
                      read, write, seal, validate, load_input)


def common_stop_checks():
    from .common_stop import checked_point, refine_pair, PrecisionReached

    class Quadratic:
        def __init__(self):
            self.counts = dict(local_calls=0, iterations=0)
            self.target = np.array([1., 2., -3., 4.])
            self.h = np.array([[4., 0., 1., 0.], [0., 3., 0., .5],
                               [1., 0., 2., 0.], [0., .5, 0., 1.]])

        def joint(self, p, gradient=False):
            d = np.asarray(p).ravel()-self.target
            value, g = float(.5*d@self.h@d), self.h@d
            return (value, g) if gradient else value

    class ToyTrace:
        def __init__(self):
            self.best, self.history = np.inf, []

        def save(self, p, value):
            self.best = min(self.best, value)

        def iteration(self, p, value, **extra):
            self.history.append((np.asarray(p).copy(), value, extra))

    cfg = dict(gtol=1e-8, common_ftol=0., maxls=25, local_maxiter=100, alternating_sweeps=12)
    m, trace = Quadratic(), ToyTrace()
    # A zero gradient in one coordinate block cannot terminate the full problem.
    m.h = np.eye(4)
    p = m.target.reshape(2, 2).copy(); p[1, 0] += 1
    checked_point(m, trace, p, cfg, 'fixture')
    try:
        checked_point(m, trace, m.target, cfg, 'fixture')
    except PrecisionReached:
        pass
    else:
        raise AssertionError('Exact common optimum not recognized')
    for method in PREP_METHODS:
        m, trace = Quadratic(), ToyTrace()
        p, _, info = refine_pair(m, trace, np.zeros((2, 2)), method, cfg)
        assert info['stop_reason'] == 'COMMON_GRADIENT_REACHED'
        assert projected_gradient(p, m.joint(p, True)[1]) <= cfg['gtol']
        hits = [i for i, (q, _, _) in enumerate(trace.history)
                if projected_gradient(q, m.joint(q, True)[1]) <= cfg['gtol']]
        assert hits == [len(trace.history)-1], 'Did not stop at first accepted qualifying state'
        limited = dict(cfg, local_maxiter=1, alternating_sweeps=1)
        m, trace = Quadratic(), ToyTrace()
        _, _, info = refine_pair(m, trace, np.zeros((2, 2)), method, limited)
        assert info['stop_reason'] != 'COMMON_GRADIENT_REACHED'
        assert info['projected_gradient'] > cfg['gtol']
    return True


def cpu_suite(out):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checks = {}
    ref = reference(); verify(source_ids(ref))
    checks['r2_frozen_sources_unchanged'] = True
    assert not torch.cuda.is_initialized()
    choices = np.array([[80, 120], [-200, 250], [320, -100], [-400, -500], [80, 120], [640, 200]])
    for count in (1, 2):
        starts, kinds = initial_starts(choices, count, 123)
        assert starts.shape == (8, count, 2)
        assert len({canonical_positions(p) for p in starts}) == 8
        assert kinds == ['spectrum']*4+['sobol']*4
        assert np.array_equal(starts, initial_starts(choices, count, 123)[0])
        assert abs(starts).max() <= 1000
    assert canonical_positions(choices[:2]) == canonical_positions(choices[:2][::-1])
    checks['actual_8_unique_4_plus_4_k1_k2'] = True
    with patch('统一模型代码.gates.g10.r3.runtime.active_project_processes',
               return_value=[dict(pid=123, command='SBL')]):
        try:
            require_exclusive()
        except RuntimeError:
            checks['active_sbl_blocks_compute'] = True
        else:
            raise AssertionError('SBL exclusion failed')
    rng = np.random.default_rng(67)
    iq = rng.normal(size=(4, 256))+1j*rng.normal(size=(4, 256))
    m = Model(iq, device='cpu')
    ref_m = ReferenceModel(iq, device='cpu')
    p = np.array([[173., 117.], [813., 213.]])
    value, grad = m.joint(p, True)
    calls = m.counts['joint_evaluations']
    assert m.joint(p) == value
    np.testing.assert_array_equal(m.joint(p, True)[1], grad)
    assert m.counts['joint_evaluations'] == calls
    np.testing.assert_allclose(value, ref_m.joint(p), atol=1e-13, rtol=0)
    a = m.response(p)
    direct = float((abs(fit(a, m.x)[1])**2).sum()/m.energy)
    np.testing.assert_allclose(value, direct, atol=1e-12, rtol=0)
    delta, fd = 1e-3, []
    for i in range(4):
        plus, minus = p.copy().ravel(), p.copy().ravel()
        plus[i] += delta; minus[i] -= delta
        fd.append((m.joint(plus)-m.joint(minus))/(2*delta))
    np.testing.assert_allclose(grad, fd, rtol=2e-4, atol=1e-9)
    checks['small_ls_svd_gradient_exact_memo'] = True
    np.testing.assert_allclose(m.joint(p[::-1]), value, atol=1e-12, rtol=0)
    for count in (1, 2):
        clean = m.response(p[:count])
        coeff = torch.as_tensor(rng.normal(size=(m.n, count))+1j*rng.normal(size=(m.n, count)))
        m.x = (clean@coeff[..., None]).squeeze(-1)
        m.energy = abs(m.x).square().sum(); m.memo.clear()
        assert m.joint(p[:count]) < 1e-20
    # A model/memo is per observation; changing data is NEVER done in the solver.
    full_iq = np.zeros((4, 8192), dtype=np.complex128); full_iq[:, 0] = 1
    full = ReferenceModel(full_iq, device='cpu')
    assert full.n == 2293 and full.sample_count == 8192
    assert full.frequency_noise_variance == 1/8192
    np.testing.assert_allclose(full.x.cpu().numpy(), 1/8192, atol=0, rtol=0)
    checks['noiseless_k1_k2_offgrid_and_band_fft_scale'] = True
    assert projected_gradient([[-1000., 1000.]], [1., -1.]) == 0
    assert projected_gradient([[-1000., 1000.]], [-1., 1.]) == 1
    truth = np.array([[0., 0.], [200., 0.]])
    assert evaluate(truth, [[20., 0.], [200., 0.]])['success']['20'] == [True, True]
    assert evaluate(truth, [[20.001, 0.], [200., 0.]])['success']['20'] == [False, True]
    assert evaluate(truth, truth[::-1])['errors_m'] == [0., 0.]
    assert evaluate(truth, [])['success']['20'] == [False, False]
    checks['box_gradient_matching_boundary_missing_outputs'] = True
    assert len(SHORT_TASKS) == 24 and len(FORMAL_TASKS) == 120
    assert len(jobs('prepare')) == 240 and len(jobs('formal')) == 480
    checks['common_gradient_first_hit_and_nonconvergence'] = common_stop_checks()
    geometry = new_layouts()
    assert len(geometry) == 4
    checks['task_counts_and_geometry'] = True
    first = make_scene(0, 40, 0)
    last = make_scene(0, 40, 15)
    np.testing.assert_array_equal(first['noise'], last['noise'])
    np.testing.assert_array_equal(first['signal'][1], last['signal'][1])
    np.testing.assert_allclose(last['signal'][0], first['signal'][0]*10**(15/20), rtol=1e-14, atol=1e-14)
    assert first['meta']['N'] == 8192 and first['meta']['sample_rate_hz'] == 50e6
    np.testing.assert_allclose(np.mean(abs(first['signal'][1])**2), .25*10**.6, rtol=1e-13)
    checks['paired_noise_weak_signal_and_inband_snr'] = True
    # Identity validation, corruption rejection and atomic Unicode JSON round trip.
    job = jobs('prepare')[0]
    config = ref['configurations'][job['method']]
    row = dict(job=job, input_sha='toy_input', truth=truth.tolist(),
               result=dict(config=config, method=job['method'], positions=truth.tolist()),
               metrics=evaluate(truth, truth))
    row['record_sha'] = seal(row)
    write(out/'中文恢复测试.json', row)
    validate(read(out/'中文恢复测试.json'), job, 'toy_input', config, truth)
    try:
        validate(row, job, 'different_input', config, truth)
    except RuntimeError:
        checks['identity_rejection_and_atomic_roundtrip'] = True
    else:
        raise AssertionError('Changed input accepted')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    capture = subprocess.run([sys.executable, '-c', "print('中文日志：恢复检查通过')"],
        capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, check=True)
    assert '中文日志：恢复检查通过' in capture.stdout
    checks['non_ascii_subprocess'] = True
    from .runner import record
    toy_clock = SimpleNamespace(state={'interruptions': []}, total=lambda: 0.,
                                flush=lambda *a, **k: None, check=lambda: None)
    toy_contract = dict(configurations=ref['configurations'])
    toy_input = (dict(meta={'engineering_fixture': True}), np.zeros((4, 8192), complex), truth, 'toy_input')
    toy_result = dict(config=config, method=job['method'], positions=truth.tolist())
    with patch('统一模型代码.gates.g10.r3.runner.load_input', return_value=toy_input), \
         patch('统一模型代码.gates.g10.r3.runner.require_exclusive'):
        with patch('统一模型代码.gates.g10.r3.solvers.run', side_effect=KeyboardInterrupt):
            try:
                record(out/'driver_fixture', job, toy_clock, toy_contract)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Interruption did not propagate')
        assert not list((out/'driver_fixture'/'rows').glob('*.json'))
        with patch('统一模型代码.gates.g10.r3.solvers.run', return_value=toy_result) as solver:
            _, computed = record(out/'driver_fixture', job, toy_clock, toy_contract)
            assert computed and solver.call_count == 1
        with patch('统一模型代码.gates.g10.r3.solvers.run', side_effect=AssertionError('Unexpected recomputation')):
            _, computed = record(out/'driver_fixture', job, toy_clock, toy_contract)
            assert not computed
    checks['driver_interruption_resume_without_recompute'] = True
    assert not torch.cuda.is_initialized()
    checks['cuda_never_initialized'] = True
    report = dict(passed=all(checks.values()), checks=checks,
                  scope='CPU engineering/numerical checks; not localization performance',
                  python=sys.version, numpy=np.__version__, torch=torch.__version__,
                  sources=source_ids(ref))
    write(out/'cpu_checks.json', report)
    torch.set_num_threads(previous_threads)
    return report


def gpu_suite(out):
    """Run only after exclusive-process check. No truth enters either solver."""
    from . import solvers
    from ..r2.band.solvers import run as old_run
    require_exclusive()
    contract = read(out/'contract.json')
    details = []
    for task in ((0, 200, 0, 'mixed'), (1, 200, 15, 'mixed'),
                 (2, 201, 15, 'mixed'), (3, 201, 0, 'mixed')):
        _, iq, _, _ = load_input(out, task, contract)
        for method in PREP_METHODS:
            cfg = contract['configurations'][method]
            old = old_run(iq, method, 2, cfg, 123)
            new = solvers.run(iq, method, 2, cfg, 123)
            no_history = solvers.run(iq, method, 2, cfg, 123, history=False)
            no_memo = solvers.run(iq, method, 2, cfg, 123, memo=False)
            for result in (no_history, no_memo):
                np.testing.assert_allclose(result['positions'], new['positions'], atol=1e-7, rtol=0)
                assert result['stop_reason'] == new['stop_reason']
            np.testing.assert_allclose(new['mechanism']['branches'][0]['initial_positions'],
                old['mechanism']['branches'][0]['initial_positions'], atol=1e-10, rtol=0)
            assert new['counts'] == no_history['counts']
            assert new['precision_met'] == (new['stop_reason'] == 'COMMON_GRADIENT_REACHED')
            np.testing.assert_allclose([e['native_objective'] for e in new['events']],
                                       [e['native_objective'] for e in no_memo['events']], atol=1e-13, rtol=0)
            accepted = new['accepted_certificates']
            hits = [i for i, item in enumerate(accepted) if item['projected_gradient'] <= cfg['gtol']]
            if hits:
                assert hits[0] >= len(accepted)-2, 'Continued after common precision'
            details.append(dict(task=list(task), method=method, old_counts=old['counts'],
                memo_counts=new['counts'], old_seconds=old['seconds'], new_seconds=new['seconds'],
                stop_reason=new['stop_reason'], projected_gradient=new['projected_gradient']))
    report = dict(passed=True, native_prefix_unchanged=True, memo_trajectory_equivalence=True,
                  history_equivalence=True, common_stop_checked=True, details=details,
                  unchanged_nomp_multistart_checks='Reuse original R3 preparation; no rerun')
    write(out/'gpu_checks.json', report)
    return report


def main():
    from .runtime import fresh
    out = fresh('cpu_checks'); out.mkdir(parents=True)
    try:
        result = cpu_suite(out)
        print({k: v for k, v in result.items() if k != 'sources'})
    except BaseException:
        write(out/'failure.json', dict(passed=False, traceback=traceback.format_exc(), scope='CPU engineering test'))
        raise
    print('CPU核验目录：', out)


if __name__ == '__main__':
    main()
