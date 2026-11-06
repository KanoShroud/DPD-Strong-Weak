"""Independent algebra, source semantics, basic behavior and paired-prefix checks."""
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ..physics import atom, fit
from ..r3.solvers import Model, run as old_run
from ..r3.checks import common_stop_checks
from ..metrics import evaluate
from .bnomp import fixed_derivatives, raw_response, coefficient, refine
from .runtime import write, load, read
from .solvers import run


def cpu(out, contract):
    checks = {}
    from .runtime import active_project_processes
    from ..r3.runtime import active_project_processes as slow_scan
    assert active_project_processes() == slow_scan()
    forbidden = lambda: (_ for _ in ()).throw(AssertionError('Non-Python command line queried'))
    fake = [SimpleNamespace(pid=-1, info={'pid': -1, 'name': 'editor.exe'}, cmdline=forbidden),
            SimpleNamespace(pid=-2, info={'pid': -2, 'name': 'python.exe'},
                cmdline=lambda: ['python', 'F:/SourceCount_DPD_E2E/test.py']),
            SimpleNamespace(pid=-3, info={'pid': -3, 'name': 'python.exe'},
                cmdline=lambda: ['python', 'C:/unrelated/script.py'])]
    with patch('psutil.process_iter', return_value=iter(fake)):
        assert [p['pid'] for p in active_project_processes()] == [-2]
    checks['process_exclusion_same_predicate_without_non_python_cmdline'] = True
    rng = np.random.default_rng(2026101701)
    iq = rng.normal(size=(4, 256))+1j*rng.normal(size=(4, 256))
    m = Model(iq, device='cpu')
    p = np.array([173.4, 117.9])
    data = m.x
    a = raw_response(m, p).squeeze(-1)
    q = coefficient(a, data).detach()
    value, gradient, hessian = fixed_derivatives(m, p, data, q)

    def fixed(z):
        a = atom(z.reshape(1, 2), m.f, m.stations, normalized=False).squeeze(-1)
        residual = data-a*q[:, None]
        return (residual.real.square()+residual.imag.square()).sum()/m.energy

    z = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    direct = fixed(z)
    g = torch.autograd.grad(direct, z)[0].detach().numpy()
    h = torch.autograd.functional.hessian(fixed, z).detach().numpy()
    np.testing.assert_allclose(value, direct.detach().numpy(), atol=1e-13, rtol=0)
    np.testing.assert_allclose(gradient, g, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(hessian, h, atol=1e-12, rtol=1e-9)
    step = .001
    finite_g, finite_h = [], []
    for i in range(2):
        d = np.eye(2)[i]*step
        high = fixed_derivatives(m, p+d, data, q)
        low = fixed_derivatives(m, p-d, data, q)
        finite_g.append((high[0]-low[0])/(2*step))
        finite_h.append((high[1]-low[1])/(2*step))
    np.testing.assert_allclose(gradient, finite_g, atol=1e-10, rtol=2e-5)
    np.testing.assert_allclose(hessian, np.array(finite_h).T, atol=1e-10, rtol=2e-5)
    checks['raw_fixed_coefficient_gradient_hessian_autograd_finite_difference'] = True
    positions = np.array([[173.4, 117.9], [-137.8, -213.6]])
    a = raw_response(m, positions)
    residual = fit(a, data)[1]
    direct_value = float(residual.abs().square().sum()/m.energy)
    joint_value, joint_gradient = m.joint(positions, True)
    np.testing.assert_allclose(joint_value, direct_value, atol=1e-12, rtol=0)
    fd = []
    for i in range(4):
        d = np.eye(4)[i].reshape(2, 2)*step
        fd.append((m.joint(positions+d)-m.joint(positions-d))/(2*step))
    np.testing.assert_allclose(joint_gradient, fd, atol=1e-10, rtol=2e-5)
    s, residual = fit(m.response(positions), data)
    z = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    projection_value = -2*(residual.detach().conj()*
        (m.response(z)@s.detach()[..., None]).squeeze(-1)).sum().real/m.energy
    envelope = torch.autograd.grad(projection_value, z)[0].detach().numpy().ravel()
    np.testing.assert_allclose(joint_gradient, envelope, atol=1e-11, rtol=1e-7)
    checks['raw_normalized_svd_profile_gradient_and_envelope'] = True
    first = positions[:1]
    trial = positions[1:]
    gain = m.conditional(trial, first)[0]
    np.testing.assert_allclose(gain, m.joint(first)-m.joint(positions), atol=1e-12, rtol=0)
    checks['conditional_score_equals_joint_explained_energy_gain'] = True
    cfg = contract['configurations']['BNOMP_REF']
    _, _, _, updates = refine(m, p, data, cfg)
    assert all(not x['accepted'] or x['fixed_q_after'] <= x['before'] for x in updates)
    checks['newton_accepts_only_same_fixed_q_objective'] = True
    for count in (1, 2):
        clean_a = m.response(positions[:count])
        s = torch.as_tensor(rng.normal(size=(m.n, count))+1j*rng.normal(size=(m.n, count)))
        m.x = (clean_a@s[..., None]).squeeze(-1)
        m.energy = m.x.abs().square().sum(); m.memo.clear()
        assert m.joint(positions[:count]) < 1e-20
    checks['offgrid_noiseless_one_two_source_objective'] = True
    full = Model(np.ones((4, 8192), complex), device='cpu')
    assert full.n == 2293 and full.sample_count == 8192 and full.frequency_noise_variance == 1/8192
    assert np.max(abs(full.f.numpy())) <= 7e6
    checks['frequency_axis_band_sample_count_noise_scale'] = True
    truth = np.array([[0., 0.], [200., 0.]])
    assert evaluate(truth, truth[::-1])['errors_m'] == [0., 0.]
    assert evaluate(truth, [[20., 0.], [200., 0.]])['success']['20'] == [True, True]
    assert evaluate(truth, [[20.001, 0.], [200., 0.]])['success']['20'] == [False, True]
    assert evaluate(truth, [])['success']['20'] == [False, False]
    checks['matching_swap_missing_and_20m_boundary'] = True
    checks['common_stopping_checks_full_not_block_gradient'] = common_stop_checks()
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    output = subprocess.run([sys.executable, '-c', "print('中文日志：R4核验通过')"],
        capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, check=True)
    assert '中文日志：R4核验通过' in output.stdout
    write(out/'中文日志检查.json', dict(message=output.stdout.strip()))
    assert read(out/'中文日志检查.json')['message'] == '中文日志：R4核验通过'
    checks['non_ascii_log_capture_and_save'] = True
    report = dict(passed=True, checks=checks, scope='numerical/engineering checks, not performance evidence')
    write(out/'cpu_checks.json', report)
    return report


def gpu(out, contract):
    cfgs = contract['configurations']
    task = [0, 200, 0, 'mixed']
    _, iq, _, _ = load(contract, task)
    pairs = []
    for method in ('NOMP', 'AP_1_ALT', 'AP_1_JOINT'):
        old = old_run(iq, method, 2, cfgs[method])
        new = run(iq, method, cfgs[method])
        np.testing.assert_allclose(new['positions'], old['positions'], rtol=0, atol=1e-7)
        np.testing.assert_allclose(new['native_objective'], old['native_objective'], rtol=0, atol=1e-13)
        assert new['stop_reason'] == old['stop_reason']
        pairs.append(dict(method=method, frozen_r3_equivalent=True))
    # Fixed-q implementation is also invariant to diagnostic recording/cache switches.
    normal = run(iq, 'BNOMP_REF', cfgs['BNOMP_REF'])
    bare = run(iq, 'BNOMP_REF', cfgs['BNOMP_REF'], history=False, memo=False)
    np.testing.assert_allclose(normal['positions'], bare['positions'], rtol=0, atol=1e-7)
    assert normal['mechanism']['native_cycles'] == bare['mechanism']['native_cycles']
    # Exact-model off-grid signals test behavior, not success on the noisy benchmark.
    source = Model(iq, sample_rate_hz=50e6, half_band_hz=7e6)
    rng = np.random.default_rng(2026101702)
    truth = np.array([[173.4, 117.9], [-137.8, -213.6]])
    fixtures = []
    for count in (1, 2):
        a = source.response(truth[:count]).cpu().numpy()
        s = rng.normal(size=(source.n, count))+1j*rng.normal(size=(source.n, count))
        x = np.einsum('flk,fk->fl', a, s)
        spectrum = np.zeros((8192, 4), complex)
        spectrum[source.frequency_indices.cpu().numpy()] = x
        clean = np.fft.ifft(spectrum*8192, axis=0).T.copy()
        for method in ('BNOMP_REF', 'NOMP'):
            result = run(clean, method, cfgs[method], count=count)
            metrics = evaluate(truth[:count], result['positions'])
            fixtures.append(dict(method=method, count=count, metrics=metrics,
                native_objective=result['native_objective'], stop_reason=result['stop_reason']))
            if not all(metrics['success']['20']):
                raise AssertionError('Basic noiseless/offgrid case failed; investigate before ranking')
    report = dict(passed=True, frozen_equivalence=pairs, fixed_q_recording_cache_equivalent=True,
                  noiseless_offgrid=fixtures, no_truth_in_solver_interface=True)
    write(out/'gpu_checks.json', report)
    return report
