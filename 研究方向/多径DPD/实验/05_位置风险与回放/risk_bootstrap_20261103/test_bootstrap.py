"""Small algebra, FFT/noise and accepted-prefix contract checks."""
import numpy as np


def run_checks(r=None):
    if r is None:
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('bootstrap_test_target', Path(__file__).with_name('runner.py'))
        r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r)
    rng = np.random.default_rng(2026110340)
    checks = {}
    a, w, p, q = rng.normal(size=(4, 4))
    dp = (np.sum((a-p)**2)-np.sum((w-p)**2))/2
    dq = (np.sum((a-q)**2)-np.sum((w-q)**2))/2
    checks['stable_label_reference_identity'] = bool(np.isclose(dp-dq, (q-p)@(a-w)))
    pp = rng.normal(size=(2, 2))
    checks['permutation_loss'] = r.loss(pp, pp[::-1]) == 0.
    checks['loss_normalization'] = np.isclose(r.loss(pp, pp+.001), 2e-6)
    mean = rng.normal(size=(5, 8192))+1j*rng.normal(size=(5, 8192))
    sigma2 = 2.3
    noise = r.generate(mean, sigma2, 0, 0)-mean
    ft = np.fft.fft(noise, axis=1, norm='ortho')
    checks['unitary_parseval'] = bool(np.isclose(np.sum(abs(ft)**2), np.sum(abs(noise)**2)))
    checks['complex_noise_total_variance'] = bool(abs(np.mean(abs(noise)**2)/sigma2-1) < .025)
    checks['real_imag_equal_variance'] = bool(abs(np.var(noise.real)/np.var(noise.imag)-1) < .04)
    checks['replicate_seed_reproducible'] = np.array_equal(r.generate(mean, sigma2, 0, 0), mean+noise)
    checks['distinct_replicates'] = not np.array_equal(r.generate(mean,sigma2,0,0), r.generate(mean,sigma2,0,1))
    st = lambda k: dict(x=[float(k)], p=(pp+k).tolist())
    solved = dict(los=st(0), history=[dict(fit=st(1), decrease=True),
                  dict(fit=st(99), decrease=False)], final=st(1), paths=[[0, 0]])
    cv = r.curve(solved, pp)
    checks['accepted_not_rejected_prefix'] = cv[0] == 0 and np.all(cv[1:] == cv[1])
    import inspect
    search_source = inspect.getsource(r.cal.ref.search)
    solve_source = inspect.getsource(r.cal.engine.solve_iq)
    checks['prefix_threshold_independent_of_cap'] = ('step*(step+1)*10*(n-1)' in search_source
                                                     and 'capacity' not in search_source)
    checks['frozen_solver_64_steps'] = 'range(1,65)' in solve_source
    checks['frozen_solver_recomputes_noise'] = "np.mean(abs(ft[:,~use])**2)" in solve_source
    checks['frozen_solver_blind_init'] = 'triple_candidates(lags)' in solve_source
    checks['noise_not_added_to_observed_iq'] = 'mean+noise' in inspect.getsource(r.generate)
    gate_case = dict(status='returned', stop='no_detected_increment', checks={},
                     final=dict(x=np.zeros(20), stationary=True))
    checks['los_gate_without_optional_closure_field'] = r.solver_gate(gate_case)
    checks['exceptions_fail_gate'] = not r.solver_gate(dict(status='numerical_or_frontend_exception'))
    empty = np.asarray([]).reshape(-1, 65)
    checks['empty_summary_shape'] = (empty-empty[:, -1:]).shape == (0, 65)
    # Conditional distribution transfer identity for two arbitrary discrete worlds.
    candidates = rng.normal(size=(11, 4))
    terminal = rng.normal(size=(11, 4))
    wp = rng.dirichlet(np.ones(11))
    wq = rng.dirichlet(np.ones(11))
    ddp = (np.sum((candidates-p)**2,1)-np.sum((terminal-p)**2,1))/2
    ddq = (np.sum((candidates-q)**2,1)-np.sum((terminal-q)**2,1))/2
    rhs = (wp-wq)@ddq+(q-p)@(wp@(candidates-terminal))
    checks['distribution_and_reference_decomposition'] = bool(np.isclose(wp@ddp-wq@ddq, rhs))
    return checks


if __name__ == '__main__':
    checks = run_checks()
    print(checks)
    assert all(checks.values())
