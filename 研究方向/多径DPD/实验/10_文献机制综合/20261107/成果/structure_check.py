"""Model-class counterexample, not a simulated localization performance run."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mechanism_probe', HERE/'probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def main():
    out = HERE.parent/'运行数据/structure_check'
    out.mkdir(parents=True, exist_ok=False)
    engine = probe.gate.api.create_engine()
    core_files = list((probe.BASE/'代码/mp_dpd').rglob('*.py'))
    identities = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in core_files}
    p0 = np.array([[-700., -500.], [650., 800.]])
    p1 = np.array([[1200., -900.], [-1000., 1300.]])
    rng = np.random.default_rng(2026110712)
    f = np.linspace(-engine.ref.B/2, engine.ref.B/2, 257)
    s = rng.normal(size=(len(f), 2))+1j*rng.normal(size=(len(f), 2))
    base = np.r_[p0.ravel()/engine.ref.UNIT, np.ones(6), np.zeros(6)]
    h0 = engine.core(base, f, np.zeros((len(f), 4), complex), [])[0]
    y0 = np.einsum('fmq,fq->fm', h0, s)
    tau0 = np.linalg.norm(engine.ref.STATIONS[:, None]-p0, axis=2)/engine.ref.C
    tau1 = np.linalg.norm(engine.ref.STATIONS[:, None]-p1, axis=2)/engine.ref.C
    shift, scale = 50e-6, 4.
    rows = []
    for eta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        epsilon = eta/(2*np.pi*engine.ref.B)
        paths, params = [], []
        for m in range(4):
            for q in range(2):
                paths += [(m, q), (m, q)]
                params += [-(1-eta), 0., epsilon*engine.ref.B,
                           1/scale, 0., (tau0[m, q]+shift-tau1[m, q])*engine.ref.B]
        x = np.r_[p1.ravel()/engine.ref.UNIT, np.ones(6), np.zeros(6), params]
        assert np.allclose(engine.ref.project(x, 8192), x, rtol=0, atol=1e-12)
        h1 = engine.core(x, f, np.zeros_like(y0), paths)[0]
        s1 = scale*np.exp(2j*np.pi*f[:, None]*shift)*s
        y1 = np.einsum('fmq,fq->fm', h1, s1)
        analytic = np.exp(-2j*np.pi*f[:, None, None]*tau1[None])*(1-(1-eta)*np.exp(-2j*np.pi*f[:, None, None]*epsilon))
        remainder = np.einsum('fmq,fq->fm', analytic, s1)
        err = float(np.linalg.norm(y1-y0)/np.linalg.norm(y0))
        identity_error = float(np.linalg.norm(y1-y0-remainder)/np.linalg.norm(y0))
        assert identity_error < 1e-11
        rows.append(dict(eta=eta, epsilon_s=epsilon, relative_observation_difference=err,
            analytic_remainder_error=identity_error, per_path_dominance=True,
            min_delay=float(np.min(x[18::3])/engine.ref.B), paths_per_link=2))
    assert rows[-1]['relative_observation_difference'] < 1e-5
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in identities.items())
    probe.gate.cfg.write(out/'result.json', dict(rows=rows, true_positions=p0, alternative_positions=p1,
        separation_rmse=float(np.sqrt(np.mean(probe.gate.errors(p1,p0)**2))),
        scope='frequency_model_counterexample_not_approved_well_separated_generator',
        implication='no_uniform_stability_over_unrestricted_near_collision_model_class',
        not_claimed=['current_cases_caused_by_this', 'well_separated_subclass_impossible', 'new_theorem_novelty'],
        source_hashes=identities, script_sha=probe.sha(Path(__file__)), source_unchanged=True))
    print(rows[-1])


if __name__ == '__main__':
    main()
