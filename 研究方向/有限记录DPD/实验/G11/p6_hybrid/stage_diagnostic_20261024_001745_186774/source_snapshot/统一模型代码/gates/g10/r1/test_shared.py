"""Shared-state invariants and resume tests; no performance qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os
from unittest.mock import patch
import numpy as np
import torch
from .config import DEFAULT
from .physics import Model
from .sbl import statistics
from .methods import run
from .scenes import make, observation, digest
from .sbl_outputs import StateStore, load_state, decode_state, replay
from .output_check import near_duplicate_groups, duplicate_identity_test
from .runtime import fresh, preflight, write, finish_audit, sha


def check_shared(out, dictionary_cache=False):
    tests = {}
    tests['duplicate_identity'] = duplicate_identity_test()
    assert max(map(len, near_duplicate_groups(np.array([[0.,0.],[.03,0.],[.06,0.]])))) == 2
    assert len(near_duplicate_groups(np.array([[0.,0.],[.05,0.]]))) == 1
    assert len(near_duplicate_groups(np.array([[0.,0.],[.050001,0.]]))) == 2
    tests['bounded_groups'] = True
    scene = make(0, 31, n=64)
    iq, _, _ = observation(scene, 'weak')
    m = Model(iq)
    m.sbl_spectral_weight = torch.ones(m.n, dtype=torch.float64, device=m.device)
    pp = np.array([[173.,117.],[173.03,117.],[500.,200.]])
    gamma = torch.tensor([.002,.003,.001], dtype=torch.float64, device=m.device)
    stats = statistics(m, pp, gamma, 32, True)
    arrays = dict(points=pp, gamma=gamma.cpu().numpy(), power=stats['power'].cpu().numpy(),
                  covariance=torch.cat(stats['covs']).cpu().numpy(), spectral_weight=np.ones(m.n))
    decoded = decode_state(m, arrays, 2)
    from ..physics import sbl_posterior
    a = m.response(pp)
    mu = sbl_posterior(a, m.x, gamma, 1/m.n)[0]
    expected = abs((a[:,:,:2]*mu[:,None,:2]).sum(-1)).square().sum(1).mean()
    np.testing.assert_allclose(decoded['group_coherent'][0], float(expected.cpu()), rtol=1e-10)
    arrays['points'] = np.array([[173.,117.],[230.,170.],[500.,200.]])
    distinct = decode_state(m, arrays, 2)
    assert distinct['outputs']['SBL'] == distinct['outputs']['SBL_GROUP_GAMMA']
    assert distinct['outputs']['SBL_GROUP_COHERENT'] == distinct['outputs']['SBL_POINT_POWER_AUX']
    tests['decode_direct_and_singletons'] = True
    cfg = dict(DEFAULT, grid=1000., sbl_grid=1000., sbl_iterations=10,
               sbl_active_atoms=2, sbl_coordinate_iterations=2, dictionary_cache=dictionary_cache)
    identity = dict(input_sha=digest(iq), config=cfg, count=1, seed=0)
    plain = run(iq, 'SBL', 1, cfg)
    full = StateStore(out/'states_full', identity, checkpoints=(5, 10))
    observed = run(iq, 'SBL', 1, cfg, state_store=full)
    np.testing.assert_allclose(plain['positions'], observed['positions'], atol=0, rtol=0)
    assert plain['stop_reason'] == observed['stop_reason']
    np.testing.assert_allclose([h['native_objective'] for h in plain['history']],
                               [h['native_objective'] for h in observed['history']], atol=0, rtol=0)
    np.testing.assert_allclose([h['positions'] for h in plain['history']],
                               [h['positions'] for h in observed['history']], atol=0, rtol=0)
    tests['observer_does_not_change_solver'] = True

    class InterruptStore(StateStore):
        def observe(self, *args, **kwargs):
            super().observe(*args, **kwargs)
            if '5' in self.manifest['roles']:
                raise InterruptedError('intentional checkpoint test')
    part = InterruptStore(out/'states_resume', identity, checkpoints=(5, 10))
    try:
        run(iq, 'SBL', 1, cfg, state_store=part)
    except InterruptedError:
        pass
    else:
        raise AssertionError('Resume test did not reach checkpoint')
    restored = StateStore(out/'states_resume', identity, checkpoints=(5, 10))
    resumed = run(iq, 'SBL', 1, cfg, state_store=restored)
    assert restored.was_resumed
    np.testing.assert_allclose(resumed['positions'], observed['positions'], atol=0, rtol=0)
    np.testing.assert_allclose([h['native_objective'] for h in resumed['history']],
                               [h['native_objective'] for h in observed['history']], atol=0, rtol=0)
    if dictionary_cache:
        # Rebuilding/refilling a cache on resume is real extra work, not new iterations.
        for field in ('iterations', 'covariance_decompositions', 'sbl_dictionary_columns'):
            assert resumed['counts'][field] == observed['counts'][field]
        tests['cache_resume_extra_response_points'] = (resumed['counts']['frequency_response_points']-
                                                       observed['counts']['frequency_response_points'])
    else:
        assert resumed['counts'] == observed['counts']
    assert resumed['stop_reason'] == observed['stop_reason']
    tests['checkpoint_resume_equal'] = True
    rep = replay(full.folder, iq, 1)
    selected = rep['states'][rep['roles']['selected']]
    np.testing.assert_allclose(selected['outputs']['SBL'], observed['positions'], atol=0, rtol=0)
    arrays, _ = load_state(full.folder, full.manifest['states'][full.manifest['roles']['selected']])
    immediate = decode_state(Model(iq), arrays, 1)
    assert immediate['outputs'] == selected['outputs']
    tests['replay_same_state'] = True
    bad = dict(full.manifest['states'][full.manifest['roles']['selected']],sha256='0'*64)
    try:
        load_state(full.folder,bad)
    except RuntimeError:
        tests['state_identity_rejection'] = True
    else:
        raise AssertionError('Incorrect state digest accepted')
    # No decoder truth argument; equal count and evaluation boundaries are checked in test_r1.
    original_hash = sha(full.path)
    again = run(iq, 'SBL', 1, cfg, state_store=StateStore(full.folder, identity, checkpoints=(5,10)))
    assert again['positions'] == observed['positions'] and sha(full.path) == original_hash
    tests['finished_resume_no_iterations'] = True
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    text = subprocess.run([sys.executable, '-c', "print('中文日志：共同状态核验')"],
                          capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, check=True)
    assert '中文日志' in text.stdout
    write(out/'中文日志核验.json', dict(stdout=text.stdout, stderr=text.stderr))
    tests['utf8_subprocess'] = True
    # Small primary baseline calls confirm the wrapper did not change method contracts.
    assert np.isfinite(np.asarray(run(iq, 'LS', 1, cfg)['positions'])).all()
    tests['baseline_wrapper'] = True
    return dict(passed=True, checks=tests)


def suite(out):
    from .test_r1 import suite as original_suite
    result = original_suite()
    if not result['passed']:
        return result
    return dict(passed=True, original=result, shared=check_shared(out))


def pipeline_smoke(out):
    from .runner import record, short_tasks
    from .runtime import Ledger, save_scene, load_scene
    from .report import generate
    assert len(short_tasks()) == 38
    scene = make(0,31,n=64)
    save_scene(out,'0_31_0',scene)
    loaded = load_scene(out,'0_31_0')
    cfg = dict(DEFAULT,grid=1000.,sbl_grid=1000.,sbl_iterations=10,
               sbl_active_atoms=2,sbl_coordinate_iterations=2)
    clock = Ledger(out)
    with patch('统一模型代码.gates.g10.r1.runner.scene_for',return_value=loaded):
        for kind in ('weak','mixed'):
            for method in ('LS','SBL'):
                record(out,clock,(0,31,0,kind),method,cfg)
        before = {str(p):sha(p) for p in (out/'rows').glob('*.json')}
        record(out,clock,(0,31,0,'weak'),'SBL',cfg)
        assert before == {str(p):sha(p) for p in (out/'rows').glob('*.json')}
    with patch('统一模型代码.gates.g10.r1.runner.run',side_effect=AssertionError('Replot must not solve')):
        generate(out,label='G10-R1 / ENGINEERING DEVELOPMENT / NOT PERFORMANCE EVIDENCE')
    assert before == {str(p):sha(p) for p in (out/'rows').glob('*.json')}
    from .runtime import read
    assert read(out/'comparison_report.json')['base_solver_rows']==4
    assert read(out/'comparison_report.json')['derived_output_rows']==4
    assert read(out/'sbl_comparison_report.json')['solver_runs']==2
    return dict(passed=True,base_solver_rows=4,derived_rows=4,offline_report=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    out = args.out or fresh('shared_tests')
    preflight(out, dict(stage='shared_state_tests'))
    try:
        report = suite(out)
        report['pipeline'] = pipeline_smoke(out)
        write(out/'report.json', report)
    except BaseException as exc:
        write(out/'report.json', dict(passed=False, error=repr(exc)))
        raise
    finish_audit(out)
    print(json.dumps(dict(out=str(out), passed=report['passed']), ensure_ascii=False))


if __name__ == '__main__':
    main()
