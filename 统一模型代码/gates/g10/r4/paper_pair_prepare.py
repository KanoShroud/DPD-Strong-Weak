"""Finite preparation: two real pairs, simulated interruption, cwd/UTF8/replot checks."""
import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from . import paper_pair_runner as run
from ..runtime import read, write, sha
from .runtime import resource_check, require_exclusive


def forbidden(*args, **kwargs):
    raise RuntimeError('Solver must not be called')


def main():
    require_exclusive(); resource_check()
    if run.ACTIVE.exists():
        raise RuntimeError('Do not prepare over an existing formal pair run')
    full = run.build_contract()
    run.check_contract(full)
    c = copy.deepcopy(full)
    c.update(preparation=True, tasks=[[0, 400, 0, 'mixed'], [3, 400, 0, 'mixed']])
    out = run.create(c, 'paper_pair_prepare')
    print('有限接入目录：'+str(out), flush=True)
    run.execute(out, stop_after=1)
    first = out/'pairs/0_400_0.json'
    first_sha = sha(first)
    seconds_before = read(out/'ledger.json')['seconds']
    calls = [0]

    def interrupt(iq, method, cfg, clock):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError('SIMULATED_INTERRUPTION_BETWEEN_PAIRED_METHODS')
        return run.solve(iq, method, cfg, clock)

    try:
        run.execute(out, solver=interrupt)
        raise AssertionError('Expected simulated interruption')
    except RuntimeError as error:
        if 'SIMULATED_INTERRUPTION' not in str(error):
            raise
    assert not (out/'pairs/3_400_0.json').exists()
    assert read(out/'status.json')['status'] == 'STOPPED'
    run.execute(out)
    assert sha(first) == first_sha
    assert read(out/'ledger.json')['seconds'] > seconds_before
    assert len(read(out/'ledger.json')['interruptions']) == 1
    # Completed rerun must not call the supplied solver.
    run.execute(out, solver=forbidden)
    row = read(first)
    original = read(Path(c['source'])/'rows/0_400_0_AP_5_JOINT.json')
    bad = copy.deepcopy(row); bad['results']['AP_5_ALT']['positions'][0][0] += 1
    try:
        run.validate(bad, c['tasks'][0], c, original)
        raise AssertionError('Corrupted row accepted')
    except RuntimeError:
        pass
    cfg = full['configuration']
    assert cfg['gtol'] == 1e-8 and cfg['common_ftol'] == 0
    for p in (out/'pairs').glob('*.json'):
        r = read(p)
        for m in run.METHODS:
            assert all(b['stop_reason'] == 'COMMON_GRADIENT_REACHED' for b in r['results'][m]['mechanism']['branches'])
        a = np.asarray(r['results'][run.METHODS[0]]['mechanism']['candidate_list'])
        b = np.asarray(r['results'][run.METHODS[1]]['mechanism']['candidate_list'])
        assert np.allclose(a, b, rtol=0, atol=1e-8)
    with patch.object(run, 'solve', side_effect=forbidden):
        run.main(['--replot', str(out)])
    # A full contract is released only after the finite engineering checks.
    write(out/'formal_contract.json', full)
    ready = dict(ready=True, fixture=str(out), formal_not_started=True, expected_pairs=2400,
        expected_solves=4800, tests=dict(real_pairs=2, original_joint_equivalence=True,
        candidate_pairing=True, branch_common_stop=True, interrupted_pair_not_committed=True,
        completed_pair_unchanged=True, cumulative_ledger_retained=True, repeated_complete_no_solver=True,
        corrupt_pair_rejected=True, replot_no_solver=True))
    write(out/'readiness.json', ready)
    write(run.RELEASE, dict(contract=str(out/'formal_contract.json'), contract_sha=sha(out/'formal_contract.json'),
        readiness=str(out/'readiness.json'), readiness_sha=sha(out/'readiness.json')))
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    tests = []
    for cwd in (run.ENTRY.parent, run.ROOT/'outputs_e2e'):
        p = subprocess.run([sys.executable, str(run.ENTRY), '--check-only'], cwd=cwd, env=env,
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        tests.append(dict(cwd=str(cwd), returncode=p.returncode, stdout=p.stdout or '', stderr=p.stderr or ''))
        if p.returncode or '放行检查通过' not in (p.stdout or ''):
            ready['ready'] = False
            write(out/'readiness.json', ready)
            raise RuntimeError('CWD/UTF8 launch check failed: '+str(tests[-1]))
    write(out/'cwd_utf8_checks.json', tests)
    ready['tests']['cwd_utf8_checks'] = True
    write(out/'readiness.json', ready)
    write(run.RELEASE, dict(contract=str(out/'formal_contract.json'), contract_sha=sha(out/'formal_contract.json'),
        readiness=str(out/'readiness.json'), readiness_sha=sha(out/'readiness.json')))
    assert not run.ACTIVE.exists()
    print('准备通过，未创建正式任务。用户入口：'+str(run.ENTRY), flush=True)


if __name__ == '__main__':
    main()
