"""Read-only result comparison and overhead attribution; no localization calls."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r4.runtime import read, write, sha, active_project_processes
from 统一模型代码.gates.g10.r3.runtime import active_project_processes as old_scan

out = Path(__file__).resolve().parent
old = out.parent/'prepare_20261017_003346_952982'
same, differences = [], []
for file in sorted((out/'rows').glob('*.json')):
    a, b = read(old/'rows'/file.name), read(file)
    assert a['input_sha'] == b['input_sha']
    np.testing.assert_array_equal(a['result']['positions'], b['result']['positions'])
    assert a['result']['native_objective'] == b['result']['native_objective']
    assert a['result']['counts'] == b['result']['counts']
    assert a['metrics'] == b['metrics']
    same.append(file.name)
    if a['job']['method'] == 'BNOMP_REF':
        differences.append(dict(task=a['job']['task'], cycles=b['result']['mechanism']['native_cycles'],
            last_energy_changes={str(n): [h['energy_before']-h['native_objective']
                for h in b['result']['history'] if h['discovered'] == n][-3:] for n in (1, 2)}))
times = {}
for name, function in [('old_all_cmdline', old_scan), ('python_only_cmdline', active_project_processes)]:
    samples = []
    for _ in range(3):
        start = time.perf_counter(); found = function(); samples.append(time.perf_counter()-start)
        assert not found
    times[name] = samples
contract = read(out/'contract.json')
for source in contract['paper_sources'].values():
    assert sha(source['path']) == source['sha256']
rows = [read(p) for p in (out/'rows').glob('*.json')]
extra = [r['result']['stages']['terminal_joint_refinement'] for r in rows if r['job']['method'] == 'NOMP_JOINT']
result = dict(passed=True, algorithm_and_input_bitwise_identical_count=len(same),
    sources=[str(old), str(out)], process_scan_seconds=times,
    bnomp_native_stopping=differences, nomp_terminal_addition_mean_seconds=float(np.mean(extra)),
    papers_unchanged=True, source_sha=sha(__file__),
    formal_status='HOLD_BNOMP_NATIVE_STOP_REVIEW',
    note='No inference, convergence extension, threshold change or formal data access in this review.')
write(out/'cost_and_equivalence_review.json', result)
print({'identical':len(same), 'scan_seconds':times, 'NOMP_joint_extra_mean_s':float(np.mean(extra))})
