"""Fresh R4 evidence, read-only R2 IQ, source seals and paired-result checks."""
from datetime import datetime
from pathlib import Path
import os
import shutil
import numpy as np
import psutil
from ..runtime import read, write, sha, verify, resource_check
from ..r2.runtime import seal, lock
from ..r2.band.runtime import load_input as load_r2
from ..r3.runtime import reference, resolve_root
from ..metrics import evaluate
from .config import ROOT, BASE, REVISION, METHODS, TASKS, jobs, configurations


def active_project_processes():
    """Same exclusion predicate as R3; avoid querying every Windows command line."""
    result = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.pid == os.getpid() or 'python' not in (proc.info['name'] or '').lower():
                continue
            command = ' '.join(proc.cmdline() or [])
            normalized = command.replace('\\', '/').lower()
            if str(ROOT).replace('\\', '/').lower() in normalized or '统一模型代码.gates.' in normalized:
                result.append(dict(pid=proc.pid, command=command))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def require_exclusive():
    active = active_project_processes()
    if active:
        raise RuntimeError('其他项目Python任务仍在运行：'+', '.join(str(p['pid']) for p in active))


def sources():
    paths = list((ROOT/'统一模型代码/gates/g10').rglob('*.py'))
    paths += list((ROOT/'统一模型代码/gates/g8').glob('*.py'))
    paths += list((ROOT/'统一模型代码/gates/g9').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.md'))
    paths += [ROOT/'运行入口/E2E/G10/G10R4近邻核验短测.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def safe_out(path):
    p = Path(path).resolve()
    if p == BASE.resolve() or not p.is_relative_to(BASE.resolve()):
        raise ValueError('Expected an isolated R4 output directory')
    return p


from migration_paths import input_path  # noqa: E402

def create():
    require_exclusive()
    ref = reference()
    verify(ref['protected'])
    resource_check()
    manifest = read(ROOT/'outputs_e2e/literature/g10_r4_nearest_20261016/source_manifest.json')
    for item in manifest.values():
        if sha(input_path(item['path']).resolve(strict=True)) != item['sha256']:
            raise RuntimeError('Source PDF identity changed')
    identity = sources()
    verify(identity)
    out = safe_out(BASE/('prepare_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    if out.exists():
        raise FileExistsError('Fresh output required')
    # Read, hash and resolve ALL 16 observations before any localization run.
    inputs = {}
    for task in TASKS:
        scene, iq, truth, count, digest = load_r2(resolve_root(ref['out']), task)
        if count != 2 or iq.shape != (4, 8192) or truth.shape != (2, 2):
            raise ValueError('Unexpected source observation')
        inputs[task_key(task)] = dict(input_sha=digest, truth=truth.tolist(), metadata=scene['meta'])
    out.mkdir(parents=True)
    contract = dict(revision=REVISION, reference=ref, configurations=configurations(ref),
        jobs=jobs(), sources=identity, paper_sources=manifest, inputs=inputs,
        scope='approved 16 already-used observations; no new formal data',
        protocol='EVAL-20261011-v2', formal_authorized=False)
    write(out/'contract.json', contract)
    for rel, digest in identity.items():
        destination = out/'source_snapshot'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, destination)
        if sha(destination) != digest:
            raise RuntimeError('Source snapshot mismatch')
    write(out/'status.json', dict(status='PREPARED', formal_started=False))
    write(BASE/'preparation.json', dict(out=str(out), contract_sha=sha(out/'contract.json')))
    return out


def check_contract(out):
    out = safe_out(out)
    c = read(out/'contract.json')
    if c['revision'] != REVISION or c['jobs'] != jobs():
        raise RuntimeError('Preparation contract changed')
    verify(c['sources'])
    ref = c['reference']; root = resolve_root(ref['out'])
    if sha(root/'contract.json') != ref['contract_sha'] or sha(root/'input_manifest.json') != ref['input_manifest_sha']:
        raise RuntimeError('Frozen reference changed')
    return c


def task_key(task):
    return f'{task[0]}_{task[1]}_{task[2]}'


def job_key(job):
    return task_key(job['task'])+'_'+job['method']


def load(c, task):
    scene, iq, truth, count, identity = load_r2(resolve_root(c['reference']['out']), task)
    registered = c['inputs'][task_key(task)]
    if count != 2 or identity != registered['input_sha'] or truth.tolist() != registered['truth']:
        raise RuntimeError('Consumed bytes differ from registered input')
    return scene, iq, truth, identity


def validate(row, job, identity, cfg, truth):
    if row.get('record_sha') != seal(row) or row['job'] != job or row['input_sha'] != identity:
        raise RuntimeError('Row identity differs')
    result = row['result']
    if result['config'] != cfg or result['method'] != job['method'] or row['truth'] != truth.tolist():
        raise RuntimeError('Solver or data fields differ')
    if row['metrics'] != evaluate(truth, result['positions']):
        raise RuntimeError('Metric replay differs')


def paired_checks(group):
    a = group['AP_1_JOINT']['result']['mechanism']
    for method in ('AP_1_ALT', 'AP_2_JOINT', 'AP_5_JOINT'):
        b = group[method]['result']['mechanism']
        np.testing.assert_array_equal(a['candidate_list'], b['candidate_list'])
        np.testing.assert_array_equal(a['first_position'], b['first_position'])
    branches = group['AP_5_JOINT']['result']['mechanism']['branches']
    for method, count in (('AP_1_JOINT', 1), ('AP_2_JOINT', 2)):
        for x, y in zip(group[method]['result']['mechanism']['branches'], branches[:count]):
            np.testing.assert_allclose(x['positions'], y['positions'], rtol=0, atol=1e-7)
            np.testing.assert_allclose(x['objective'], y['objective'], rtol=0, atol=1e-13)
            assert x['stop_reason'] == y['stop_reason']
    native, joint = (group[m]['result'] for m in ('NOMP', 'NOMP_JOINT'))
    np.testing.assert_array_equal(native['positions'], joint['mechanism']['prefix_positions'])
    assert native['counts'] == joint['mechanism']['prefix_counts']
    assert native['native_objective'] == joint['mechanism']['prefix_objective']
    assert joint['native_objective'] <= native['native_objective']+1e-13
    assert group['AP_5_JOINT']['result']['native_objective'] <= group['AP_2_JOINT']['result']['native_objective']+1e-13
    assert group['AP_2_JOINT']['result']['native_objective'] <= group['AP_1_JOINT']['result']['native_objective']+1e-13


def audit(out):
    c = check_contract(out)
    grouped, hashes = {}, {}
    for job in c['jobs']:
        _, _, truth, identity = load(c, job['task'])
        path = out/'rows'/(job_key(job)+'.json')
        row = read(path)
        validate(row, job, identity, c['configurations'][job['method']], truth)
        grouped.setdefault(task_key(job['task']), {})[job['method']] = row
        hashes[path.name] = sha(path)
    if set(hashes) != {p.name for p in (out/'rows').glob('*.json')}:
        raise RuntimeError('Unexpected result file')
    for group in grouped.values():
        if set(group) != set(METHODS):
            raise RuntimeError('Incomplete paired case')
        paired_checks(group)
    result = dict(passed=True, cases=len(grouped), solves=len(hashes), rows=hashes,
        input_identity_passed=True, metric_replay_passed=True, nested_ap_branches_passed=True,
        unchanged_nomp_prefix_passed=True, frozen_sources_passed=True)
    write(out/'final_audit_report.json', result)
    return result


__all__ = ['read', 'write', 'sha', 'seal', 'lock', 'require_exclusive', 'resource_check',
           'create', 'check_contract', 'job_key', 'task_key', 'load', 'validate', 'audit']
