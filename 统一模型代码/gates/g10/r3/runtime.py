"""Additive source contract, input identity and resource-isolated R3 runs."""
from datetime import datetime
from pathlib import Path
import os
import shutil
import time
import numpy as np
import psutil
from ..runtime import read, write, sha, verify, save_scene, load_scene
from ..r1.scenes import make as make_scene, observation, digest
from ..r2.runtime import seal, lock
from ..r2.band.runtime import load_input as load_r2_input
from ..r2.band.scenes import geometry_valid, layouts as r2_layouts
from ..metrics import evaluate
from .config import (ROOT, BASE, R2_BASE, ENTRY, REVISION, GEOMETRY_SEED,
                     WAVEFORM_SEED, METHODS, jobs, FORMAL_TASKS)


def resolve_root(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT/p).resolve(strict=True)


def reference():
    path = resolve_root(read(R2_BASE/'active_run.json')['out'])
    if not path.is_relative_to(R2_BASE.resolve()):
        raise ValueError('R2 reference outside registered root')
    contract = read(path/'contract.json')
    verify(contract['sources']); verify(contract['band_sources'])
    if sha(path/'input_manifest.json') != contract['input_manifest_sha']:
        raise RuntimeError('R2 input manifest changed')
    if not (path/'main_batch_report').is_dir():
        raise RuntimeError('R2 main batch not finalized')
    configurations = {}
    for method in METHODS:
        original = 'MULTISTART' if method == 'MULTISTART_FIXED' else method
        candidates = [j['config'] for j in contract['jobs'] if j['method'] == original]
        if not candidates or any(c != candidates[0] for c in candidates):
            raise RuntimeError('R2 configuration not unique: '+original)
        configurations[method] = dict(candidates[0])
    configurations['MULTISTART_FIXED']['starts'] = 8
    for method in ('AP_1_ALT', 'AP_1_JOINT'):
        configurations[method].update(common_stop=True, common_ftol=0.)
    a, b = configurations['AP_1_ALT'], configurations['AP_1_JOINT']
    if a != b:
        raise RuntimeError('AP pair must share all configuration values')
    return dict(out=str(path), contract_sha=sha(path/'contract.json'),
                input_manifest_sha=sha(path/'input_manifest.json'),
                configurations=configurations, protected=contract['sources'] | contract['band_sources'])


def active_project_processes():
    result = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == os.getpid() or 'python' not in (proc.info['name'] or '').lower():
                continue
            command = ' '.join(proc.info['cmdline'] or [])
            normalized = command.replace('\\', '/').lower()
            if str(ROOT).replace('\\', '/').lower() in normalized or '统一模型代码.gates.' in normalized:
                result.append(dict(pid=proc.pid, command=command))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def require_exclusive():
    active = active_project_processes()
    if active:
        raise RuntimeError('R2/SBL或其他项目任务仍在运行；R3不启动GPU。PID: '
                           +', '.join(str(p['pid']) for p in active))


def source_ids(ref):
    # R2 verifies only its recorded keys. New R3 files are not modifications to them.
    result = dict(ref['protected'])
    paths = list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.md'))+[ENTRY]
    for p in paths:
        key = str(p.relative_to(ROOT))
        if key in ref['protected']:
            raise RuntimeError('R3 attempted to overlap frozen R2 file: '+key)
        result[key] = sha(p)
    return result


def new_layouts():
    rng, result = np.random.default_rng(GEOMETRY_SEED), []
    old = np.asarray(r2_layouts())
    for group in range(4):
        for _ in range(100000):
            p = rng.uniform(-950, 950, (2, 2))
            if geometry_valid(p, group) and not any(np.array_equal(p, q) for q in old):
                result.append(p.tolist())
                break
        else:
            raise RuntimeError('Geometry generation failed')
    return result


def fresh(stage):
    return BASE/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))


def safe_out(path):
    p = Path(path).resolve()
    if not p.is_relative_to(BASE.resolve()) or p == BASE.resolve():
        raise ValueError('Output must be a child of the isolated R3 root')
    return p


def create(out, stage, ref, preparation=None):
    require_exclusive()
    out = safe_out(out)
    if out.exists():
        raise FileExistsError('Fresh output required: '+str(out))
    sources = source_ids(ref)
    verify(sources)
    previous = None
    pointer = BASE/'preparation.json'
    if stage == 'prepare' and pointer.exists():
        prior = resolve_root(read(pointer)['out'])
        old = read(prior/'contract.json')
        # Keep one original-stop preparation as a sealed, read-only reference.
        if old['revision'] == 'G10_R3_AP_COST_V1':
            for rel, identity in old['sources'].items():
                if sha(prior/'source_snapshot'/rel) != identity:
                    raise RuntimeError('Prior source snapshot changed: '+rel)
            prior_audit = read(prior/'final_audit_report.json')
            for name, identity in prior_audit['rows'].items():
                if sha(prior/'rows'/name) != identity:
                    raise RuntimeError('Prior row changed: '+name)
            previous = dict(out=str(prior), contract_sha=sha(prior/'contract.json'),
                            audit_sha=sha(prior/'final_audit_report.json'),
                            report_sha=sha(prior/'comparison_report.json'))
        else:
            previous = old.get('previous_preparation')
    out.mkdir(parents=True)
    contract = dict(revision=REVISION, stage=stage, jobs=jobs(stage), reference=ref,
        sources=sources, configurations=ref['configurations'], preparation=preparation,
        geometry_seed=GEOMETRY_SEED, waveform_seed=WAVEFORM_SEED,
        layouts=new_layouts(), protocol='EVAL-20261011-v2', created=time.time(),
        previous_preparation=previous)
    write(out/'contract.json', contract)
    for rel, identity in sources.items():
        target = out/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
        if sha(target) != identity:
            raise RuntimeError('Source snapshot differs')
    inputs = {}
    if stage == 'formal':
        for task in FORMAL_TASKS:
            group, repeat, gap, _ = task
            scene = make_scene(group, repeat, gap, positions=contract['layouts'][group], seed=WAVEFORM_SEED)
            scene['meta'].update(role='r3_independent', geometry_seed=GEOMETRY_SEED)
            key = task_key(task)
            info = save_scene(out, key, scene)
            inputs[key] = dict(metadata_sha=sha(out/'inputs'/f'{key}.json'), payload_sha=info['sha256'])
        write(out/'input_manifest.json', inputs)
        contract['input_manifest_sha'] = sha(out/'input_manifest.json')
        write(out/'contract.json', contract)
    write(out/'status.json', dict(status='PREPARED', stage=stage, completed=0))
    return out


def check_contract(out):
    out = safe_out(out)
    c = read(out/'contract.json')
    if c['revision'] != REVISION or c['jobs'] != jobs(c['stage']):
        raise RuntimeError('Contract/revision/job list changed')
    verify(c['sources'])
    if c.get('previous_preparation'):
        prior = c['previous_preparation']; root = resolve_root(prior['out'])
        for name, key in (('contract.json', 'contract_sha'), ('final_audit_report.json', 'audit_sha'),
                          ('comparison_report.json', 'report_sha')):
            if sha(root/name) != prior[key]:
                raise RuntimeError('Original-stop evidence changed: '+name)
    r = c['reference']; root = resolve_root(r['out'])
    if sha(root/'contract.json') != r['contract_sha'] or sha(root/'input_manifest.json') != r['input_manifest_sha']:
        raise RuntimeError('R2 frozen reference changed')
    if c['stage'] == 'formal' and sha(out/'input_manifest.json') != c['input_manifest_sha']:
        raise RuntimeError('R3 input manifest changed')
    return c


def task_key(task):
    return f'{task[0]}_{task[1]}_{task[2]:g}'


def job_key(job):
    return f"{task_key(job['task'])}_{job['method']}_t{job['repetition']}"


def load_input(out, task, contract=None):
    c = contract or read(out/'contract.json')
    if c['stage'] == 'prepare':
        scene, iq, truth, count, identity = load_r2_input(resolve_root(c['reference']['out']), task)
        if count != 2:
            raise ValueError('Expected K=2')
        return scene, iq, truth, identity
    key = task_key(task)
    manifest = read(out/'input_manifest.json')[key]
    metadata = out/'inputs'/f'{key}.json'
    if sha(metadata) != manifest['metadata_sha'] or read(metadata)['sha256'] != manifest['payload_sha']:
        raise RuntimeError('Input metadata changed')
    scene = load_scene(out, key)  # same bytes are hashed and consumed
    iq, truth, count = observation(scene, 'mixed')
    if count != 2:
        raise ValueError('Expected K=2')
    return scene, iq, truth, digest(iq)


def validate(row, job, identity, config, truth):
    if row.get('record_sha') != seal(row) or row['job'] != job or row['input_sha'] != identity:
        raise RuntimeError('Saved row identity/configuration differs')
    result = row['result']
    if result['config'] != config or result['method'] != job['method'] or row['truth'] != truth.tolist():
        raise RuntimeError('Saved solver/data fields differ')
    if row['metrics'] != evaluate(truth, result['positions']):
        raise RuntimeError('Metric replay differs')


def audit(out, complete=True):
    c = check_contract(out)
    planned, hashes, grouped, cached = set(), {}, {}, {}
    for job in c['jobs']:
        name = job_key(job)+'.json'; planned.add(name)
        p = out/'rows'/name
        if not p.exists():
            if complete:
                raise RuntimeError('Missing row '+name)
            continue
        task = tuple(job['task'])
        if task not in cached:
            _, _, truth, identity = load_input(out, task, c)
            cached[task] = truth, identity
        truth, identity = cached[task]
        row = read(p)
        validate(row, job, identity, c['configurations'][job['method']], truth)
        grouped.setdefault((task, job['repetition']), {})[job['method']] = row
        hashes[name] = sha(p)
    if any(p.name not in planned for p in (out/'rows').glob('*.json')):
        raise RuntimeError('Unplanned row')
    for values in grouped.values():
        if all(m in values for m in ('AP_1_ALT', 'AP_1_JOINT')):
            a, b = (values[m]['result']['mechanism']['branches'][0]['initial_positions']
                    for m in ('AP_1_ALT', 'AP_1_JOINT'))
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
    report = dict(identity_passed=True, metric_replay_passed=True, ap_prefix_passed=True,
        complete=complete, rows=hashes, expected=len(c['jobs']), completed=len(hashes))
    write(out/'final_audit_report.json', report)
    return report


__all__ = ['read', 'write', 'sha', 'verify', 'seal', 'lock', 'resolve_root', 'reference',
           'require_exclusive', 'active_project_processes', 'source_ids', 'new_layouts',
           'fresh', 'safe_out', 'create', 'check_contract', 'task_key', 'job_key',
           'load_input', 'validate', 'audit']
