"""Frozen job/input/source identities and atomic per-case recovery."""
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np
from ...runtime import read, write, sha, verify, preflight, save_scene, load_scene
from ...r1.runtime import Ledger
from ..runtime import seal, lock
from ...metrics import evaluate
from .config import BASE, ROOT, REVISION, jobs, DEFAULT_CONFIG, METHODS
from .scenes import make, layouts, observation, digest, band_energy


def fresh(stage):
    return BASE/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))


def source_ids():
    paths = list((ROOT/'统一模型代码/gates/g10').rglob('*.py'))
    paths += list((ROOT/'统一模型代码/gates/g10/r2/band').glob('*.md'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def key(task):
    return f'{task[0]}_{task[1]}_{task[2]:g}'


def tag(task, method, variant=0):
    return f'{key(task)}_{task[3]}_{method}_v{variant}'


def job_plan(formal=False, frozen=None):
    plan = []
    for task, method, variant, changes in jobs(formal):
        cfg = dict(DEFAULT_CONFIG, **((frozen or {}).get(method, {}) if formal else changes))
        plan.append(dict(task=list(task), method=method, variant=variant, config=cfg,
                         seed=int(np.random.SeedSequence([2026101611, *task[:3]]).generate_state(1)[0])))
    # SBL follows the completed primary comparison; no interleaving slow cases.
    return sorted(plan, key=lambda j: (j['method'] == 'SBL', j['task'], METHODS.index(j['method']), j['variant']))


def create(out, formal=False, frozen=None, budget_hours=20.):
    out = Path(out).resolve()
    if not out.is_relative_to(BASE.resolve()) or out == BASE.resolve():
        raise ValueError('Output must be a fresh isolated R2/band run')
    plan = job_plan(formal, frozen)
    preflight(out, dict(stage='independent' if formal else 'prepare', revision=REVISION,
        jobs=plan, band_sources=source_ids(), budget_hours=budget_hours,
        layouts=layouts() if formal else None, formal_started=formal))
    contract = read(out/'contract.json')
    for rel, identity in (contract['sources'] | contract['band_sources']).items():
        target = out/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
        if sha(target) != identity:
            raise RuntimeError('Source snapshot mismatch')
    inputs = {}
    for job in plan:
        task = job['task']
        name = key(task)
        if name in inputs:
            continue
        scene = make(task, formal)
        info = save_scene(out, name, scene)
        inputs[name] = dict(metadata_sha=sha(out/'inputs'/f'{name}.json'), payload_sha=info['sha256'],
                            band_energy=band_energy(scene))
    write(out/'input_manifest.json', inputs)
    contract['input_manifest_sha'] = sha(out/'input_manifest.json')
    write(out/'contract.json', contract)
    write(out/'status.json', dict(status='PREPARED_INPUTS', formal_started=formal, completed=0, planned=len(plan)))
    return out


def check_contract(out):
    c = read(out/'contract.json')
    if c['revision'] != REVISION or sha(out/'input_manifest.json') != c['input_manifest_sha']:
        raise RuntimeError('Contract/input manifest changed')
    verify(c['sources']); verify(c['band_sources'])
    return c


def load_input(out, task):
    name = key(task)
    identity = read(out/'input_manifest.json')[name]
    metadata = out/'inputs'/f'{name}.json'
    if sha(metadata) != identity['metadata_sha'] or read(metadata)['sha256'] != identity['payload_sha']:
        raise RuntimeError('Input metadata changed')
    scene = load_scene(out, name)  # hashes and consumes the same bytes
    iq, truth, count = observation(scene, task[3])
    return scene, iq, truth, count, digest(iq)


def validate(row, job, identity):
    if row.get('record_sha') != seal(row) or row['input_sha'] != identity or row['job'] != job:
        raise RuntimeError('Row/input/configuration identity mismatch')
    if row['metrics'] != evaluate(row['truth'], row['result']['positions']):
        raise RuntimeError('Saved metrics differ')
    if row['result']['config'] != job['config'] or row['method'] != job['method']:
        raise RuntimeError('Solver configuration differs')


def audit(out, complete=False):
    contract = check_contract(out)
    hashes, paired, identities = {}, {}, {}
    for job in contract['jobs']:
        task, method = job['task'], job['method']
        path = out/'rows'/f"{tag(task, method, job['variant'])}.json"
        if not path.exists():
            if complete:
                raise RuntimeError('Missing expected row: '+path.name)
            continue
        task_key = tuple(task)
        if task_key not in identities:
            identities[task_key] = load_input(out, task)[-1]
        row = read(path)
        validate(row, job, identities[task_key])
        hashes[path.name] = sha(path)
        if job['variant'] == 0:
            paired.setdefault(task_key, {})[method] = row
    if set(hashes) != {p.name for p in (out/'rows').glob('*.json')}:
        raise RuntimeError('Unexpected row file')
    for group in paired.values():
        prefixes = [group[m]['result']['mechanism']['prefix_positions'] for m in
                    ('NOMP', 'NOMP_EXTRA12', 'NOMP_JOINT') if m in group]
        for p in prefixes[1:]:
            np.testing.assert_allclose(p, prefixes[0], rtol=0, atol=1e-8)
        for suffix in ('ALT', 'JOINT'):
            if 'AP_1_'+suffix in group and 'AP_5_'+suffix in group:
                a = group['AP_1_'+suffix]['result']
                b = group['AP_5_'+suffix]['result']
                np.testing.assert_allclose(a['mechanism']['branches'][0]['final_positions'],
                    b['mechanism']['branches'][0]['final_positions'], rtol=0, atol=1e-8)
    result = dict(identity_passed=True, paired_checks_passed=True, complete=complete,
                  rows=hashes, n_rows=len(hashes), planned=len(contract['jobs']))
    write(out/'final_audit_report.json', result)
    return result


__all__ = ['read', 'write', 'sha', 'verify', 'seal', 'lock', 'Ledger', 'fresh', 'source_ids',
           'key', 'tag', 'job_plan', 'create', 'check_contract', 'load_input', 'validate', 'audit']
