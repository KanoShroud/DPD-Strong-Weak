"""R2 identity and one-case recovery, isolated from frozen R1 outputs."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import psutil
import numpy as np
from ..runtime import read, write, sha, verify, preflight as old_preflight, load_scene
from ..r1.runtime import Ledger
from ..r1.scenes import observation, digest
from ..metrics import evaluate
from .config import ROOT, BASE, R1_BASE, CONFIG, METHODS, tasks

__all__ = ['read', 'write', 'sha', 'verify', 'Ledger', 'fresh', 'preflight',
           'reference_manifest', 'load_input', 'seal', 'validate_row', 'audit', 'lock']


def fresh(stage):
    return BASE/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))


def sources():
    files = list((ROOT/'统一模型代码/gates/g10/r1').glob('*.py'))
    files += list((ROOT/'统一模型代码/gates/g10/r2').glob('*.py'))
    files += list((ROOT/'统一模型代码/gates/g10/r2').glob('*.md'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def preflight(out, contract):
    out = Path(out).resolve()
    if not out.is_relative_to(BASE.resolve()) or out == BASE.resolve():
        raise ValueError('R2 output outside isolated root')
    old_preflight(out, dict(contract, r2_sources=sources()))
    info = read(out/'contract.json')
    for rel, identity in (info['sources'] | info['r2_sources']).items():
        target = out/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
        if sha(target) != identity:
            raise RuntimeError('Source snapshot changed: '+rel)


def reference_manifest():
    run = Path(read(R1_BASE/'active_run.json')['out']).resolve(strict=True)
    if not run.is_relative_to(R1_BASE.resolve()):
        raise RuntimeError('R1 reference outside frozen R1 root')
    if read(run/'status.json')['status'] != 'COMPLETED':
        raise RuntimeError('R1 reference incomplete')
    final = read(run/'final_audit_report.json')
    if not final['identity_passed']:
        raise RuntimeError('R1 audit failed')
    contract = read(run/'contract.json')
    verify(contract['sources']); verify(contract['r1_sources'])
    entries = {}
    for group, realization, gap in tasks('develop'):
        key = f'{group}_{realization}_{gap:g}'
        meta = run/'inputs'/f'{key}.json'
        info = read(meta)
        payload = (run/info['path']).resolve(strict=True)
        if not payload.is_relative_to(run) or sha(payload) != final['inputs'][meta.name]:
            raise RuntimeError('R1 input identity mismatch: '+key)
        if info['sha256'] != final['inputs'][meta.name]:
            raise RuntimeError('R1 metadata identity mismatch')
        refs = {}
        for method in ('AP', 'BNOMP'):
            path = run/'rows'/f'{key}_mixed_{method}.json'
            expected = final['rows'][path.name]
            if sha(path) != expected:
                raise RuntimeError('Historical reference row changed')
            refs[method] = dict(path=str(path), sha256=expected)
        entries[key] = dict(metadata=str(meta), metadata_sha=sha(meta),
                            payload=str(payload), payload_sha=sha(payload), references=refs)
    return dict(run=str(run), audit_sha=sha(run/'final_audit_report.json'),
                contract_sha=sha(run/'contract.json'), entries=entries)


def load_input(out, task):
    key = f'{task[0]}_{task[1]}_{task[2]:g}'
    contract = read(out/'contract.json')
    if contract['stage'] == 'prepare':
        scene = load_scene(out, key)
        refs = {}
    else:
        entry = read(out/'input_manifest.json')['entries'][key]
        if sha(entry['metadata']) != entry['metadata_sha']:
            raise RuntimeError('Metadata changed: '+key)
        info = read(entry['metadata'])
        if info['sha256'] != entry['payload_sha']:
            raise RuntimeError('Payload reference changed')
        # load_scene checks and consumes the SAME byte buffer.
        scene = load_scene(Path(entry['metadata']).parents[1], key)
        refs = entry['references']
    iq, truth, count = observation(scene, 'mixed')
    if count != 2 or iq.shape != (4, CONFIG['N']) or scene['meta']['sample_rate_hz'] != CONFIG['sample_rate_hz']:
        raise RuntimeError('Observation contract mismatch')
    return scene, iq, truth, digest(iq), refs


def seal(row):
    payload = {k: v for k, v in row.items() if k != 'record_sha'}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def validate_row(row, identity=None):
    if row.get('record_sha') != seal(row):
        raise RuntimeError('Saved row checksum differs')
    if row['result']['config'] != CONFIG or row['method'] not in METHODS:
        raise RuntimeError('Saved configuration differs')
    if identity is not None and row['input_sha'] != identity:
        raise RuntimeError('Saved input differs')
    if row['metrics'] != evaluate(row['truth'], row['result']['positions']):
        raise RuntimeError('Saved metric differs')


def audit(out, expected=None):
    contract = read(out/'contract.json')
    verify(contract['sources']); verify(contract['r2_sources'])
    hashes, grouped, inputs = {}, {}, {}
    for path in sorted((out/'rows').glob('*.json')):
        row = read(path); validate_row(row)
        key = tuple(row['task'])
        if key not in inputs:
            _, _, _, identity, _ = load_input(out, key)
            inputs[key] = identity
        if row['input_sha'] != inputs[key]:
            raise RuntimeError('Input bytes changed after solve')
        hashes[path.name] = sha(path)
        grouped.setdefault(key, {})[row['method']] = row
    for group in grouped.values():
        available = [group[m]['result']['mechanism']['prefix_positions'] for m in METHODS if m.startswith('NOMP') and m in group]
        for p in available[1:]:
            np.testing.assert_allclose(p, available[0], rtol=0, atol=1e-8)
        for suffix in ('ALT', 'JOINT'):
            one, five = f'AP_1_{suffix}', f'AP_5_{suffix}'
            if one in group and five in group:
                a, b = group[one]['result'], group[five]['result']
                np.testing.assert_allclose(a['mechanism']['branches'][0]['final_positions'],
                                           b['mechanism']['branches'][0]['final_positions'], rtol=0, atol=1e-8)
                if b['events'][-1]['native_objective'] > a['events'][-1]['native_objective']+1e-12:
                    raise RuntimeError('Candidate containment inequality violated')
        if contract['stage'] == 'develop' and 'NOMP' in group:
            row = group['NOMP']
            reference = row['references']['BNOMP']
            if sha(reference['path']) != reference['sha256']:
                raise RuntimeError('Historical NOMP changed')
            old = read(reference['path'])
            np.testing.assert_allclose(row['result']['positions'], old['result']['positions'], rtol=0, atol=1e-7)
            if row['input_sha'] != old['input_sha']:
                raise RuntimeError('Historical input pairing differs')
    if expected is not None and (len(hashes) != expected or any(len(g) != len(METHODS) for g in grouped.values())):
        raise RuntimeError('Incomplete paired task manifest')
    result = dict(identity_passed=True, paired_checks_passed=True, rows=hashes,
                  n_rows=len(hashes), n_inputs=len(inputs), complete=expected is not None,
                  sources=contract['sources'] | contract['r2_sources'])
    write(out/'final_audit_report.json', result)
    return result


@contextmanager
def lock(out):
    path = out/'process.lock'
    if path.exists():
        info = read(path)
        try:
            alive = abs(psutil.Process(info['pid']).create_time()-info['created']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError('Another process is using this run')
        path.unlink()  # Exact stale lock only; never remove evidence.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
