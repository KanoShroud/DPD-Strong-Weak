"""Approved R4: 2400 core cases, paired 240-case panorama; no data-based tuning."""
from pathlib import Path
import copy
import numpy as np
from ..r2.band.scenes import geometry_valid
from .config import ROOT, BASE
from .runtime import read, sha, check_contract

ENTRY = ROOT/'运行入口/E2E/G10/G10R4完整比较一键运行.py'
REVISION = 'G10_R4_FORMAL_CORE2400_PANORAMA240_V1'
CORE = ('NOMP', 'NOMP_JOINT', 'AP_1_ALT', 'AP_1_JOINT', 'AP_2_JOINT', 'AP_5_JOINT')
EXTRA = ('BNOMP_REF', 'PROFILE_DIAG', 'LS', 'MUSIC', 'MVDR', 'AP_FULL', 'MULTISTART_FIXED', 'GA')
METHODS = CORE+EXTRA+('SBL',)
GAPS = (0, 3, 6, 9, 12, 15)
REPEATS = tuple(range(400, 420))
SUBSET_REPEATS = (400, 410)
GEOMETRY_SEED, WAVEFORM_SEED, OPTIMIZER_SEED = 20261017041, 20261017042, 20261017043
LIMIT_SECONDS = 12*3600
TIMES = (.25, .5, 1., 2., 3., 5., 15., 30., 60., 120., 300.)


def layouts():
    rng, result = np.random.default_rng(GEOMETRY_SEED), []
    for category in range(4):
        for number in range(5):
            for _ in range(100000):
                p = rng.uniform(-950, 950, (2, 2))
                if geometry_valid(p, category):
                    result.append(dict(id=len(result), category=category, positions=p.tolist(),
                        role='analysis' if number < 3 else 'locked_validation'))
                    break
            else:
                raise RuntimeError('No geometry satisfying the approved constraints')
    return result


def tasks():
    return [(g, r, gap, 'mixed') for g in range(20) for r in REPEATS for gap in GAPS]


def key(task):
    return f'{task[0]}_{task[1]}_{task[2]}'


def job_key(job):
    return key(job['task'])+'_'+job['method']


def jobs(stage):
    methods = CORE if stage == 'core' else EXTRA if stage == 'supplement' else ('SBL',)
    result = []
    for i, task in enumerate(tasks()):
        if stage != 'core' and task[1] not in SUBSET_REPEATS:
            continue
        order = methods[i % len(methods):]+methods[:i % len(methods)]
        result.extend(dict(task=list(task), method=m, stage=stage) for m in order)
    return result


def solver_seed(task):
    return int(np.random.SeedSequence([OPTIMIZER_SEED, *task[:3]]).generate_state(1)[0])


def frozen_configurations():
    pointer = read(BASE/'preparation.json')
    prep = Path(pointer['out']).resolve(strict=True)
    if sha(prep/'contract.json') != pointer['contract_sha']:
        raise RuntimeError('Original preparation identity changed')
    original = check_contract(prep)
    cfgs = copy.deepcopy(original['configurations'])
    for m in ('NOMP', 'NOMP_JOINT'):
        cfgs[m]['cycles'] = 200
    cfgs['BNOMP_REF']['bnomp_cycles'] = 10000
    cfgs['PROFILE_DIAG'] = copy.deepcopy(cfgs['BNOMP_REF'])
    r2 = Path(original['reference']['out'])
    r2 = (r2 if r2.is_absolute() else ROOT/r2).resolve(strict=True)
    old = read(r2/'contract.json')
    for method in EXTRA+('SBL',):
        if method in cfgs:
            continue
        name = 'MULTISTART' if method == 'MULTISTART_FIXED' else method
        found = [j['config'] for j in old['jobs'] if j['method']==name]
        if not found or any(c!=found[0] for c in found):
            raise RuntimeError('Nonunique frozen configuration: '+name)
        cfgs[method] = copy.deepcopy(found[0])
    cfgs['MULTISTART_FIXED']['starts'] = 8
    return cfgs, dict(preparation=str(prep), preparation_sha=sha(prep/'contract.json'),
        r2=str(r2), r2_contract_sha=sha(r2/'contract.json'),
        r2_manifest_sha=sha(r2/'input_manifest.json'))
