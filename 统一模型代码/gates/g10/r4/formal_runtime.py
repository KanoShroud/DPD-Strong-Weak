"""Fresh formal evidence, hashed consumed input, staged resumability and accounting."""
from pathlib import Path
from datetime import datetime
import copy
import shutil
import sys
import time
import numpy as np
import torch
from ..runtime import save_scene, load_scene, verify
from ..r1.runtime import Ledger
from ..r1.scenes import make, observation, digest
from ..metrics import evaluate
from .runtime import read, write, sha, seal, lock, require_exclusive, resource_check, paired_checks
from .formal_protocol import (ROOT, BASE, ENTRY, REVISION, CORE, LIMIT_SECONDS,
    SUBSET_REPEATS, WAVEFORM_SEED, GEOMETRY_SEED, jobs, tasks, key, job_key)


def fresh(prefix):
    return BASE/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))


def safe(path):
    p=Path(path).resolve()
    if not p.is_relative_to(BASE.resolve()) or p==BASE.resolve():
        raise ValueError('Expected isolated R4 output directory')
    return p


def sources():
    files=list((ROOT/'统一模型代码/gates/g10').rglob('*.py'))
    files+=list((ROOT/'统一模型代码/gates/g8').glob('*.py'))
    files+=list((ROOT/'统一模型代码/gates/g9').rglob('*.py'))+[ENTRY]
    files+=[Path(__file__).with_name('formal_contract.md')]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def environment():
    return dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), threads=torch.get_num_threads())


def snapshot(out, identity):
    for rel,digest_value in identity.items():
        p=out/'source_snapshot'/rel; p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,p)
        if sha(p)!=digest_value:
            raise RuntimeError('Source copy differs')


def read_release():
    item=read(BASE/'formal_release.json')
    prep=safe(item['out']).resolve(strict=True)
    ready=read(prep/'readiness.json')
    if not ready.get('ready') or sha(prep/'readiness.json')!=item['readiness_sha']:
        raise RuntimeError('Formal preparation has not been released')
    c=read(prep/'contract.json')
    if sha(prep/'contract.json')!=item['contract_sha'] or c['revision']!=REVISION:
        raise RuntimeError('Release contract identity differs')
    verify(c['sources'])
    if environment()!=c['environment']:
        raise RuntimeError('Execution environment differs from the checked preparation')
    return prep,c,ready


def create(prep, release):
    out=safe(fresh('formal'))
    if out.exists():
        raise FileExistsError('Fresh formal directory required')
    out.mkdir()
    c=copy.deepcopy(release)
    c.update(stage='formal', preparation=str(prep), preparation_sha=sha(prep/'contract.json'),
             expected={s:len(jobs(s)) for s in ('core','supplement','sbl')})
    write(out/'contract.json',c)
    snapshot(out,c['sources'])
    write(out/'status.json',dict(status='CREATED', next_stage='core', completed_stages=[]))
    write(BASE/'formal_run.json',dict(out=str(out),contract_sha=sha(out/'contract.json')))
    return out


def check(out):
    out=safe(out).resolve(strict=True)
    c=read(out/'contract.json')
    if c['revision']!=REVISION:
        raise RuntimeError('Unknown formal contract')
    verify(c['sources'])
    if environment()!=c['environment']:
        raise RuntimeError('Environment identity differs')
    prep=Path(c['preparation']).resolve(strict=True)
    if sha(prep/'contract.json')!=c['preparation_sha']:
        raise RuntimeError('Preparation reference changed')
    return c


def ensure_input(out,task,c):
    name=key(task); meta=out/'inputs'/(name+'.json')
    if not meta.exists():
        g,repeat,gap,_=task
        layout=c['layouts'][g]
        scene=make(g,repeat,gap,positions=layout['positions'],seed=WAVEFORM_SEED)
        scene['meta'].update(role=layout['role'], category=layout['category'],geometry_seed=GEOMETRY_SEED)
        save_scene(out,name,scene)
    info=read(meta)
    scene=load_scene(out,name)  # hash and consume the SAME payload bytes
    g,repeat,gap,_=task
    expected=c['layouts'][g]
    if (info['seed']!=WAVEFORM_SEED or info['group']!=g or info['realization']!=repeat
            or info['gap_db']!=gap or info['positions']!=expected['positions']
            or info['role']!=expected['role'] or info['category']!=expected['category']):
        raise RuntimeError('Input semantics differ from manifest')
    iq,truth,count=observation(scene,'mixed')
    if count!=2 or iq.shape!=(4,8192):
        raise ValueError('Input shape differs')
    identity=dict(iq_sha=digest(iq),metadata_sha=sha(meta),payload_sha=info['sha256'])
    return iq,truth,identity


def validate(row,job,c):
    if row.get('record_sha')!=seal(row) or row['job']!=job or row['result']['method']!=job['method']:
        raise RuntimeError('Result record identity differs')
    if row['result']['config']!=c['configurations'][job['method']]:
        raise RuntimeError('Result configuration differs')
    if row['metrics']!=evaluate(row['truth'],row['result']['positions']):
        raise RuntimeError('Metric replay differs')


def audit(out,c,stages):
    inputs,rows,grouped={}, {}, {}
    for stage in stages:
        for job in jobs(stage):
            path=out/'rows'/(job_key(job)+'.json')
            row=read(path); validate(row,job,c)
            task=tuple(job['task']); name=key(task)
            if name not in inputs:
                _,truth,identity=ensure_input(out,task,c)
                inputs[name]=dict(identity=identity,truth=truth.tolist())
            if row['input_identity']!=inputs[name]['identity'] or row['truth']!=inputs[name]['truth']:
                raise RuntimeError('Input/result pairing differs')
            rows[path.name]=sha(path)
            if stage=='core':
                grouped.setdefault(name,{})[job['method']]=row
    for group in grouped.values():
        paired_checks(group)
    # No row from a future stage is required for an earlier stage audit.
    verify(c['sources'])
    report=dict(passed=True,stages=stages,solves=len(rows),rows=rows,inputs=inputs,
        paired_prefixes=True,metric_replay=True,sources_unchanged=True)
    write(out/('final_audit_report.json' if 'sbl' in stages else stages[-1]+'_audit_report.json'),report)
    return report


def manifest_at_end(out,c):
    entries={}
    for task in tasks():
        _,_,identity=ensure_input(out,task,c)
        entries[key(task)]=identity
    write(out/'input_manifest.json',entries)
    return sha(out/'input_manifest.json')


def compact_result(result,keep_map=False):
    result=copy.deepcopy(result)
    if not keep_map:
        result.pop('native_map',None)
    # Keep each native cycle, accepted output and optimizer diagnostics. Avoid
    # storing a second thousands-step list of local Newton details per REF case.
    updates=result.pop('updates',None)
    mech=result.get('mechanism',{})
    fixed=mech.pop('fixed_q_updates',None)
    if updates or fixed:
        result['local_update_summary']=dict(calls=len(updates or fixed),
            accepted=sum(bool(u['accepted']) for u in (updates or fixed)),
            small_steps=sum(u['step_m']<.01 for u in (updates or fixed)))
    return result


def publication_case(task):
    return task[0] in (0,5,10,15) and task[1]==SUBSET_REPEATS[0] and task[2] in (0,15)


__all__=['read','write','sha','seal','lock','require_exclusive','resource_check','Ledger',
         'sources','snapshot','environment','fresh','safe','read_release','create','check',
         'ensure_input','validate','audit','manifest_at_end','compact_result','publication_case',
         'LIMIT_SECONDS','ROOT','BASE','time','CORE']
