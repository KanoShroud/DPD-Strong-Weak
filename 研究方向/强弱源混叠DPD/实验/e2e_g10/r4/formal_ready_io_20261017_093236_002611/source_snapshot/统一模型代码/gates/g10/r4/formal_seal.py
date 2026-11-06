"""Package passed numerical checks into a complete formal manifest; no new solves."""
import copy
import shutil
from pathlib import Path
from ..runtime import verify
from .runtime import read,write,sha
from .formal_protocol import (BASE,ENTRY,GEOMETRY_SEED,WAVEFORM_SEED,OPTIMIZER_SEED,
    SUBSET_REPEATS,REPEATS,tasks,jobs)
from .formal_runtime import fresh,sources,snapshot,environment
from .formal_prepare import command_check
from .formal_entry_checks import main as check_stages


def main(refresh_orchestration=False):
    if (BASE/'formal_run.json').exists():
        raise RuntimeError('An existing formal run requires a separate migration review')
    pointer=read(BASE/'formal_release.json')
    prep=Path(pointer['out']).resolve(strict=True)
    old=read(prep/'contract.json')
    if old['stage']!='preparation' and not refresh_orchestration:
        raise RuntimeError('Expected a numerical preparation, not an already sealed release')
    assert sha(prep/'contract.json')==pointer['contract_sha']
    assert sha(prep/'readiness.json')==pointer['readiness_sha']
    assert read(prep/'readiness.json')['ready']
    audit=read(prep/'final_audit_report.json')
    assert audit['passed'] and audit['solves']==60
    current=sources()
    changed={p:dict(before=h,after=current.get(p)) for p,h in old['sources'].items() if current.get(p)!=h}
    if refresh_orchestration:
        allowed={'统一模型代码/gates/g10/r4/'+name for name in
                 ('formal_runner.py','formal_entry_checks.py','formal_seal.py','formal_contract.md')}
        allowed.add(str(ENTRY.relative_to(ENTRY.parents[3])))
        assert {p.replace('\\','/') for p in changed}<={p.replace('\\','/') for p in allowed}, 'Unexpected scientific source change'
        verify({p:h for p,h in old['sources'].items() if p not in changed})
    else:
        verify(old['sources'])
    assert environment()==old['environment']
    for name,digest in audit['rows'].items():
        assert sha(prep/'rows'/name)==digest
    out=fresh('formal_ready'); out.mkdir()
    c=copy.deepcopy(old)
    evidence=Path(old.get('numerical_evidence_preparation',str(prep)))
    c.update(stage='formal_release',preparation_tasks=c.get('preparation_tasks',c['tasks']),tasks=tasks(),
        seeds=dict(geometry=GEOMETRY_SEED,waveform=WAVEFORM_SEED,optimizer=OPTIMIZER_SEED),
        repeats=list(REPEATS),subset_repeats=list(SUBSET_REPEATS),
        numerical_evidence_preparation=str(evidence),numerical_evidence_contract_sha=sha(evidence/'contract.json'),
        numerical_evidence_audit_sha=sha(evidence/'final_audit_report.json'),sources=current,
        execution_mode='continuous_all_stages',previous_release=str(prep),orchestration_changes=changed)
    assert len(c['tasks'])==2400
    assert c['expected']=={s:len(jobs(s)) for s in ('core','supplement','sbl')}
    write(out/'contract.json',c); snapshot(out,c['sources'])
    for name in ('rows','ga_stop'):
        shutil.copytree(prep/name,out/name)
    for name in ('final_audit_report.json','budget_estimate.json','ga_stop_review.json',
                 'adapter_equivalence.json','recovery_checks.json','console_check.json','warmup.json'):
        shutil.copyfile(prep/name,out/name)
    try:
        check_stages(out)
        verify(c['sources'])
        ready=read(prep/'readiness.json')
        ready.update(formal_manifest_verified=True,stage_transition_checks=True,
                     numerical_evidence=str(evidence),packaging_only=True)
        write(out/'readiness.json',ready)
        write(BASE/'formal_release.json',dict(out=str(out),contract_sha=sha(out/'contract.json'),
            readiness_sha=sha(out/'readiness.json')))
        write(out/'working_directory_check.json',command_check(['--check-only'],ENTRY.parent))
        write(out/'status.json',dict(status='READY_FOR_USER',formal_started=False,numerical_evidence=str(prep)))
        summary=(f'# G10-R4 正式入口准备完成\n\n'
            f'- 正式运行尚未启动；未生成正式 IQ。\n'
            f'- 核心：2400 条观测 × 6 版本；共同子集：240 条观测，补充 8 版本与 SBL。\n'
            f'- 总求解次数：16560；预计 {ready["forecast_hours"]:.2f} 小时，已批准预算 12 小时。\n'
            f'- 一次点击连续完成核心、补充参照及 SBL；阶段报告分别保存。\n'
            f'- 数值短测证据：{evidence}\n'
            f'- 本目录只补齐正式清单、种子并核验真实入口，不重跑数值实验。\n'
            f'- 工程模拟覆盖分批衔接、已完成任务复用和预算边界。\n')
        (out/'准备报告.md').write_text(summary,encoding='utf-8')
        print(str(out)); print(summary)
    except BaseException:
        write(out/'readiness.json',dict(ready=False,status='FAILED_PACKAGING_RETAINED',formal_started=False))
        write(out/'status.json',dict(status='FAILED_PACKAGING_RETAINED'))
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh-orchestration',action='store_true')
    main(parser.parse_args().refresh_orchestration)
