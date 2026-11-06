"""User-run formal comparison; one launch runs core, supplements, then SBL."""
import argparse
from pathlib import Path
import time
import traceback
from ..metrics import evaluate
from . import formal_adapter as adapter
from .formal_protocol import jobs, job_key, solver_seed, LIMIT_SECONDS, BASE
from .formal_runtime import (read,write,sha,seal,lock,require_exclusive,resource_check,Ledger,
    read_release,create,check,ensure_input,validate,audit,manifest_at_end,compact_result,publication_case)
from .formal_report import generate


def record(out,job,c,clock):
    iq,truth,identity=ensure_input(out,job['task'],c)
    path=out/'rows'/(job_key(job)+'.json')
    if path.exists():
        row=read(path); validate(row,job,c)
        if row['input_identity']!=identity or row['truth']!=truth.tolist():
            raise RuntimeError('Completed row/input identity mismatch')
        return row,False
    require_exclusive()
    clock.flush(dict(tag=job_key(job),ledger_start_seconds=clock.total()))
    keep_map=publication_case(job['task'])
    result=adapter.run(iq,job['method'],c['configurations'][job['method']],solver_seed(job['task']),clock.check,keep_map)
    row=dict(job=job,input_identity=identity,truth=truth.tolist(),
        result=compact_result(result,keep_map),metrics=evaluate(truth,result['positions']))
    row['record_sha']=seal(row)
    write(path,row); validate(read(path),job,c)
    clock.flush()
    return row,True


def execute(out):
    c=check(out); require_exclusive(); resource_check()
    state=read(out/'status.json')
    if state['status']=='COMPLETED':
        print('正式比较已完成，不重复运行。结果：'+str(out),flush=True); return
    if 'core' in state.get('completed_stages',[]):
        print('本次继续综合比较，先非SBL补充，后SBL；核心结果不重算。',flush=True)
    else:
        print('本次连续运行：核心主批 → 非SBL补充 → SBL；各阶段自动保存报告。',flush=True)
    stages=state.get('completed_stages',[])
    with lock(out):
        clock=Ledger(out)
        try:
            for stage in ('core','supplement','sbl'):
                if stage in stages:
                    continue
                plan=jobs(stage)
                started=time.perf_counter(); last=0.; computed=0; used_seconds=0.
                for i,job in enumerate(plan,1):
                    if clock.total()>=LIMIT_SECONDS:
                        clock.flush()
                        write(out/'status.json',dict(status='BUDGET_PAUSED',next_stage=stage,completed_stages=stages,seconds=clock.total()))
                        print('\n累计12小时预算已到；未截断当前算法。已保存结果，需确认增加预算后再继续。',flush=True)
                        return
                    row,new=record(out,job,c,clock)
                    computed+=int(new)
                    if new:
                        used_seconds+=row['result']['complete_seconds']
                    now=time.perf_counter()
                    if now-last>=20 or i==len(plan):
                        expected=sum(c['cost_means'][j['method']] for j in plan[i:])
                        done_expected=sum(c['cost_means'][j['method']] for j in plan[:i])
                        ratio=used_seconds/max(done_expected,1e-9) if computed==i else 1.
                        eta=expected*max(.5,ratio)
                        filled=int(24*i/len(plan))
                        print((f"\rG10-R4 {stage} [{'#'*filled}{'-'*(24-filled)}] {i}/{len(plan)}"
                              f" | 累计{clock.total()/3600:.2f}h | 本批ETA {eta/60:.1f}min | {job_key(job)}").ljust(170),
                              end='\n' if i==len(plan) else '',flush=True)
                        last=now
                print(f'{stage}求解结束，进行身份审计和统计绘图。',flush=True)
                stages=stages+[stage]
                audit(out,c,stages)
                if stage=='core':
                    digest_value=manifest_at_end(out,c)
                    write(out/'input_manifest_identity.json',dict(sha256=digest_value))
                generate(out,c,stages)
                clock.flush()
                write(out/(stage+'_timing.json'),dict(seconds=time.perf_counter()-started,newly_computed=computed))
                if stage=='core':
                    write(out/'status.json',dict(status='CORE_COMPLETE',next_stage='supplement',completed_stages=stages,seconds=clock.total()))
                    (out/'运行摘要.md').write_text('# G10-R4核心主批完成\n\n核心2400条×6版本已完成。请回读reports/core/运行摘要.md、comparison_report.json及core_audit_report.json。\n\n程序自动继续240条综合比较，SBL最后运行。\n',encoding='utf-8')
                    print('核心主批已完成，报告已保存；现在自动继续补充比较。\n'+str(out),flush=True)
                    continue
                write(out/'status.json',dict(status='SUPPLEMENT_COMPLETE' if stage=='supplement' else 'COMPLETED',
                    next_stage='sbl' if stage=='supplement' else None,completed_stages=stages,seconds=clock.total()))
                if stage=='supplement':
                    print('非SBL综合补充已完成，报告已保存；现在开始SBL慢批。',flush=True)
            # Final input manifest must still describe the exact consumed bytes.
            expected=read(out/'input_manifest_identity.json')['sha256']
            if sha(out/'input_manifest.json')!=expected:
                raise RuntimeError('Input manifest identity changed')
            (out/'运行摘要.md').write_text('# G10-R4全部完成\n\n核心比较与共同240条综合比较均已完成。回读reports/sbl/运行摘要.md、comparison_report.json及根目录final_audit_report.json。\n\n核心/综合样本规模分别报告，不跨样本排名。旧阶段报告保留。\n',encoding='utf-8')
            clock.flush()
            write(out/'status.json',dict(status='COMPLETED',completed_stages=stages,seconds=clock.total()))
            print('G10-R4全部完成：'+str(out),flush=True)
        except BaseException:
            clock.flush(keep=True)
            write(out/'status.json',dict(status='STOPPED',completed_stages=[s for s in stages if (out/'reports'/s/'comparison_report.json').exists()],
                error=traceback.format_exc(),seconds=clock.total()))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description='G10-R4完整比较：一次点击依次运行核心、补充及SBL；中文路径与日志。')
    parser.add_argument('--prepare',action='store_true',help='仅开发接入检查，由Codex使用')
    parser.add_argument('--check-only',action='store_true',help='只读检查放行配置，不生成正式数据')
    parser.add_argument('--replot',type=Path,help='仅根据保存结果重新统计绘图')
    args=parser.parse_args(argv)
    if args.prepare:
        from .formal_prepare import main as prepare
        prepare(); return
    if args.replot:
        out=args.replot.resolve(strict=True); c=read(out/'contract.json')
        stages=read(out/'status.json')['completed_stages']
        generate(out,c,stages); return
    prep,c,ready=read_release()
    if args.check_only:
        resource_check()
        print(f"放行检查通过。预计{ready['forecast_hours']:.2f}小时；预算12小时。没有生成正式观测。",flush=True)
        return
    require_exclusive(); resource_check()
    pointer=BASE/'formal_run.json'
    if pointer.exists():
        item=read(pointer); out=Path(item['out']).resolve(strict=True)
        if sha(out/'contract.json')!=item['contract_sha']:
            raise RuntimeError('Active formal contract changed')
    else:
        out=create(prep,c)
    print('结果目录：'+str(out),flush=True)
    execute(out)


if __name__=='__main__':
    main()
