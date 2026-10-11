"""保留历史runner/solver；修复版单独合同、目录、入口。"""
import argparse
import hashlib
import os
import time
import traceback

import numpy as np
import torch

from .metrics import evaluate
from .runner import Jobs, key_hash, last_positions, begin, environment, prepare, formal
from .runtime import BASE, read, write, verify, identity
from .scenes import make
from .solver_v2 import run


class RepairJobs(Jobs):
    def __init__(self,out,initial=0.):
        super().__init__(out,initial)
        path=out/'ledger.json'
        self.starting_charge=read(path).get('starting_charge',initial) if path.exists() else initial

    def charge(self,value):
        write(self.out/'ledger.json',dict(charged_seconds=value,starting_charge=self.starting_charge))

    def task(self,tag,iq,truth,method,budget,metadata,hr=(2,1e-4)):
        self.guard()
        path=self.out/'rows'/f'{tag}.json'
        request=dict(tag=tag,method=method,budget=budget,metadata=metadata,hr=list(hr),
            solver='v2',iq_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest())
        if path.exists():
            row=read(path)
            seal=row.pop('row_sha256',None)
            if seal!=key_hash(row) or row['request_hash']!=key_hash(request):
                raise RuntimeError('修复版恢复身份不一致')
            row['row_sha256']=seal
            return row
        before=time.perf_counter()
        self.charge(self.consumed+before-self.start+budget+1)
        result=run(iq,method,budget,count=len(truth),guard=self.guard,hr_config=hr)
        row=dict(request_hash=key_hash(request),**metadata,method=method,budget_seconds=budget,
            **result,metrics=evaluate(truth,last_positions(result,budget)))
        row['row_sha256']=key_hash(row)
        write(path,row)
        self.charge(self.consumed+time.perf_counter()-self.start)
        if time.perf_counter()-self.last_progress>=15:
            self.last_progress=time.perf_counter()
            print(f'\r修复版 {tag} | 累计 {(self.consumed+time.perf_counter()-self.start)/60:.1f} min',end='',flush=True)
        return row


def smoke(out,jobs):
    scene=make('smoke',1,0,.5,3,12)
    rows=[]
    for n in (4096,16384):
        for method in ('A','B','C','DPD','HR'):
            for count in (1,2):
                iq=(scene['signal'][1,:, :n] if count==1 else scene['signal'][:,:, :n].sum(0))+scene['noise'][:,:n]
                truth=scene['positions'][1:] if count==1 else scene['positions']
                rows.append(jobs.task(f'{n}_{method}_{count}',iq,truth,method,15.,dict(N=n,K=count)))
    write(out/'report.json',dict(status='SMOKE_COMPLETED',rows=rows,scope='工程验证，不作性能结论'))


def cost_calibration(out,jobs):
    path=out/'frozen_contract.json'
    if not path.exists():
        return
    contract=read(path)
    write(out/'pre_cost_contract.json',contract)
    print('\n阶段4/4：实测混合与单源各方法成本，不把已收敛任务按满预算估算')
    timing=[]
    for group in range(4):
        for overlap in (.5,1.):
            scene=make('dev',group,0,overlap,contract['selected_snr_db'],12)
            for n in (4096,16384):
                for kind in ('mixed','strong','weak'):
                    ids=[0,1] if kind=='mixed' else [0] if kind=='strong' else [1]
                    iq=scene['signal'][ids,:,:n].sum(0)+scene['noise'][:,:n]
                    for method in ('A','B','C','DPD','HR'):
                        timing.append(jobs.task(f'cost_{group}_{overlap}_{n}_{kind}_{method}',
                            iq,scene['positions'][ids],method,2*contract['T0_seconds'][str(n)],
                            dict(group=group,overlap=overlap,N=n,kind=kind),tuple(contract['hr'][str(n)])))
    estimate=0.
    details=[]
    for n in (4096,16384):
        for method in ('A','B','C','DPD','HR'):
            for kind,count in (('mixed',256),('strong',128),('weak',128)):
                times=[r['seconds'] for r in timing if r['N']==n and r['method']==method and r['kind']==kind]
                seconds=float(np.mean(times))
                estimate+=count*seconds
                details.append(dict(N=n,method=method,kind=kind,mean_seconds=seconds,formal_count=count))
    used=read(out/'ledger.json')['charged_seconds']
    contract.update(solver_version='g9_v2_common_dpd_proposal',development_seconds=used,
        maximum_budget_estimate_seconds=contract['conservative_formal_seconds'],
        measured_formal_estimate_seconds=estimate,conservative_formal_seconds=estimate*1.3,
        cost_estimate_details=details,status='READY' if used+estimate*1.3<=8*3600 else 'BUDGET_REVIEW_REQUIRED')
    write(path,contract)
    write(out/'report.json',contract)
    print(f'\n成本标定：预计{estimate/3600:.2f}小时，含30%预留{estimate*1.3/3600:.2f}小时；{contract["status"]}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','prepare','formal'),required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--contract')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.zeros(1,device='cuda').sum().item()
    out,source=begin('v2_'+args.stage,args.run_id)
    contract=read(args.contract) if args.contract else None
    if args.stage=='formal':
        if contract is None or contract.get('solver_version')!='g9_v2_common_dpd_proposal':
            raise ValueError('正式比较缺少修复版合同')
        verify(contract['sources'])
        verify([contract['dev_contract']])
        if environment()!=contract['environment']:
            raise RuntimeError('设备/环境与冻结合同不符')
        previous=contract['development_seconds']
    else:
        previous=read(BASE/'score_diagnosis_20261011_01/ledger.json')['cumulative_seconds']
        for ledger in BASE.glob('repair_smoke_*/ledger.json'):
            if ledger.parent==out:
                continue
            previous+=read(ledger)['charged_seconds']-read(ledger)['starting_charge']
    jobs=RepairJobs(out,previous)
    try:
        if args.stage=='smoke':
            smoke(out,jobs)
        elif args.stage=='prepare':
            prepare(out,jobs)
            cost_calibration(out,jobs)
        else:
            formal(out,jobs,contract)
        verify(source['sources'])
        if args.stage!='formal':
            write(out/'final_audit_report.json',dict(status='PASS',sources=source['sources'],
                report=identity(out/'report.json'),test_access=False,scope='修复版'+args.stage))
    except Exception:
        write(out/f'failure_{time.time_ns()}.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    os.environ.setdefault('PYTHONUTF8','1')
    main()
