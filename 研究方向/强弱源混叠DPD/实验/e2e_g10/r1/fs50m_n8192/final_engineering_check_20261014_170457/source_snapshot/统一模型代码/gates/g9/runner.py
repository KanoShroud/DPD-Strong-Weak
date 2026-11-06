"""开发准备与正式比较分离；正式阶段仅允许读取已冻结开发合同。"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

from .metrics import evaluate
from .runtime import BASE, Guard, identity, location, preflight, read, source_manifest, verify, write
from .scenes import make
from .solver import run

METHODS = ('A', 'B', 'C', 'DPD', 'HR')
LENGTHS = (4096, 16384)
FACTORS = (.5, 1., 2.)


def key_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def last_positions(result, seconds):
    events = [e for e in result['events'] if e['seconds'] <= seconds]
    return events[-1]['positions'] if events else []


class Jobs:
    def __init__(self, out, initial=0.):
        self.out = out
        self.start = time.perf_counter()
        ledger = out/'ledger.json'
        self.consumed = read(ledger)['charged_seconds'] if ledger.exists() else initial
        self.guard = Guard(8*3600, self.consumed)
        self.last_progress = 0.

    def task(self, tag, iq, truth, method, budget, metadata, hr=(2, 1e-4)):
        self.guard()
        path = self.out/'rows'/f'{tag}.json'
        request = dict(tag=tag, method=method, budget=budget, metadata=metadata, hr=list(hr),
                       iq_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest())
        if path.exists():
            row = read(path)
            seal = row.pop('row_sha256', None)
            if seal != key_hash(row) or row['request_hash'] != key_hash(request):
                raise RuntimeError('恢复请求身份不一致')
            row['row_sha256'] = seal
            return row
        before = time.perf_counter()
        # 中断时保守计入未完成任务的时间额度，恢复不免费重置总预算。
        write(self.out/'ledger.json', dict(charged_seconds=self.consumed+before-self.start+budget+1))
        result = run(iq, method, budget, count=len(truth), guard=self.guard, hr_config=hr)
        row = dict(request_hash=key_hash(request), **metadata, method=method,
                   budget_seconds=budget, **result,
                   metrics=evaluate(truth, last_positions(result, budget)))
        row['row_sha256'] = key_hash(row)
        write(path, row)
        write(self.out/'ledger.json', dict(charged_seconds=self.consumed+time.perf_counter()-self.start))
        if time.perf_counter()-self.last_progress >= 15:
            self.last_progress = time.perf_counter()
            print(f'\r进度 {tag} | 累计计算 {(self.consumed+time.perf_counter()-self.start)/60:.1f} min', end='', flush=True)
        return row


def environment():
    return dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                device=torch.cuda.get_device_name(), threads=torch.get_num_threads())


def begin(stage, run_id):
    out = location(BASE/run_id)
    if out.exists():
        contract = read(out/'source_contract.json')
        if contract['stage'] != stage:
            raise RuntimeError('目录阶段不匹配')
        verify(contract['sources'])
    else:
        out = preflight(out)
        contract = dict(stage=stage, sources=source_manifest(), environment=environment())
        write(out/'source_contract.json', contract)
    return out, contract


def smoke(out, jobs):
    scene = make('smoke', 1, 0, .5, 0, 12)
    results = []
    for n in LENGTHS:
        iq = scene['signal'][1, :, :n]+scene['noise'][:, :n]
        results.append(jobs.task(f'smoke_single_{n}', iq, scene['positions'][1:], 'A', 15., dict(N=n)))
    iq = scene['signal'][:, :, :4096].sum(0)+scene['noise'][:, :4096]
    for method in METHODS:
        results.append(jobs.task('smoke_'+method, iq, scene['positions'], method, 15., dict(N=4096)))
    report = dict(status='SMOKE_COMPLETED', scope='工程与吞吐，不作为定位性能结论', results=results)
    write(out/'report.json', report)
    print('\n短测完成：'+str(out/'report.json'))


def prepare(out, jobs):
    qualification = []
    selected_snr = None
    print('阶段1/3：孤立源条件核验（开发集，保持原波形/噪声配对）')
    for snr in (0, 3, 6):
        rows = []
        for group in range(16):
            for overlap in (.5, 1.):
                scene = make('dev', group, 0, overlap, snr, 12)
                for n in LENGTHS:
                    for source in (0, 1):
                        iq = scene['signal'][source, :, :n]+scene['noise'][:, :n]
                        metadata = dict(group=group, overlap=overlap, N=n, source=source, snr=snr)
                        rows.append(jobs.task(f'qual_{snr}_{group}_{overlap}_{n}_{source}', iq,
                            scene['positions'][source:source+1], 'A', 20., metadata))
        rates = {str(n):float(np.mean([all(r['metrics']['success']['100']) for r in rows
                                      if r['N'] == n and r['source'] == 1])) for n in LENGTHS}
        qualification.append(dict(snr=snr, weak_success=rates))
        print(f'\n孤立弱源 SNR={snr} dB：{rates}', flush=True)
        if min(rates.values()) >= .9:
            selected_snr = snr
            break
    if selected_snr is None:
        write(out/'report.json', dict(status='SCENARIO_NOT_READY', qualification=qualification,
            reason='0/3/6 dB孤立弱源参照未同时达到90%；暂停正式比较，不自动增加SNR或改门槛'))
        return
    print('阶段2/3：开发集AP完整耗时与HR配置校准')
    timing, hr_selected, baseline_rows = {}, {}, []
    for n in LENGTHS:
        calibration = {}
        ap_times = []
        for group in range(16):
            for overlap in (.5, 1.):
                scene = make('dev', group, 0, overlap, selected_snr, 12)
                iq = scene['signal'][:, :, :n].sum(0)+scene['noise'][:, :n]
                meta = dict(group=group, overlap=overlap, N=n, gap=12, snr=selected_snr)
                row = jobs.task(f'timing_{group}_{overlap}_{n}', iq, scene['positions'], 'A', 40., meta)
                ap_times.append(row['seconds'])
                for segments in (2, 4, 8):
                    for loading in (1e-4, 1e-2):
                        config = (segments, loading)
                        r = jobs.task(f'hr_{group}_{overlap}_{n}_{segments}_{loading}', iq,
                            scene['positions'], 'HR', 20., meta, config)
                        calibration.setdefault(config, []).append(r['metrics']['gospa_m'])
        timing[str(n)] = float(np.median(ap_times))
        hr_selected[str(n)] = list(min(calibration, key=lambda c: (np.mean(calibration[c]), c)))
    print('\n阶段3/3：冻结常规搜索对照、预算与合同')
    for method in ('A', 'B'):
        for group in range(16):
            for overlap in (.5, 1.):
                scene = make('dev', group, 0, overlap, selected_snr, 12)
                for n in LENGTHS:
                    iq = scene['signal'][:, :, :n].sum(0)+scene['noise'][:, :n]
                    meta = dict(group=group, overlap=overlap, N=n)
                    baseline_rows.append(jobs.task(f'baseline_{method}_{group}_{overlap}_{n}', iq,
                        scene['positions'], method, timing[str(n)], meta))
    best_baseline = min(('A', 'B'), key=lambda method: (
        -np.mean([all(r['metrics']['success']['100']) for r in baseline_rows if r['method'] == method]),
        np.mean([r['metrics']['gospa_m'] for r in baseline_rows if r['method'] == method])))
    used = read(out/'ledger.json')['charged_seconds']
    # 256条混合/N；5方法一次运行到2T0，三个预算从同一anytime轨迹读取。
    # 单独强弱对照：每N共256条单源，所有方法运行到相同2T0。
    upper = sum((256+256)*len(METHODS)*2*timing[str(n)] for n in LENGTHS)*1.15
    contract = dict(status='READY' if used+upper <= 8*3600 else 'BUDGET_REVIEW_REQUIRED',
        protocol='EVAL-20261010-v1', selected_snr_db=selected_snr, qualification=qualification,
        T0_seconds=timing, hr=hr_selected, best_baseline=best_baseline,
        development_seconds=used, conservative_formal_seconds=upper, total_limit_seconds=8*3600,
        factors=list(FACTORS), groups=32, realizations=2, gaps=[6, 12], overlaps=[.5, 1.],
        sources=source_manifest(), dev_contract=identity(out/'source_contract.json'),
        environment=environment(), main_radius_m=100, auxiliary_radius_m=[30, 50],
        gospa=dict(p=2, c=100, alpha=2), seed=2026101009,
        response_weight=.25, global_exploration_every=4, cells_per_search=48,
        starts=5, ap_cycles=5, local_maxiter=30, candidate_exclusion_m=30,
        check_role='check', test_access=False)
    write(out/'frozen_contract.json', contract)
    write(out/'report.json', contract)
    print('\n准备完成：'+contract['status'])
    print(f'正式运行保守上限 {upper/3600:.2f} h；开发已用 {used/3600:.2f} h')


def formal(out, jobs, contract):
    from .report import finalize
    verify(contract['sources'])
    if contract['status'] != 'READY':
        raise RuntimeError('开发合同未放行正式比较：'+contract['status'])
    rows = []
    total = 32*2*2*2
    print(f'正式比较：{total}组长度/重叠/随机实现；含2档功率差与单源对照')
    for group in range(32):
        for realization in range(2):
            for overlap in (.5, 1.):
                for gap in (6, 12):
                    scene = make('check', group, realization, overlap, contract['selected_snr_db'], gap)
                    for n in LENGTHS:
                        kinds = ('mixed', 'strong', 'weak') if gap == 12 else ('mixed',)
                        for kind in kinds:
                            ids = [0, 1] if kind == 'mixed' else [0] if kind == 'strong' else [1]
                            iq = scene['signal'][ids, :, :n].sum(0)+scene['noise'][:, :n]
                            truth = scene['positions'][ids]
                            for method in METHODS:
                                meta = dict(group=group, realization=realization, overlap=overlap,
                                            gap=gap, N=n, kind=kind)
                                tag = f'{group}_{realization}_{overlap}_{gap}_{n}_{kind}_{method}'
                                raw = jobs.task(tag, iq, truth, method, 2*contract['T0_seconds'][str(n)],
                                                meta, tuple(contract['hr'][str(n)]))
                                for factor in FACTORS:
                                    cutoff = factor*contract['T0_seconds'][str(n)]
                                    rows.append(dict(**meta, method=method, factor=factor,
                                        seconds=min(raw['seconds'], cutoff), budget_seconds=cutoff,
                                        score_points=raw['score_points'] if factor == 2 else None,
                                        response_points=raw['response_points'] if factor == 2 else None,
                                        updates=raw['updates'] if factor == 2 else None,
                                        metrics=evaluate(truth, last_positions(raw, cutoff))))
        print(f'\n布局完成 {group+1}/32', flush=True)
    write(out/'evaluation_rows.json', rows)
    report = finalize(out, rows, contract)
    verify(contract['sources'])
    write(out/'final_audit_report.json', dict(status='PASS', test_access=False,
        rows=len(rows), source_identity='PASS', report=identity(out/'comparison_report.json')))
    print('\n正式比较完成：'+str(out/'comparison_report.json'))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'prepare', 'formal'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--contract')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.init()
    torch.zeros(1, device='cuda').sum().item()
    out, source = begin(args.stage, args.run_id)
    contract = read(args.contract) if args.contract else None
    if args.stage == 'formal' and contract is None:
        raise ValueError('正式运行必须指定冻结合同')
    previous = contract['development_seconds'] if contract else 0.
    if args.stage == 'prepare':
        # 工程短测同样计入本Gate计算预算；只读已经完成的短测账本。
        for path in sorted(BASE.glob('smoke_*/ledger.json')):
            previous += read(path)['charged_seconds']
    if args.stage == 'formal':
        verify([contract['dev_contract']])
        if environment() != contract['environment']:
            raise RuntimeError('运行环境与冻结合同不同，先复核计时可比性')
    jobs = Jobs(out, previous)
    try:
        if args.stage == 'smoke':
            smoke(out, jobs)
        elif args.stage == 'prepare':
            prepare(out, jobs)
        else:
            formal(out, jobs, contract)
        verify(source['sources'])
    except Exception as error:
        failure = 'failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'
        write(out/failure, dict(status='FAILED', error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUTF8', '1')
    main()
