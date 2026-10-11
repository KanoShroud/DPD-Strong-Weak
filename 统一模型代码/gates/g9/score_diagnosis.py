"""真值仅用于归因；旧DPD预测初始化的结果独立登记为非真值分支。"""
import hashlib
import time

import numpy as np
import torch

from .runner import key_hash, last_positions
from .runtime import BASE, Guard, identity, preflight, read, source_manifest, verify, write
from .scenes import make
from .solver import Score, local


def main():
    torch.set_num_threads(4)
    prior = BASE/'prepare_20261010_01'
    bridge = BASE/'bridge_20261010_01'
    verify(read(bridge/'final_audit_report.json')['sources'])
    verify(read(bridge/'final_audit_report.json')['inputs'])
    out = preflight(BASE/'score_diagnosis_20261011_01')
    sources = source_manifest()
    inputs = [identity(bridge/'report.json')]
    start = time.perf_counter()
    previous = read(bridge/'ledger.json')['cumulative_seconds']
    guard = Guard(8*3600, previous)
    rows = []
    write(out/'contract.json', dict(sources=sources, scope='开发单源评分归因及非真值初始化检查',
        oracle_policy='truth与truth_local仅诊断；deploy_seed来自旧DPD预测', prior_seconds=previous))
    last = 0.
    for index, path in enumerate(sorted((prior/'rows').glob('qual_*.json'))):
        guard()
        old = read(path)
        seal = old.pop('row_sha256')
        assert key_hash(old) == seal
        meta = {k:old[k] for k in ('group','overlap','N','source','snr')}
        scene = make('dev', meta['group'], 0, meta['overlap'], meta['snr'], 12)
        iq = scene['signal'][meta['source'], :, :meta['N']]+scene['noise'][:, :meta['N']]
        request = dict(tag=path.stem, method='A', budget=20., metadata=meta, hr=[2,1e-4],
                       iq_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest())
        assert key_hash(request) == old['request_hash']
        linked = bridge/'rows'/path.name
        inputs.append(identity(linked))
        data = {r['method']:r for r in read(linked)}
        truth = scene['positions'][meta['source']]
        new = np.asarray(last_positions(old, 20.))[0]
        seed = np.asarray(data['old_old']['positions'])[0]
        calc = Score(iq, guard)
        true_score, returned_score, seed_score = calc.evaluate([truth, new, seed])
        # 非真值分支：旧DPD定位输出作为连续精修初值，保留原候选防止下降。
        refined = local(calc, seed, [])
        refined_score = calc.evaluate([refined])[0]
        if refined_score < seed_score:
            refined, refined_score = seed, seed_score
        oracle = local(calc, truth, [])
        row = dict(**meta, old_error_m=float(np.linalg.norm(seed-truth)),
            new_error_m=float(np.linalg.norm(new-truth)),
            truth_score=float(true_score), returned_score=float(returned_score),
            truth_higher=bool(true_score > returned_score+1e-10),
            deploy_seed_error_m=float(np.linalg.norm(refined-truth)),
            deploy_seed_score=float(refined_score),
            deploy_seed_position=refined.tolist(),
            oracle_local_error_m=float(np.linalg.norm(oracle-truth)),
            oracle_local_score=float(calc.evaluate([oracle])[0]))
        rows.append(row)
        if time.perf_counter()-last > 15:
            print(f'\r评分核验 {index+1}/384', end='', flush=True)
            last = time.perf_counter()
    summary = []
    for snr in (0,3,6):
        for n in (4096,16384):
            for source in (0,1):
                subset = [r for r in rows if r['snr']==snr and r['N']==n and r['source']==source]
                failures = [r for r in subset if r['new_error_m']>100]
                summary.append(dict(snr=snr,N=n,source=source,cases=len(subset),
                    new_failures=len(failures), truth_higher_failures=sum(r['truth_higher'] for r in failures),
                    deploy_seed_success=sum(r['deploy_seed_error_m']<=100 for r in subset),
                    oracle_local_success=sum(r['oracle_local_error_m']<=100 for r in subset)))
    write(out/'rows.json', rows)
    write(out/'report.json', dict(status='COMPLETED', summary=summary, test_access=False))
    verify(sources)
    verify(inputs)
    elapsed = time.perf_counter()-start
    write(out/'ledger.json', dict(seconds_this_run=elapsed,cumulative_seconds=previous+elapsed))
    write(out/'final_audit_report.json',dict(status='PASS',sources=sources,inputs=inputs,
        report=identity(out/'report.json'), regenerated_iq_request_identity='PASS',test_access=False))
    print('\n'+str(out/'report.json'))


if __name__=='__main__':
    main()
