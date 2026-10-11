"""G9同IQ三组衔接诊断；保持既有评分/搜索源码与历史结果不变。"""
import argparse
import hashlib
import time
import traceback

import numpy as np
import torch

from ..g8.physics import Spectrum, search
from .metrics import evaluate
from .runner import key_hash
from .runtime import BASE, Guard, identity, preflight, read, source_manifest, verify, write
from .scenes import make
from .solver import Score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    prior = (BASE/'prepare_20261010_01').resolve(strict=True)
    audit = read(prior/'final_audit_report.json')
    verify(audit['sources'])
    verify([audit['report'], audit['summary']])
    previous = read(prior/'ledger.json')['charged_seconds']
    for ledger in BASE.glob('bridge_*/ledger.json'):
        previous += read(ledger)['seconds_this_run']
    out = preflight(BASE/args.run_id)
    sources = source_manifest()
    guard = Guard(8*3600, previous)
    rows = []
    inputs = []
    started = time.perf_counter()
    write(out/'contract.json', dict(scope='同IQ单源三组对照，无训练，无check/test',
        short=args.short, sources=sources, prior_audit=identity(prior/'final_audit_report.json'),
        previous_seconds=previous, methods=['old_old', 'new_old', 'new_new_reused'],
        primary_radius_m=100, snr=[0, 3, 6], role='dev', compare_time='复用A耗时为历史实测，仅描述，不作严格加速判定'))
    try:
        paths = sorted((prior/'rows').glob('qual_*.json'))
        if args.short:
            paths = paths[:2]
        last_log = 0.
        cache_key, scene = None, None
        for index, path in enumerate(paths):
            guard()
            old = read(path)
            seal = old.pop('row_sha256')
            if key_hash(old) != seal:
                raise RuntimeError('历史结果摘要不一致：'+str(path))
            group, overlap, n, source, snr = (old[k] for k in ('group', 'overlap', 'N', 'source', 'snr'))
            key = (group, overlap, snr)
            if cache_key != key:
                scene = make('dev', group, 0, overlap, snr, 12)
                cache_key = key
            iq = scene['signal'][source, :, :n]+scene['noise'][:, :n]
            truth = scene['positions'][source:source+1]
            metadata = dict(group=group, overlap=overlap, N=n, source=source, snr=snr)
            iq_hash = hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
            request = dict(tag=path.stem, method='A', budget=20., metadata=metadata,
                           hr=[2, 1e-4], iq_sha256=iq_hash)
            if key_hash(request) != old['request_hash']:
                raise RuntimeError('重新生成IQ不匹配历史请求：'+str(path))
            inputs.append(identity(path))
            reused = dict(**metadata, method='new_new_reused', metrics=old['metrics'],
                          seconds=old['seconds'], iq_sha256=iq_hash)
            result = [reused]
            for method in ('old_old', 'new_old'):
                torch.cuda.synchronize()
                start = time.perf_counter()
                calc = Spectrum(iq, guard=guard) if method == 'old_old' else Score(iq, guard)
                answer = search(calc, 1)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                result.append(dict(**metadata, method=method, seconds=elapsed,
                    positions=answer['positions'], metrics=evaluate(truth, answer['positions']),
                    iq_sha256=iq_hash))
            rows.extend(result)
            write(out/'rows'/path.name, result)
            elapsed = time.perf_counter()-started
            write(out/'ledger.json', dict(seconds_this_run=elapsed, cumulative_seconds=previous+elapsed))
            if elapsed-last_log >= 15 or index == len(paths)-1:
                print(f'\r配对核验 {index+1}/{len(paths)} | {elapsed:.1f}s', end='', flush=True)
                last_log = elapsed
        summary = []
        for snr in (0, 3, 6):
            for n in (4096, 16384):
                for source in (0, 1):
                    for method in ('old_old', 'new_old', 'new_new_reused'):
                        selected = [r for r in rows if r['snr'] == snr and r['N'] == n
                                    and r['source'] == source and r['method'] == method]
                        if not selected:
                            continue
                        errors = [e for r in selected for e in r['metrics']['errors_m'] if e is not None]
                        summary.append(dict(snr=snr,N=n,source=source,method=method,cases=len(selected),
                            success=sum(all(r['metrics']['success']['100']) for r in selected),
                            rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                            median_error_m=float(np.median(errors)) if errors else None,
                            mean_seconds=float(np.mean([r['seconds'] for r in selected]))))
        write(out/'report.json', dict(status='SMOKE_COMPLETED' if args.short else 'COMPLETED',
              scope='G9单源同IQ衔接，不是混合场景正式比较', records=len(paths), summary=summary,
              seconds_this_run=time.perf_counter()-started, test_access=False))
        verify(sources)
        verify(inputs)
        write(out/'final_audit_report.json', dict(status='PASS', input_records=len(inputs),
              regenerated_iq_request_identity='PASS', row_seals='PASS', sources=sources, inputs=inputs,
              report=identity(out/'report.json'), test_access=False, training=False))
        print('\n完成：'+str(out/'report.json'))
    except Exception:
        elapsed = time.perf_counter()-started
        write(out/'ledger.json', dict(seconds_this_run=elapsed, cumulative_seconds=previous+elapsed))
        write(out/'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
