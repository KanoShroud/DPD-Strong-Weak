"""Frozen 24-seen-case regression; no held-out realization access."""
import argparse
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import stable

cfg, ad, HERE = stable.cfg, stable.ad, stable.HERE
INPUT = HERE.parent/'inside_pilot_20261104/batch_v2'
DEV = ['IN_100_R00_L2_P075', 'IN_101_R00_L3_P150', 'IN_104_R00_L2_P075', 'IN_103_R00_L0']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['dev', 'paired'], required=True)
    args = parser.parse_args()
    qa = cfg.read(HERE/'correctness_v1.json')
    assert qa['passed']
    for p, digest in qa['source_hashes'].items():
        assert cfg.sha(p) == digest
    stable.inside.rr.check_hashes(cfg.read(INPUT/'identity.json'))
    tags = DEV if args.phase == 'dev' else sorted(p.parent.name for p in (INPUT/'cases').glob('*/evaluation.json'))
    if args.phase == 'paired':
        assert cfg.read(HERE/'dev_v1/summary.json')['numerical_gate_passed'], 'Development gate failed'
    out = HERE/(args.phase+'_v1')
    identity = ad.source_hashes()
    for p in list(HERE.glob('*.py'))+[HERE/'protocol.md', HERE/'correctness_v1.json']:
        identity[str(p.resolve())] = cfg.sha(p)
    if out.exists():
        assert cfg.read(out/'identity.json') == identity
    else:
        out.mkdir()
        cfg.write(out/'identity.json', identity)
        cfg.write(out/'contract.json', dict(tags=tags, variants=['OLD', 'ORTH_SAFE'], threads=2,
                   input=str(INPUT.resolve(strict=True)), args=sys.argv))
    rows = []
    with threadpool_limits(limits=2):
        engines = dict(OLD=stable.inside.Engine(), ORTH_SAFE=stable.Engine())
        for i, tag in enumerate(tags):
            src = INPUT/'cases'/tag
            assert '_R00_' in tag
            for file, digest in cfg.read(src/'receipt.json').items():
                assert cfg.sha(src/file) == digest
            with np.load(src/'iq.npz', allow_pickle=False) as z:
                iq = z['iq']
            assert cfg.digest(iq) == cfg.read(src/'evaluation.json')['iq_sha']
            order = ['OLD', 'ORTH_SAFE'] if i % 2 == 0 else ['ORTH_SAFE', 'OLD']
            for name in order:
                dst = out/(tag+'_'+name+'.json')
                if dst.exists():
                    value = cfg.read(dst)
                else:
                    print('START', tag, name, flush=True)
                    start = time.perf_counter()
                    e = engines[name]
                    try:
                        result = e.mp(iq)
                        value = dict(tag=tag, variant=name, status='returned', result=result)
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        value = dict(tag=tag, variant=name, status='failed', exception=repr(exc),
                            traceback=traceback.format_exc(), seconds=time.perf_counter()-start)
                    value['iq_sha'] = cfg.digest(iq)
                    cfg.write(dst, value)  # Decision before truth evaluation.
                erow = cfg.read(src/'evaluation.json')
                with np.load(INPUT/'banks'/f"{erow['layout']:02d}_00.npz", allow_pickle=False) as z:
                    truth = z['p']
                row = dict(tag=tag, variant=name, status=value['status'], feasible=False,
                           rmse=None, numerical_consistent=False, normal=False, stop=value.get('exception'))
                if value['status'] == 'returned':
                    result = value['result']; final = result['final']
                    x = np.array(final['x']); e = engines[name]
                    f, y, n, _ = ad.b.fc.cal.spectral(iq)
                    r, j = e.evaluate(x, f, y, result['paths'])
                    cost, g = e.objective(x, f, y, result['paths'])
                    pg = float(np.max(abs(x-e.ref.project(x-j.T@r, n))))
                    err = stable.inside.rr.evaluation(final['p'], truth)
                    row.update(rmse=float(np.sqrt(np.mean(np.square(err)))), errors=err,
                        normal=result['normal'], stop=result['stop'], seconds=result['seconds'],
                        cost=cost, pg=pg, K=len(result['paths']),
                        feasible=bool(np.isfinite(x).all() and np.linalg.norm(x-e.ref.project(x, n)) < 1e-7),
                        numerical_consistent=bool(np.allclose(g, j.T@r, atol=1e-9, rtol=1e-5) and np.isclose(cost, final['cost'], rtol=1e-9, atol=1e-10)),
                        gradient_difference=float(np.linalg.norm(g-j.T@r)),
                        rejected=result.get('invalid_trial_rejections', 0))
                else:
                    row['seconds'] = value['seconds']
                rows.append(row)
                cfg.write(out/'rows.json', rows)
                print('END', tag, name, row['rmse'], row['stop'], row['seconds'], flush=True)
    repaired = [r for r in rows if r['variant'] == 'ORTH_SAFE']
    gate = all(r['status'] == 'returned' and r['feasible'] and r['numerical_consistent'] for r in repaired)
    stable.inside.rr.check_hashes(identity)
    cfg.write(out/'summary.json', dict(completed=len(rows), numerical_gate_passed=gate,
                                     rows=rows, unchanged=True))
    print('NUMERICAL GATE', gate, flush=True)


if __name__ == '__main__':
    main()
