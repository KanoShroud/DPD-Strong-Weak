"""Four prespecified regression cases; extend only after reviewing this gate."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
import projected_fallback as pf
import pair

stable = pf.stable
cfg = stable.cfg


def main():
    out = stable.HERE/'projected_dev_v2'
    out.mkdir(exist_ok=False)
    assert cfg.read(stable.HERE/'paired_v1/summary.json')['numerical_gate_passed']
    identity = stable.ad.source_hashes()
    cfg.write(out/'identity.json', identity)
    engine = pf.Engine()
    rows = []
    for tag in pair.DEV:
        src = pair.INPUT/'cases'/tag
        for name, digest in cfg.read(src/'receipt.json').items():
            assert cfg.sha(src/name) == digest
        with np.load(src/'iq.npz', allow_pickle=False) as z:
            iq = z['iq']
        print('START', tag, flush=True)
        begin = time.perf_counter()
        result = engine.mp(iq)
        cfg.write(out/(tag+'.json'), result)
        ev = cfg.read(src/'evaluation.json')
        with np.load(pair.INPUT/'banks'/f"{ev['layout']:02d}_00.npz") as z:
            truth = z['p']
        err = stable.inside.rr.evaluation(result['final']['p'], truth)
        fallback = [h['fit'] for h in result['history'] if 'projected_fallback_history' in h.get('fit', {})]
        row = dict(tag=tag, rmse=float(np.sqrt(np.mean(np.square(err)))), normal=result['normal'],
                   stop=result['stop'], seconds=time.perf_counter()-begin, K=len(result['paths']),
                   fallback_count=len(fallback), pg=result['final']['projected_gradient'],
                   all_within_original_iterations=all(x['iterations'] <= 300 for x in fallback),
                   all_monotone=all(h['cost_after'] < h['cost_before'] for x in fallback for h in x['projected_fallback_history']))
        rows.append(row)
        cfg.write(out/'rows.json', rows)
        print(row, flush=True)
    stable.inside.rr.check_hashes(identity)
    cfg.write(out/'summary.json', dict(rows=rows, completed=4, unchanged=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
