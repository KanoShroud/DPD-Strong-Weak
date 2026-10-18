"""Native solver diagnostic on approved geometry; explicit engineering/performance labels."""
import argparse
import json
from .config import METHODS, DEFAULT
from .runtime import fresh, preflight, write, save_scene, load_scene, Ledger, finish_audit
from .scenes import make, observation, digest
from .methods import run, at_time
from ..metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--groups', type=int, nargs='+', default=[0])
    parser.add_argument('--kind', default='weak', choices=['weak', 'strong', 'mixed', 'clean_weak'])
    parser.add_argument('--gap', type=float, default=15.)
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--update', default='fp_half', choices=['em', 'fp_half'])
    parser.add_argument('--sbl-iterations', type=int, default=DEFAULT['sbl_iterations'])
    parser.add_argument('--weight', default='uniform', choices=['uniform', 'excess_power'])
    parser.add_argument('--segments', type=int, default=8)
    parser.add_argument('--coordinate-scale', type=float, default=DEFAULT['sbl_coordinate_scale'])
    parser.add_argument('--active-atoms', type=int, default=DEFAULT['sbl_active_atoms'])
    parser.add_argument('--coordinate-iterations', type=int, default=DEFAULT['sbl_coordinate_iterations'])
    parser.add_argument('--spectral-model', choices=['flat', 'observed'], default='flat')
    parser.add_argument('--fft-chunk', type=int, default=DEFAULT['fft_chunk'])
    parser.add_argument('--no-dictionary-cache', action='store_true')
    parser.add_argument('--dictionary-cache', action='store_true', help='Opt in to exact full dictionary cache')
    args = parser.parse_args()
    cfg = dict(DEFAULT, sbl_update=args.update, sbl_iterations=args.sbl_iterations,
               music_weight=args.weight, segments=args.segments, sbl_coordinate_scale=args.coordinate_scale,
               sbl_active_atoms=args.active_atoms, sbl_coordinate_iterations=args.coordinate_iterations,
               sbl_spectral_model=args.spectral_model, sbl_progress=True,
               fft_chunk=args.fft_chunk, dictionary_cache=args.dictionary_cache and not args.no_dictionary_cache)
    out = fresh('native_probe')
    preflight(out, dict(stage='R1_NATIVE_DIAGNOSTIC', config=cfg, arguments=vars(args),
                        note='No formal claims; no per-sample time cap'))
    clock = Ledger(out)
    print('OUT='+str(out), flush=True)
    for group in args.groups:
        key = f'{group}_{args.realization}_{args.gap:g}'
        save_scene(out, key, make(group, args.realization, args.gap))
        scene = load_scene(out, key)
        kind = 'weak' if args.kind == 'clean_weak' else args.kind
        iq, truth, count = observation(scene, kind)
        if args.kind == 'clean_weak':
            iq = scene['signal'][1].copy()
        for method in args.methods:
            clock.flush(dict(group=group, method=method, kind=args.kind))
            print(f'START {group} {method} {args.kind}', flush=True)
            result = run(iq, method, count, cfg, seed=102+group, guard=clock.check)
            maps = result.pop('native_map')
            if maps is not None:
                write(out/'maps'/f'{group}_{method}.json', maps)
            row = dict(scene=scene['meta'], method=method, kind=args.kind,
                       input_sha=digest(iq), truth=truth.tolist(), result=result,
                       metrics=evaluate(truth, result['positions']),
                       at_times={str(t): evaluate(truth, at_time(result, t)) for t in (.5,1,2,3,5)})
            write(out/'rows'/f'{group}_{method}.json', row)
            clock.flush()
            print(json.dumps(dict(method=method, group=group, seconds=result['seconds'],
                                  stop=result['stop_reason'], iterations=result['counts']['iterations'],
                                  errors=row['metrics']['errors_m']), ensure_ascii=False), flush=True)
    finish_audit(out)
    print('DONE '+str(out), flush=True)


if __name__ == '__main__':
    main()
