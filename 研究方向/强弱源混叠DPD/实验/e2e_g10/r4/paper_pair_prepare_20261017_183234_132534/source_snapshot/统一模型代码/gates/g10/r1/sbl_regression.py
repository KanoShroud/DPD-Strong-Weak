"""Finite SBL-only regression; no full benchmark and no truth-based selection."""
import argparse
from .config import DEFAULT
from .runtime import fresh, preflight, finish_audit, Ledger, write
from .runner import record
from .test_r1 import suite


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['singles', 'mixed', 'repeat'], required=True)
    p.add_argument('--active-atoms', type=int, default=8)
    p.add_argument('--iterations', type=int, default=300)
    args = p.parse_args()
    cfg = dict(DEFAULT, sbl_update='fp_half', sbl_active_atoms=args.active_atoms,
               sbl_coordinate_scale=1000., sbl_coordinate_iterations=20,
               sbl_progress=True, sbl_iterations=args.iterations)
    if args.stage == 'singles':
        tasks = [((g, 0, 0, 'weak'), clean) for g in (0, 1, 2) for clean in (True, False)]
    elif args.stage == 'mixed':
        tasks = [((g, 0, gap, 'mixed'), False) for g in (0, 1, 2, 3) for gap in (0, 15)]
    else:
        tasks = [((g, 1, 0, 'weak'), False) for g in (0, 3)]
    out = fresh('sbl_regression_'+args.stage)
    preflight(out, dict(stage='SBL_FINITE_REGRESSION', arguments=vars(args), config=cfg,
                        tasks=tasks, no_formal_release=True))
    tests = suite(); write(out/'numerical_tests.json', tests)
    if not tests['passed']:
        raise RuntimeError('Numerical check failed')
    clock = Ledger(out); rows = []
    print('OUT='+str(out), flush=True)
    for i, (task, clean) in enumerate(tasks):
        print(f'{i+1}/{len(tasks)} SBL {task} clean={clean}', flush=True)
        r = record(out, clock, task, 'SBL', cfg, clean, suffix='_clean' if clean else '')
        item = dict(tag=r['tag'], errors_m=r['metrics']['errors_m'],
                    success20=all(r['metrics']['success']['20']),
                    seconds=r['result']['seconds'], stop=r['result']['stop_reason'],
                    iterations=r['result']['counts']['iterations'])
        rows.append(item); write(out/'summary.json', dict(rows=rows, formal_ready=False))
        print(item, flush=True)
    clock.flush(); finish_audit(out)
    print('DONE '+str(out), flush=True)


if __name__ == '__main__':
    main()
