"""Fixed-input engineering probe; no research batch is launched."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))


def run(folder, output):
    opened = set()

    def audit(event, args):
        if event == 'open' and isinstance(args[0], str):
            p = Path(args[0]).resolve()
            if p.suffix == '.pyc' and p.parent.name == '__pycache__':
                p = p.parent.parent/(p.name.split('.')[0]+'.py')
            if p.suffix == '.py' and p.is_relative_to(ROOT):
                opened.add(str(p))

    sys.addaudithook(audit)
    if str(folder) == 'api':
        from 研究方向.多径DPD.代码.mp_dpd.api import runtime
        module = runtime()
    else:
        sys.path.insert(0, str(folder))
        spec = importlib.util.spec_from_file_location('migration_stable', folder/'stable.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    engine = module.Engine()
    cfg, ad = module.cfg, module.ad
    rng = np.random.default_rng(20261106)
    freq = np.linspace(-cfg.B/2, cfg.B/2, 41)
    y = rng.normal(size=(41, 4))+1j*rng.normal(size=(41, 4))
    x = np.r_[[-1.1, -.8, 1.2, 1.5], np.ones(6), np.zeros(6)]
    data = {}
    for n in (0, 1, 2):
        paths = [(i, i % 2) for i in range(n)]
        v = np.r_[x, np.tile([.2, .1, 5.], n)]
        loss, grad = engine.objective(v, freq, y, paths)
        data[f'objective_{n}'] = np.r_[loss, grad]
        data[f'project_{n}'] = engine.ref.project(v, cfg.N)
    layout = module.inside.layouts()[0]
    bank = cfg.make_bank(layout, 'pilot', 0, ad.wave.synthesize, ad.wave.rrc)
    iq, powers = cfg.observation(bank, cfg.CONDITIONS[2])
    data['iq'] = iq
    data['powers'] = powers
    data['layouts'] = np.array([v['p'] for v in module.inside.layouts()])
    lags = engine.predicted(np.array([[1000., 1700.]]))[0][0]
    data['triples'] = engine.triples(np.tile(lags[:, None], (1, 6)))
    # Exercise the entire observation-only initialization, LOS, MP, and Flood.
    bank0 = cfg.observation(bank, cfg.CONDITIONS[0])[0]
    solved = engine.mp(bank0)
    flood = engine.flood_solve(bank0)
    print('MP fields', list(solved), 'Flood', list(flood), flush=True)
    data['mp_p'] = np.asarray(solved['final']['x'][:4]).reshape(2, 2)*engine.ref.UNIT
    data['flood_p'] = np.asarray(flood['p'])
    data['los_p'] = np.asarray(solved['los']['p'])
    for label in ('L1_P075', 'L2_P075'):
        condition = next(v for v in cfg.CONDITIONS if v['name'] == label)
        obs = cfg.observation(bank, condition)[0]
        result = engine.mp(obs)
        data[label+'_p'] = np.asarray(result['final']['p'])
        data[label+'_x'] = np.asarray(result['final']['x'])
        data[label+'_paths'] = np.asarray(result['paths'])
        print(label, result['stop'], 'paths', len(result['paths']), flush=True)
    f, yy, n, _ = ad.b.fc.cal.spectral(bank0)
    points = rng.uniform(-3000, 3000, (128, 2))
    kernel = ad.kernel_module.CrossSpectrumKernel(yy.T, f, n, cfg.FS)
    data['xia_direct'] = ad.xia.scores(points, cfg.STATIONS, yy.T, f, n)
    data['xia_kernel'] = kernel.scores(points, cfg.STATIONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data)
    sources = sorted(opened | {str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                      if getattr(m, '__file__', None)
                      and Path(m.__file__).resolve().is_relative_to(ROOT)
                      and str(m.__file__).endswith('.py')})
    output.with_suffix('.json').write_text(json.dumps(dict(sources=sources,
        mp_status={k: v for k, v in solved.items() if isinstance(v, (str, bool, int, float))},
        flood_status={k: v for k, v in flood.items() if isinstance(v, (str, bool, int, float))}),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('SOURCES', len(sources), 'OUTPUT', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.folder, args.output)
