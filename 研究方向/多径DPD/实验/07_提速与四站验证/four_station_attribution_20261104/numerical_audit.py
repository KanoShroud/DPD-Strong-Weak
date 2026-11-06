"""Fixed-state algebra checks only; no estimator changes or optimization."""
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'four_station_physical_20261104'))
import scene as cfg  # noqa: E402
import adapters as ad  # noqa: E402


def main():
    pilot = cfg.HERE/'pilot_20261104_182457_012169'
    e = ad.Engine()

    def orthogonal_core(x, f, y, paths):
        d = list(e.core(x, f, y, paths))
        h = d[0]; norms = np.linalg.norm(h, axis=1)
        u, singular, vh = np.linalg.svd(h/norms[:, None], full_matrices=False)
        assert (singular[:, -1]/singular[:, 0] > 1e-12).all()
        inv = ((vh.conj().transpose(0, 2, 1)/singular[:, None]) @ u.conj().transpose(0, 2, 1))/norms[:, :, None]
        s = (inv@y[:, :, None])[:, :, 0]
        r = y-(u@(u.conj().transpose(0, 2, 1)@y[:, :, None]))[:, :, 0]
        d[1:4] = [inv, s, r]
        return tuple(d)

    ev = ad.clone(e.evaluate, bindings={'core': orthogonal_core})
    ob = ad.clone(e.objective, bindings={'core': orthogonal_core})
    records = []
    for tag in ['IN_00_R00_L0', 'IN_00_R00_L1_P075', 'MIX_02_R00_L2_P100', 'OUT_01_R00_L3_P125']:
        path = pilot/'cases'/tag
        result = cfg.read(path/'MP_decision.json')['result']
        with np.load(path/'iq.npz') as z:
            iq = z['iq']
        f, y, n, _ = ad.b.fc.cal.spectral(iq)
        x = np.array(result['final']['x']); paths = result['paths']
        oldr, oldj = e.evaluate(x, f, y, paths)
        newr, newj = ev(x, f, y, paths)
        oldcost, oldg = e.objective(x, f, y, paths)
        newcost, newg = ob(x, f, y, paths)
        h = e.core(x, f, y, paths)[0]
        u = np.linalg.svd(h/np.linalg.norm(h, axis=1)[:, None], full_matrices=False)[0]
        oldres = e.core(x, f, y, paths)[3]
        newres = orthogonal_core(x, f, y, paths)[3]
        orth = lambda r: float(np.linalg.norm(u.conj().transpose(0, 2, 1)@r[:, :, None])/np.linalg.norm(y))
        pg = lambda g: float(np.max(abs(x-e.ref.project(x-g, n))))
        records.append(dict(tag=tag, old_cost=oldcost, orthogonal_cost=newcost,
            old_gradient_disagreement=float(np.linalg.norm(oldj.T@oldr-oldg)),
            orthogonal_gradient_disagreement=float(np.linalg.norm(newj.T@newr-newg)),
            old_pg_jtr=pg(oldj.T@oldr), old_pg_envelope=pg(oldg),
            orthogonal_pg_jtr=pg(newj.T@newr), orthogonal_pg_envelope=pg(newg),
            old_residual_nonorthogonality=orth(oldres), orthogonal_residual_nonorthogonality=orth(newres),
            maximum_gain=float(np.max(abs(x[4:e.D])))))
    cfg.write(HERE/'fixed_state_numerics.json', dict(source_sha=cfg.sha(__file__),
        purpose='Algebraically equivalent fixed-state check only; not a modified estimator or performance result', rows=records))
    for r in records:
        print(r)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
