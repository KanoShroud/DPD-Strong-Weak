"""2D restriction of the already checked unknown-gain LOS profile-LS model."""
import importlib.util
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize, least_squares, linear_sum_assignment

BASE_PATH = (Path(__file__).resolve().parents[2] /
             'research_admission_20261025/core_multipath/model.py')
spec = importlib.util.spec_from_file_location('frozen_los_reference', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def xyz(p):
    p = np.atleast_2d(p)
    return np.column_stack([p, np.zeros(len(p))])


def columns(k, m):
    return np.r_[np.array([[3*q, 3*q+1] for q in range(k)]).ravel(),
                 np.arange(3*k, 3*k+2*k*(m-1))]


def expand(x, k, m):
    full = np.zeros(3*k+2*k*(m-1))
    full[columns(k, m)] = x
    return full


class Model2D:
    def __init__(self, y, frequency, stations):
        self.original = base.Model(y, frequency, xyz(stations))
        self.stations = stations
        self.frequency = frequency
        self.y = y

    def single(self, z):
        value, grad = self.original.single(np.r_[z*1000, 0.])
        return value, grad[:2]

    def candidates(self):
        axis = np.arange(-1000, 1001, 20.)
        grid = np.array(np.meshgrid(axis, axis, indexing='ij')).reshape(2, -1).T
        # The spectrum itself is well defined at a station; its derivative is not.
        with np.errstate(invalid='ignore', divide='ignore'):
            values = self.original.grid_scores(xyz(grid))
        selected = []
        for i in np.argsort(-values, kind='stable'):
            if all(np.linalg.norm(grid[i]-p) >= 60 for p in selected):
                selected.append(grid[i])
            if len(selected) == 6:
                break
        refined = []
        for p in selected:
            if np.min(np.linalg.norm(p-self.stations, axis=1)) < 1e-6:
                # Explicit singular-derivative status, not truth-based exclusion.
                continue
            opt = minimize(self.single, p/1000, jac=True, method='L-BFGS-B',
                           bounds=[(-1., 1.)]*2,
                           options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-12, 'maxls': 30})
            point = opt.x*1000
            if not np.isfinite(opt.fun):
                continue
            value, gain = self.original.single(np.r_[point, 0.], True)
            if all(np.linalg.norm(point-r['p']) > 1 for r in refined):
                refined.append({'p': point, 'g': gain, 'score': value, 'status': str(opt.message)})
        return sorted(refined, key=lambda r: -r['score'])

    def fit(self, positions, gains):
        k, m = len(positions), len(self.stations)
        idx = columns(k, m)
        initial = base.pack(xyz(positions), gains)[idx]
        cache = {}
        trace = []

        def compute(z):
            if 'x' not in cache or not np.array_equal(z, cache['x']):
                r, jac = self.original.residual_jac(expand(z, k, m), k)
                jac = jac[:, idx]
                cache.update(x=z.copy(), r=r, jac=jac)
                trace.append([float(r@r), float(np.max(abs(2*jac.T@r)))])
            return cache['r'], cache['jac']

        bounds = [(-1., 1.)]*(2*k) + [(-np.inf, np.inf)]*(2*k*(m-1))
        opt = least_squares(lambda z: compute(z)[0], initial, jac=lambda z: compute(z)[1],
                            bounds=np.array(bounds).T, x_scale='jac', method='trf',
                            max_nfev=300, gtol=1e-7, ftol=1e-12, xtol=1e-12)
        full = expand(opt.x, k, m)
        p, g = base.unpack(full, k, m)
        value, grad = self.original.value_grad(full, k)
        return {'positions': p[:, :2].tolist(), 'loss': value,
                'gradient_inf': float(np.max(abs(grad[idx]))),
                'status': str(opt.message), 'success': bool(opt.success),
                'nfev': opt.nfev, 'trace': trace, 'g_real': g.real.tolist(),
                'g_imag': g.imag.tolist()}


def solve(y, frequency, stations, k):
    started = time.perf_counter()
    model = Model2D(y, frequency, stations)
    candidates = model.candidates()
    if not candidates:
        return {'positions': None, 'status': 'NO_FIRST_CANDIDATE', 'seconds': time.perf_counter()-started}
    first = candidates[0]
    branches = []
    if k == 1:
        branches.append(model.fit(np.array([first['p']]), np.array([first['g']])))
    else:
        full = base.pack(xyz(first['p']), first['g'][None])
        residual = model.original.components(full, 1)[-1]
        second = Model2D(residual.T, frequency, stations).candidates()
        for candidate in second[:3]:
            if np.linalg.norm(first['p']-candidate['p']) > 1:
                branches.append(model.fit(np.array([first['p'], candidate['p']]),
                                          np.array([first['g'], candidate['g']])))
    if not branches:
        return {'positions': None, 'status': 'NO_SECOND_CANDIDATE', 'seconds': time.perf_counter()-started}
    best = min(branches, key=lambda r: r['loss'])
    return {'positions': best['positions'], 'status': best['status'], 'selected': best,
            'branches': branches, 'seconds': time.perf_counter()-started,
            'version': 'LOS_UNKNOWN_GAIN_LS_2D_RESIDUAL3'}


def evaluate(pred, truth):
    if pred is None or np.asarray(pred).shape != np.asarray(truth).shape:
        return {'complete': False, 'errors_m': None, 'rmse_m': None, 'within20': False}
    p = np.array(pred)
    cost = np.sum((p[:, None]-truth[None])**2, axis=-1)
    i, j = linear_sum_assignment(cost)
    errors = np.zeros(len(truth))
    errors[j] = np.sqrt(cost[i, j])
    return {'complete': True, 'errors_m': errors.tolist(),
            'rmse_m': float(np.sqrt(np.mean(errors**2))),
            'within20': bool(np.all(errors <= 20))}
