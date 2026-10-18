"""Read-only SBL state observation, compact recovery and paired output replay."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from ..runtime import safe
from ..physics import shortlist, atom
from .runtime import read, write, sha
from .output_check import near_duplicate_groups

CHECKPOINTS = (25, 50, 100, 200, 300)
VARIANTS = ('SBL', 'SBL_GROUP_GAMMA', 'SBL_GROUP_COHERENT', 'SBL_POINT_POWER_AUX')
STAT_FIELDS = ('em', 'power', 'numerator', 'denominator')


def save_arrays(path, arrays):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    with pending.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def load_state(folder, entry):
    path = (Path(folder)/entry['path']).resolve(strict=True)
    if not path.is_relative_to(Path(folder).resolve()):
        raise ValueError('State path escaped its sample directory')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry['sha256']:
        raise RuntimeError('SBL state identity changed: '+str(path))
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files if k != 'metadata'}
        meta = json.loads(str(data['metadata'].item()))
    return arrays, meta


class StateStore:
    """Observer only: no truth, no changes to solver updates or stopping."""
    def __init__(self, folder, identity, checkpoints=CHECKPOINTS):
        self.folder = safe(folder)
        self.identity = identity
        self.checkpoints = tuple(checkpoints)
        self.path = self.folder/'manifest.json'
        self.manifest = read(self.path) if self.path.exists() else dict(
            version='SBL_SHARED_STATE_V1', identity=identity,
            checkpoints=list(checkpoints), states={}, roles={})
        if self.manifest['identity'] != identity or self.manifest['checkpoints'] != list(checkpoints):
            raise RuntimeError('SBL state belongs to another input/configuration')
        self.best = None
        self.was_resumed = False

    @contextmanager
    def recording(self, t):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            t.recording_seconds += time.perf_counter()-start

    def _payload(self, m, t, pp, gamma, stats, loop):
        arrays = dict(points=pp.copy(), gamma=gamma.detach().cpu().numpy().copy(),
                      covariance=torch.cat(stats['covs']).detach().cpu().numpy().copy(),
                      spectral_weight=m.sbl_spectral_weight.cpu().numpy().copy())
        arrays.update({k: stats[k].detach().cpu().numpy().copy() for k in STAT_FIELDS})
        meta = dict(loop=loop, evidence=float(stats['evidence']), sample_rate_hz=m.sample_rate_hz,
                    n=m.n, noise_variance=1/m.n, counts=dict(m.counts),
                    local_optimizers=list(getattr(m, 'local_optimizers', [])),
                    trace=dict(algorithm_seconds=t.elapsed(), best=t.best, events=list(t.events),
                               history=list(t.history), stages=dict(t.stages), reason=t.reason),
                    # Counts and algorithm clock do not include observer copies / writes.
                    recording_seconds=t.recording_seconds)
        return arrays, meta

    def _save(self, payload):
        arrays, meta = payload
        key = f"iter_{meta['loop']['iteration']:04d}"
        path = self.folder/(key+'.npz')
        if key not in self.manifest['states']:
            save_arrays(path, dict(arrays, metadata=np.asarray(json.dumps(meta, ensure_ascii=False))))
            self.manifest['states'][key] = dict(path=path.name, sha256=sha(path),
                                                iteration=meta['loop']['iteration'])
        return key

    def observe(self, m, t, pp, gamma, stats, loop, improved, final=False):
        with self.recording(t):
            # Take the algorithm timestamp before observer work, not midway through it.
            stamp = t.elapsed()
            payload = None
            if improved or final or loop['iteration'] in self.checkpoints:
                payload = self._payload(m, t, pp, gamma, stats, loop)
                payload[1]['trace']['algorithm_seconds'] = stamp
            if improved:
                self.best = payload
            if final or loop['iteration'] in self.checkpoints:
                key = self._save(payload)
                if loop['iteration'] in self.checkpoints:
                    self.manifest['roles'][str(loop['iteration'])] = key
                best_key = self._save(self.best)
                self.manifest['resume'] = dict(current=key, best=best_key, finished=final)
                if final:
                    self.manifest['roles']['terminal'] = key
                    self.manifest['roles']['selected'] = best_key
                write(self.path, self.manifest)

    def restore(self, m, t):
        checkpoint = self.manifest.get('resume')
        if checkpoint is None:
            return None
        arrays, meta = load_state(self.folder, self.manifest['states'][checkpoint['current']])
        self.best = load_state(self.folder, self.manifest['states'][checkpoint['best']])
        self.was_resumed = True
        t.resume_setup_seconds = t.elapsed()
        trace = meta['trace']
        t.prefix_seconds = trace['algorithm_seconds']
        t.start = time.perf_counter()
        t.recording_seconds = 0.
        t.best, t.events, t.history, t.stages = trace['best'], trace['events'], trace['history'], trace['stages']
        t.reason = trace['reason']
        m.counts = meta['counts']
        m.local_optimizers = meta['local_optimizers']
        m.sbl_spectral_weight = torch.tensor(arrays['spectral_weight'], device=m.device)
        stats = {k: torch.tensor(arrays[k], device=m.device) for k in STAT_FIELDS}
        stats.update(evidence=meta['evidence'], covs=[torch.tensor(arrays['covariance'], device=m.device)])
        return arrays['points'], torch.tensor(arrays['gamma'], device=m.device), stats, meta['loop'], checkpoint['finished']


def decode_state(m, arrays, count, tolerance=.05):
    """All outputs from one state. Only group members need posterior reconstruction."""
    def sync():
        if str(m.device).startswith('cuda'):
            torch.cuda.synchronize()
    pp, gamma = arrays['points'], arrays['gamma']
    power = arrays['power']
    outputs, timings = {}, {}
    sync(); start = time.perf_counter()
    take = shortlist(pp, gamma, count)
    outputs['SBL'] = pp[take].tolist()
    timings['SBL'] = time.perf_counter()-start
    start = time.perf_counter()
    groups = near_duplicate_groups(pp, tolerance)
    centers = np.asarray([pp[g[0]].copy() if len(g) == 1 else np.average(pp[g], axis=0, weights=gamma[g]) for g in groups])
    group_gamma = np.asarray([gamma[g].sum() for g in groups])
    grouping_seconds = time.perf_counter()-start
    start = time.perf_counter()
    outputs['SBL_GROUP_GAMMA'] = centers[shortlist(centers, group_gamma, count)].tolist()
    timings['SBL_GROUP_GAMMA'] = grouping_seconds+time.perf_counter()-start
    sync(); start = time.perf_counter()
    coherent = np.asarray([power[g].sum() for g in groups])
    multi = [(i, g) for i, g in enumerate(groups) if len(g) > 1]
    if multi:
        ids = np.concatenate([g for _, g in multi])
        offsets = np.cumsum([0]+[len(g) for _, g in multi])
        covariance = torch.as_tensor(arrays['covariance'], device=m.device)
        gg = torch.as_tensor(gamma[ids], device=m.device)
        w = torch.as_tensor(arrays['spectral_weight'], device=m.device)
        sums = np.zeros(len(multi))
        for lo in range(0, m.n, 128):
            a = atom(torch.as_tensor(pp[ids], device=m.device), m.f[lo:lo+128], m.stations)
            a = a*w[lo:lo+128].sqrt()[:, None, None]
            chol = torch.linalg.cholesky(covariance[lo:lo+128])
            invx = torch.cholesky_solve(m.x[lo:lo+128, :, None], chol).squeeze(-1)
            mu = gg[None]*(a.conj()*invx[:, :, None]).sum(1)
            for q, (_, g) in enumerate(multi):
                sl = slice(offsets[q], offsets[q+1])
                reconstructed = (a[:, :, sl]*mu[:, None, sl]).sum(-1)
                sums[q] += float(abs(reconstructed).square().sum().cpu())/m.n
        for (i, _), value in zip(multi, sums):
            coherent[i] = value
    outputs['SBL_GROUP_COHERENT'] = centers[shortlist(centers, coherent, count)].tolist()
    sync()
    timings['SBL_GROUP_COHERENT'] = grouping_seconds+time.perf_counter()-start
    start = time.perf_counter()
    outputs['SBL_POINT_POWER_AUX'] = pp[shortlist(pp, power, count)].tolist()
    timings['SBL_POINT_POWER_AUX'] = time.perf_counter()-start
    return dict(outputs=outputs, output_seconds=timings,
                group_count=len(groups), duplicate_groups=len(multi),
                duplicate_atoms=sum(len(g) for _, g in multi),
                grouped_points=centers.tolist(), group_gamma=group_gamma.tolist(),
                group_coherent=coherent.tolist(), grouping_seconds=grouping_seconds,
                extra_covariance_decompositions=m.n if multi else 0,
                extra_frequency_response_points=m.n*sum(len(g) for _, g in multi))


def replay(folder, iq, count):
    from .physics import Model
    manifest = read(Path(folder)/'manifest.json')
    m = Model(iq, sample_rate_hz=manifest['identity']['config']['sample_rate_hz'])
    states = {}
    for role, key in manifest['roles'].items():
        if key not in states:
            arrays, meta = load_state(folder, manifest['states'][key])
            states[key] = dict(decode_state(m, arrays, count), iteration=meta['loop']['iteration'],
                               solver_seconds=meta['trace']['algorithm_seconds'],
                               native_objective=meta['evidence'], state_sha=manifest['states'][key]['sha256'])
            for field in ('grouped_points', 'group_gamma', 'group_coherent'):
                states[key].pop(field)
    # Filling later budgets carries the terminal solution; it never invents iterations.
    roles = dict(manifest['roles'])
    for iteration in CHECKPOINTS:
        roles.setdefault(str(iteration), roles['terminal'])
    return dict(version=manifest['version'], roles=roles, states=states,
                manifest_sha=sha(Path(folder)/'manifest.json'))
