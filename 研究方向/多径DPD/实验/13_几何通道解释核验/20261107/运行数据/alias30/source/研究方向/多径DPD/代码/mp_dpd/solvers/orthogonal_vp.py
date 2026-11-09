# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Same-model orthogonal VP evaluation and invalid-trial rejection."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
pass  # Explicit package imports replace historical search-path mutation.
import 研究方向.多径DPD.代码.mp_dpd.models.inside_scene as inside  # noqa: E402

cfg, ad = inside.cfg, inside.ad


def guarded_objective(x, f, y, paths):
    try:
        return raw_objective(x, f, y, paths)  # noqa: F821 -- per-instance binding
    except ValueError as exc:
        if str(exc) != 'station coincidence':
            raise
        invalid_trials.append(dict(x=np.asarray(x).copy(), paths=list(paths)))  # noqa: F821
        return np.inf, np.zeros_like(x)


class Engine(inside.Engine):
    def __init__(self, reject_invalid=True):
        super().__init__()
        self.invalid_trials = []
        self.basis = None

        def linear_projection(h, y):
            # A common per-frequency scale preserves the original rank criterion.
            scale = np.max(abs(h), axis=(1, 2))
            if not np.all(scale > 0):
                raise ValueError('zero channel matrix')
            u, s, vh = np.linalg.svd(h/scale[:, None, None], full_matrices=False)
            keep = s > 1e-12*s[:, :1]
            reciprocal = np.divide(1., s, out=np.zeros_like(s), where=keep)
            inv = ((vh.conj().transpose(0, 2, 1)*reciprocal[:, None]) @ u.conj().transpose(0, 2, 1))/scale[:, None, None]
            waveform = (inv@y[:, :, None])[:, :, 0]
            basis = u*keep[:, None]
            residual = y-(basis@(basis.conj().transpose(0, 2, 1)@y[:, :, None]))[:, :, 0]
            self.basis = (h, basis, keep)
            return inv, waveform, residual

        def complement(h, ds):
            assert self.basis[0] is h
            u = self.basis[1]
            return ds-u@(u.conj().transpose(0, 2, 1)@ds)

        self.linear_projection = linear_projection
        self.core = ad.clone(self.core, [(
            'inv=np.linalg.pinv(h,rcond=1e-12)\n    s=(inv@y[:,:,None])[:,:,0]\n    r=y-(h@s[:,:,None])[:,:,0]',
            'inv,s,r=linear_projection(h,y)')], {'linear_projection': linear_projection})
        self.evaluate.__globals__['core'] = self.core
        changed = ad.clone(self.evaluate, [('jp=ds-h@(inv@ds)', 'jp=complement(h,ds)')], {'complement': complement})
        # Preserve existing bound function identities throughout the frozen import graph.
        self.evaluate.__code__ = changed.__code__
        self.evaluate.__globals__['complement'] = complement
        self.evaluate.generated_source = changed.generated_source
        self.objective.__globals__['core'] = self.core
        self.raw_objective = ad.clone(self.objective)
        if reject_invalid:
            self.objective.__code__ = guarded_objective.__code__
            self.objective.__globals__.update(raw_objective=self.raw_objective, invalid_trials=self.invalid_trials)

    def mp(self, iq):
        self.invalid_trials.clear()
        result = super().mp(iq)
        result['invalid_trial_rejections'] = len(self.invalid_trials)
        result['numerical_variant'] = 'ORTHOGONAL_VP_ORIGINAL_RANK_AND_PG'
        return result
