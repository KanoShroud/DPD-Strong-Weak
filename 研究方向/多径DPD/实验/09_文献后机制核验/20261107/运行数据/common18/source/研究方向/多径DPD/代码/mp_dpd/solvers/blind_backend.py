# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Saved geometry -> conventional LOS/MP DPD; no truth accepted by solve."""
from pathlib import Path
from datetime import datetime
import json
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from 研究方向.多径DPD.代码.mp_dpd.initialization.tdoa_candidates import ROOT, B, sha, write, signature
from 研究方向.多径DPD.代码.mp_dpd.initialization.blind_initialization import rank_one, los_vector, local_los, seed_paths, residual, feasible
from 研究方向.多径DPD.代码.mp_dpd.solvers.boundary_update import run, state

IQ=ROOT/'finite_iq_clues_20261029_204430_528859'
INIT=ROOT/'robust_geometry_20261029_210612_452075'
PROTOCOL=ROOT/'盲后端短测协议.md'


def solve(p,f,y):
    before=signature(p,f,y)
    start=time.perf_counter()
    gains=np.column_stack([rank_one(y,f,position)[0] for position in p])
    v0=los_vector(gains)
    pp,vv,meta=local_los(p,v0,f,y)
    los_time=time.perf_counter()-start
    seed_start=time.perf_counter()
    seeded,seed_meta=seed_paths(pp,vv,f,y)
    seed_time=time.perf_counter()-seed_start
    rr,_=residual(seeded,pp,f,y)
    fit=run(pp,seeded,f,y)
    assert before==signature(p,f,y)
    return dict(initial=p,initial_v=v0,los_p=pp,los_v=vv,los=meta,los_seconds=los_time,
          seeded_v=seeded,seed=seed_meta,seed_seconds=seed_time,seed_cost=.5*float(rr@rr),
          mp=fit,seconds=time.perf_counter()-start,input_signature=before)


def errors(p,truth):
    d=np.linalg.norm(np.asarray(p)[:,None]-truth[None],axis=2)
    i,j=linear_sum_assignment(d*d)
    e=np.zeros(2)
    e[j]=d[i,j]
    return e


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
