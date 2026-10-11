"""修复共同初始化：旧DPD只提候选，新投影目标决定更新与最终解。"""
import time

import numpy as np
import torch
from scipy.optimize import minimize

from ..g8.physics import Spectrum, search, peaks
from .solver import Score, Deadline, candidates, far, local


def run(iq, method, budget, count=2, guard=lambda: None, hr_config=(2, 1e-4)):
    torch.cuda.synchronize()
    started = time.perf_counter()
    events, updates = [], 0
    score_points, response_points, proposal_points = 0, 0, 0
    calc = None

    def check():
        guard()
        if time.perf_counter()-started >= budget:
            raise Deadline()

    def save(positions):
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if elapsed <= budget:
            events.append(dict(seconds=elapsed, positions=np.asarray(positions).tolist()))

    class CountedSpectrum(Spectrum):
        def evaluate(self, points, *args, **kwargs):
            nonlocal proposal_points
            value = super().evaluate(points, *args, **kwargs)
            proposal_points += len(points)
            return value

    try:
        proposal = CountedSpectrum(iq, method='hr' if method == 'HR' else 'dpd',
            segments=hr_config[0], loading=hr_config[1], guard=check)
        found = search(proposal, count)
        if method in ('DPD', 'HR'):
            positions = np.asarray(found['positions'])
            save(positions)
            # 保留旧粗细搜索的有效解，再尝试原生评分局部优化；不接受下降。
            for k in range(count):
                initial = positions[k].copy()
                others = np.delete(positions,k,axis=0)
                scale = max(float(proposal.evaluate(initial[None])[0]),1e-30)
                def objective(p):
                    check()
                    return -float(proposal.evaluate(np.asarray(p)[None])[0])/scale
                bounds = [(max(-1000,v-50),min(1000,v+50)) for v in initial]
                refined = minimize(objective,initial,method='Powell',bounds=bounds,
                    options=dict(maxiter=15,xtol=.1,ftol=1e-7)).x
                if far(refined[None],others)[0] and objective(refined)<=objective(initial):
                    positions[k]=refined
                save(positions)
        else:
            pts, values = np.asarray(found['candidates']), np.asarray(found['scores'])
            starts = pts[peaks(pts,values,5)]
            del proposal
            calc = Score(iq, check)
            best = -np.inf
            for initial in starts:
                calc.set_sources([])
                positions = [local(calc,initial,[])]
                if count==2:
                    calc.set_sources(positions)
                    second = candidates(calc,positions,method,1)[0]
                    positions.append(local(calc,second,positions))
                positions=np.asarray(positions)
                # 一旦完整位置集合可用即发布，后续只发布更优的联合目标解。
                calc.set_sources(positions)
                score=1-float(calc.residual.abs().square().sum().cpu())/calc.energy
                if score>best:
                    best=score
                    save(positions)
                for iteration in range(5 if count==2 else 1):
                    for k in range(count):
                        others=np.delete(positions,k,axis=0)
                        calc.set_sources(others)
                        point=positions[k]
                        if iteration in (2,4):
                            candidate=candidates(calc,others,method,1)[0]
                            if calc.evaluate([candidate])[0]>calc.evaluate([point])[0]:
                                point=candidate
                        refined=local(calc,point,others)
                        if calc.evaluate([refined])[0]>=calc.evaluate([positions[k]])[0]:
                            positions[k]=refined
                        updates+=1
                    calc.set_sources(positions)
                    score=1-float(calc.residual.abs().square().sum().cpu())/calc.energy
                    if score>best:
                        best=score
                        save(positions)
    except Deadline:
        pass
    torch.cuda.synchronize()
    if calc is not None:
        score_points, response_points=calc.points,calc.response_points
    return dict(events=events,seconds=time.perf_counter()-started,
        score_points=score_points+proposal_points,response_points=response_points,
        proposal_points=proposal_points,updates=updates,solver_version='g9_v2_common_dpd_proposal',
        status='BUDGET_EXHAUSTED' if time.perf_counter()-started>=budget else 'FINISHED')
