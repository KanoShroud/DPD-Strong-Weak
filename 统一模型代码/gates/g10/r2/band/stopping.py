"""Observation-only SBL stopping; no ground truth or success radius enters."""
from itertools import permutations
import numpy as np


def movement(old, new):
    a, b = np.asarray(old).reshape(-1, 2), np.asarray(new).reshape(-1, 2)
    if len(a) != len(b):
        return float('inf')
    return min(float(np.linalg.norm(a-b[list(p)], axis=1).max()) for p in permutations(range(len(b))))


class StopController:
    def __init__(self, cfg):
        self.override_legacy = cfg['sbl_stop_rule'] == 'candidate'
        self.cfg, self.previous, self.stable = cfg, None, 0
        self.checkpoints, self.first_stop = [], None

    def update(self, iteration, value, gamma, positions, seconds):
        if iteration % 5:
            return False
        weights = gamma.detach().cpu().numpy().copy()
        current = dict(iteration=iteration, objective=float(value), positions=np.asarray(positions).tolist(), seconds=seconds)
        if self.previous is not None:
            old, previous_gamma = self.previous
            current['objective_change'] = abs(value-old['objective'])/max(1., abs(old['objective']))
            current['gamma_change'] = float(abs(weights-previous_gamma).sum()/max(previous_gamma.sum(), 1e-30))
            current['output_movement_m'] = movement(old['positions'], positions)
            good = (iteration >= self.cfg['sbl_stop_minimum'] and current['objective_change'] < 1e-5
                    and current['gamma_change'] < 1e-2
                    and current['output_movement_m'] < self.cfg['sbl_position_tolerance'])
            self.stable = self.stable+1 if good else 0
        current['consecutive'] = self.stable
        current['would_stop'] = self.stable >= self.cfg['sbl_stop_patience']
        if current['would_stop'] and self.first_stop is None:
            self.first_stop = current.copy()
        self.checkpoints.append(current)
        self.previous = current, weights
        return self.override_legacy and current['would_stop']
