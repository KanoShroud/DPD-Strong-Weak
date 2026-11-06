"""Formula tests only; no localization samples or ground-truth solver input."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import core


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = PROJECT / "outputs_e2e/research_admission_20261027/localization_identifiability/neighbor_closure_20261030/Friedlander2014.txt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    checks = []
    rng = np.random.default_rng(2026110207)

    def z(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    def check(name, passed, value=None):
        checks.append({"name": name, "passed": bool(passed), "value": value})
        if not passed:
            raise AssertionError(name)

    for q in (1, 2):
        c, d, n = 3 * q, 31, 5
        op, y, gains = z((c, d, n)), z(d), z(c)
        index = np.repeat(np.arange(q), 3)
        wave, rank, _ = core.waveform_ls(y, op, gains, index, q)
        explicit = np.concatenate([sum(gains[j] * op[j] for j in range(c) if index[j] == k) for k in range(q)], axis=1)
        reference = np.linalg.lstsq(explicit, y, rcond=None)[0]
        check(f"Q{q}_common_wave_matches_lstsq", np.allclose(wave.ravel(), reference, atol=1e-12))
        check(f"Q{q}_full_rank", rank == q * n)
        means = core.component_means(op, gains, wave, index)
        check(f"Q{q}_sum_equals_shared_design", np.allclose(means.sum(axis=0), explicit @ wave.ravel()))
        check(f"Q{q}_wave_normal_equations", np.linalg.norm(explicit.conj().T @ (y - means.sum(axis=0))) < 1e-10)
        fitted_g, gain_rank, _ = core.gain_ls(y, op, wave, index)
        gain_design = np.column_stack([op[j] @ wave[index[j]] for j in range(c)])
        check(f"Q{q}_gain_matches_lstsq", np.allclose(fitted_g, np.linalg.lstsq(gain_design, y, rcond=None)[0]))
        check(f"Q{q}_gain_rank", gain_rank == c)
        cost = [core.residual_objective(y, means)]
        for _ in range(20):
            gains, _, _ = core.gain_ls(y, op, wave, index)
            cost.append(core.residual_objective(y, core.component_means(op, gains, wave, index)))
            wave, _, _ = core.waveform_ls(y, op, gains, index, q)
            cost.append(core.residual_objective(y, core.component_means(op, gains, wave, index)))
        check(f"Q{q}_alternating_LS_monotone", np.max(np.diff(cost)) < 1e-10, float(np.max(np.diff(cost))))
        beta = np.arange(1, c + 1, dtype=float)
        beta /= beta.sum()
        means = core.component_means(op, gains, wave, index)
        complete = core.expectation(y, means, beta)
        check(f"Q{q}_E_sum_is_observation", np.allclose(complete.sum(axis=0), y, atol=1e-12))
        old_surrogate = core.weighted_surrogate(complete, means, beta)
        old_cost = core.residual_objective(y, means)
        check(f"Q{q}_surrogate_touches", np.isclose(old_surrogate, old_cost, atol=1e-11))
        gaps = []
        for _ in range(100):
            candidate = means + .1 * z(means.shape)
            q_delta = core.weighted_surrogate(complete, candidate, beta) - old_surrogate
            f_delta = core.residual_objective(y, candidate) - old_cost
            gaps.append(q_delta - f_delta)
        check(f"Q{q}_majorization_100_candidates", min(gaps) > -1e-11, float(min(gaps)))
        em_wave, _, _ = core.waveform_surrogate_ls(complete, op, gains, index, q, beta)
        em_means = core.component_means(op, gains, em_wave, index)
        check(f"Q{q}_shared_EM_M_step_decreases_surrogate", core.weighted_surrogate(complete, em_means, beta) <= old_surrogate + 1e-10)
        check(f"Q{q}_shared_EM_M_step_decreases_observed_cost", core.residual_objective(y, em_means) <= old_cost + 1e-10)
        cov = core.conditional_component_covariance(beta, .4)
        check(f"Q{q}_conditional_cov_Hermitian", np.allclose(cov, cov.T.conj()))
        check(f"Q{q}_conditional_cov_PSD", np.linalg.eigvalsh(cov).min() > -1e-13)
        check(f"Q{q}_conditional_cov_zero_sum_noise", np.linalg.norm(cov @ np.ones(c)) < 1e-13)
        check(f"Q{q}_conditional_cov_rank", np.linalg.matrix_rank(cov, tol=1e-12) == c - 1)

    # Station-local gain columns reproduce independent per-station Eq.18 solves.
    m, k, samples, n = 3, 2, 9, 4
    op = np.zeros((m * k, m * samples, n), dtype=complex)
    for station in range(m):
        for path in range(k):
            op[station * k + path, station * samples:(station + 1) * samples] = z((samples, n))
    y, wave = z(m * samples), z((1, n))
    gains, _, _ = core.gain_ls(y, op, wave, np.zeros(m * k, dtype=int))
    local = []
    for station in range(m):
        rows = slice(station * samples, (station + 1) * samples)
        matrix = np.column_stack([op[station * k + j, rows] @ wave[0] for j in range(k)])
        local.extend(np.linalg.lstsq(matrix, y[rows], rcond=None)[0])
    check("station_gain_LS_matches_Eq18", np.allclose(gains, local))

    # Deliberately incompatible per-path waveforms: different statistical model.
    op = z((2, 25, 4))
    independent_wave = z((2, 4))
    y = sum(op[j] @ independent_wave[j] for j in range(2))
    common, _, _ = core.waveform_ls(y, op, np.ones(2), np.zeros(2, dtype=int), 1)
    shared_cost = core.residual_objective(y, core.component_means(op, np.ones(2), common, np.zeros(2, dtype=int)))
    independent_design = np.concatenate(op, axis=1)
    independent_fit = np.linalg.lstsq(independent_design, y, rcond=None)[0]
    independent_cost = float(np.linalg.norm(y - independent_design @ independent_fit) ** 2)
    check("per_path_wave_is_not_common_wave_model", shared_cost > 1 and independent_cost < 1e-20,
          {"shared_cost": shared_cost, "independent_cost": independent_cost})

    # Rank deficiency is returned, not hidden by a ridge or assumed invertibility.
    op = np.ones((1, 7, 3), dtype=complex)
    wave, rank, _ = core.waveform_ls(z(7), op, np.ones(1), np.array([0]), 1)
    check("rank_deficiency_reported", rank == 1 and np.isfinite(wave).all())
    for label, beta in (("zero", [0, 1]), ("negative", [-1, 2]), ("unnormalized", [.2, .2]), ("nan", [np.nan, 1])):
        try:
            core.expectation(np.ones(4), np.zeros((2, 4)), beta)
        except ValueError:
            check(f"reject_beta_{label}", True)
        else:
            check(f"reject_beta_{label}", False)

    out = HERE / ("test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(exist_ok=False)
    result = {
        "scope": "mathematical_unit_checks_only_not_localization_or_paper_replication",
        "seed": 2026110207, "passed": True, "checks": checks,
        "count": len(checks), "python": sys.version, "numpy": np.__version__,
        "sources_sha256": {str(p.relative_to(PROJECT)): sha(p) for p in (HERE / "core.py", HERE / "selftest.py", SOURCE)},
        "global_optimality_proven": False, "full_geolocator_implemented": False,
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EM数学内核：{len(checks)}项检查通过；未运行定位实验。")
    print(out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
