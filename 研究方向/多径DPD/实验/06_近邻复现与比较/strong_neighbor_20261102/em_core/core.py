"""Isolated linear/EM primitives; not a complete Friedlander geolocator.

All arrays are complex baseband quantities. Operators have shape (C, D, N),
one C component acting on N samples of its source, producing D observations.
Several components may share a source; their waveform MUST remain shared.
Dense matrices are for formula tests, not a proposed large-data implementation.
"""

from __future__ import annotations

import numpy as np


def _complex(x):
    x = np.asarray(x, dtype=np.complex128)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite array")
    return x


def _beta(beta, count):
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (count,) or not np.isfinite(beta).all():
        raise ValueError("beta shape/nonfinite")
    if np.any(beta <= 0) or not np.isclose(beta.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError("beta must be positive and sum to one")
    return beta


def _topology(operators, source_index, n_sources):
    operators = _complex(operators)
    index = np.asarray(source_index)
    if operators.ndim != 3 or not all(operators.shape):
        raise ValueError("operators must have nonempty C,D,N axes")
    if not isinstance(n_sources, (int, np.integer)) or n_sources < 1:
        raise ValueError("n_sources must be positive integer")
    if index.shape != (operators.shape[0],) or index.dtype.kind not in "iu":
        raise ValueError("source_index must have C integer entries")
    if np.any(index < 0) or np.any(index >= n_sources):
        raise ValueError("source_index outside source range")
    if set(index.tolist()) != set(range(n_sources)):
        raise ValueError("every source must have at least one component")
    return operators, index


def design_shared(operators, gains, source_index, n_sources):
    """A=[sum_{c in source 0}g_c T_c, ...]; Eq.9/19 extension to Q sources."""
    operators, index = _topology(operators, source_index, n_sources)
    gains = _complex(gains)
    if gains.shape != (operators.shape[0],):
        raise ValueError("gains must have C entries")
    grouped = np.zeros((n_sources, *operators.shape[1:]), dtype=complex)
    np.add.at(grouped, index, gains[:, None, None] * operators)
    return grouped.transpose(1, 0, 2).reshape(operators.shape[1], -1)


def waveform_ls(y, operators, gains, source_index, n_sources):
    """Observed-data common-waveform LS, Eq.19/43 for Q=1, extension for Q>1.

    Rank deficiency uses the minimum-norm LS solution, not a claim of unique
    waveform recovery. Rank is returned for downstream identifiability checks.
    """
    y = _complex(y)
    design = design_shared(operators, gains, source_index, n_sources)
    if y.shape != (design.shape[0],):
        raise ValueError("y must have D entries")
    value, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    return value.reshape(n_sources, -1), int(rank), singular


def component_means(operators, gains, waves, source_index):
    waves = _complex(waves)
    if waves.ndim != 2:
        raise ValueError("waves must have Q,N axes")
    operators, index = _topology(operators, source_index, waves.shape[0])
    gains = _complex(gains)
    if waves.shape[1] != operators.shape[2] or gains.shape != (operators.shape[0],):
        raise ValueError("gain/wave shape mismatch")
    return gains[:, None] * np.einsum("cdn,cn->cd", operators, waves[index])


def gain_ls(y, operators, waves, source_index):
    """Fixed shared waveforms, unconstrained complex gain LS (Eq.18 analogue).

    Components can be station-local via zero rows in operators; that reproduces
    per-station LS. This is NOT LOS-dominance constrained gain optimization.
    """
    waves = _complex(waves)
    if waves.ndim != 2:
        raise ValueError("waves must have Q,N axes")
    operators, index = _topology(operators, source_index, waves.shape[0])
    y = _complex(y)
    if waves.shape[1] != operators.shape[2] or y.shape != (operators.shape[1],):
        raise ValueError("y/wave shape mismatch")
    design = np.einsum("cdn,cn->dc", operators, waves[index])
    gains, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    return gains, int(rank), singular


def expectation(y, means, beta):
    """Friedlander Eq.42 mean; beta strictly positive for weighted surrogate."""
    means = _complex(means)
    y = _complex(y)
    if means.ndim != 2 or y.shape != (means.shape[1],):
        raise ValueError("means must have C,D axes and y D entries")
    beta = _beta(beta, means.shape[0])
    return means + beta[:, None] * (y - means.sum(axis=0))


def conditional_component_covariance(beta, sigma2):
    """Independent latent-noise augmentation, NOT an equation in Friedlander.

    Independent w_c~CN(0,beta_c*sigma2 I) conditioned on sum w_c has covariance
    sigma2*(diag(beta)-beta beta.T) for each observation coordinate.
    """
    beta = np.asarray(beta, dtype=float)
    beta = _beta(beta, len(beta))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    return sigma2 * (np.diag(beta) - np.outer(beta, beta))


def residual_objective(y, means):
    means = _complex(means)
    y = _complex(y)
    if means.ndim != 2 or y.shape != (means.shape[1],):
        raise ValueError("means/y shape mismatch")
    residual = y - means.sum(axis=0)
    return float(np.vdot(residual, residual).real)


def weighted_surrogate(complete_means, candidate_means, beta):
    """Sum ||zhat_c-mu_c||²/beta_c, covariance constant omitted.

    This is a negative-EM-Q surrogate in noise-power units (no 1/sigma2).
    With zhat=expectation(y,mu_old,beta), it touches F at mu_old and
    majorizes F at all candidate means. Decreasing it guarantees F decrease;
    arbitrary per-component updates that do not decrease it have no guarantee.
    """
    complete_means, candidate_means = _complex(complete_means), _complex(candidate_means)
    if complete_means.ndim != 2 or complete_means.shape != candidate_means.shape:
        raise ValueError("surrogate shape mismatch")
    beta = _beta(beta, complete_means.shape[0])
    return float(np.sum(np.abs(complete_means - candidate_means) ** 2 / beta[:, None]))


def waveform_surrogate_ls(complete_means, operators, gains, source_index, n_sources, beta):
    """Exact shared-waveform minimizer of the frozen E-step surrogate.

    Separate from observed-data Eq.43 LS: these subproblems are not identical.
    This extension is provided to test a valid shared-waveform EM M-step.
    """
    operators, index = _topology(operators, source_index, n_sources)
    gains, target = _complex(gains), _complex(complete_means)
    c, d, n = operators.shape
    beta = _beta(beta, c)
    if gains.shape != (c,) or target.shape != (c, d):
        raise ValueError("gain/complete_mean shape mismatch")
    matrix = np.zeros((c, d, n_sources, n), dtype=complex)
    matrix[np.arange(c), :, index, :] = gains[:, None, None] * operators
    matrix = (matrix / np.sqrt(beta)[:, None, None, None]).reshape(c * d, n_sources * n)
    rhs = (target / np.sqrt(beta)[:, None]).reshape(c * d)
    wave, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    return wave.reshape(n_sources, n), int(rank), singular
