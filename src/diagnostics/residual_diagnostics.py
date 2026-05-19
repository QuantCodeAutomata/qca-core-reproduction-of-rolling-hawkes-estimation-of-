"""
Residual diagnostics for fitted Hawkes processes.

Implements the paper's residual time-transformation and Kolmogorov-Smirnov
uniformity test for model adequacy assessment.

Rejection rule (paper's conservative criterion):
    Reject a window only if ALL 50 bootstrap realizations are rejected at 5%.
    Equivalently: reject if max KS p-value across 50 fits < 0.05.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest, uniform
from typing import List, Tuple

from src.models.hawkes_exp_truncated import compute_compensator


def compute_uniformized_residuals(
    times: np.ndarray,
    T: float,
    mu: float,
    n: float,
    beta: float,
) -> np.ndarray:
    """
    Compute uniformized residuals U_i = 1 - exp(-Delta_i).

    Under the fitted model, U_i should be i.i.d. Uniform[0,1].

    Steps:
    1. Compute compensator xi_i at each event time.
    2. Compute gaps Delta_i = xi_i - xi_{i-1} (with xi_0 = 0 implicitly).
    3. Uniformize: U_i = 1 - exp(-Delta_i).

    Parameters
    ----------
    times : np.ndarray
        Sorted event times in [0, T].
    T : float
        Window length in seconds.
    mu : float
        Background intensity.
    n : float
        Branching ratio.
    beta : float
        Decay rate.

    Returns
    -------
    np.ndarray
        Uniformized residuals U_i in [0, 1].
    """
    xi = compute_compensator(times, T, mu, n, beta)
    # Prepend 0 for the first gap
    xi_extended = np.concatenate([[0.0], xi])
    delta = np.diff(xi_extended)
    # Clip to avoid numerical issues
    delta = np.clip(delta, 1e-15, None)
    U = 1.0 - np.exp(-delta)
    return U


def ks_test_uniformity(U: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test of uniformity on [0,1].

    Parameters
    ----------
    U : np.ndarray
        Uniformized residuals.

    Returns
    -------
    Tuple[float, float]
        (ks_statistic, p_value)
    """
    if len(U) < 2:
        return np.nan, np.nan
    stat, pval = kstest(U, "uniform")
    return float(stat), float(pval)


def evaluate_window_adequacy(
    ks_pvalues: List[float],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Apply the paper's conservative rejection rule.

    Reject a window only if ALL bootstrap realizations are rejected at alpha.
    Equivalently: reject if max(ks_pvalues) < alpha.

    Parameters
    ----------
    ks_pvalues : List[float]
        KS p-values from each bootstrap realization.
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    Tuple[bool, float]
        (accepted, max_ks_pvalue)
        accepted = True if the window is NOT rejected.
    """
    valid_pvals = [p for p in ks_pvalues if np.isfinite(p)]
    if len(valid_pvals) == 0:
        return False, np.nan
    max_pval = max(valid_pvals)
    accepted = max_pval >= alpha
    return accepted, max_pval


def run_diagnostics_single(
    times: np.ndarray,
    T: float,
    mu: float,
    n: float,
    beta: float,
) -> Tuple[float, float]:
    """
    Run full residual diagnostics for a single fitted Hawkes model.

    Parameters
    ----------
    times : np.ndarray
        Sorted event times in [0, T].
    T : float
        Window length in seconds.
    mu, n, beta : float
        Fitted parameters.

    Returns
    -------
    Tuple[float, float]
        (ks_statistic, ks_pvalue)
    """
    U = compute_uniformized_residuals(times, T, mu, n, beta)
    return ks_test_uniformity(U)
