"""
Univariate Hawkes process with constant background intensity and exponential kernel.

Model: lambda(t) = mu + n*beta * sum_{t_j < t} exp(-beta*(t - t_j))

Parameterization: mu > 0, beta > 0, 0 <= n < 1 (branching ratio)
Kernel: h(t) = alpha * exp(-beta*t) with alpha = n*beta

Truncated-window likelihood: no pre-window excitation is imported.
This matches the paper's estimation approach and is acknowledged to bias n downward.

Custom implementation — Context7 found no library equivalent for truncated-window
likelihood with within-second bootstrap and custom residual rejection logic.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Recursive excitation computation (O(N) per likelihood call)
# ---------------------------------------------------------------------------

def _compute_excitation_recursive(
    times: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute R_i = sum_{j < i} exp(-beta*(t_i - t_j)) recursively in O(N).

    Parameters
    ----------
    times : np.ndarray
        Sorted event times (float64, seconds).
    beta : float
        Exponential decay rate.

    Returns
    -------
    np.ndarray
        R[i] for i = 0, ..., N-1.  R[0] = 0 by convention.
    """
    N = len(times)
    R = np.zeros(N, dtype=np.float64)
    if N <= 1:
        return R
    for i in range(1, N):
        dt = times[i] - times[i - 1]
        R[i] = (1.0 + R[i - 1]) * np.exp(-beta * dt)
    return R


def log_likelihood_truncated(
    times: np.ndarray,
    T: float,
    mu: float,
    n: float,
    beta: float,
) -> float:
    """
    Log-likelihood of a truncated-window univariate Hawkes process.

    No pre-window excitation is assumed (truncated window).

    ell(mu, n, beta) =
        sum_i log(mu + n*beta*R_i)
        - mu*T
        - n * sum_j (1 - exp(-beta*(T - t_j)))

    Parameters
    ----------
    times : np.ndarray
        Sorted event times in [0, T] (float64, seconds).
    T : float
        Window length in seconds.
    mu : float
        Background intensity (events/second).
    n : float
        Branching ratio (0 <= n < 1).
    beta : float
        Exponential decay rate (1/second).

    Returns
    -------
    float
        Log-likelihood value.
    """
    if len(times) == 0:
        return -mu * T

    R = _compute_excitation_recursive(times, beta)
    alpha = n * beta

    # Sum of log intensities
    intensities = mu + alpha * R
    if np.any(intensities <= 0):
        return -np.inf
    log_sum = np.sum(np.log(intensities))

    # Compensator
    compensator = mu * T + n * np.sum(1.0 - np.exp(-beta * (T - times)))

    return log_sum - compensator


def _neg_loglik(
    params_transformed: np.ndarray,
    times: np.ndarray,
    T: float,
) -> float:
    """
    Negative log-likelihood in transformed parameter space.

    Transformation: log(mu), logit(n), log(beta) to enforce constraints.
    """
    log_mu, logit_n, log_beta = params_transformed
    mu = np.exp(log_mu)
    n = 1.0 / (1.0 + np.exp(-logit_n))
    beta = np.exp(log_beta)
    ll = log_likelihood_truncated(times, T, mu, n, beta)
    return -ll if np.isfinite(ll) else 1e15


def fit_hawkes_exp(
    times: np.ndarray,
    T: float,
    mu_init: Optional[float] = None,
    n_init: float = 0.5,
    beta_init: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """
    Fit a univariate Hawkes process with exponential kernel via MLE.

    Uses L-BFGS-B on transformed parameters (log/logit) to enforce constraints.

    Parameters
    ----------
    times : np.ndarray
        Sorted event times in [0, T] (float64, seconds).
    T : float
        Window length in seconds.
    mu_init : float, optional
        Initial background intensity. Defaults to 10-30% of event rate.
    n_init : float
        Initial branching ratio. Default 0.5.
    beta_init : float, optional
        Initial decay rate. Defaults to 1/(characteristic decay ~5s).
    max_iter : int
        Maximum optimizer iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict with keys:
        mu_hat, n_hat, beta_hat, loglik, success, message, n_events
    """
    N = len(times)
    if N < 2:
        return {
            "mu_hat": np.nan,
            "n_hat": np.nan,
            "beta_hat": np.nan,
            "loglik": np.nan,
            "success": False,
            "message": "Too few events",
            "n_events": N,
        }

    event_rate = N / T
    if mu_init is None:
        mu_init = max(event_rate * 0.2, 1e-6)
    if beta_init is None:
        beta_init = 1.0 / max(5.0, T / N)

    # Clip n_init away from boundaries
    n_init = np.clip(n_init, 0.01, 0.99)

    x0 = np.array([
        np.log(mu_init),
        np.log(n_init / (1.0 - n_init)),
        np.log(beta_init),
    ])

    result = minimize(
        _neg_loglik,
        x0,
        args=(times, T),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol * 1e-3},
    )

    log_mu, logit_n, log_beta = result.x
    mu_hat = np.exp(log_mu)
    n_hat = 1.0 / (1.0 + np.exp(-logit_n))
    beta_hat = np.exp(log_beta)
    loglik = -result.fun

    return {
        "mu_hat": mu_hat,
        "n_hat": n_hat,
        "beta_hat": beta_hat,
        "loglik": loglik,
        "success": result.success,
        "message": result.message,
        "n_events": N,
    }


def fit_hawkes_exp_multi_start(
    times: np.ndarray,
    T: float,
    n_starts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Fit Hawkes model with multiple random initializations, return best result.

    Parameters
    ----------
    times : np.ndarray
        Sorted event times in [0, T].
    T : float
        Window length in seconds.
    n_starts : int
        Number of random starting points.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    dict
        Best fit result (highest log-likelihood).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = len(times)
    event_rate = max(N / T, 1e-6)

    best_result = None
    best_ll = -np.inf

    # Default initialization
    candidates = [
        (event_rate * 0.2, 0.5, 1.0 / 5.0),
        (event_rate * 0.1, 0.3, 1.0 / 10.0),
        (event_rate * 0.3, 0.7, 1.0 / 2.0),
    ]
    # Random starts
    for _ in range(n_starts - len(candidates)):
        mu_r = event_rate * rng.uniform(0.05, 0.5)
        n_r = rng.uniform(0.1, 0.9)
        beta_r = rng.uniform(0.05, 2.0)
        candidates.append((mu_r, n_r, beta_r))

    for mu_i, n_i, beta_i in candidates:
        res = fit_hawkes_exp(times, T, mu_init=mu_i, n_init=n_i, beta_init=beta_i)
        if res["success"] and np.isfinite(res["loglik"]) and res["loglik"] > best_ll:
            best_ll = res["loglik"]
            best_result = res

    if best_result is None:
        # Return last result even if not converged
        best_result = fit_hawkes_exp(times, T)

    return best_result


def compute_compensator(
    times: np.ndarray,
    T: float,
    mu: float,
    n: float,
    beta: float,
) -> np.ndarray:
    """
    Compute the compensator (integrated intensity) at each event time.

    xi_i = mu*t_i + n * sum_{j < i} (1 - exp(-beta*(t_i - t_j)))

    This is used for residual time transformation diagnostics.

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
        Compensator values xi_i for i = 0, ..., N-1.
    """
    N = len(times)
    xi = np.zeros(N, dtype=np.float64)
    if N == 0:
        return xi

    # Recursive computation of sum_{j<i}(1 - exp(-beta*(t_i - t_j)))
    # Let S_i = sum_{j<i}(1 - exp(-beta*(t_i - t_j)))
    # S_0 = 0
    # S_i = S_{i-1}*exp(-beta*dt) + (1 - exp(-beta*dt)) * (i-1) ... no, use direct recursion
    # Actually: S_i = sum_{j<i}(1 - exp(-beta*(t_i-t_j)))
    #               = (i-1) - sum_{j<i} exp(-beta*(t_i-t_j))
    #               = (i-1) - R_i
    # where R_i = sum_{j<i} exp(-beta*(t_i-t_j))

    R = _compute_excitation_recursive(times, beta)

    for i in range(N):
        S_i = i - R[i]  # sum_{j<i}(1 - exp(-beta*(t_i-t_j)))
        xi[i] = mu * times[i] + n * S_i

    return xi
