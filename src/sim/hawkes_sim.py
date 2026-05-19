"""
Hawkes process simulation using Ogata's thinning algorithm.

Simulates a univariate Hawkes process with constant background intensity
and exponential kernel: lambda(t) = mu + n*beta * sum_{t_j<t} exp(-beta*(t-t_j))

Also provides a wrapper around tick.hawkes.SimuHawkesExpKernels when available.

Custom Ogata thinning — Context7 confirmed tick provides SimuHawkesExpKernels,
but custom implementation used for full control over truncation and reproducibility.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def simulate_hawkes_ogata(
    mu: float,
    n: float,
    beta: float,
    T: float,
    seed: Optional[int] = None,
    max_events: int = 1_000_000,
) -> np.ndarray:
    """
    Simulate a univariate Hawkes process via Ogata's thinning algorithm.

    Model: lambda(t) = mu + alpha * sum_{t_j < t} exp(-beta*(t - t_j))
    where alpha = n * beta.

    Parameters
    ----------
    mu : float
        Background intensity (events/second).
    n : float
        Branching ratio (0 <= n < 1).
    beta : float
        Exponential decay rate (1/second).
    T : float
        Simulation horizon (seconds).
    seed : int, optional
        Random seed.
    max_events : int
        Safety cap on number of events.

    Returns
    -------
    np.ndarray
        Sorted event times in [0, T].
    """
    assert 0 <= n < 1, "Branching ratio must be in [0, 1)"
    assert mu > 0, "Background intensity must be positive"
    assert beta > 0, "Decay rate must be positive"

    rng = np.random.default_rng(seed)
    alpha = n * beta

    events = []
    t = 0.0
    lambda_star = mu  # Upper bound on intensity

    while t < T and len(events) < max_events:
        # Draw next candidate time
        u = rng.uniform()
        dt = -np.log(u) / lambda_star
        t_candidate = t + dt

        if t_candidate > T:
            break

        # Compute current intensity at t_candidate
        lambda_t = mu + alpha * sum(
            np.exp(-beta * (t_candidate - s)) for s in events
        )

        # Thinning acceptance
        u2 = rng.uniform()
        if u2 <= lambda_t / lambda_star:
            events.append(t_candidate)
            t = t_candidate
            # Update upper bound
            lambda_star = lambda_t + alpha
        else:
            t = t_candidate
            # Recompute upper bound (intensity can only decrease between events)
            lambda_star = mu + alpha * sum(
                np.exp(-beta * (t - s)) for s in events
            )

    return np.array(events, dtype=np.float64)


def simulate_hawkes_ogata_fast(
    mu: float,
    n: float,
    beta: float,
    T: float,
    seed: Optional[int] = None,
    max_events: int = 1_000_000,
) -> np.ndarray:
    """
    Fast Ogata thinning using recursive intensity update.

    Uses the recursive formula for the excitation sum to avoid O(N^2) computation.

    Parameters
    ----------
    mu : float
        Background intensity.
    n : float
        Branching ratio.
    beta : float
        Decay rate.
    T : float
        Simulation horizon (seconds).
    seed : int, optional
        Random seed.
    max_events : int
        Safety cap.

    Returns
    -------
    np.ndarray
        Sorted event times in [0, T].
    """
    assert 0 <= n < 1, "Branching ratio must be in [0, 1)"
    assert mu > 0, "Background intensity must be positive"
    assert beta > 0, "Decay rate must be positive"

    rng = np.random.default_rng(seed)
    alpha = n * beta

    events = []
    t = 0.0
    R = 0.0  # Recursive excitation sum: sum_{j} exp(-beta*(t - t_j))
    lambda_star = mu  # Upper bound

    while t < T and len(events) < max_events:
        u = rng.uniform()
        if u == 0:
            u = 1e-15
        dt = -np.log(u) / lambda_star
        t_candidate = t + dt

        if t_candidate > T:
            break

        # Update R to t_candidate
        R_candidate = R * np.exp(-beta * (t_candidate - t))
        lambda_t = mu + alpha * R_candidate

        u2 = rng.uniform()
        if u2 <= lambda_t / lambda_star:
            events.append(t_candidate)
            t = t_candidate
            R = R_candidate + 1.0  # Add new event contribution
            lambda_star = lambda_t + alpha
        else:
            t = t_candidate
            R = R_candidate
            lambda_star = lambda_t  # Intensity can only decrease

    return np.array(events, dtype=np.float64)


def simulate_hawkes_tick(
    mu: float,
    n: float,
    beta: float,
    T: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Hawkes process using tick library (if available).

    Using tick.hawkes.SimuHawkesExpKernels — Context7 confirmed.

    Parameters
    ----------
    mu : float
        Background intensity.
    n : float
        Branching ratio.
    beta : float
        Decay rate.
    T : float
        Simulation horizon (seconds).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Sorted event times in [0, T].
    """
    try:
        from tick.hawkes import SimuHawkesExpKernels
        # tick's adjacency parameter is the kernel norm (branching ratio n),
        # NOT alpha = n * beta. The kernel is: adjacency * decays * exp(-decays * t)
        # so kernel integral = adjacency = n.
        sim = SimuHawkesExpKernels(
            adjacency=[[n]],
            decays=[[beta]],
            baseline=[mu],
            end_time=T,
            seed=seed if seed is not None else 42,
            verbose=False,
        )
        sim.simulate()
        timestamps = sim.timestamps[0]
        return np.sort(np.array(timestamps, dtype=np.float64))
    except ImportError:
        logger.warning("tick not available, falling back to Ogata thinning")
        return simulate_hawkes_ogata_fast(mu, n, beta, T, seed=seed)


def simulate_hawkes(
    mu: float,
    n: float,
    beta: float,
    T: float,
    seed: Optional[int] = None,
    use_tick: bool = True,
) -> np.ndarray:
    """
    Simulate a Hawkes process, preferring tick if available.

    Parameters
    ----------
    mu : float
        Background intensity.
    n : float
        Branching ratio.
    beta : float
        Decay rate.
    T : float
        Simulation horizon (seconds).
    seed : int, optional
        Random seed.
    use_tick : bool
        Try tick first if True.

    Returns
    -------
    np.ndarray
        Sorted event times in [0, T].
    """
    if use_tick:
        return simulate_hawkes_tick(mu, n, beta, T, seed=seed)
    return simulate_hawkes_ogata_fast(mu, n, beta, T, seed=seed)


def expected_event_count(mu: float, n: float, T: float) -> float:
    """
    Expected number of events in [0, T] for a stationary Hawkes process.

    E[N(T)] = mu * T / (1 - n)

    Parameters
    ----------
    mu : float
        Background intensity.
    n : float
        Branching ratio.
    T : float
        Horizon.

    Returns
    -------
    float
        Expected event count.
    """
    assert 0 <= n < 1
    return mu * T / (1.0 - n)


def calibrate_params_for_target_count(
    target_count: float,
    n: float,
    beta: float,
    T: float,
) -> Tuple[float, float, float]:
    """
    Calibrate mu so that expected event count matches target.

    Parameters
    ----------
    target_count : float
        Desired expected number of events.
    n : float
        Branching ratio.
    beta : float
        Decay rate.
    T : float
        Horizon.

    Returns
    -------
    Tuple[float, float, float]
        (mu, n, beta)
    """
    mu = target_count * (1.0 - n) / T
    return mu, n, beta
