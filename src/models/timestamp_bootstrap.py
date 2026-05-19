"""
Timestamp-randomization bootstrap for second-rounded event times.

When timestamps are rounded to the nearest second, events sharing the same
second stamp are redistributed uniformly within that second.

Paper's procedure:
    For each window, identify all events sharing the same second.
    For each of 50 bootstrap realizations, redistribute events sharing second s
    by sampling i.i.d. Uniform(s, s+1) times, then sort all event times.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def randomize_within_seconds(
    times: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Redistribute events sharing the same second stamp uniformly within that second.

    For each unique second s in the event times, all events with time in [s, s+1)
    are assigned i.i.d. Uniform(s, s+1) positions, then the full array is sorted.

    Parameters
    ----------
    times : np.ndarray
        Event times (float64, seconds). May contain ties at second boundaries.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Randomized and sorted event times.
    """
    if rng is None:
        rng = np.random.default_rng()

    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return times.copy()

    # Floor to identify second bins
    seconds = np.floor(times).astype(np.int64)
    unique_seconds = np.unique(seconds)

    randomized = times.copy()
    for s in unique_seconds:
        mask = seconds == s
        count = np.sum(mask)
        if count > 1:
            # Redistribute uniformly within [s, s+1)
            randomized[mask] = s + rng.uniform(0.0, 1.0, size=count)

    return np.sort(randomized)


def generate_bootstrap_realizations(
    times: np.ndarray,
    n_bootstrap: int = 50,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate multiple within-second randomized bootstrap realizations.

    Parameters
    ----------
    times : np.ndarray
        Second-rounded event times in a window.
    n_bootstrap : int
        Number of bootstrap realizations. Paper uses 50.
    seed : int, optional
        Base random seed for reproducibility.

    Returns
    -------
    List[np.ndarray]
        List of n_bootstrap randomized event-time arrays.
    """
    rng = np.random.default_rng(seed)
    realizations = []
    for _ in range(n_bootstrap):
        child_rng = np.random.default_rng(rng.integers(0, 2**31))
        realizations.append(randomize_within_seconds(times, rng=child_rng))
    return realizations


def round_to_seconds(times: np.ndarray) -> np.ndarray:
    """
    Round continuous event times to the nearest second.

    Parameters
    ----------
    times : np.ndarray
        Continuous event times (float64, seconds).

    Returns
    -------
    np.ndarray
        Times rounded to nearest second (float64).
    """
    return np.round(times).astype(np.float64)
