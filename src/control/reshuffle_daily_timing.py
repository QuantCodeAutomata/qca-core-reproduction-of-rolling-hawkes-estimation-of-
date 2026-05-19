"""
Reshuffled-timing control experiment.

For each accepted trading day d, preserves the daily event count N_d while
replacing event times with N_d i.i.d. Uniform draws over the session interval.

This destroys within-day temporal dependence while preserving daily activity,
allowing a test of whether the Hawkes branching ratio n captures genuine
self-excitation rather than merely high trading intensity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regular trading session: 9:30-16:15 NY = 6h45m = 24300 seconds
SESSION_DURATION_S = 6.75 * 3600.0  # 24300 seconds


def reshuffle_day_events(
    n_events: int,
    session_duration_s: float = SESSION_DURATION_S,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate N_d i.i.d. Uniform event times over the session interval.

    Preserves daily event count while destroying temporal dependence.

    Parameters
    ----------
    n_events : int
        Number of events to generate (= original daily count).
    session_duration_s : float
        Session duration in seconds.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Sorted event times in [0, session_duration_s].
    """
    if rng is None:
        rng = np.random.default_rng()
    times = rng.uniform(0.0, session_duration_s, size=n_events)
    return np.sort(times)


def build_reshuffled_event_series(
    daily_event_counts: Dict[str, int],
    session_duration_s: float = SESSION_DURATION_S,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Build a full reshuffled event series for all accepted trading days.

    Parameters
    ----------
    daily_event_counts : Dict[str, int]
        Mapping from date string to daily event count.
    session_duration_s : float
        Session duration in seconds.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from date string to reshuffled event times (seconds since open).
    """
    rng = np.random.default_rng(seed)
    reshuffled = {}
    for date_str, n_events in sorted(daily_event_counts.items()):
        child_rng = np.random.default_rng(rng.integers(0, 2**31))
        reshuffled[date_str] = reshuffle_day_events(n_events, session_duration_s, child_rng)
    return reshuffled


def extract_daily_event_counts(
    event_df: pd.DataFrame,
    date_col: str = "date",
) -> Dict[str, int]:
    """
    Extract daily event counts from the cleaned event series.

    Parameters
    ----------
    event_df : pd.DataFrame
        Cleaned event series.
    date_col : str
        Date column name.

    Returns
    -------
    Dict[str, int]
        Mapping from date string to event count.
    """
    counts = event_df.groupby(date_col).size()
    return {str(d): int(c) for d, c in counts.items()}


def run_reshuffled_pipeline(
    daily_event_counts: Dict[str, int],
    window_length_min: int = 10,
    step_min: int = 5,
    n_bootstrap: int = 50,
    session_duration_s: float = SESSION_DURATION_S,
    base_seed: int = 42,
    n_starts: int = 3,
    min_events: int = 10,
    max_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full rolling Hawkes estimation pipeline on reshuffled event times.

    Parameters
    ----------
    daily_event_counts : Dict[str, int]
        Daily event counts from the empirical series.
    window_length_min : int
        Window length in minutes.
    step_min : int
        Step size in minutes.
    n_bootstrap : int
        Bootstrap realizations per window.
    session_duration_s : float
        Session duration in seconds.
    base_seed : int
        Base random seed.
    n_starts : int
        MLE starting points.
    min_events : int
        Minimum events per window.
    max_days : int, optional
        Limit number of days (for testing).

    Returns
    -------
    pd.DataFrame
        Bootstrap-level results for all windows across all days.
    """
    from src.rolling.window_pipeline import process_day_rolling

    reshuffled = build_reshuffled_event_series(
        daily_event_counts, session_duration_s, seed=base_seed
    )

    all_results = []
    dates = sorted(reshuffled.keys())
    if max_days is not None:
        dates = dates[:max_days]

    for i, date_str in enumerate(dates):
        day_times = reshuffled[date_str]
        logger.info(f"Processing reshuffled day {date_str} ({i+1}/{len(dates)})")
        try:
            day_results = process_day_rolling(
                day_times_seconds=day_times,
                trade_date=date_str,
                window_length_min=window_length_min,
                step_min=step_min,
                session_duration_s=session_duration_s,
                n_bootstrap=n_bootstrap,
                base_seed=base_seed + i * 10000,
                n_starts=n_starts,
                min_events=min_events,
            )
            if not day_results.empty:
                all_results.append(day_results)
        except Exception as e:
            logger.warning(f"Day {date_str} failed: {e}")

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
