"""
Flash-crash case study: April 27 vs May 6, 2010.

Compares rolling Hawkes branching-ratio estimates between:
- April 27, 2010: large but exogenous market move
- May 6, 2010: endogenous flash-crash episode

Uses previous-day 5%-95% benchmark bands for event rate and branching ratio.

Timezone handling: all times in America/New_York (Eastern).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SESSION_START_HOUR = 9.5    # 9:30
SESSION_END_HOUR = 16.25    # 16:15
SESSION_DURATION_S = (SESSION_END_HOUR - SESSION_START_HOUR) * 3600.0  # 24300s


def build_minute_price_series(
    event_df: pd.DataFrame,
    date_str: str,
    timestamp_col: str = "timestamp",
    price_col: str = "mid_price",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Build minute-by-minute mid-price series for a single day.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event series with timestamp and mid_price columns.
    date_str : str
        Date to extract (YYYY-MM-DD).
    timestamp_col : str
        Timestamp column name.
    price_col : str
        Price column name.
    tz : str
        Timezone.

    Returns
    -------
    pd.DataFrame
        Minute-level OHLC-style price series.
    """
    df = event_df.copy()
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz)

    day_mask = df[timestamp_col].dt.date == pd.Timestamp(date_str).date()
    day_df = df[day_mask].copy()

    if day_df.empty:
        return pd.DataFrame()

    day_df = day_df.set_index(timestamp_col)
    minute_price = day_df[price_col].resample("1min").last().ffill()
    minute_price.name = "price"
    return minute_price.reset_index()


def build_minute_volume_series(
    trade_df: pd.DataFrame,
    date_str: str,
    timestamp_col: str = "timestamp",
    volume_col: str = "size",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Build minute-by-minute volume series for a single day.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Trade data with timestamp and size columns.
    date_str : str
        Date to extract.
    timestamp_col : str
        Timestamp column name.
    volume_col : str
        Volume column name.
    tz : str
        Timezone.

    Returns
    -------
    pd.DataFrame
        Minute-level volume series.
    """
    df = trade_df.copy()
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz)

    day_mask = df[timestamp_col].dt.date == pd.Timestamp(date_str).date()
    day_df = df[day_mask].copy()

    if day_df.empty:
        return pd.DataFrame()

    day_df = day_df.set_index(timestamp_col)
    minute_vol = day_df[volume_col].resample("1min").sum().fillna(0)
    minute_vol.name = "volume"
    return minute_vol.reset_index()


def extract_day_event_times(
    event_df: pd.DataFrame,
    date_str: str,
    timestamp_col: str = "timestamp",
    tz: str = "America/New_York",
) -> np.ndarray:
    """
    Extract event times as seconds since session open for a single day.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event series.
    date_str : str
        Date to extract.
    timestamp_col : str
        Timestamp column name.
    tz : str
        Timezone.

    Returns
    -------
    np.ndarray
        Sorted event times in seconds since 9:30 NY.
    """
    df = event_df.copy()
    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz)

    day_mask = df[timestamp_col].dt.date == pd.Timestamp(date_str).date()
    day_df = df[day_mask].copy().sort_values(timestamp_col)

    if day_df.empty:
        return np.array([])

    session_open = pd.Timestamp(date_str, tz=tz).replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    times_s = (day_df[timestamp_col] - session_open).dt.total_seconds().values
    # Filter to session
    mask = (times_s >= 0) & (times_s <= SESSION_DURATION_S)
    return np.sort(times_s[mask])


def compute_rolling_estimates_for_day(
    day_times_seconds: np.ndarray,
    trade_date: str,
    window_length_min: int = 10,
    step_min: int = 5,
    n_bootstrap: int = 50,
    base_seed: int = 42,
    n_starts: int = 3,
    min_events: int = 5,
) -> pd.DataFrame:
    """
    Compute rolling Hawkes estimates for a single case-study day.

    Parameters
    ----------
    day_times_seconds : np.ndarray
        Event times in seconds since session open.
    trade_date : str
        Date string.
    window_length_min : int
        Window length in minutes.
    step_min : int
        Step size in minutes.
    n_bootstrap : int
        Bootstrap realizations.
    base_seed : int
        Random seed.
    n_starts : int
        MLE starting points.
    min_events : int
        Minimum events per window.

    Returns
    -------
    pd.DataFrame
        Aggregated window estimates.
    """
    from src.rolling.window_pipeline import process_day_rolling, aggregate_window_estimates

    raw = process_day_rolling(
        day_times_seconds=day_times_seconds,
        trade_date=trade_date,
        window_length_min=window_length_min,
        step_min=step_min,
        session_duration_s=SESSION_DURATION_S,
        n_bootstrap=n_bootstrap,
        base_seed=base_seed,
        n_starts=n_starts,
        min_events=min_events,
    )

    if raw.empty:
        return pd.DataFrame()

    # Include all windows (not just accepted) for case study
    agg = aggregate_window_estimates(raw, accepted_only=False)
    return agg


def build_benchmark_bands(
    prev_day_estimates: pd.DataFrame,
    n_col: str = "n_hat_median",
    event_rate_col: str = "event_count",
    window_length_min: int = 10,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.DataFrame:
    """
    Build 5%-95% benchmark bands from the previous trading day.

    For each window position (window_start_min), compute quantile bands
    across the previous day's bootstrap-based estimates.

    Parameters
    ----------
    prev_day_estimates : pd.DataFrame
        Aggregated estimates from the previous day.
    n_col : str
        Branching ratio column.
    event_rate_col : str
        Event count column.
    window_length_min : int
        Window length in minutes.
    lower_q, upper_q : float
        Quantile bounds.

    Returns
    -------
    pd.DataFrame
        Benchmark bands with columns: window_start_min, n_lower, n_upper,
        rate_lower, rate_upper
    """
    if prev_day_estimates.empty:
        return pd.DataFrame()

    df = prev_day_estimates.copy()

    # Support both window_start_s (seconds) and window_start_min columns
    if "window_start_s" in df.columns and "window_start_min" not in df.columns:
        df["window_start_min"] = df["window_start_s"] / 60.0

    # Compute event rate per minute
    df["event_rate_per_min"] = df[event_rate_col] / window_length_min

    records = []
    for ws_min, group in df.groupby("window_start_min"):
        n_vals = group[n_col].dropna()
        rate_vals = group["event_rate_per_min"].dropna()
        if len(n_vals) > 0:
            records.append({
                "window_start_min": ws_min,
                "n_lower": n_vals.quantile(lower_q),
                "n_upper": n_vals.quantile(upper_q),
                "n_median": n_vals.median(),
                "rate_lower": rate_vals.quantile(lower_q) if len(rate_vals) > 0 else np.nan,
                "rate_upper": rate_vals.quantile(upper_q) if len(rate_vals) > 0 else np.nan,
                "rate_median": rate_vals.median() if len(rate_vals) > 0 else np.nan,
            })

    return pd.DataFrame(records)


def generate_synthetic_case_study_data(
    case_date: str,
    n_target: float,
    events_per_day: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic event times for a case-study day.

    Parameters
    ----------
    case_date : str
        Date string (for labeling only).
    n_target : float
        Target branching ratio.
    events_per_day : int
        Target event count.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Synthetic event times in seconds since session open.
    """
    from src.sim.hawkes_sim import simulate_hawkes, calibrate_params_for_target_count

    T = SESSION_DURATION_S
    beta = 1.0 / 5.0  # 5-second characteristic decay
    mu, n, beta = calibrate_params_for_target_count(events_per_day, n_target, beta, T)

    times = simulate_hawkes(mu, n, beta, T, seed=seed, use_tick=True)
    return times
