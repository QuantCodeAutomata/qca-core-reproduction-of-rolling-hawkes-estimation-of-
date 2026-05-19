"""
Rolling-window Hawkes estimation pipeline.

Creates overlapping windows of length T in {10, 20, 30} minutes with a
5-minute step across each accepted day. Within each window, uses only events
strictly inside the window, rebasing time to [0, T].

No pre-window excitation is imported (truncated-window approach).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.hawkes_exp_truncated import fit_hawkes_exp_multi_start
from src.models.timestamp_bootstrap import generate_bootstrap_realizations
from src.diagnostics.residual_diagnostics import (
    run_diagnostics_single,
    evaluate_window_adequacy,
)

logger = logging.getLogger(__name__)


def extract_window_events(
    day_times_seconds: np.ndarray,
    window_start_s: float,
    window_end_s: float,
) -> np.ndarray:
    """
    Extract events strictly inside a window and rebase to [0, T].

    Parameters
    ----------
    day_times_seconds : np.ndarray
        All event times for the day in seconds since session open.
    window_start_s : float
        Window start in seconds since session open.
    window_end_s : float
        Window end in seconds since session open.

    Returns
    -------
    np.ndarray
        Rebased event times in [0, T] where T = window_end_s - window_start_s.
    """
    mask = (day_times_seconds > window_start_s) & (day_times_seconds <= window_end_s)
    events = day_times_seconds[mask] - window_start_s
    return np.sort(events)


def estimate_window_bootstrap(
    window_times: np.ndarray,
    T_seconds: float,
    n_bootstrap: int = 50,
    seed: Optional[int] = None,
    n_starts: int = 3,
) -> List[Dict]:
    """
    Estimate Hawkes parameters for a window using bootstrap randomization.

    For each of n_bootstrap realizations:
    1. Randomize within-second timestamps
    2. Fit Hawkes MLE
    3. Compute KS diagnostic

    Parameters
    ----------
    window_times : np.ndarray
        Event times in [0, T_seconds] (may be second-rounded).
    T_seconds : float
        Window length in seconds.
    n_bootstrap : int
        Number of bootstrap realizations.
    seed : int, optional
        Random seed.
    n_starts : int
        Number of MLE starting points.

    Returns
    -------
    List[Dict]
        One dict per bootstrap realization with keys:
        bootstrap_id, mu_hat, n_hat, beta_hat, loglik, ks_stat, ks_pvalue,
        success, n_events
    """
    bootstrap_times = generate_bootstrap_realizations(
        window_times, n_bootstrap=n_bootstrap, seed=seed
    )

    results = []
    for b_id, b_times in enumerate(bootstrap_times):
        fit = fit_hawkes_exp_multi_start(b_times, T_seconds, n_starts=n_starts)
        ks_stat, ks_pval = np.nan, np.nan
        if fit["success"] and np.isfinite(fit["n_hat"]):
            ks_stat, ks_pval = run_diagnostics_single(
                b_times, T_seconds, fit["mu_hat"], fit["n_hat"], fit["beta_hat"]
            )
        results.append({
            "bootstrap_id": b_id,
            "mu_hat": fit["mu_hat"],
            "n_hat": fit["n_hat"],
            "beta_hat": fit["beta_hat"],
            "loglik": fit["loglik"],
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pval,
            "success": fit["success"],
            "n_events": fit["n_events"],
        })
    return results


def process_day_rolling(
    day_times_seconds: np.ndarray,
    trade_date: str,
    window_length_min: int = 10,
    step_min: int = 5,
    session_duration_s: float = 6.75 * 3600,  # 9:30-16:15 = 6h45m
    n_bootstrap: int = 50,
    base_seed: int = 42,
    n_starts: int = 3,
    min_events: int = 10,
) -> pd.DataFrame:
    """
    Process all rolling windows for a single trading day.

    Parameters
    ----------
    day_times_seconds : np.ndarray
        Event times in seconds since session open (9:30 NY).
    trade_date : str
        Date string for labeling.
    window_length_min : int
        Window length in minutes.
    step_min : int
        Step size in minutes.
    session_duration_s : float
        Total session duration in seconds.
    n_bootstrap : int
        Bootstrap realizations per window.
    base_seed : int
        Base random seed.
    n_starts : int
        MLE starting points.
    min_events : int
        Minimum events required to attempt estimation.

    Returns
    -------
    pd.DataFrame
        One row per (window, bootstrap) with all estimation results.
    """
    T_seconds = window_length_min * 60.0
    step_seconds = step_min * 60.0

    window_starts = np.arange(0, session_duration_s - T_seconds + 1, step_seconds)
    all_records = []

    for w_idx, ws in enumerate(window_starts):
        we = ws + T_seconds
        window_times = extract_window_events(day_times_seconds, ws, we)

        if len(window_times) < min_events:
            continue

        seed = base_seed + w_idx * 1000
        bootstrap_results = estimate_window_bootstrap(
            window_times, T_seconds, n_bootstrap=n_bootstrap, seed=seed, n_starts=n_starts
        )

        ks_pvalues = [r["ks_pvalue"] for r in bootstrap_results]
        accepted, max_ks_pval = evaluate_window_adequacy(ks_pvalues)

        for r in bootstrap_results:
            r.update({
                "trade_date": trade_date,
                "window_start_s": ws,
                "window_end_s": we,
                "window_start_min": ws / 60.0,
                "window_end_min": we / 60.0,
                "window_length_min": window_length_min,
                "event_count": len(window_times),
                "accepted_window": accepted,
                "max_ks_pvalue": max_ks_pval,
            })
        all_records.extend(bootstrap_results)

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def aggregate_window_estimates(
    results_df: pd.DataFrame,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """
    Aggregate bootstrap estimates per window to a single summary row.

    Uses median across bootstrap realizations as the central estimate
    (documented convention: paper is ambiguous, median chosen for robustness).

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw bootstrap-level results.
    accepted_only : bool
        If True, only include accepted windows.

    Returns
    -------
    pd.DataFrame
        One row per window with median estimates and quantile bands.
    """
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    if accepted_only:
        df = df[df["accepted_window"]].copy()

    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "trade_date", "window_start_s", "window_end_s",
        "window_length_min", "event_count", "accepted_window", "max_ks_pvalue"
    ]

    agg = df.groupby(group_cols).agg(
        n_hat_median=("n_hat", "median"),
        n_hat_mean=("n_hat", "mean"),
        n_hat_q10=("n_hat", lambda x: x.quantile(0.10)),
        n_hat_q90=("n_hat", lambda x: x.quantile(0.90)),
        mu_hat_median=("mu_hat", "median"),
        mu_hat_mean=("mu_hat", "mean"),
        beta_hat_median=("beta_hat", "median"),
        loglik_median=("loglik", "median"),
        n_bootstrap_success=("success", "sum"),
    ).reset_index()

    return agg


def build_two_month_aggregate(
    window_agg_df: pd.DataFrame,
    date_col: str = "trade_date",
    n_col: str = "n_hat_median",
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Build centered two-month aggregate of branching ratio estimates.

    For each month t, averages estimates from [t-1 month, t+1 month].

    Parameters
    ----------
    window_agg_df : pd.DataFrame
        Aggregated window estimates.
    date_col : str
        Date column.
    n_col : str
        Branching ratio column.
    freq : str
        Resampling frequency.

    Returns
    -------
    pd.DataFrame
        Monthly centered averages with columns: month, n_hat_2m_mean, n_hat_2m_median
    """
    if window_agg_df.empty or date_col not in window_agg_df.columns:
        return pd.DataFrame(columns=["month", "n_hat_2m_mean", "n_hat_2m_median",
                                     "n_hat_2m_q10", "n_hat_2m_q90", "n_windows"])
    df = window_agg_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Monthly grid
    monthly_dates = pd.date_range(
        df.index.min().to_period("M").to_timestamp(),
        df.index.max().to_period("M").to_timestamp(),
        freq=freq,
    )

    records = []
    for month_end in monthly_dates:
        month_start = month_end - pd.DateOffset(months=1)
        month_end_plus = month_end + pd.DateOffset(months=1)
        mask = (df.index >= month_start) & (df.index <= month_end_plus)
        subset = df.loc[mask, n_col].dropna()
        if len(subset) > 0:
            records.append({
                "month": month_end,
                "n_hat_2m_mean": subset.mean(),
                "n_hat_2m_median": subset.median(),
                "n_hat_2m_q10": subset.quantile(0.10),
                "n_hat_2m_q90": subset.quantile(0.90),
                "n_windows": len(subset),
            })

    return pd.DataFrame(records)
