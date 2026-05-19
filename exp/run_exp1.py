"""
Experiment 1: Core rolling Hawkes estimation of market endogeneity.

Reproduces the paper's main empirical result that the Hawkes branching ratio n,
estimated on rolling windows of ES mid-price-change events, rises materially
from 1998 to 2010.

Since raw CME ES data is not available, this experiment uses synthetic data
calibrated to match the paper's reported statistics:
- ~150 events per 10-min window in 1998
- ~890 events per 10-min window in 2008
- ~350 events per 10-min window in 2010

The synthetic data simulates the secular rise in n from ~0.3 (1998) to ~0.7-0.8 (2007+).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sim.hawkes_sim import simulate_hawkes, calibrate_params_for_target_count
from src.rolling.window_pipeline import (
    process_day_rolling,
    aggregate_window_estimates,
    build_two_month_aggregate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Session parameters
SESSION_DURATION_S = 6.75 * 3600.0  # 9:30-16:15 = 24300s
BETA_INV_S = 5.0  # 5-second characteristic decay


def get_year_params(year: int) -> tuple:
    """
    Return (n_target, events_per_10min) calibrated to paper's reported statistics.

    Paper benchmarks:
        1998: ~150 events/10min, n ~0.3
        2002: ~300 events/10min, n ~0.6
        2006: ~600 events/10min, n ~0.7
        2008: ~890 events/10min, n ~0.75
        2010: ~350 events/10min, n ~0.75
    """
    # Linear interpolation of n over years
    year_n = {
        1998: 0.30,
        1999: 0.35,
        2000: 0.38,
        2001: 0.50,
        2002: 0.58,
        2003: 0.62,
        2004: 0.65,
        2005: 0.68,
        2006: 0.70,
        2007: 0.72,
        2008: 0.75,
        2009: 0.76,
        2010: 0.75,
    }
    year_events = {
        1998: 150,
        1999: 180,
        2000: 200,
        2001: 250,
        2002: 300,
        2003: 350,
        2004: 450,
        2005: 550,
        2006: 650,
        2007: 750,
        2008: 890,
        2009: 600,
        2010: 350,
    }
    n = year_n.get(year, 0.5)
    events = year_events.get(year, 400)
    return n, events


def simulate_year_days(
    year: int,
    n_days: int = 20,
    base_seed: int = 0,
) -> list:
    """
    Simulate n_days of ES mid-price-change events for a given year.

    Returns list of (date_str, times_seconds) tuples.
    """
    n_target, events_per_10min = get_year_params(year)
    events_per_day = events_per_10min * int(SESSION_DURATION_S / 600)  # scale to full day

    beta = 1.0 / BETA_INV_S
    T = SESSION_DURATION_S
    mu, n, beta = calibrate_params_for_target_count(events_per_day, n_target, beta, T)

    trading_days = pd.bdate_range(
        start=f"{year}-01-05",
        end=f"{year}-12-31",
        freq="B",
    )[:n_days]

    days_data = []
    for i, day in enumerate(trading_days):
        seed = base_seed + year * 1000 + i
        times = simulate_hawkes(mu, n, beta, T, seed=seed, use_tick=True)
        days_data.append((str(day.date()), times))

    return days_data


def run_exp1(
    window_lengths: list = [10, 20, 30],
    n_days_per_year: int = 15,
    n_bootstrap: int = 50,
    years: list = None,
    n_starts: int = 3,
) -> dict:
    """
    Run Experiment 1: rolling Hawkes estimation across 1998-2010.

    Parameters
    ----------
    window_lengths : list
        Window lengths in minutes.
    n_days_per_year : int
        Number of synthetic days per year.
    n_bootstrap : int
        Bootstrap realizations per window.
    years : list, optional
        Years to process. Defaults to 1998-2010.
    n_starts : int
        MLE starting points.

    Returns
    -------
    dict
        Results keyed by window length.
    """
    if years is None:
        years = list(range(1998, 2011))

    all_results = {wl: [] for wl in window_lengths}
    window_counts = {wl: 0 for wl in window_lengths}
    rejection_counts = {wl: 0 for wl in window_lengths}

    for year in years:
        logger.info(f"Processing year {year}")
        days_data = simulate_year_days(year, n_days=n_days_per_year, base_seed=year * 100)

        for date_str, day_times in days_data:
            for wl in window_lengths:
                try:
                    day_results = process_day_rolling(
                        day_times_seconds=day_times,
                        trade_date=date_str,
                        window_length_min=wl,
                        step_min=5,
                        session_duration_s=SESSION_DURATION_S,
                        n_bootstrap=n_bootstrap,
                        base_seed=hash(date_str) % (2**31),
                        n_starts=n_starts,
                        min_events=10,
                    )
                    if not day_results.empty:
                        # Count windows and rejections
                        unique_windows = day_results.drop_duplicates(
                            subset=["window_start_s", "window_end_s"]
                        )
                        window_counts[wl] += len(unique_windows)
                        rejection_counts[wl] += (~unique_windows["accepted_window"]).sum()

                        agg = aggregate_window_estimates(day_results, accepted_only=True)
                        if not agg.empty:
                            all_results[wl].append(agg)
                except Exception as e:
                    logger.warning(f"Day {date_str} wl={wl} failed: {e}")

    # Combine results
    combined = {}
    for wl in window_lengths:
        if all_results[wl]:
            df = pd.concat(all_results[wl], ignore_index=True)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            combined[wl] = df
            n_windows = window_counts[wl]
            n_rejected = rejection_counts[wl]
            rej_rate = n_rejected / n_windows * 100 if n_windows > 0 else 0
            logger.info(
                f"Window {wl}min: {n_windows} windows, "
                f"{n_rejected} rejected ({rej_rate:.2f}%)"
            )
        else:
            combined[wl] = pd.DataFrame()

    return combined, window_counts, rejection_counts


def plot_long_run_n(
    combined: dict,
    window_lengths: list = [10, 20, 30],
    save_path: str = None,
) -> None:
    """
    Plot long-run centered two-month average of branching ratio n.
    """
    fig, axes = plt.subplots(len(window_lengths), 1, figsize=(14, 4 * len(window_lengths)),
                              sharex=True)
    if len(window_lengths) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, wl, color in zip(axes, window_lengths, colors):
        df = combined.get(wl, pd.DataFrame())
        if df.empty:
            ax.set_title(f"{wl}-min windows (no data)")
            continue

        two_month = build_two_month_aggregate(df, n_col="n_hat_median")
        if two_month.empty:
            continue

        ax.fill_between(
            two_month["month"],
            two_month["n_hat_2m_q10"],
            two_month["n_hat_2m_q90"],
            alpha=0.3,
            color=color,
            label="10%-90% band",
        )
        ax.plot(
            two_month["month"],
            two_month["n_hat_2m_mean"],
            color=color,
            linewidth=2,
            label=f"{wl}-min mean",
        )
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylabel("Branching ratio n", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f"Rolling {wl}-min Hawkes branching ratio (centered 2-month avg)", fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Year", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved long-run n plot to {save_path}")
    plt.close()


def plot_intraday_profile(
    combined: dict,
    target_month: str = "2009-03",
    window_length: int = 10,
    save_path: str = None,
) -> None:
    """
    Plot intraday profiles of mu and n for a target month.
    """
    df = combined.get(window_length, pd.DataFrame())
    if df.empty:
        logger.warning("No data for intraday profile")
        return

    # Filter to target month
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    mask = df["trade_date"].dt.to_period("M").astype(str) == target_month
    month_df = df[mask].copy()

    if month_df.empty:
        logger.warning(f"No data for month {target_month}")
        return

    # Group by window start time
    month_df["window_start_min_of_day"] = month_df["window_start_s"] / 60.0

    intraday_n = month_df.groupby("window_start_min_of_day")["n_hat_median"].agg(
        ["mean", "median", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]
    )
    intraday_n.columns = ["mean", "median", "q10", "q90"]

    intraday_mu = month_df.groupby("window_start_min_of_day")["mu_hat_median"].agg(
        ["mean", "median"]
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # n intraday
    ax1.fill_between(
        intraday_n.index,
        intraday_n["q10"],
        intraday_n["q90"],
        alpha=0.3,
        color="#1f77b4",
        label="10%-90% band",
    )
    ax1.plot(intraday_n.index, intraday_n["mean"], color="#1f77b4", linewidth=2, label="Mean n")
    ax1.set_ylabel("Branching ratio n", fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"Intraday profile of n — {target_month} ({window_length}-min windows)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # mu intraday
    ax2.plot(intraday_mu.index, intraday_mu["mean"], color="#ff7f0e", linewidth=2, label="Mean mu")
    ax2.set_ylabel("Background intensity mu (events/s)", fontsize=11)
    ax2.set_xlabel("Minutes since session open", fontsize=11)
    ax2.set_title(f"Intraday profile of mu — {target_month} ({window_length}-min windows)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved intraday profile to {save_path}")
    plt.close()


def compute_summary_stats(
    combined: dict,
    window_counts: dict,
    rejection_counts: dict,
) -> pd.DataFrame:
    """
    Compute summary statistics for the experiment.
    """
    records = []
    for wl in sorted(combined.keys()):
        df = combined.get(wl, pd.DataFrame())
        n_windows = window_counts.get(wl, 0)
        n_rejected = rejection_counts.get(wl, 0)
        rej_rate = n_rejected / n_windows * 100 if n_windows > 0 else 0

        if not df.empty:
            # Early period (1998-2000)
            early = df[df["trade_date"].dt.year <= 2000]["n_hat_median"]
            # Late period (2007-2010)
            late = df[df["trade_date"].dt.year >= 2007]["n_hat_median"]

            records.append({
                "window_min": wl,
                "n_windows_total": n_windows,
                "n_rejected": n_rejected,
                "rejection_rate_pct": round(rej_rate, 2),
                "n_early_mean": round(early.mean(), 3) if len(early) > 0 else np.nan,
                "n_late_mean": round(late.mean(), 3) if len(late) > 0 else np.nan,
                "n_overall_mean": round(df["n_hat_median"].mean(), 3),
                "n_overall_median": round(df["n_hat_median"].median(), 3),
            })
        else:
            records.append({
                "window_min": wl,
                "n_windows_total": n_windows,
                "n_rejected": n_rejected,
                "rejection_rate_pct": round(rej_rate, 2),
            })

    return pd.DataFrame(records)


def main():
    logger.info("=" * 60)
    logger.info("Experiment 1: Core Rolling Hawkes Estimation")
    logger.info("=" * 60)

    # Run with reduced parameters for feasibility
    combined, window_counts, rejection_counts = run_exp1(
        window_lengths=[10, 20, 30],
        n_days_per_year=10,   # 10 days per year × 13 years = 130 days
        n_bootstrap=50,
        years=list(range(1998, 2011)),
        n_starts=3,
    )

    # Summary statistics
    summary = compute_summary_stats(combined, window_counts, rejection_counts)
    logger.info("\nSummary Statistics:")
    logger.info(summary.to_string())
    summary.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)

    # Save raw results
    for wl, df in combined.items():
        if not df.empty:
            df.to_parquet(DATA_DIR / f"exp1_results_wl{wl}.parquet", index=False)

    # Plots
    plot_long_run_n(
        combined,
        window_lengths=[10, 20, 30],
        save_path=str(RESULTS_DIR / "exp1_long_run_n.png"),
    )

    # Intraday profile for 2009 (using 10-min windows)
    plot_intraday_profile(
        combined,
        target_month="2009-01",
        window_length=10,
        save_path=str(RESULTS_DIR / "exp1_intraday_profile.png"),
    )

    logger.info("Experiment 1 complete.")
    return combined, summary


if __name__ == "__main__":
    main()
