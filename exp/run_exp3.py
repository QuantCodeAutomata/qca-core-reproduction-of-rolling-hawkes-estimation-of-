"""
Experiment 3: Reshuffled timing control.

Tests whether the Hawkes branching ratio n captures temporal self-excitation
rather than merely high trading intensity or volume.

Preserves each day's total event count while destroying within-day temporal
dependence by replacing event times with i.i.d. Uniform draws.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sim.hawkes_sim import simulate_hawkes, calibrate_params_for_target_count
from src.control.reshuffle_daily_timing import (
    extract_daily_event_counts,
    run_reshuffled_pipeline,
)
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

SESSION_DURATION_S = 6.75 * 3600.0


def generate_empirical_daily_counts(
    years: list = None,
    n_days_per_year: int = 10,
    base_seed: int = 42,
) -> dict:
    """
    Generate synthetic empirical daily event counts for 1998-2010.

    Returns dict mapping date_str -> event_count.
    """
    if years is None:
        years = list(range(1998, 2011))

    year_events = {
        1998: 150 * 40,  # ~150 events/10min × 40 windows/day
        1999: 180 * 40,
        2000: 200 * 40,
        2001: 250 * 40,
        2002: 300 * 40,
        2003: 350 * 40,
        2004: 450 * 40,
        2005: 550 * 40,
        2006: 650 * 40,
        2007: 750 * 40,
        2008: 890 * 40,
        2009: 600 * 40,
        2010: 350 * 40,
    }

    rng = np.random.default_rng(base_seed)
    daily_counts = {}

    for year in years:
        mean_count = year_events.get(year, 400 * 40)
        trading_days = pd.bdate_range(
            start=f"{year}-01-05",
            end=f"{year}-12-31",
            freq="B",
        )[:n_days_per_year]

        for day in trading_days:
            count = int(rng.poisson(mean_count))
            daily_counts[str(day.date())] = count

    return daily_counts


def run_empirical_pipeline(
    years: list = None,
    n_days_per_year: int = 10,
    window_length_min: int = 10,
    n_bootstrap: int = 50,
    n_starts: int = 3,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Run empirical Hawkes estimation pipeline on synthetic ES data.
    """
    if years is None:
        years = list(range(1998, 2011))

    from exp.run_exp1 import get_year_params

    all_results = []
    for year in years:
        n_target, events_per_10min = get_year_params(year)
        events_per_day = events_per_10min * int(SESSION_DURATION_S / 600)
        beta = 1.0 / 5.0
        mu, n, beta = calibrate_params_for_target_count(events_per_day, n_target, beta, SESSION_DURATION_S)

        trading_days = pd.bdate_range(
            start=f"{year}-01-05",
            end=f"{year}-12-31",
            freq="B",
        )[:n_days_per_year]

        for i, day in enumerate(trading_days):
            seed = base_seed + year * 1000 + i
            times = simulate_hawkes(mu, n, beta, SESSION_DURATION_S, seed=seed, use_tick=True)
            date_str = str(day.date())

            try:
                day_results = process_day_rolling(
                    day_times_seconds=times,
                    trade_date=date_str,
                    window_length_min=window_length_min,
                    step_min=5,
                    session_duration_s=SESSION_DURATION_S,
                    n_bootstrap=n_bootstrap,
                    base_seed=seed,
                    n_starts=n_starts,
                    min_events=10,
                )
                if not day_results.empty:
                    agg = aggregate_window_estimates(day_results, accepted_only=True)
                    if not agg.empty:
                        all_results.append(agg)
            except Exception as e:
                logger.warning(f"Empirical day {date_str} failed: {e}")

    if not all_results:
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def run_exp3(
    years: list = None,
    n_days_per_year: int = 10,
    window_length_min: int = 10,
    n_bootstrap: int = 50,
    n_starts: int = 3,
    base_seed: int = 42,
) -> tuple:
    """
    Run Experiment 3: Reshuffled timing control.

    Returns
    -------
    tuple
        (empirical_df, reshuffled_df)
    """
    if years is None:
        years = list(range(1998, 2011))

    logger.info("Running empirical pipeline...")
    empirical_df = run_empirical_pipeline(
        years=years,
        n_days_per_year=n_days_per_year,
        window_length_min=window_length_min,
        n_bootstrap=n_bootstrap,
        n_starts=n_starts,
        base_seed=base_seed,
    )
    logger.info(f"Empirical: {len(empirical_df)} window estimates")

    logger.info("Generating daily event counts for reshuffling...")
    daily_counts = generate_empirical_daily_counts(
        years=years,
        n_days_per_year=n_days_per_year,
        base_seed=base_seed + 99999,
    )
    logger.info(f"Total days for reshuffling: {len(daily_counts)}")

    logger.info("Running reshuffled pipeline...")
    reshuffled_raw = run_reshuffled_pipeline(
        daily_event_counts=daily_counts,
        window_length_min=window_length_min,
        step_min=5,
        n_bootstrap=n_bootstrap,
        session_duration_s=SESSION_DURATION_S,
        base_seed=base_seed + 12345,
        n_starts=n_starts,
        min_events=10,
    )

    reshuffled_df = pd.DataFrame()
    if not reshuffled_raw.empty:
        reshuffled_df = aggregate_window_estimates(reshuffled_raw, accepted_only=True)
        if not reshuffled_df.empty:
            reshuffled_df["trade_date"] = pd.to_datetime(reshuffled_df["trade_date"])
    logger.info(f"Reshuffled: {len(reshuffled_df)} window estimates")

    return empirical_df, reshuffled_df


def plot_empirical_vs_reshuffled(
    empirical_df: pd.DataFrame,
    reshuffled_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Plot empirical vs reshuffled branching ratio over time.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    if not empirical_df.empty:
        emp_2m = build_two_month_aggregate(empirical_df, n_col="n_hat_median")
        if not emp_2m.empty:
            ax.fill_between(
                emp_2m["month"],
                emp_2m["n_hat_2m_q10"],
                emp_2m["n_hat_2m_q90"],
                alpha=0.2,
                color="#1f77b4",
            )
            ax.plot(
                emp_2m["month"],
                emp_2m["n_hat_2m_mean"],
                color="#1f77b4",
                linewidth=2.5,
                label="Empirical n (2-month avg)",
            )

    if not reshuffled_df.empty:
        res_2m = build_two_month_aggregate(reshuffled_df, n_col="n_hat_median")
        if not res_2m.empty:
            ax.fill_between(
                res_2m["month"],
                res_2m["n_hat_2m_q10"],
                res_2m["n_hat_2m_q90"],
                alpha=0.2,
                color="#d62728",
            )
            ax.plot(
                res_2m["month"],
                res_2m["n_hat_2m_mean"],
                color="#d62728",
                linewidth=2.5,
                linestyle="--",
                label="Reshuffled n (2-month avg)",
            )

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Branching ratio n", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Empirical vs Reshuffled Branching Ratio (10-min windows)\n"
        "Reshuffling preserves daily counts but destroys temporal dependence",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")
    plt.close()


def compute_comparison_stats(
    empirical_df: pd.DataFrame,
    reshuffled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute comparison statistics between empirical and reshuffled n.
    """
    records = []

    for label, df in [("empirical", empirical_df), ("reshuffled", reshuffled_df)]:
        if df.empty:
            continue
        df = df.copy()
        df["year"] = df["trade_date"].dt.year

        for year, group in df.groupby("year"):
            n_vals = group["n_hat_median"].dropna()
            records.append({
                "series": label,
                "year": year,
                "n_mean": n_vals.mean(),
                "n_median": n_vals.median(),
                "n_std": n_vals.std(),
                "n_windows": len(n_vals),
            })

    return pd.DataFrame(records)


def main():
    logger.info("=" * 60)
    logger.info("Experiment 3: Reshuffled Timing Control")
    logger.info("=" * 60)

    empirical_df, reshuffled_df = run_exp3(
        years=list(range(1998, 2011)),
        n_days_per_year=10,
        window_length_min=10,
        n_bootstrap=50,
        n_starts=3,
        base_seed=42,
    )

    # Save results
    if not empirical_df.empty:
        empirical_df.to_parquet(DATA_DIR / "exp3_empirical.parquet", index=False)
    if not reshuffled_df.empty:
        reshuffled_df.to_parquet(DATA_DIR / "exp3_reshuffled.parquet", index=False)

    # Comparison stats
    stats = compute_comparison_stats(empirical_df, reshuffled_df)
    stats.to_csv(RESULTS_DIR / "exp3_comparison_stats.csv", index=False)
    logger.info("\nComparison Statistics:")
    logger.info(stats.to_string())

    # Plot
    plot_empirical_vs_reshuffled(
        empirical_df,
        reshuffled_df,
        save_path=str(RESULTS_DIR / "exp3_empirical_vs_reshuffled.png"),
    )

    logger.info("Experiment 3 complete.")
    return empirical_df, reshuffled_df


if __name__ == "__main__":
    main()
