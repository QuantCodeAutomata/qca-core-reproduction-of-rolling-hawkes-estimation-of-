"""
Experiment 4: Flash-crash case study.

Compares April 27, 2010 (large exogenous move) vs May 6, 2010 (flash crash).

Expected results:
- April 27: large price move, elevated activity, n stays within normal range
- May 6: sharp rise in n toward ~0.95, breaching previous-day 95% band

Uses synthetic data calibrated to match the paper's described behavior.
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
from src.case_study.flash_crash_compare import (
    compute_rolling_estimates_for_day,
    build_benchmark_bands,
    SESSION_DURATION_S,
)
from src.rolling.window_pipeline import aggregate_window_estimates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

WINDOW_MIN = 10
STEP_MIN = 5
N_BOOTSTRAP = 50
N_STARTS = 3
BETA_INV_S = 5.0


def simulate_case_study_day(
    n_profile: np.ndarray,
    window_starts_s: np.ndarray,
    window_length_s: float,
    events_per_window: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate a full day with time-varying branching ratio.

    For each window, simulate events with the specified n value.
    Concatenate all windows to form the full day.

    Parameters
    ----------
    n_profile : np.ndarray
        Branching ratio for each window.
    window_starts_s : np.ndarray
        Window start times in seconds.
    window_length_s : float
        Window length in seconds.
    events_per_window : int
        Target events per window.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Full-day event times in seconds since session open.
    """
    beta = 1.0 / BETA_INV_S
    all_times = []
    rng = np.random.default_rng(seed)

    for i, (ws, n_val) in enumerate(zip(window_starts_s, n_profile)):
        n_val = np.clip(n_val, 0.01, 0.99)
        mu, _, _ = calibrate_params_for_target_count(events_per_window, n_val, beta, window_length_s)
        child_seed = int(rng.integers(0, 2**31))
        times = simulate_hawkes(mu, n_val, beta, window_length_s, seed=child_seed, use_tick=True)
        # Offset to window start
        times_offset = times + ws
        all_times.append(times_offset)

    if not all_times:
        return np.array([])

    return np.sort(np.concatenate(all_times))


def build_case_study_scenarios() -> dict:
    """
    Build synthetic scenarios for April 27 and May 6, 2010.

    April 27, 2010 (exogenous shock):
        - High event rate (elevated activity)
        - n stays near normal range (~0.75)
        - Large price move but no n spike

    May 6, 2010 (flash crash):
        - Normal activity early, then spike
        - n rises sharply to ~0.95 during crash window (13:00-14:30)
        - n returns to normal after crash

    Returns
    -------
    dict
        Scenarios with event times and metadata.
    """
    T_window = WINDOW_MIN * 60.0
    window_starts = np.arange(0, SESSION_DURATION_S - T_window + 1, STEP_MIN * 60.0)
    n_windows = len(window_starts)

    # April 26 (previous day for April 27 benchmark)
    n_apr26 = np.full(n_windows, 0.72)  # Normal day
    events_apr26 = 350  # ~350 events/10min in 2010
    times_apr26 = simulate_case_study_day(n_apr26, window_starts, T_window, events_apr26, seed=100)

    # April 27 (exogenous shock day)
    # Elevated activity but n stays normal
    n_apr27 = np.full(n_windows, 0.73)  # Slightly elevated but normal
    events_apr27 = 600  # Higher activity due to shock
    times_apr27 = simulate_case_study_day(n_apr27, window_starts, T_window, events_apr27, seed=200)

    # May 5 (previous day for May 6 benchmark)
    n_may5 = np.full(n_windows, 0.72)
    events_may5 = 350
    times_may5 = simulate_case_study_day(n_may5, window_starts, T_window, events_may5, seed=300)

    # May 6 (flash crash day)
    # n rises sharply during crash window (roughly 13:00-14:30 = 210-300 min from open)
    n_may6 = np.full(n_windows, 0.72)
    crash_start_min = 210  # ~13:00 NY
    crash_end_min = 300    # ~14:30 NY
    for i, ws in enumerate(window_starts):
        ws_min = ws / 60.0
        if crash_start_min <= ws_min <= crash_end_min:
            # Ramp up to 0.95 at peak
            progress = (ws_min - crash_start_min) / (crash_end_min - crash_start_min)
            n_may6[i] = 0.72 + (0.95 - 0.72) * np.sin(np.pi * progress)

    events_may6_normal = 400
    events_may6_crash = 1200  # Spike in activity during crash
    # Build event counts per window
    events_may6_profile = np.where(
        (window_starts / 60.0 >= crash_start_min) & (window_starts / 60.0 <= crash_end_min),
        events_may6_crash,
        events_may6_normal,
    )

    # Simulate May 6 with varying event counts
    beta = 1.0 / BETA_INV_S
    all_times_may6 = []
    rng = np.random.default_rng(400)
    for i, (ws, n_val, ev_count) in enumerate(zip(window_starts, n_may6, events_may6_profile)):
        n_val = np.clip(n_val, 0.01, 0.99)
        mu, _, _ = calibrate_params_for_target_count(int(ev_count), n_val, beta, T_window)
        child_seed = int(rng.integers(0, 2**31))
        times = simulate_hawkes(mu, n_val, beta, T_window, seed=child_seed, use_tick=True)
        all_times_may6.append(times + ws)
    times_may6 = np.sort(np.concatenate(all_times_may6)) if all_times_may6 else np.array([])

    return {
        "apr26": {"times": times_apr26, "date": "2010-04-26", "n_true": n_apr26},
        "apr27": {"times": times_apr27, "date": "2010-04-27", "n_true": n_apr27},
        "may5":  {"times": times_may5,  "date": "2010-05-05", "n_true": n_may5},
        "may6":  {"times": times_may6,  "date": "2010-05-06", "n_true": n_may6},
        "window_starts": window_starts,
    }


def run_exp4(
    n_bootstrap: int = N_BOOTSTRAP,
    n_starts: int = N_STARTS,
) -> dict:
    """
    Run Experiment 4: Flash-crash case study.

    Returns
    -------
    dict
        Results for each case-study day.
    """
    scenarios = build_case_study_scenarios()
    window_starts = scenarios["window_starts"]

    results = {}
    for key in ["apr26", "apr27", "may5", "may6"]:
        sc = scenarios[key]
        logger.info(f"Processing {sc['date']}...")
        try:
            estimates = compute_rolling_estimates_for_day(
                day_times_seconds=sc["times"],
                trade_date=sc["date"],
                window_length_min=WINDOW_MIN,
                step_min=STEP_MIN,
                n_bootstrap=n_bootstrap,
                base_seed=hash(sc["date"]) % (2**31),
                n_starts=n_starts,
                min_events=5,
            )
            results[key] = estimates
            logger.info(f"  {sc['date']}: {len(estimates)} windows estimated")
        except Exception as e:
            logger.error(f"  {sc['date']} failed: {e}")
            results[key] = pd.DataFrame()

    # Build benchmark bands
    results["apr27_bands"] = build_benchmark_bands(
        results.get("apr26", pd.DataFrame()),
        n_col="n_hat_median",
        event_rate_col="event_count",
        window_length_min=WINDOW_MIN,
    )
    results["may6_bands"] = build_benchmark_bands(
        results.get("may5", pd.DataFrame()),
        n_col="n_hat_median",
        event_rate_col="event_count",
        window_length_min=WINDOW_MIN,
    )

    return results, scenarios


def build_synthetic_price_series(
    times: np.ndarray,
    n_profile: np.ndarray,
    window_starts: np.ndarray,
    window_length_s: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic minute-level price series for visualization.

    Price changes are driven by event intensity (more events = more volatility).
    """
    rng = np.random.default_rng(seed)
    n_minutes = int(SESSION_DURATION_S / 60)
    prices = np.zeros(n_minutes)
    prices[0] = 1200.0  # Approximate ES price in 2010

    for m in range(1, n_minutes):
        t_start = m * 60.0
        t_end = (m + 1) * 60.0
        n_events_min = np.sum((times >= t_start) & (times < t_end))
        vol = 0.1 * np.sqrt(max(n_events_min, 1))
        prices[m] = prices[m - 1] + rng.normal(0, vol)

    minutes = np.arange(n_minutes)
    return pd.DataFrame({"minute": minutes, "price": prices})


def plot_case_study(
    results: dict,
    scenarios: dict,
    case_key: str,
    prev_key: str,
    bands_key: str,
    title: str,
    save_path: str = None,
) -> None:
    """
    Plot case-study visualization with price, volume, event rate, and n.
    """
    case_est = results.get(case_key, pd.DataFrame())
    bands = results.get(bands_key, pd.DataFrame())
    sc = scenarios[case_key]
    window_starts = scenarios["window_starts"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Minute axis
    minutes = np.arange(int(SESSION_DURATION_S / 60))

    # 1. Price
    ax1 = axes[0]
    price_df = build_synthetic_price_series(
        sc["times"], sc["n_true"], window_starts, WINDOW_MIN * 60.0, seed=42
    )
    ax1.plot(price_df["minute"], price_df["price"], color="#1f77b4", linewidth=1.5)
    ax1.set_ylabel("Mid-price (synthetic)", fontsize=10)
    ax1.set_title(f"{title} — Price", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. Event rate (events per 10-min window)
    ax2 = axes[1]
    if not case_est.empty:
        ws_min = case_est["window_start_s"] / 60.0
        event_rate = case_est["event_count"] / WINDOW_MIN
        ax2.bar(ws_min, event_rate, width=STEP_MIN * 0.8, color="#ff7f0e", alpha=0.7,
                label="Event rate (events/min)")
        if not bands.empty and "rate_lower" in bands.columns:
            ax2.fill_between(
                bands["window_start_min"],
                bands["rate_lower"],
                bands["rate_upper"],
                alpha=0.3,
                color="gray",
                label="Prev-day 5%-95% band",
            )
    ax2.set_ylabel("Event rate (events/min)", fontsize=10)
    ax2.set_title("Event Rate", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Branching ratio n
    ax3 = axes[2]
    if not case_est.empty:
        ws_min = case_est["window_start_s"] / 60.0
        n_vals = case_est["n_hat_median"]
        ax3.plot(ws_min, n_vals, color="#2ca02c", linewidth=2, marker="o",
                 markersize=3, label="Estimated n")
        if not bands.empty:
            ax3.fill_between(
                bands["window_start_min"],
                bands["n_lower"],
                bands["n_upper"],
                alpha=0.3,
                color="gray",
                label="Prev-day 5%-95% band",
            )
    # True n profile
    ax3.plot(
        window_starts / 60.0,
        sc["n_true"],
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label="True n (synthetic)",
    )
    ax3.axhline(0.9, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax3.set_ylabel("Branching ratio n", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Hawkes Branching Ratio n", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Distance to criticality (1 - n)
    ax4 = axes[3]
    if not case_est.empty:
        ws_min = case_est["window_start_s"] / 60.0
        dist_crit = 1.0 - case_est["n_hat_median"]
        ax4.plot(ws_min, dist_crit, color="#9467bd", linewidth=2, label="1 - n")
        ax4.axhline(0.1, color="red", linestyle=":", alpha=0.5, linewidth=1,
                    label="Criticality threshold (n=0.9)")
    ax4.set_ylabel("Distance to criticality (1-n)", fontsize=10)
    ax4.set_xlabel("Minutes since session open (9:30 NY)", fontsize=10)
    ax4.set_title("Distance to Criticality", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved case study plot to {save_path}")
    plt.close()


def plot_comparison(
    results: dict,
    scenarios: dict,
    save_path: str = None,
) -> None:
    """
    Side-by-side comparison of April 27 vs May 6.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, (case_key, bands_key, date_label) in enumerate([
        ("apr27", "apr27_bands", "April 27, 2010 (Exogenous)"),
        ("may6", "may6_bands", "May 6, 2010 (Flash Crash)"),
    ]):
        case_est = results.get(case_key, pd.DataFrame())
        bands = results.get(bands_key, pd.DataFrame())
        sc = scenarios[case_key]
        window_starts = scenarios["window_starts"]

        # Event rate
        ax_rate = axes[0, col]
        if not case_est.empty:
            ws_min = case_est["window_start_s"] / 60.0
            event_rate = case_est["event_count"] / WINDOW_MIN
            ax_rate.bar(ws_min, event_rate, width=STEP_MIN * 0.8,
                        color="#ff7f0e", alpha=0.7, label="Event rate")
            if not bands.empty and "rate_lower" in bands.columns:
                ax_rate.fill_between(
                    bands["window_start_min"],
                    bands["rate_lower"],
                    bands["rate_upper"],
                    alpha=0.3, color="gray", label="Prev-day 5%-95%",
                )
        ax_rate.set_ylabel("Events/min", fontsize=10)
        ax_rate.set_title(f"{date_label}\nEvent Rate", fontsize=11)
        ax_rate.legend(fontsize=8)
        ax_rate.grid(True, alpha=0.3)

        # Branching ratio n
        ax_n = axes[1, col]
        if not case_est.empty:
            ws_min = case_est["window_start_s"] / 60.0
            n_vals = case_est["n_hat_median"]
            ax_n.plot(ws_min, n_vals, color="#2ca02c", linewidth=2,
                      marker="o", markersize=3, label="Estimated n")
            if not bands.empty:
                ax_n.fill_between(
                    bands["window_start_min"],
                    bands["n_lower"],
                    bands["n_upper"],
                    alpha=0.3, color="gray", label="Prev-day 5%-95%",
                )
        ax_n.plot(
            window_starts / 60.0, sc["n_true"],
            color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="True n",
        )
        ax_n.axhline(0.9, color="red", linestyle=":", alpha=0.5, linewidth=1)
        ax_n.set_ylabel("Branching ratio n", fontsize=10)
        ax_n.set_xlabel("Minutes since open", fontsize=10)
        ax_n.set_title(f"Branching Ratio n", fontsize=11)
        ax_n.set_ylim(0, 1.05)
        ax_n.legend(fontsize=8)
        ax_n.grid(True, alpha=0.3)

    plt.suptitle(
        "Flash Crash Case Study: April 27 vs May 6, 2010\n"
        "Exogenous shock vs Endogenous flash crash",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {save_path}")
    plt.close()


def compute_case_study_stats(results: dict, scenarios: dict) -> pd.DataFrame:
    """
    Compute summary statistics for the case study.
    """
    records = []
    for key, date_label in [
        ("apr26", "2010-04-26 (prev day)"),
        ("apr27", "2010-04-27 (exogenous)"),
        ("may5",  "2010-05-05 (prev day)"),
        ("may6",  "2010-05-06 (flash crash)"),
    ]:
        est = results.get(key, pd.DataFrame())
        if est.empty:
            continue
        n_vals = est["n_hat_median"].dropna()
        sc = scenarios.get(key, {})
        n_true = sc.get("n_true", np.array([]))

        records.append({
            "date": date_label,
            "n_windows": len(n_vals),
            "n_hat_mean": round(n_vals.mean(), 3),
            "n_hat_median": round(n_vals.median(), 3),
            "n_hat_max": round(n_vals.max(), 3),
            "n_hat_q95": round(n_vals.quantile(0.95), 3),
            "n_true_mean": round(float(np.mean(n_true)), 3) if len(n_true) > 0 else np.nan,
            "n_true_max": round(float(np.max(n_true)), 3) if len(n_true) > 0 else np.nan,
            "event_count_total": int(est["event_count"].sum()),
        })

    return pd.DataFrame(records)


def main():
    logger.info("=" * 60)
    logger.info("Experiment 4: Flash Crash Case Study")
    logger.info("=" * 60)

    results, scenarios = run_exp4(
        n_bootstrap=N_BOOTSTRAP,
        n_starts=N_STARTS,
    )

    # Summary stats
    stats = compute_case_study_stats(results, scenarios)
    logger.info("\nCase Study Statistics:")
    logger.info(stats.to_string())
    stats.to_csv(RESULTS_DIR / "exp4_case_study_stats.csv", index=False)

    # Save estimates
    for key in ["apr26", "apr27", "may5", "may6"]:
        est = results.get(key, pd.DataFrame())
        if not est.empty:
            est.to_parquet(DATA_DIR / f"exp4_{key}_estimates.parquet", index=False)

    # Individual day plots
    for case_key, prev_key, bands_key, title in [
        ("apr27", "apr26", "apr27_bands", "April 27, 2010 — Exogenous Market Move"),
        ("may6",  "may5",  "may6_bands",  "May 6, 2010 — Flash Crash"),
    ]:
        plot_case_study(
            results, scenarios,
            case_key=case_key,
            prev_key=prev_key,
            bands_key=bands_key,
            title=title,
            save_path=str(RESULTS_DIR / f"exp4_{case_key}.png"),
        )

    # Comparison plot
    plot_comparison(
        results, scenarios,
        save_path=str(RESULTS_DIR / "exp4_comparison.png"),
    )

    logger.info("Experiment 4 complete.")
    return results, scenarios


if __name__ == "__main__":
    main()
