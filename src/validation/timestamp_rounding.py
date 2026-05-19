"""
Synthetic validation of within-second timestamp randomization.

Demonstrates that rounding event times to seconds and redistributing them
uniformly within each second produces only modest parameter-estimation error.

Targets the paper's reported approximate standard deviations:
    mu: ~1.4%, n: ~0.6%, beta: ~3.5%
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.sim.hawkes_sim import simulate_hawkes, calibrate_params_for_target_count
from src.models.hawkes_exp_truncated import fit_hawkes_exp_multi_start
from src.models.timestamp_bootstrap import (
    round_to_seconds,
    generate_bootstrap_realizations,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter scenarios (representative of ES empirical windows)
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (label, n, beta_inv_s, target_count_per_10min)
    ("low_endogeneity",    0.3, 5.0,  150),   # ~1998 levels
    ("medium_endogeneity", 0.6, 5.0,  400),   # ~2005 levels
    ("high_endogeneity",   0.8, 5.0,  890),   # ~2008 levels
]


def run_single_mc_replication(
    mu: float,
    n_true: float,
    beta_true: float,
    T: float,
    n_bootstrap: int = 50,
    seed: int = 0,
    n_starts: int = 3,
) -> Dict:
    """
    Run a single Monte Carlo replication of the timestamp-rounding validation.

    Steps:
    1. Simulate exact continuous-time events
    2. Fit to exact times (baseline)
    3. Round to seconds
    4. Generate n_bootstrap randomized reconstructions
    5. Fit to each reconstruction
    6. Compute errors

    Parameters
    ----------
    mu : float
        True background intensity.
    n_true : float
        True branching ratio.
    beta_true : float
        True decay rate.
    T : float
        Window length in seconds.
    n_bootstrap : int
        Bootstrap realizations.
    seed : int
        Random seed.
    n_starts : int
        MLE starting points.

    Returns
    -------
    Dict
        Replication results with exact fit, bootstrap fits, and errors.
    """
    # Step 1: Simulate exact times
    exact_times = simulate_hawkes(mu, n_true, beta_true, T, seed=seed, use_tick=True)

    # Step 2: Fit to exact times
    exact_fit = fit_hawkes_exp_multi_start(exact_times, T, n_starts=n_starts)

    # Step 3: Round to seconds
    rounded_times = round_to_seconds(exact_times)

    # Step 4: Generate bootstrap reconstructions
    bootstrap_times_list = generate_bootstrap_realizations(
        rounded_times, n_bootstrap=n_bootstrap, seed=seed + 10000
    )

    # Step 5: Fit to each reconstruction
    bootstrap_fits = []
    for b_times in bootstrap_times_list:
        fit = fit_hawkes_exp_multi_start(b_times, T, n_starts=n_starts)
        bootstrap_fits.append(fit)

    # Step 6: Aggregate bootstrap estimates
    n_hats = np.array([f["n_hat"] for f in bootstrap_fits if f["success"] and np.isfinite(f["n_hat"])])
    mu_hats = np.array([f["mu_hat"] for f in bootstrap_fits if f["success"] and np.isfinite(f["mu_hat"])])
    beta_hats = np.array([f["beta_hat"] for f in bootstrap_fits if f["success"] and np.isfinite(f["beta_hat"])])

    result = {
        "seed": seed,
        "n_events_exact": len(exact_times),
        "n_events_rounded": len(rounded_times),
        # True parameters
        "mu_true": mu,
        "n_true": n_true,
        "beta_true": beta_true,
        # Exact-time fit
        "mu_exact": exact_fit["mu_hat"],
        "n_exact": exact_fit["n_hat"],
        "beta_exact": exact_fit["beta_hat"],
        "exact_success": exact_fit["success"],
        # Bootstrap aggregate (median)
        "mu_boot_median": float(np.median(mu_hats)) if len(mu_hats) > 0 else np.nan,
        "n_boot_median": float(np.median(n_hats)) if len(n_hats) > 0 else np.nan,
        "beta_boot_median": float(np.median(beta_hats)) if len(beta_hats) > 0 else np.nan,
        "n_boot_success": len(n_hats),
        # Bootstrap std
        "mu_boot_std": float(np.std(mu_hats)) if len(mu_hats) > 1 else np.nan,
        "n_boot_std": float(np.std(n_hats)) if len(n_hats) > 1 else np.nan,
        "beta_boot_std": float(np.std(beta_hats)) if len(beta_hats) > 1 else np.nan,
    }

    # Relative errors vs true parameters
    if np.isfinite(result["n_boot_median"]) and n_true > 0:
        result["n_rel_err_vs_true"] = (result["n_boot_median"] - n_true) / n_true
    else:
        result["n_rel_err_vs_true"] = np.nan

    if np.isfinite(result["mu_boot_median"]) and mu > 0:
        result["mu_rel_err_vs_true"] = (result["mu_boot_median"] - mu) / mu
    else:
        result["mu_rel_err_vs_true"] = np.nan

    if np.isfinite(result["beta_boot_median"]) and beta_true > 0:
        result["beta_rel_err_vs_true"] = (result["beta_boot_median"] - beta_true) / beta_true
    else:
        result["beta_rel_err_vs_true"] = np.nan

    return result


def run_monte_carlo_validation(
    scenario_label: str,
    n_true: float,
    beta_inv_s: float,
    target_count: float,
    T: float = 600.0,  # 10 minutes
    n_replications: int = 100,
    n_bootstrap: int = 50,
    base_seed: int = 42,
    n_starts: int = 3,
) -> pd.DataFrame:
    """
    Run full Monte Carlo validation for a single parameter scenario.

    Parameters
    ----------
    scenario_label : str
        Scenario name for labeling.
    n_true : float
        True branching ratio.
    beta_inv_s : float
        Characteristic decay time in seconds (1/beta).
    target_count : float
        Target expected event count per window.
    T : float
        Window length in seconds.
    n_replications : int
        Number of Monte Carlo replications.
    n_bootstrap : int
        Bootstrap realizations per replication.
    base_seed : int
        Base random seed.
    n_starts : int
        MLE starting points.

    Returns
    -------
    pd.DataFrame
        One row per replication with all estimation results.
    """
    beta_true = 1.0 / beta_inv_s
    mu, _, _ = calibrate_params_for_target_count(target_count, n_true, beta_true, T)

    logger.info(
        f"Scenario '{scenario_label}': mu={mu:.4f}, n={n_true}, beta={beta_true:.4f}, "
        f"T={T}s, expected_count={target_count}"
    )

    records = []
    for rep in range(n_replications):
        seed = base_seed + rep * 137
        try:
            result = run_single_mc_replication(
                mu, n_true, beta_true, T,
                n_bootstrap=n_bootstrap,
                seed=seed,
                n_starts=n_starts,
            )
            result["scenario"] = scenario_label
            result["replication"] = rep
            records.append(result)
        except Exception as e:
            logger.warning(f"Replication {rep} failed: {e}")

    return pd.DataFrame(records)


def summarize_validation_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize Monte Carlo validation results.

    Computes mean, std, and quantiles of relative estimation errors.

    Parameters
    ----------
    df : pd.DataFrame
        Raw replication results.

    Returns
    -------
    pd.DataFrame
        Summary statistics per scenario.
    """
    if df.empty:
        return pd.DataFrame()

    summary_records = []
    for scenario, group in df.groupby("scenario"):
        rec = {"scenario": scenario}
        for param in ["mu", "n", "beta"]:
            rel_err_col = f"{param}_rel_err_vs_true"
            boot_std_col = f"{param}_boot_std"
            true_col = f"{param}_true"

            if rel_err_col in group.columns:
                errs = group[rel_err_col].dropna()
                rec[f"{param}_rel_err_mean"] = errs.mean()
                rec[f"{param}_rel_err_std"] = errs.std()
                rec[f"{param}_rel_err_q10"] = errs.quantile(0.10)
                rec[f"{param}_rel_err_q90"] = errs.quantile(0.90)

            if boot_std_col in group.columns and true_col in group.columns:
                # Relative std of bootstrap estimates
                true_val = group[true_col].iloc[0]
                if true_val > 0:
                    rel_std = group[boot_std_col].dropna() / true_val
                    rec[f"{param}_boot_rel_std_mean"] = rel_std.mean()

        rec["n_replications"] = len(group)
        rec["n_true"] = group["n_true"].iloc[0]
        rec["mu_true"] = group["mu_true"].iloc[0]
        rec["beta_true"] = group["beta_true"].iloc[0]
        summary_records.append(rec)

    return pd.DataFrame(summary_records)
