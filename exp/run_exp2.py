"""
Experiment 2: Synthetic validation of within-second timestamp randomization.

Demonstrates that rounding event times to seconds and redistributing them
uniformly within each second produces only modest parameter-estimation error.

Targets the paper's reported approximate standard deviations:
    mu: ~1.4%, n: ~0.6%, beta: ~3.5%
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.timestamp_rounding import (
    SCENARIOS,
    run_monte_carlo_validation,
    summarize_validation_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def run_exp2(
    n_replications: int = 50,
    n_bootstrap: int = 50,
    T: float = 600.0,
    base_seed: int = 42,
    n_starts: int = 3,
) -> tuple:
    """
    Run Experiment 2: Monte Carlo validation of timestamp rounding.

    Parameters
    ----------
    n_replications : int
        Monte Carlo replications per scenario.
    n_bootstrap : int
        Bootstrap realizations per replication.
    T : float
        Window length in seconds (600 = 10 min).
    base_seed : int
        Base random seed.
    n_starts : int
        MLE starting points.

    Returns
    -------
    tuple
        (all_results_df, summary_df)
    """
    all_dfs = []

    for label, n_true, beta_inv_s, target_count in SCENARIOS:
        logger.info(f"Running scenario: {label} (n={n_true}, beta_inv={beta_inv_s}s, target={target_count})")
        df = run_monte_carlo_validation(
            scenario_label=label,
            n_true=n_true,
            beta_inv_s=beta_inv_s,
            target_count=target_count,
            T=T,
            n_replications=n_replications,
            n_bootstrap=n_bootstrap,
            base_seed=base_seed,
            n_starts=n_starts,
        )
        all_dfs.append(df)
        logger.info(f"Scenario {label}: {len(df)} replications completed")

    all_results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    summary = summarize_validation_results(all_results)

    return all_results, summary


def plot_error_distributions(
    all_results: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Plot distributions of relative estimation errors across scenarios.
    """
    if all_results.empty:
        logger.warning("No results to plot")
        return

    scenarios = all_results["scenario"].unique()
    params = ["n", "mu", "beta"]
    param_labels = {
        "n": "Branching ratio n",
        "mu": "Background intensity μ",
        "beta": "Decay rate β",
    }

    fig, axes = plt.subplots(len(params), len(scenarios),
                              figsize=(5 * len(scenarios), 4 * len(params)))
    if len(scenarios) == 1:
        axes = axes.reshape(-1, 1)
    if len(params) == 1:
        axes = axes.reshape(1, -1)

    for j, scenario in enumerate(scenarios):
        sc_df = all_results[all_results["scenario"] == scenario]
        for i, param in enumerate(params):
            ax = axes[i, j]
            col = f"{param}_rel_err_vs_true"
            if col in sc_df.columns:
                errs = sc_df[col].dropna() * 100  # Convert to %
                ax.hist(errs, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
                ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
                ax.axvline(errs.mean(), color="orange", linestyle="-", linewidth=1.5,
                           label=f"Mean: {errs.mean():.1f}%")
                ax.set_xlabel(f"Relative error (%)", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title(f"{param_labels[param]}\n{scenario}", fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

    plt.suptitle("Relative Estimation Error: Rounded vs True Parameters\n(Within-second Bootstrap Validation)",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved error distribution plot to {save_path}")
    plt.close()


def plot_bootstrap_stability(
    all_results: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Plot bootstrap estimate stability (std across 50 realizations).
    """
    if all_results.empty:
        return

    scenarios = all_results["scenario"].unique()
    params = ["n", "mu", "beta"]

    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 5))
    if len(params) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, param, color in zip(axes, params, colors):
        std_col = f"{param}_boot_std"
        true_col = f"{param}_true"

        data_by_scenario = []
        labels = []
        for scenario in scenarios:
            sc_df = all_results[all_results["scenario"] == scenario]
            if std_col in sc_df.columns and true_col in sc_df.columns:
                true_val = sc_df[true_col].iloc[0]
                if true_val > 0:
                    rel_std = (sc_df[std_col].dropna() / true_val * 100).values
                    data_by_scenario.append(rel_std)
                    labels.append(scenario.replace("_", "\n"))

        if data_by_scenario:
            bp = ax.boxplot(data_by_scenario, labels=labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("Bootstrap std / true value (%)", fontsize=10)
            ax.set_title(f"Bootstrap stability: {param}", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Bootstrap Estimate Stability Across Scenarios", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved bootstrap stability plot to {save_path}")
    plt.close()


def main():
    logger.info("=" * 60)
    logger.info("Experiment 2: Timestamp Rounding Bootstrap Validation")
    logger.info("=" * 60)

    all_results, summary = run_exp2(
        n_replications=50,
        n_bootstrap=50,
        T=600.0,
        base_seed=42,
        n_starts=3,
    )

    logger.info("\nValidation Summary:")
    logger.info(summary.to_string())

    # Save results
    if not all_results.empty:
        all_results.to_parquet(DATA_DIR / "exp2_mc_results.parquet", index=False)
    summary.to_csv(RESULTS_DIR / "exp2_summary.csv", index=False)

    # Plots
    plot_error_distributions(
        all_results,
        save_path=str(RESULTS_DIR / "exp2_error_distributions.png"),
    )
    plot_bootstrap_stability(
        all_results,
        save_path=str(RESULTS_DIR / "exp2_bootstrap_stability.png"),
    )

    logger.info("Experiment 2 complete.")
    return all_results, summary


if __name__ == "__main__":
    main()
