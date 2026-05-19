"""
Tests for rolling window pipeline (src/rolling/window_pipeline.py).

Tests verify:
- Window extraction correctness
- Pipeline produces valid output structure
- Aggregation logic
"""

import numpy as np
import pandas as pd
import pytest

from src.rolling.window_pipeline import (
    extract_window_events,
    estimate_window_bootstrap,
    process_day_rolling,
    aggregate_window_estimates,
    build_two_month_aggregate,
)


class TestExtractWindowEvents:
    def test_basic_extraction(self):
        """Events strictly inside window should be extracted."""
        times = np.array([0.0, 100.0, 300.0, 500.0, 700.0])
        events = extract_window_events(times, window_start_s=0.0, window_end_s=600.0)
        # Strictly inside (0, 600]
        assert 100.0 in events - 0.0  # rebased
        assert 300.0 in events - 0.0

    def test_rebasing(self):
        """Events should be rebased to [0, T]."""
        times = np.array([100.0, 200.0, 300.0])
        events = extract_window_events(times, window_start_s=100.0, window_end_s=400.0)
        # Rebased: 200-100=100, 300-100=200
        assert np.all(events >= 0)
        assert np.all(events <= 300.0)

    def test_empty_window(self):
        """Window with no events should return empty array."""
        times = np.array([100.0, 200.0])
        events = extract_window_events(times, window_start_s=500.0, window_end_s=600.0)
        assert len(events) == 0

    def test_boundary_exclusion(self):
        """Events at window_start should be excluded (strictly inside)."""
        times = np.array([0.0, 100.0, 600.0])
        events = extract_window_events(times, window_start_s=0.0, window_end_s=600.0)
        # t=0 is NOT strictly inside (0, 600]
        # t=600 IS included (<=)
        assert 0.0 not in events
        assert 600.0 - 0.0 in events  # rebased

    def test_sorted_output(self):
        """Output should be sorted."""
        times = np.array([50.0, 150.0, 250.0, 350.0])
        events = extract_window_events(times, window_start_s=0.0, window_end_s=400.0)
        assert np.all(np.diff(events) >= 0)


class TestEstimateWindowBootstrap:
    def test_returns_correct_number_of_results(self):
        """Should return n_bootstrap results."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 100))
        results = estimate_window_bootstrap(times, T_seconds=600.0, n_bootstrap=10, seed=42)
        assert len(results) == 10

    def test_result_structure(self):
        """Each result should have required keys."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 100))
        results = estimate_window_bootstrap(times, T_seconds=600.0, n_bootstrap=5, seed=42)
        required_keys = ["bootstrap_id", "mu_hat", "n_hat", "beta_hat", "loglik",
                         "ks_stat", "ks_pvalue", "success", "n_events"]
        for r in results:
            for key in required_keys:
                assert key in r, f"Missing key: {key}"

    def test_bootstrap_ids_sequential(self):
        """Bootstrap IDs should be 0, 1, 2, ..."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 100))
        results = estimate_window_bootstrap(times, T_seconds=600.0, n_bootstrap=5, seed=42)
        ids = [r["bootstrap_id"] for r in results]
        assert ids == list(range(5))

    def test_n_hat_in_valid_range(self):
        """Estimated n should be in [0, 1) for successful fits."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 150))
        results = estimate_window_bootstrap(times, T_seconds=600.0, n_bootstrap=10, seed=42)
        for r in results:
            if r["success"] and np.isfinite(r["n_hat"]):
                assert 0 <= r["n_hat"] < 1


class TestProcessDayRolling:
    """Use short session (30 min = 1800s) to keep tests fast."""

    SHORT_SESSION = 1800.0  # 30 minutes

    def _make_times(self, n: int = 300, seed: int = 42) -> np.ndarray:
        return np.sort(np.random.default_rng(seed).uniform(0, self.SHORT_SESSION, n))

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        result = process_day_rolling(
            self._make_times(), "2009-01-05", window_length_min=10, step_min=5,
            n_bootstrap=3, n_starts=1, min_events=10,
            session_duration_s=self.SHORT_SESSION,
        )
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        """Output should have required columns."""
        result = process_day_rolling(
            self._make_times(), "2009-01-05", window_length_min=10, step_min=5,
            n_bootstrap=3, n_starts=1, min_events=10,
            session_duration_s=self.SHORT_SESSION,
        )
        if not result.empty:
            required = ["trade_date", "window_start_s", "window_end_s",
                        "window_length_min", "event_count", "accepted_window",
                        "n_hat", "mu_hat", "beta_hat"]
            for col in required:
                assert col in result.columns, f"Missing column: {col}"

    def test_window_length_correct(self):
        """Window length should match specified value."""
        result = process_day_rolling(
            self._make_times(), "2009-01-05", window_length_min=10, step_min=5,
            n_bootstrap=3, n_starts=1, min_events=10,
            session_duration_s=self.SHORT_SESSION,
        )
        if not result.empty:
            assert (result["window_length_min"] == 10).all()

    def test_empty_day(self):
        """Day with no events should return empty DataFrame."""
        result = process_day_rolling(
            np.array([]), "2009-01-05", window_length_min=10, step_min=5,
            n_bootstrap=3, n_starts=1, min_events=10,
            session_duration_s=self.SHORT_SESSION,
        )
        assert result.empty

    def test_too_few_events_per_window(self):
        """Day with very few events should return empty or minimal results."""
        times = np.array([100.0, 200.0, 300.0])  # Only 3 events
        result = process_day_rolling(
            times, "2009-01-05", window_length_min=10, step_min=5,
            n_bootstrap=3, n_starts=1, min_events=10,
            session_duration_s=self.SHORT_SESSION,
        )
        # Should be empty since no window has >= 10 events
        assert result.empty


class TestAggregateWindowEstimates:
    SHORT_SESSION = 1800.0

    def _make_raw(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, self.SHORT_SESSION, 300))
        return process_day_rolling(
            times, "2009-01-05", window_length_min=10, step_min=5,
            n_bootstrap=3, n_starts=1, min_events=10,
            session_duration_s=self.SHORT_SESSION,
        )

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        raw = self._make_raw()
        agg = aggregate_window_estimates(raw, accepted_only=False)
        assert isinstance(agg, pd.DataFrame)

    def test_fewer_rows_than_raw(self):
        """Aggregated should have fewer rows than raw (one per window)."""
        raw = self._make_raw()
        if not raw.empty:
            agg = aggregate_window_estimates(raw, accepted_only=False)
            assert len(agg) <= len(raw)

    def test_empty_input(self):
        """Empty input should return empty DataFrame."""
        agg = aggregate_window_estimates(pd.DataFrame(), accepted_only=False)
        assert agg.empty


class TestBuildTwoMonthAggregate:
    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2009-01-01", periods=100, freq="B"),
            "n_hat_median": np.random.default_rng(42).uniform(0.3, 0.8, 100),
        })
        result = build_two_month_aggregate(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        """Should have month and n_hat columns."""
        df = pd.DataFrame({
            "trade_date": pd.date_range("2009-01-01", periods=50, freq="B"),
            "n_hat_median": np.random.default_rng(42).uniform(0.3, 0.8, 50),
        })
        result = build_two_month_aggregate(df)
        if not result.empty:
            assert "month" in result.columns
            assert "n_hat_2m_mean" in result.columns

    def test_empty_input(self):
        """Empty input should return empty DataFrame."""
        result = build_two_month_aggregate(pd.DataFrame())
        assert result.empty
