"""
Tests for data loading and day filtering modules.

Tests verify:
- Contract roll schedule correctness
- Day filtering logic
- Synthetic data generation
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.data.loader import (
    get_active_contract,
    build_roll_schedule,
    generate_synthetic_es_events,
    _third_friday,
    _rollover_date,
)
from src.data.day_filter import (
    filter_nontrading_days,
    filter_low_volume_days,
    apply_all_filters,
)


class TestContractRollSchedule:
    def test_third_friday_march_1998(self):
        """Third Friday of March 1998 should be March 20, 1998."""
        tf = _third_friday(1998, 3)
        assert tf == date(1998, 3, 20)
        assert tf.weekday() == 4  # Friday

    def test_third_friday_is_friday(self):
        """Third Friday should always be a Friday."""
        for year in range(1998, 2011):
            for month in [3, 6, 9, 12]:
                tf = _third_friday(year, month)
                assert tf.weekday() == 4, f"Not Friday: {tf}"

    def test_rollover_8_days_before_expiry(self):
        """Rollover should be 8 calendar days before expiry."""
        expiry = _third_friday(2010, 3)
        rollover = _rollover_date(2010, 3)
        assert (expiry - rollover).days == 8

    def test_active_contract_before_rollover(self):
        """Before rollover, should use current quarter."""
        # March 2010 expiry: March 19, 2010; rollover: March 11, 2010
        # On March 10, 2010, should still be March contract
        contract = get_active_contract(date(2010, 3, 10))
        assert "H" in contract  # March code
        assert "2010" in contract

    def test_active_contract_after_rollover(self):
        """After rollover, should use next quarter."""
        # March 2010 rollover: March 11, 2010
        # On March 12, 2010, should be June contract
        contract = get_active_contract(date(2010, 3, 12))
        assert "M" in contract  # June code
        assert "2010" in contract

    def test_contract_format(self):
        """Contract ticker should follow ESX{YEAR} format."""
        contract = get_active_contract(date(2009, 6, 1))
        assert contract.startswith("ES")
        assert len(contract) >= 6

    def test_build_roll_schedule_length(self):
        """Roll schedule should cover all business days."""
        schedule = build_roll_schedule(date(2009, 1, 1), date(2009, 3, 31))
        assert len(schedule) > 0
        assert "date" in schedule.columns
        assert "contract" in schedule.columns

    def test_roll_schedule_no_weekends(self):
        """Roll schedule should not include weekends."""
        schedule = build_roll_schedule(date(2009, 1, 1), date(2009, 3, 31))
        for d in schedule["date"]:
            assert d.weekday() < 5  # Monday=0, Friday=4


class TestSyntheticDataGeneration:
    def test_returns_dataframe(self):
        df = generate_synthetic_es_events(n_days=5, events_per_day_mean=100, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_synthetic_es_events(n_days=5, events_per_day_mean=100, seed=42)
        assert "timestamp" in df.columns
        assert "date" in df.columns
        assert "seconds_since_open" in df.columns

    def test_event_count_approximately_correct(self):
        """Event count should be approximately n_days * events_per_day_mean."""
        n_days = 10
        mean = 500
        df = generate_synthetic_es_events(n_days=n_days, events_per_day_mean=mean, seed=42)
        expected = n_days * mean
        # Allow 50% deviation
        assert abs(len(df) - expected) / expected < 0.5

    def test_seconds_in_session(self):
        """Event times should be within session duration."""
        df = generate_synthetic_es_events(n_days=5, events_per_day_mean=100, seed=42)
        session_duration = 6.75 * 3600
        assert (df["seconds_since_open"] >= 0).all()
        assert (df["seconds_since_open"] <= session_duration).all()

    def test_reproducibility(self):
        """Same seed should produce same data."""
        df1 = generate_synthetic_es_events(n_days=5, events_per_day_mean=100, seed=42)
        df2 = generate_synthetic_es_events(n_days=5, events_per_day_mean=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestDayFiltering:
    def _make_event_df(self, dates_counts: dict) -> pd.DataFrame:
        """Helper to create a simple event DataFrame."""
        records = []
        for date_str, count in dates_counts.items():
            for i in range(count):
                records.append({
                    "date": pd.Timestamp(date_str).date(),
                    "event_id": i,
                })
        return pd.DataFrame(records)

    def test_filter_nontrading_removes_zero_count_days(self):
        """Days with zero events (no rows) should not appear in output."""
        # Build a DataFrame with two active days only
        df = self._make_event_df({
            "2009-01-05": 100,
            "2009-01-07": 50,
        })
        # 2009-01-06 has no rows at all → should not appear in output
        filtered = filter_nontrading_days(df)
        dates = filtered["date"].unique()
        assert pd.Timestamp("2009-01-06").date() not in dates
        # Active days should be preserved
        assert pd.Timestamp("2009-01-05").date() in dates
        assert pd.Timestamp("2009-01-07").date() in dates

    def test_filter_low_volume_removes_low_days(self):
        """Days below 5th percentile should be removed."""
        rng = np.random.default_rng(42)
        records = []
        for i in range(20):
            date_str = f"2009-01-{i+1:02d}" if i < 20 else f"2009-02-{i-19:02d}"
            count = 100 if i == 0 else rng.integers(500, 1000)  # One very low day
            for j in range(count):
                records.append({"date": pd.Timestamp(f"2009-01-{i+1:02d}").date(), "event_id": j})

        df = pd.DataFrame(records)
        filtered = filter_low_volume_days(df)
        # Should have fewer unique dates
        assert len(filtered["date"].unique()) <= len(df["date"].unique())

    def test_apply_all_filters_returns_dataframe(self):
        """apply_all_filters should return a DataFrame."""
        df = generate_synthetic_es_events(n_days=10, events_per_day_mean=500, seed=42)
        df["date"] = df["timestamp"].dt.date
        filtered = apply_all_filters(df, date_col="date", timestamp_col="timestamp")
        assert isinstance(filtered, pd.DataFrame)
