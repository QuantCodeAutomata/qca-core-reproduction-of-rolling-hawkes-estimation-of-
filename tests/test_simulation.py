"""
Tests for Hawkes simulation module (src/sim/hawkes_sim.py).

Tests verify:
- Simulated events are within [0, T]
- Event count is approximately correct
- Sorted output
- Edge cases
"""

import numpy as np
import pytest

from src.sim.hawkes_sim import (
    simulate_hawkes_ogata_fast,
    expected_event_count,
    calibrate_params_for_target_count,
)


class TestSimulateHawkesOgataFast:
    def test_events_in_range(self):
        """All events must be in [0, T]."""
        times = simulate_hawkes_ogata_fast(mu=0.5, n=0.3, beta=0.5, T=600.0, seed=42)
        assert np.all(times >= 0)
        assert np.all(times <= 600.0)

    def test_sorted_output(self):
        """Output must be sorted."""
        times = simulate_hawkes_ogata_fast(mu=0.5, n=0.3, beta=0.5, T=600.0, seed=42)
        assert np.all(np.diff(times) >= 0)

    def test_approximate_event_count(self):
        """Event count should be approximately E[N(T)] = mu*T/(1-n)."""
        mu, n, beta, T = 0.5, 0.3, 0.5, 600.0
        expected = expected_event_count(mu, n, T)
        times = simulate_hawkes_ogata_fast(mu, n, beta, T, seed=42)
        # Allow 50% deviation (stochastic)
        assert abs(len(times) - expected) / expected < 0.5

    def test_reproducibility(self):
        """Same seed should produce same result."""
        t1 = simulate_hawkes_ogata_fast(mu=0.3, n=0.5, beta=0.3, T=300.0, seed=42)
        t2 = simulate_hawkes_ogata_fast(mu=0.3, n=0.5, beta=0.3, T=300.0, seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        t1 = simulate_hawkes_ogata_fast(mu=0.3, n=0.5, beta=0.3, T=300.0, seed=1)
        t2 = simulate_hawkes_ogata_fast(mu=0.3, n=0.5, beta=0.3, T=300.0, seed=2)
        assert not np.array_equal(t1, t2)

    def test_high_n_more_events(self):
        """Higher branching ratio should produce more events on average."""
        mu, beta, T = 0.2, 0.5, 600.0
        n_seeds = 5
        counts_low = []
        counts_high = []
        for seed in range(n_seeds):
            t_low = simulate_hawkes_ogata_fast(mu, 0.2, beta, T, seed=seed)
            t_high = simulate_hawkes_ogata_fast(mu, 0.7, beta, T, seed=seed)
            counts_low.append(len(t_low))
            counts_high.append(len(t_high))
        assert np.mean(counts_high) > np.mean(counts_low)

    def test_invalid_n_raises(self):
        """n >= 1 should raise assertion error."""
        with pytest.raises(AssertionError):
            simulate_hawkes_ogata_fast(mu=0.5, n=1.0, beta=0.5, T=100.0)

    def test_invalid_mu_raises(self):
        """mu <= 0 should raise assertion error."""
        with pytest.raises(AssertionError):
            simulate_hawkes_ogata_fast(mu=0.0, n=0.5, beta=0.5, T=100.0)

    def test_zero_horizon(self):
        """Zero horizon should produce no events."""
        times = simulate_hawkes_ogata_fast(mu=0.5, n=0.3, beta=0.5, T=0.0, seed=42)
        assert len(times) == 0

    def test_very_short_horizon(self):
        """Very short horizon should produce few or no events."""
        times = simulate_hawkes_ogata_fast(mu=0.001, n=0.1, beta=0.5, T=0.1, seed=42)
        assert len(times) <= 5


class TestExpectedEventCount:
    def test_poisson_case(self):
        """For n=0, expected count = mu * T."""
        assert abs(expected_event_count(mu=0.5, n=0.0, T=600.0) - 300.0) < 1e-10

    def test_increases_with_n(self):
        """Expected count should increase with n."""
        mu, T = 0.5, 600.0
        counts = [expected_event_count(mu, n, T) for n in [0.1, 0.3, 0.5, 0.7, 0.9]]
        assert all(counts[i] < counts[i+1] for i in range(len(counts)-1))

    def test_invalid_n_raises(self):
        with pytest.raises(AssertionError):
            expected_event_count(mu=0.5, n=1.0, T=600.0)


class TestCalibrateParams:
    def test_expected_count_matches_target(self):
        """Calibrated mu should give expected count equal to target."""
        target = 200.0
        n, beta, T = 0.5, 0.3, 600.0
        mu, n_out, beta_out = calibrate_params_for_target_count(target, n, beta, T)
        actual_expected = expected_event_count(mu, n_out, T)
        assert abs(actual_expected - target) / target < 1e-6

    def test_returns_positive_mu(self):
        mu, n, beta = calibrate_params_for_target_count(100.0, 0.3, 0.5, 600.0)
        assert mu > 0

    def test_preserves_n_and_beta(self):
        n_in, beta_in = 0.6, 0.4
        mu, n_out, beta_out = calibrate_params_for_target_count(150.0, n_in, beta_in, 600.0)
        assert n_out == n_in
        assert beta_out == beta_in
