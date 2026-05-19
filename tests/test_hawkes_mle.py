"""
Tests for the Hawkes MLE module (src/models/hawkes_exp_truncated.py).

Tests verify:
- Log-likelihood computation correctness
- Parameter estimation accuracy
- Compensator computation
- Edge cases (empty data, single event, extreme parameters)
"""

import numpy as np
import pytest

from src.models.hawkes_exp_truncated import (
    _compute_excitation_recursive,
    log_likelihood_truncated,
    fit_hawkes_exp,
    fit_hawkes_exp_multi_start,
    compute_compensator,
)


# ---------------------------------------------------------------------------
# Tests for _compute_excitation_recursive
# ---------------------------------------------------------------------------

class TestExcitationRecursive:
    def test_empty_times(self):
        R = _compute_excitation_recursive(np.array([]), beta=1.0)
        assert len(R) == 0

    def test_single_event(self):
        R = _compute_excitation_recursive(np.array([1.0]), beta=1.0)
        assert len(R) == 1
        assert R[0] == 0.0  # No predecessors

    def test_two_events(self):
        times = np.array([0.0, 1.0])
        beta = 1.0
        R = _compute_excitation_recursive(times, beta)
        assert R[0] == 0.0
        # R[1] = exp(-beta * (t1 - t0)) = exp(-1)
        assert abs(R[1] - np.exp(-1.0)) < 1e-10

    def test_three_events_recursive_property(self):
        times = np.array([0.0, 1.0, 2.0])
        beta = 0.5
        R = _compute_excitation_recursive(times, beta)
        # R[2] = (1 + R[1]) * exp(-beta * (t2 - t1))
        expected_R2 = (1.0 + R[1]) * np.exp(-beta * 1.0)
        assert abs(R[2] - expected_R2) < 1e-10

    def test_nonnegative(self):
        times = np.sort(np.random.default_rng(42).uniform(0, 100, 50))
        R = _compute_excitation_recursive(times, beta=0.3)
        assert np.all(R >= 0)

    def test_matches_direct_computation(self):
        """Verify recursive formula matches direct sum."""
        times = np.array([0.0, 1.0, 3.0, 6.0])
        beta = 0.5
        R = _compute_excitation_recursive(times, beta)

        # Direct computation
        for i in range(len(times)):
            direct = sum(np.exp(-beta * (times[i] - times[j])) for j in range(i))
            assert abs(R[i] - direct) < 1e-10, f"Mismatch at i={i}: {R[i]} vs {direct}"


# ---------------------------------------------------------------------------
# Tests for log_likelihood_truncated
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def test_empty_events(self):
        ll = log_likelihood_truncated(np.array([]), T=600.0, mu=0.1, n=0.5, beta=0.2)
        # Should equal -mu * T
        assert abs(ll - (-0.1 * 600.0)) < 1e-10

    def test_finite_for_valid_params(self):
        times = np.array([10.0, 50.0, 100.0, 200.0])
        ll = log_likelihood_truncated(times, T=600.0, mu=0.01, n=0.5, beta=0.2)
        assert np.isfinite(ll)

    def test_negative_infinity_for_zero_mu(self):
        times = np.array([10.0, 50.0])
        # With mu=0 and n=0, intensity is 0 at first event
        ll = log_likelihood_truncated(times, T=600.0, mu=0.0, n=0.0, beta=0.2)
        assert ll == -np.inf or not np.isfinite(ll)

    def test_decreases_with_wrong_params(self):
        """True params should give higher likelihood than wrong params."""
        rng = np.random.default_rng(42)
        # Generate events from a Poisson process (n=0)
        mu_true = 0.5
        T = 600.0
        n_events = rng.poisson(mu_true * T)
        times = np.sort(rng.uniform(0, T, n_events))

        ll_true = log_likelihood_truncated(times, T, mu=mu_true, n=0.0, beta=1.0)
        ll_wrong = log_likelihood_truncated(times, T, mu=mu_true * 10, n=0.0, beta=1.0)
        # True mu should give higher likelihood
        assert ll_true > ll_wrong

    def test_compensator_structure(self):
        """Log-likelihood should be finite for reasonable parameters."""
        times = np.linspace(1, 599, 100)
        ll = log_likelihood_truncated(times, T=600.0, mu=0.1, n=0.3, beta=0.5)
        assert np.isfinite(ll)
        assert ll < 0  # Log-likelihood is typically negative


# ---------------------------------------------------------------------------
# Tests for fit_hawkes_exp
# ---------------------------------------------------------------------------

class TestFitHawkesExp:
    def test_too_few_events(self):
        result = fit_hawkes_exp(np.array([1.0]), T=600.0)
        assert not result["success"]
        assert np.isnan(result["n_hat"])

    def test_returns_valid_params(self):
        """Fit should return valid parameters for reasonable data."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 100))
        result = fit_hawkes_exp(times, T=600.0)
        assert result["n_events"] == 100
        if result["success"]:
            assert 0 <= result["n_hat"] < 1
            assert result["mu_hat"] > 0
            assert result["beta_hat"] > 0

    def test_n_in_valid_range(self):
        """Branching ratio must be in [0, 1)."""
        rng = np.random.default_rng(123)
        times = np.sort(rng.uniform(0, 600, 200))
        result = fit_hawkes_exp(times, T=600.0)
        if result["success"]:
            assert 0 <= result["n_hat"] < 1

    def test_mu_positive(self):
        """Background intensity must be positive."""
        rng = np.random.default_rng(456)
        times = np.sort(rng.uniform(0, 600, 150))
        result = fit_hawkes_exp(times, T=600.0)
        if result["success"]:
            assert result["mu_hat"] > 0

    def test_beta_positive(self):
        """Decay rate must be positive."""
        rng = np.random.default_rng(789)
        times = np.sort(rng.uniform(0, 600, 150))
        result = fit_hawkes_exp(times, T=600.0)
        if result["success"]:
            assert result["beta_hat"] > 0

    def test_loglik_finite(self):
        """Log-likelihood should be finite for successful fits."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 100))
        result = fit_hawkes_exp(times, T=600.0)
        if result["success"]:
            assert np.isfinite(result["loglik"])

    def test_recovers_poisson_params(self):
        """For Poisson data (n≈0), should recover approximately correct mu."""
        rng = np.random.default_rng(42)
        mu_true = 0.5
        T = 600.0
        n_events = rng.poisson(mu_true * T)
        times = np.sort(rng.uniform(0, T, n_events))

        result = fit_hawkes_exp(times, T=T, n_init=0.01)
        if result["success"]:
            # mu should be close to true value (within 50% for Poisson data)
            assert abs(result["mu_hat"] - mu_true) / mu_true < 0.5


# ---------------------------------------------------------------------------
# Tests for fit_hawkes_exp_multi_start
# ---------------------------------------------------------------------------

class TestFitHawkesExpMultiStart:
    def test_returns_best_result(self):
        """Multi-start should return a valid result."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 150))
        result = fit_hawkes_exp_multi_start(times, T=600.0, n_starts=3)
        assert "n_hat" in result
        assert "mu_hat" in result
        assert "beta_hat" in result

    def test_better_than_single_start(self):
        """Multi-start should achieve at least as good likelihood as single start."""
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 600, 200))
        T = 600.0

        single = fit_hawkes_exp(times, T)
        multi = fit_hawkes_exp_multi_start(times, T, n_starts=5)

        if single["success"] and multi["success"]:
            assert multi["loglik"] >= single["loglik"] - 1e-6  # Allow tiny numerical diff


# ---------------------------------------------------------------------------
# Tests for compute_compensator
# ---------------------------------------------------------------------------

class TestComputeCompensator:
    def test_empty_times(self):
        xi = compute_compensator(np.array([]), T=600.0, mu=0.1, n=0.5, beta=0.2)
        assert len(xi) == 0

    def test_monotone_increasing(self):
        """Compensator should be monotonically increasing."""
        times = np.array([10.0, 50.0, 100.0, 200.0, 400.0])
        xi = compute_compensator(times, T=600.0, mu=0.1, n=0.5, beta=0.2)
        assert np.all(np.diff(xi) > 0)

    def test_nonnegative(self):
        """Compensator should be non-negative."""
        times = np.linspace(1, 599, 50)
        xi = compute_compensator(times, T=600.0, mu=0.05, n=0.3, beta=0.1)
        assert np.all(xi >= 0)

    def test_pure_poisson_case(self):
        """For n=0, compensator should equal mu * t."""
        times = np.array([10.0, 50.0, 100.0, 200.0])
        mu = 0.1
        xi = compute_compensator(times, T=600.0, mu=mu, n=0.0, beta=1.0)
        expected = mu * times
        np.testing.assert_allclose(xi, expected, rtol=1e-10)

    def test_length_matches_input(self):
        """Output length should match input length."""
        times = np.linspace(1, 599, 75)
        xi = compute_compensator(times, T=600.0, mu=0.1, n=0.4, beta=0.3)
        assert len(xi) == len(times)


# ---------------------------------------------------------------------------
# Integration test: simulate and recover parameters
# ---------------------------------------------------------------------------

class TestSimulateAndRecover:
    def test_parameter_recovery_low_n(self):
        """Test parameter recovery for low branching ratio."""
        from src.sim.hawkes_sim import simulate_hawkes_ogata_fast

        mu_true, n_true, beta_true = 0.3, 0.3, 0.5
        T = 600.0
        times = simulate_hawkes_ogata_fast(mu_true, n_true, beta_true, T, seed=42)

        if len(times) < 10:
            pytest.skip("Too few events for reliable estimation")

        result = fit_hawkes_exp_multi_start(times, T, n_starts=5)
        if result["success"]:
            # n should be in reasonable range
            assert abs(result["n_hat"] - n_true) < 0.3

    def test_parameter_recovery_high_n(self):
        """Test parameter recovery for high branching ratio."""
        from src.sim.hawkes_sim import simulate_hawkes_ogata_fast

        mu_true, n_true, beta_true = 0.1, 0.7, 0.3
        T = 600.0
        times = simulate_hawkes_ogata_fast(mu_true, n_true, beta_true, T, seed=123)

        if len(times) < 10:
            pytest.skip("Too few events for reliable estimation")

        result = fit_hawkes_exp_multi_start(times, T, n_starts=5)
        if result["success"]:
            assert 0 <= result["n_hat"] < 1
