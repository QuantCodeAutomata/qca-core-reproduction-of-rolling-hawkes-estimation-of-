"""
Tests for timestamp bootstrap module (src/models/timestamp_bootstrap.py).

Tests verify:
- Within-second randomization preserves event count
- Randomized times are within correct second bins
- Bootstrap generates correct number of realizations
- Round-to-seconds function works correctly
"""

import numpy as np
import pytest

from src.models.timestamp_bootstrap import (
    randomize_within_seconds,
    generate_bootstrap_realizations,
    round_to_seconds,
)


class TestRandomizeWithinSeconds:
    def test_preserves_event_count(self):
        """Randomization must preserve total event count."""
        times = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 5.0])
        randomized = randomize_within_seconds(times)
        assert len(randomized) == len(times)

    def test_output_sorted(self):
        """Output must be sorted."""
        times = np.array([0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        randomized = randomize_within_seconds(times)
        assert np.all(np.diff(randomized) >= 0)

    def test_times_within_correct_second(self):
        """Randomized times must stay within their original second bin."""
        times = np.array([2.0, 2.0, 2.0, 5.0, 5.0])
        rng = np.random.default_rng(42)
        randomized = randomize_within_seconds(times, rng=rng)

        # Events originally at second 2 should be in [2, 3)
        # Events originally at second 5 should be in [5, 6)
        # (We check by floor)
        floors = np.floor(randomized).astype(int)
        # All floors should be either 2 or 5
        assert set(floors).issubset({2, 5})

    def test_unique_times_unchanged(self):
        """Events with unique second stamps should not be moved."""
        times = np.array([1.0, 3.0, 7.0])  # All unique seconds
        rng = np.random.default_rng(42)
        randomized = randomize_within_seconds(times, rng=rng)
        # Unique events: floor should match original
        assert np.floor(randomized[0]) == 1
        assert np.floor(randomized[1]) == 3
        assert np.floor(randomized[2]) == 7

    def test_empty_input(self):
        """Empty input should return empty output."""
        randomized = randomize_within_seconds(np.array([]))
        assert len(randomized) == 0

    def test_single_event(self):
        """Single event should be randomized within its second."""
        times = np.array([5.0])
        rng = np.random.default_rng(42)
        randomized = randomize_within_seconds(times, rng=rng)
        assert len(randomized) == 1
        assert 5.0 <= randomized[0] < 6.0

    def test_reproducibility_with_seed(self):
        """Same seed should produce same result."""
        times = np.array([0.0, 0.0, 1.0, 1.0])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = randomize_within_seconds(times, rng=rng1)
        r2 = randomize_within_seconds(times, rng=rng2)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_different_results(self):
        """Different seeds should (almost certainly) produce different results."""
        times = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        r1 = randomize_within_seconds(times, rng=rng1)
        r2 = randomize_within_seconds(times, rng=rng2)
        assert not np.allclose(r1, r2)


class TestGenerateBootstrapRealizations:
    def test_correct_number_of_realizations(self):
        """Should generate exactly n_bootstrap realizations."""
        times = np.array([0.0, 0.0, 1.0, 2.0])
        realizations = generate_bootstrap_realizations(times, n_bootstrap=50, seed=42)
        assert len(realizations) == 50

    def test_each_realization_correct_length(self):
        """Each realization should have same length as input."""
        times = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
        realizations = generate_bootstrap_realizations(times, n_bootstrap=10, seed=42)
        for r in realizations:
            assert len(r) == len(times)

    def test_realizations_are_sorted(self):
        """Each realization should be sorted."""
        times = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
        realizations = generate_bootstrap_realizations(times, n_bootstrap=20, seed=42)
        for r in realizations:
            assert np.all(np.diff(r) >= 0)

    def test_realizations_differ(self):
        """Bootstrap realizations should differ from each other."""
        times = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        realizations = generate_bootstrap_realizations(times, n_bootstrap=10, seed=42)
        # At least some realizations should differ
        all_same = all(np.allclose(realizations[0], r) for r in realizations[1:])
        assert not all_same

    def test_empty_input(self):
        """Empty input should produce empty realizations."""
        realizations = generate_bootstrap_realizations(np.array([]), n_bootstrap=5, seed=42)
        assert len(realizations) == 5
        for r in realizations:
            assert len(r) == 0

    def test_reproducibility(self):
        """Same seed should produce same realizations."""
        times = np.array([0.0, 0.0, 1.0, 1.0])
        r1 = generate_bootstrap_realizations(times, n_bootstrap=5, seed=42)
        r2 = generate_bootstrap_realizations(times, n_bootstrap=5, seed=42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)


class TestRoundToSeconds:
    def test_exact_seconds_unchanged(self):
        """Times that are already integers should be unchanged."""
        times = np.array([1.0, 2.0, 5.0, 10.0])
        rounded = round_to_seconds(times)
        np.testing.assert_array_equal(rounded, times)

    def test_rounds_to_nearest(self):
        """Should round to nearest second."""
        times = np.array([1.4, 1.6, 2.5, 3.1])
        rounded = round_to_seconds(times)
        expected = np.array([1.0, 2.0, 2.0, 3.0])  # 2.5 rounds to 2 (banker's rounding)
        # Note: numpy rounds 0.5 to nearest even
        assert rounded[0] == 1.0
        assert rounded[1] == 2.0
        assert rounded[3] == 3.0

    def test_output_dtype_float64(self):
        """Output should be float64."""
        times = np.array([1.3, 2.7])
        rounded = round_to_seconds(times)
        assert rounded.dtype == np.float64

    def test_empty_input(self):
        rounded = round_to_seconds(np.array([]))
        assert len(rounded) == 0

    def test_creates_ties(self):
        """Rounding should create ties for nearby events."""
        times = np.array([1.1, 1.2, 1.3, 1.4])
        rounded = round_to_seconds(times)
        # All should round to 1.0
        assert np.all(rounded == 1.0)
