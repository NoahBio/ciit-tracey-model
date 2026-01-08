"""
Unit tests for configuration utilities in src/config.py.

Tests all utility functions including matrix sampling, memory weighting,
bond calculation, and success threshold computation.
"""

import pytest
import numpy as np

from src.config import (
    sample_u_matrix,
    get_memory_weights,
    rs_to_bond,
    calculate_success_threshold,
    U_MIN,
    U_MAX,
    U_MATRIX,
    MEMORY_SIZE,
    BOND_ALPHA,
    BOND_OFFSET,
)


# ============================================================================
# TEST sample_u_matrix()
# ============================================================================

class TestSampleUMatrix:
    """Test utility matrix sampling function."""

    def test_sample_u_matrix_shape(self):
        """Should return 8x8 array."""
        u_matrix = sample_u_matrix()
        assert u_matrix.shape == (8, 8)

    def test_sample_u_matrix_within_bounds(self):
        """All sampled values should be within [U_MIN, U_MAX]."""
        u_matrix = sample_u_matrix(random_state=42)

        # Check all values are within bounds
        assert np.all(u_matrix >= U_MIN), "Some values below U_MIN"
        assert np.all(u_matrix <= U_MAX), "Some values above U_MAX"

    def test_sample_u_matrix_reproducibility(self):
        """Same seed should produce same matrix."""
        u_matrix1 = sample_u_matrix(random_state=42)
        u_matrix2 = sample_u_matrix(random_state=42)

        np.testing.assert_array_equal(u_matrix1, u_matrix2)

    def test_sample_u_matrix_different_seeds(self):
        """Different seeds should produce different matrices."""
        u_matrix1 = sample_u_matrix(random_state=42)
        u_matrix2 = sample_u_matrix(random_state=123)

        # At least some values should differ
        assert not np.array_equal(u_matrix1, u_matrix2)

    def test_sample_u_matrix_accepts_random_state(self):
        """Should accept np.random.RandomState object."""
        rng = np.random.RandomState(42)
        u_matrix = sample_u_matrix(random_state=rng)

        assert u_matrix.shape == (8, 8)
        assert np.all(u_matrix >= U_MIN)
        assert np.all(u_matrix <= U_MAX)


# ============================================================================
# TEST get_memory_weights()
# ============================================================================

class TestGetMemoryWeights:
    """Test memory weighting function for recency bias."""

    def test_memory_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        weights = get_memory_weights()
        assert np.isclose(np.sum(weights), 1.0)

    def test_memory_weights_recency_bias(self):
        """Recent interactions should have higher weights than older ones."""
        weights = get_memory_weights()

        # Most recent (last element) should have highest weight
        assert weights[-1] == np.max(weights)

        # Oldest (first element) should have lowest weight
        assert weights[0] == np.min(weights)

    def test_memory_weights_monotonic_increasing(self):
        """Weights should be monotonically increasing (older → newer)."""
        weights = get_memory_weights()

        # Check each weight >= previous weight
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i-1], \
                f"Weight decrease at index {i}: {weights[i-1]:.6f} → {weights[i]:.6f}"

    def test_memory_weights_correct_length(self):
        """Should return array of length n_interactions."""
        weights = get_memory_weights(n_interactions=MEMORY_SIZE)
        assert len(weights) == MEMORY_SIZE

        weights_custom = get_memory_weights(n_interactions=30)
        assert len(weights_custom) == 30

    def test_memory_weights_all_positive(self):
        """All weights should be positive."""
        weights = get_memory_weights()
        assert np.all(weights > 0)

    def test_memory_weights_ratio(self):
        """Most recent should have ~2x weight of oldest (sqrt weighting)."""
        weights = get_memory_weights()

        ratio = weights[-1] / weights[0]

        # With sqrt(t) weighting: ratio should be close to 2.0
        # Formula: (1 + sqrt(1)) / (1 + sqrt(0)) = 2 / 1 = 2.0
        assert 1.8 < ratio < 2.2

    def test_memory_weights_uses_global_config(self):
        """Test that get_memory_weights() respects global RECENCY_WEIGHTING_FACTOR."""
        import src.config as config_module

        # Save original value
        original = config_module.RECENCY_WEIGHTING_FACTOR

        try:
            # Test default behavior
            config_module.RECENCY_WEIGHTING_FACTOR = 3
            weights_default = get_memory_weights(n_interactions=50)

            # Test explicit override
            weights_explicit = get_memory_weights(n_interactions=50, recency_weighting_factor=2)

            # Should differ (3x vs 2x ratio)
            assert not np.allclose(weights_default, weights_explicit)

        finally:
            # Restore original
            config_module.RECENCY_WEIGHTING_FACTOR = original

    def test_memory_weights_explicit_override(self):
        """Test explicit recency_weighting_factor overrides global config."""
        weights_1 = get_memory_weights(n_interactions=50, recency_weighting_factor=1)
        weights_5 = get_memory_weights(n_interactions=50, recency_weighting_factor=5)

        # Ratio of newest:oldest should differ significantly
        ratio_1 = weights_1[-1] / weights_1[0]
        ratio_5 = weights_5[-1] / weights_5[0]

        assert abs(ratio_1 - 1.5) < 0.1, f"Expected ratio ~1.5, got {ratio_1:.2f}"
        assert abs(ratio_5 - 5.0) < 0.5, f"Expected ratio ~5.0, got {ratio_5:.2f}"


# ============================================================================
# TEST rs_to_bond()
# ============================================================================

class TestRsToBond:
    """Test relationship satisfaction to bond transformation."""

    def test_rs_to_bond_range(self):
        """Bond should always be in [0, 1]."""
        rs_min, rs_max = -70, 70

        # Test extreme values
        bond_min = rs_to_bond(rs_min, rs_min, rs_max)
        bond_max = rs_to_bond(rs_max, rs_min, rs_max)

        assert 0.0 <= bond_min <= 1.0
        assert 0.0 <= bond_max <= 1.0

        # Test intermediate values
        for rs in np.linspace(rs_min, rs_max, 20):
            bond = rs_to_bond(rs, rs_min, rs_max)
            assert 0.0 <= bond <= 1.0, f"Bond {bond:.4f} out of range for RS={rs:.2f}"

    def test_rs_to_bond_at_extremes(self):
        """Minimum RS should give ~0 bond, maximum RS should give ~1 bond."""
        rs_min, rs_max = -70, 70

        bond_at_min = rs_to_bond(rs_min, rs_min, rs_max)
        bond_at_max = rs_to_bond(rs_max, rs_min, rs_max)

        # Due to sigmoid with offset=0.8, extremes won't be at 0 and 1
        # But should show clear separation
        assert bond_at_min < 0.2, f"Bond at rs_min is {bond_at_min:.4f}, expected < 0.2"
        assert bond_at_max > 0.8, f"Bond at rs_max is {bond_at_max:.4f}, expected > 0.8"
        assert bond_at_max > bond_at_min + 0.5, "Should have substantial range"

    def test_rs_to_bond_monotonicity(self):
        """Higher RS should always give higher or equal bond."""
        rs_min, rs_max = -70, 70

        rs_values = np.linspace(rs_min, rs_max, 50)
        bond_values = [rs_to_bond(rs, rs_min, rs_max) for rs in rs_values]

        # Check monotonicity
        for i in range(1, len(bond_values)):
            assert bond_values[i] >= bond_values[i-1], \
                f"Bond decreased: RS {rs_values[i-1]:.2f}→{rs_values[i]:.2f}, " \
                f"bond {bond_values[i-1]:.4f}→{bond_values[i]:.4f}"

    def test_rs_to_bond_inflection_point(self):
        """Sigmoid should have inflection point near offset."""
        rs_min, rs_max = -70, 70

        # At offset (default 0.8), normalized RS = 0.8, should give bond ≈ 0.5
        # after the 2*(rs_normalized - offset) shift
        rs_at_offset = rs_min + BOND_OFFSET * (rs_max - rs_min)
        bond_at_offset = rs_to_bond(rs_at_offset, rs_min, rs_max, offset=BOND_OFFSET)

        # Sigmoid at 0 gives 0.5
        assert 0.4 < bond_at_offset < 0.6, \
            f"Bond at offset RS should be near 0.5, got {bond_at_offset:.4f}"

    def test_rs_to_bond_custom_alpha(self):
        """Higher alpha should give steeper sigmoid."""
        rs_min, rs_max = -70, 70
        rs_mid = (rs_min + rs_max) / 2

        bond_alpha_5 = rs_to_bond(rs_mid, rs_min, rs_max, alpha=5)
        bond_alpha_10 = rs_to_bond(rs_mid, rs_min, rs_max, alpha=10)

        # With higher alpha, should be closer to extremes (steeper)
        # Both should still be in [0, 1], but alpha affects steepness
        assert 0.0 <= bond_alpha_5 <= 1.0
        assert 0.0 <= bond_alpha_10 <= 1.0

    def test_rs_to_bond_custom_offset(self):
        """Different offset should shift the inflection point."""
        rs_min, rs_max = -70, 70

        rs_at_50pct = rs_min + 0.5 * (rs_max - rs_min)

        # With offset=0.5, bond should be near 0.5 at 50th percentile
        bond_offset_50 = rs_to_bond(rs_at_50pct, rs_min, rs_max, offset=0.5)

        assert 0.4 < bond_offset_50 < 0.6


# ============================================================================
# TEST calculate_success_threshold()
# ============================================================================

class TestCalculateSuccessThreshold:
    """Test success threshold calculation."""

    def test_success_threshold_within_bounds(self):
        """Threshold should be between min and max RS."""
        u_matrix = sample_u_matrix(random_state=42)

        threshold = calculate_success_threshold(u_matrix, percentile=0.8)

        rs_min = u_matrix.min()
        rs_max = u_matrix.max()

        assert rs_min <= threshold <= rs_max, \
            f"Threshold {threshold:.2f} outside [{rs_min:.2f}, {rs_max:.2f}]"

    def test_success_threshold_at_percentile(self):
        """Threshold should be at correct percentile of range."""
        u_matrix = sample_u_matrix(random_state=42)

        rs_min = u_matrix.min()
        rs_max = u_matrix.max()

        # Test different percentiles
        for percentile in [0.0, 0.25, 0.5, 0.75, 1.0]:
            threshold = calculate_success_threshold(u_matrix, percentile=percentile)

            expected = rs_min + percentile * (rs_max - rs_min)

            np.testing.assert_almost_equal(threshold, expected, decimal=6)

    def test_success_threshold_90th_percentile(self):
        """Default 90th percentile should be correct."""
        u_matrix = sample_u_matrix(random_state=42)

        threshold = calculate_success_threshold(u_matrix)

        rs_min = u_matrix.min()
        rs_max = u_matrix.max()
        expected = rs_min + 0.9 * (rs_max - rs_min)

        np.testing.assert_almost_equal(threshold, expected)

    def test_success_threshold_invalid_percentile_raises(self):
        """Invalid percentile should raise ValueError."""
        u_matrix = sample_u_matrix(random_state=42)

        # Percentile < 0
        with pytest.raises(ValueError, match="percentile must be between 0 and 1"):
            calculate_success_threshold(u_matrix, percentile=-0.1)

        # Percentile > 1
        with pytest.raises(ValueError, match="percentile must be between 0 and 1"):
            calculate_success_threshold(u_matrix, percentile=1.5)

    def test_success_threshold_different_matrices(self):
        """Different matrices should give different thresholds."""
        u_matrix1 = sample_u_matrix(random_state=42)
        u_matrix2 = sample_u_matrix(random_state=123)

        threshold1 = calculate_success_threshold(u_matrix1)
        threshold2 = calculate_success_threshold(u_matrix2)

        # Should be different (unless extremely unlikely)
        assert threshold1 != threshold2

    def test_success_threshold_with_u_matrix_constant(self):
        """With deterministic U_MATRIX, result should be deterministic."""
        threshold = calculate_success_threshold(U_MATRIX, percentile=0.8)

        # Should be reproducible
        threshold2 = calculate_success_threshold(U_MATRIX, percentile=0.8)

        assert threshold == threshold2
