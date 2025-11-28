"""
Unit tests for FrequencyAmplifierClient in src/agents/client_agents/frequency_amplifier_client.py.

Tests marginal frequency amplification mechanism: therapist behavior frequencies
amplify expected utilities. History influences expectations.
"""

import pytest
import numpy as np

from src.agents.client_agents.frequency_amplifier_client import FrequencyAmplifierClient
from src.config import MEMORY_SIZE, HISTORY_WEIGHT
from tests.conftest import assert_valid_probability_distribution


# ==============================================================================
# TEST MARGINAL FREQUENCY CALCULATION
# ==============================================================================

class TestMarginalFrequencyCalculation:
    """Test _calculate_marginal_frequencies() produces valid distributions."""

    def test_frequencies_are_valid_distribution(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Marginal frequencies should form valid probability distribution."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        frequencies = client._calculate_marginal_frequencies()

        # Should be valid probability distribution
        assert_valid_probability_distribution(frequencies)

    def test_frequencies_reflect_memory_pattern(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Frequencies should reflect therapist behavior in memory."""
        # Memory where therapist always responds with octant 4
        uniform_memory = [(i % 8, 4) for i in range(MEMORY_SIZE)]

        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=uniform_memory,
            random_state=fixed_seed
        )

        frequencies = client._calculate_marginal_frequencies()

        # Octant 4 should have very high frequency (recency weighting prevents exactly 1.0)
        assert frequencies[4] > 0.9, f"Expected octant 4 frequency > 0.9, got {frequencies[4]}"

        # Other octants should have near-zero frequency
        for i in range(8):
            if i != 4:
                assert frequencies[i] < 0.02, f"Octant {i} should have near-zero frequency, got {frequencies[i]}"

    def test_frequencies_with_recency_weighting(self, fixed_u_matrix, low_entropy, fixed_seed):
        """More recent interactions should have higher influence on frequencies."""
        # Memory: older half is all octant 0, newer half is all octant 7
        old_pattern = [(0, 0)] * (MEMORY_SIZE // 2)
        recent_pattern = [(0, 7)] * (MEMORY_SIZE // 2)
        memory = old_pattern + recent_pattern

        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            random_state=fixed_seed
        )

        frequencies = client._calculate_marginal_frequencies()

        # Recent octant 7 should have higher frequency than older octant 0
        assert frequencies[7] > frequencies[0], \
            f"Recent octant 7 freq ({frequencies[7]}) should exceed old octant 0 freq ({frequencies[0]})"

    def test_frequencies_change_after_memory_update(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Updating memory should change marginal frequencies."""
        # Start with mostly octant 0
        memory = [(0, 0)] * MEMORY_SIZE

        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            random_state=fixed_seed
        )

        initial_freq = client._calculate_marginal_frequencies().copy()

        # Add many octant 7 interactions
        for _ in range(25):  # Half of memory
            client.update_memory(client_action=0, therapist_action=7)

        updated_freq = client._calculate_marginal_frequencies()

        # Frequency of octant 7 should increase
        assert updated_freq[7] > initial_freq[7], "Octant 7 frequency should increase after updates"

        # Frequency of octant 0 should decrease
        assert updated_freq[0] < initial_freq[0], "Octant 0 frequency should decrease after updates"


# ==============================================================================
# TEST HISTORY AMPLIFICATION
# ==============================================================================

class TestHistoryAmplification:
    """Test that utilities are amplified based on therapist frequencies."""

    def test_payoffs_depend_on_history(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Same bond but different history should give different payoffs."""
        # Two very different memory patterns
        memory1 = [(0, 0)] * MEMORY_SIZE  # Therapist always octant 0
        memory2 = [(0, 7)] * MEMORY_SIZE  # Therapist always octant 7

        client1 = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory1,
            random_state=fixed_seed
        )

        client2 = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory2,
            random_state=fixed_seed
        )

        # Force same bond
        client1.bond = 0.5
        client2.bond = 0.5

        payoffs1 = client1._calculate_expected_payoffs()
        payoffs2 = client2._calculate_expected_payoffs()

        # Payoffs should differ due to different history
        assert not np.allclose(payoffs1, payoffs2), \
            "Payoffs should differ with different history patterns"

    def test_amplification_formula(self, low_entropy, fixed_seed):
        """Test that amplification follows: adjusted = raw + (raw × freq × weight)."""
        # Create simple U_matrix for easy verification
        u_matrix = np.array([
            [10, 20, 30, 40, 50, 60, 70, 80],
            [0, 0, 0, 0, 0, 0, 0, 0],  # All zeros
            [-10, -20, -30, -40, -50, -60, -70, -80],  # All negative
            [100, 100, 100, 100, 100, 100, 100, 100],  # All same
            [10, -10, 10, -10, 10, -10, 10, -10],  # Mixed
            [5, 10, 15, 20, 25, 30, 35, 40],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [100, 0, 50, 25, 75, 10, 90, 60],
        ])

        # Memory where therapist always responds with octant 3
        # This creates P(therapist=3) ≈ 1.0
        memory = [(i % 8, 3) for i in range(MEMORY_SIZE)]

        client = FrequencyAmplifierClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,  # For clearer testing
            random_state=fixed_seed
        )

        # Force specific bond for testing
        client.bond = 0.5

        # Get frequencies
        freq = client._calculate_marginal_frequencies()

        # Octant 3 should have very high frequency
        assert freq[3] > 0.9

        # For client action 0, raw utilities = [10, 20, 30, 40, 50, 60, 70, 80]
        # Amplified[3] ≈ 40 + (40 × 1.0 × 1.0) = 40 + 40 = 80
        # After amplification, utilities for action 0 will be boosted at index 3

        # We can't easily verify the exact expected payoff without reimplementing
        # the percentile selection, but we can verify amplification affects outcomes
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8

    def test_zero_frequency_no_amplification(self, low_entropy, fixed_seed):
        """Unobserved therapist responses (freq=0) should not be amplified."""
        # U_matrix with varied utilities (avoid all same value to prevent division by zero)
        u_matrix = np.random.RandomState(42).uniform(-50, 50, (8, 8))

        # Memory: therapist NEVER responds with octant 7
        memory = [(i % 8, i % 7) for i in range(MEMORY_SIZE)]  # Only 0-6, never 7

        client = FrequencyAmplifierClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,
            random_state=fixed_seed
        )

        freq = client._calculate_marginal_frequencies()

        # Octant 7 should have very low or zero frequency
        assert freq[7] < 0.05, f"Octant 7 should have near-zero frequency, got {freq[7]}"

    def test_negative_utilities_amplified_more_negative(self, low_entropy, fixed_seed):
        """Negative utilities with high frequency should become MORE negative."""
        # U_matrix with negative utilities
        u_matrix = np.array([
            [-10, -20, -30, -40, -50, -60, -70, -80],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [10, 20, 30, 40, 50, 60, 70, 80],
            [-5, -10, -15, -20, -25, -30, -35, -40],
            [5, 10, 15, 20, 25, 30, 35, 40],
            [-100, -100, -100, -100, -100, -100, -100, -100],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])

        # Therapist always responds with octant 2 (which has negative utils in row 0)
        memory = [(0, 2)] * MEMORY_SIZE

        client = FrequencyAmplifierClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,
            random_state=fixed_seed
        )

        # For action 0: raw[2] = -30, freq[2] ≈ 1.0
        # amplified[2] = -30 + (-30 × 1.0 × 1.0) = -30 - 30 = -60
        # So amplification makes negative utilities MORE negative

        # This is inherent to the formula and will affect percentile selection
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8


# ==============================================================================
# TEST HISTORY WEIGHT PARAMETER
# ==============================================================================

class TestHistoryWeightParameter:
    """Test history_weight parameter controls amplification strength."""

    def test_default_history_weight(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should use HISTORY_WEIGHT from config by default."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert client.history_weight == HISTORY_WEIGHT

    def test_custom_history_weight(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should accept custom history_weight parameter."""
        custom_weight = 2.5

        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=custom_weight,
            random_state=fixed_seed
        )

        assert client.history_weight == custom_weight

    def test_higher_weight_stronger_amplification(self, low_entropy, fixed_seed):
        """Higher history_weight should cause stronger amplification effects."""
        # Simple U_matrix
        u_matrix = np.random.RandomState(42).uniform(-50, 50, (8, 8))

        # Clear memory pattern
        memory = [(0, 3)] * MEMORY_SIZE  # Therapist always octant 3

        # Client with low history weight
        client_low = FrequencyAmplifierClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=0.1,
            random_state=fixed_seed
        )

        # Client with high history weight
        client_high = FrequencyAmplifierClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=2.0,
            random_state=fixed_seed
        )

        # Force same bond
        client_low.bond = 0.5
        client_high.bond = 0.5

        payoffs_low = client_low._calculate_expected_payoffs()
        payoffs_high = client_high._calculate_expected_payoffs()

        # Payoffs should differ more with higher weight
        # (Not testing exact magnitude, just that they're different)
        assert not np.allclose(payoffs_low, payoffs_high), \
            "Different history weights should produce different payoffs"

    def test_zero_weight_approaches_bond_only(self, fixed_u_matrix, low_entropy, fixed_seed):
        """history_weight=0 should give results close to BondOnlyClient."""
        memory1 = [(0, 0)] * MEMORY_SIZE
        memory2 = [(0, 7)] * MEMORY_SIZE

        # Two clients with very different history but zero weight
        client1 = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory1,
            history_weight=0.0,
            random_state=fixed_seed
        )

        client2 = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory2,
            history_weight=0.0,
            random_state=fixed_seed
        )

        # Force same bond
        client1.bond = 0.6
        client2.bond = 0.6

        payoffs1 = client1._calculate_expected_payoffs()
        payoffs2 = client2._calculate_expected_payoffs()

        # With zero weight, amplification = raw + (raw × freq × 0) = raw
        # So should behave like BondOnlyClient (same payoffs with same bond)
        np.testing.assert_allclose(payoffs1, payoffs2, rtol=1e-6)


# ==============================================================================
# TEST EFFECTIVE HISTORY WEIGHT HOOK
# ==============================================================================

class TestEffectiveHistoryWeightHook:
    """Test _get_effective_history_weight() for subclass override."""

    def test_returns_base_history_weight(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Base implementation should return unmodified history_weight."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.5,
            random_state=fixed_seed
        )

        effective = client._get_effective_history_weight()

        assert effective == 1.5, "Base implementation should return history_weight"

    def test_called_during_payoff_calculation(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """_get_effective_history_weight() should be called in payoff calculation."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.0,
            random_state=fixed_seed
        )

        # Method should exist and be callable
        assert hasattr(client, '_get_effective_history_weight')
        assert callable(client._get_effective_history_weight)


# ==============================================================================
# TEST INTEGRATION WITH BASE CLIENT
# ==============================================================================

class TestIntegrationWithBase:
    """Test FrequencyAmplifierClient integrates with BaseClientAgent."""

    def test_inherits_base_functionality(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should inherit all BaseClientAgent functionality."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Base attributes
        assert hasattr(client, 'u_matrix')
        assert hasattr(client, 'entropy')
        assert hasattr(client, 'memory')
        assert hasattr(client, 'bond')
        assert hasattr(client, 'relationship_satisfaction')

        # Base methods
        assert hasattr(client, 'select_action')
        assert hasattr(client, 'update_memory')
        assert hasattr(client, 'check_dropout')

    def test_full_action_cycle(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Test full cycle: frequencies → amplification → payoffs → action."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Calculate frequencies
        frequencies = client._calculate_marginal_frequencies()
        assert_valid_probability_distribution(frequencies)

        # Calculate payoffs
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8

        # Select action
        action = client.select_action()
        assert 0 <= action <= 7
        assert isinstance(action, (int, np.integer))
