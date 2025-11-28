"""
Unit tests for BondWeightedConditionalAmplifier in
src/agents/client_agents/bond_weighted_conditional_amplifier_client.py.

Tests bond-modulated history influence with conditional frequencies:
effective_weight = (bond ** bond_power) × history_weight.
Extends ConditionalAmplifierClient with bond-scaled history weight.
"""

import pytest
import numpy as np

from src.agents.client_agents.bond_weighted_conditional_amplifier_client import BondWeightedConditionalAmplifier
from src.config import MEMORY_SIZE, HISTORY_WEIGHT


# ==============================================================================
# TEST BOND-WEIGHTED CONDITIONAL MECHANISM
# ==============================================================================

class TestBondWeightedConditionalMechanism:
    """Test bond-based scaling of history weight with conditional frequencies."""

    def test_inherits_from_conditional_amplifier(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should inherit from ConditionalAmplifierClient."""
        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Should have ConditionalAmplifier methods
        assert hasattr(client, '_calculate_conditional_frequencies')
        assert hasattr(client, '_get_effective_history_weight')
        assert hasattr(client, 'smoothing_alpha')

        # Should NOT have marginal frequency method (that's in FrequencyAmplifier)
        assert not hasattr(client, '_calculate_marginal_frequencies')

    def test_has_bond_power_parameter(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should have bond_power parameter."""
        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            bond_power=2.5,
            random_state=fixed_seed
        )

        assert hasattr(client, 'bond_power')
        assert client.bond_power == 2.5

    def test_default_bond_power(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Default bond_power should be 1.0 (linear scaling)."""
        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert client.bond_power == 1.0

    def test_has_smoothing_parameter(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should have smoothing_alpha from ConditionalAmplifier."""
        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            smoothing_alpha=0.5,
            random_state=fixed_seed
        )

        assert client.smoothing_alpha == 0.5

    def test_effective_weight_formula(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Effective weight should be (bond ** bond_power) × history_weight."""
        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.5,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Set specific bond for testing
        client.bond = 0.6

        effective = client._get_effective_history_weight()

        # Should be: (0.6 ** 2.0) × 1.5 = 0.36 × 1.5 = 0.54
        expected = (0.6 ** 2.0) * 1.5
        np.testing.assert_almost_equal(effective, expected, decimal=6)


# ==============================================================================
# TEST BOND SCALING WITH CONDITIONALS
# ==============================================================================

class TestBondScalingWithConditionals:
    """Test how bond level affects history influence with conditional frequencies."""

    def test_low_bond_reduces_conditional_history_influence(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Low bond should reduce history influence even with conditional frequencies."""
        memory = [(0, 4)] * MEMORY_SIZE  # Strong conditional pattern

        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=2.0,
            bond_power=3.0,
            random_state=fixed_seed
        )

        # Low bond
        client.bond = 0.3
        effective_low = client._get_effective_history_weight()

        # Should be: (0.3 ** 3.0) × 2.0 = 0.027 × 2.0 = 0.054
        expected = (0.3 ** 3.0) * 2.0
        np.testing.assert_almost_equal(effective_low, expected, decimal=6)
        assert effective_low < 0.1, "Low bond should strongly reduce history weight"

    def test_high_bond_maintains_conditional_history_influence(self, fixed_u_matrix, low_entropy, fixed_seed):
        """High bond should maintain strong history influence."""
        memory = [(0, 4)] * MEMORY_SIZE

        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,
            bond_power=1.0,
            random_state=fixed_seed
        )

        # High bond
        client.bond = 0.95
        effective_high = client._get_effective_history_weight()

        # Should be: (0.95 ** 1.0) × 1.0 = 0.95
        np.testing.assert_almost_equal(effective_high, 0.95, decimal=6)

    def test_bond_power_controls_steepness_with_conditionals(
        self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """Higher bond_power should create steeper drop-off at low bond."""
        # Client with linear scaling (bond_power=1.0)
        client_linear = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.0,
            bond_power=1.0,
            random_state=fixed_seed
        )

        # Client with cubic scaling (bond_power=3.0)
        client_cubic = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.0,
            bond_power=3.0,
            random_state=fixed_seed
        )

        # Test at low bond (0.4)
        client_linear.bond = 0.4
        client_cubic.bond = 0.4

        eff_linear = client_linear._get_effective_history_weight()
        eff_cubic = client_cubic._get_effective_history_weight()

        # Linear: 0.4 × 1.0 = 0.4
        # Cubic: 0.4³ × 1.0 = 0.064
        np.testing.assert_almost_equal(eff_linear, 0.4, decimal=6)
        np.testing.assert_almost_equal(eff_cubic, 0.064, decimal=6)

        # Cubic should be much lower
        assert eff_cubic < eff_linear / 4, "Higher bond_power should reduce weight more at low bond"


# ==============================================================================
# TEST PAYOFF DIFFERENCES WITH CONDITIONALS
# ==============================================================================

class TestPayoffDifferencesWithConditionals:
    """Test that bond affects payoffs through conditional history weighting."""

    def test_same_history_different_bonds_different_payoffs(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Same conditional pattern but different bonds should give different payoffs."""
        # Strong conditional pattern: client 0 → therapist 4
        memory = [(0, 4)] * 25 + [(1, 3)] * 25

        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.5,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Calculate payoffs at low bond
        client.bond = 0.2
        payoffs_low_bond = client._calculate_expected_payoffs()

        # Calculate payoffs at high bond
        client.bond = 0.9
        payoffs_high_bond = client._calculate_expected_payoffs()

        # Payoffs should differ due to different history influence
        assert not np.allclose(payoffs_low_bond, payoffs_high_bond), \
            "Different bonds should produce different payoffs"

    def test_low_bond_approaches_bond_only_with_conditionals(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Very low bond should make behavior approach BondOnlyClient."""
        # Two very different conditional patterns
        memory1 = [(i % 8, 0) for i in range(MEMORY_SIZE)]
        memory2 = [(i % 8, 7) for i in range(MEMORY_SIZE)]

        client1 = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory1,
            history_weight=1.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        client2 = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory2,
            history_weight=1.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Very low bond → effective weight ≈ 0
        client1.bond = 0.01
        client2.bond = 0.01

        payoffs1 = client1._calculate_expected_payoffs()
        payoffs2 = client2._calculate_expected_payoffs()

        # With near-zero effective weight, should behave similarly
        np.testing.assert_allclose(payoffs1, payoffs2, rtol=0.01)

    def test_conditional_specificity_preserved(self, low_entropy, fixed_seed):
        """Bond weighting should preserve conditional specificity."""
        # Pattern: client 0 → therapist 4, client 1 → therapist 3
        memory = [(0, 4)] * 25 + [(1, 3)] * 25

        u_matrix = np.random.RandomState(42).uniform(-50, 50, (8, 8))

        client = BondWeightedConditionalAmplifier(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,
            bond_power=1.0,
            smoothing_alpha=0.1,
            random_state=fixed_seed
        )

        # Even with bond weighting, conditional distributions should differ by action
        client.bond = 0.7

        freq_given_0 = client._calculate_conditional_frequencies(0)
        freq_given_1 = client._calculate_conditional_frequencies(1)

        # P(4|0) should be higher than P(4|1)
        assert freq_given_0[4] > freq_given_1[4], \
            "Conditional specificity should be preserved"


# ==============================================================================
# TEST INTEGRATION
# ==============================================================================

class TestIntegration:
    """Test integration with base functionality."""

    def test_inherits_all_base_functionality(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should have all BaseClientAgent and ConditionalAmplifier functionality."""
        client = BondWeightedConditionalAmplifier(
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

        # ConditionalAmplifier attributes
        assert hasattr(client, 'history_weight')
        assert hasattr(client, 'smoothing_alpha')

        # BondWeighted attributes
        assert hasattr(client, 'bond_power')

        # Methods
        assert hasattr(client, 'select_action')
        assert hasattr(client, '_calculate_conditional_frequencies')

    def test_full_action_cycle_with_all_features(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Test full cycle with bond-weighted conditional history."""
        client = BondWeightedConditionalAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.2,
            smoothing_alpha=0.15,
            bond_power=1.8,
            random_state=fixed_seed
        )

        # Calculate conditional frequencies for each action
        for action in range(8):
            freq = client._calculate_conditional_frequencies(action)
            assert len(freq) == 8
            assert np.isclose(np.sum(freq), 1.0)

        # Get effective weight (should be scaled by bond)
        effective_weight = client._get_effective_history_weight()
        assert effective_weight <= client.history_weight

        # Calculate payoffs
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8

        # Select action
        action = client.select_action()
        assert 0 <= action <= 7
