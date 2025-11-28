"""
Unit tests for BondWeightedFrequencyAmplifier in
src/agents/client_agents/bond_weighted_frequency_amplifier_client.py.

Tests bond-modulated history influence: effective_weight = (bond ** bond_power) × history_weight.
Extends FrequencyAmplifierClient with bond-scaled history weight.
"""

import pytest
import numpy as np

from src.agents.client_agents.bond_weighted_frequency_amplifier_client import BondWeightedFrequencyAmplifier
from src.config import MEMORY_SIZE, HISTORY_WEIGHT


# ==============================================================================
# TEST BOND-WEIGHTED HISTORY MECHANISM
# ==============================================================================

class TestBondWeightedMechanism:
    """Test bond-based scaling of history weight."""

    def test_inherits_from_frequency_amplifier(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should inherit from FrequencyAmplifierClient."""
        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Should have FrequencyAmplifier methods
        assert hasattr(client, '_calculate_marginal_frequencies')
        assert hasattr(client, '_get_effective_history_weight')

        # Should NOT have conditional methods (those are in ConditionalAmplifier)
        assert not hasattr(client, '_calculate_conditional_frequencies')

    def test_has_bond_power_parameter(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should have bond_power parameter."""
        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            bond_power=2.0,
            random_state=fixed_seed
        )

        assert hasattr(client, 'bond_power')
        assert client.bond_power == 2.0

    def test_default_bond_power(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Default bond_power should be 1.0 (linear scaling)."""
        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert client.bond_power == 1.0

    def test_effective_weight_formula(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Effective weight should be (bond ** bond_power) × history_weight."""
        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=2.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Set specific bond for testing
        client.bond = 0.5

        effective = client._get_effective_history_weight()

        # Should be: (0.5 ** 2.0) × 2.0 = 0.25 × 2.0 = 0.5
        expected = (0.5 ** 2.0) * 2.0
        np.testing.assert_almost_equal(effective, expected, decimal=6)


# ==============================================================================
# TEST BOND SCALING BEHAVIOR
# ==============================================================================

class TestBondScaling:
    """Test how bond level affects history influence."""

    def test_low_bond_reduces_history_influence(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Low bond should reduce history influence."""
        memory = [(0, 0)] * MEMORY_SIZE  # Strong history pattern

        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=2.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Low bond
        client.bond = 0.2
        effective_low = client._get_effective_history_weight()

        # Should be: (0.2 ** 2.0) × 2.0 = 0.04 × 2.0 = 0.08
        assert effective_low < 0.1, f"Low bond should reduce history weight, got {effective_low}"

    def test_high_bond_increases_history_influence(self, fixed_u_matrix, low_entropy, fixed_seed):
        """High bond should maintain strong history influence."""
        memory = [(0, 0)] * MEMORY_SIZE

        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,
            bond_power=1.0,
            random_state=fixed_seed
        )

        # High bond
        client.bond = 0.9
        effective_high = client._get_effective_history_weight()

        # Should be: (0.9 ** 1.0) × 1.0 = 0.9
        np.testing.assert_almost_equal(effective_high, 0.9, decimal=6)

    def test_bond_power_controls_steepness(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Higher bond_power should create steeper drop-off at low bond."""
        # Client with linear scaling (bond_power=1.0)
        client_linear = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.0,
            bond_power=1.0,
            random_state=fixed_seed
        )

        # Client with quadratic scaling (bond_power=2.0)
        client_quadratic = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Test at low bond (0.3)
        client_linear.bond = 0.3
        client_quadratic.bond = 0.3

        eff_linear = client_linear._get_effective_history_weight()
        eff_quadratic = client_quadratic._get_effective_history_weight()

        # Linear: 0.3 × 1.0 = 0.3
        # Quadratic: 0.3² × 1.0 = 0.09
        np.testing.assert_almost_equal(eff_linear, 0.3, decimal=6)
        np.testing.assert_almost_equal(eff_quadratic, 0.09, decimal=6)

        # Quadratic should be much lower
        assert eff_quadratic < eff_linear, "Higher bond_power should reduce weight more at low bond"


# ==============================================================================
# TEST PAYOFF DIFFERENCES
# ==============================================================================

class TestPayoffDifferences:
    """Test that bond affects payoffs through history weighting."""

    def test_same_history_different_bonds_different_payoffs(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Same history but different bonds should give different payoffs."""
        memory = [(0, 4)] * MEMORY_SIZE  # Strong pattern

        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=2.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        # Calculate payoffs at low bond
        client.bond = 0.3
        payoffs_low_bond = client._calculate_expected_payoffs()

        # Calculate payoffs at high bond
        client.bond = 0.9
        payoffs_high_bond = client._calculate_expected_payoffs()

        # Payoffs should differ due to different history influence
        assert not np.allclose(payoffs_low_bond, payoffs_high_bond), \
            "Different bonds should produce different payoffs"

    def test_low_bond_approaches_bond_only(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Very low bond should make behavior approach BondOnlyClient."""
        memory1 = [(0, 0)] * MEMORY_SIZE
        memory2 = [(0, 7)] * MEMORY_SIZE

        # Two clients with different history but very low bond
        client1 = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory1,
            history_weight=1.0,
            bond_power=2.0,
            random_state=fixed_seed
        )

        client2 = BondWeightedFrequencyAmplifier(
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

        # With near-zero effective weight, should behave similarly to BondOnly
        # (not identical due to numerical precision, but very close)
        np.testing.assert_allclose(payoffs1, payoffs2, rtol=0.01)


# ==============================================================================
# TEST INTEGRATION
# ==============================================================================

class TestIntegration:
    """Test integration with base functionality."""

    def test_inherits_all_base_functionality(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should have all BaseClientAgent functionality."""
        client = BondWeightedFrequencyAmplifier(
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

        # FrequencyAmplifier attributes
        assert hasattr(client, 'history_weight')

        # BondWeighted attributes
        assert hasattr(client, 'bond_power')

        # Methods
        assert hasattr(client, 'select_action')
        assert hasattr(client, '_calculate_marginal_frequencies')

    def test_full_action_cycle(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Test full cycle with bond-weighted history."""
        client = BondWeightedFrequencyAmplifier(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.0,
            bond_power=1.5,
            random_state=fixed_seed
        )

        # Calculate frequencies
        frequencies = client._calculate_marginal_frequencies()
        assert len(frequencies) == 8

        # Get effective weight (should be scaled by bond)
        effective_weight = client._get_effective_history_weight()
        assert effective_weight <= client.history_weight

        # Calculate payoffs
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8

        # Select action
        action = client.select_action()
        assert 0 <= action <= 7
