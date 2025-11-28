"""
Unit tests for BondOnlyClient in src/agents/client_agents/bond_only_client.py.

Tests bond-only expectation mechanism: bond determines percentile of raw
utilities, completely ignoring interaction history/frequencies.
"""

import pytest
import numpy as np

from src.agents.client_agents import BondOnlyClient
from src.config import MEMORY_SIZE
from tests.conftest import (
    assert_valid_octant,
    assert_bond_in_range,
)


# ==============================================================================
# TEST BOND-ONLY PAYOFF CALCULATION
# ==============================================================================

class TestBondOnlyPayoffCalculation:
    """Test that BondOnlyClient ignores history and uses only bond."""

    def test_payoffs_ignore_history(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Same bond should give same payoffs regardless of memory pattern."""
        # Two very different memory patterns
        complementary_memory = [(0, 4), (1, 3), (2, 2)] * (MEMORY_SIZE // 3) + [(0, 4), (1, 3)]  # Pad to MEMORY_SIZE
        conflictual_memory = [(0, 0)] * MEMORY_SIZE

        client1 = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        client2 = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=conflictual_memory,
            random_state=fixed_seed
        )

        # Force same bond for both (memory creates different RS/bond initially)
        # Save initial bonds
        bond1_initial = client1.bond
        bond2_initial = client2.bond

        # They should be different initially due to different memory
        assert bond1_initial != bond2_initial

        # Directly set bond to same value for testing payoff calculation
        client1.bond = 0.5
        client2.bond = 0.5

        payoffs1 = client1._calculate_expected_payoffs()
        payoffs2 = client2._calculate_expected_payoffs()

        # Payoffs should be identical despite different memory
        np.testing.assert_array_equal(payoffs1, payoffs2)

    def test_payoffs_only_depend_on_bond_and_umatrix(self, low_entropy, complementary_memory, fixed_seed):
        """Different U_matrices should give different payoffs with same bond."""
        u_matrix1 = np.random.RandomState(42).uniform(-70, 70, (8, 8))
        u_matrix2 = np.random.RandomState(123).uniform(-70, 70, (8, 8))

        client1 = BondOnlyClient(
            u_matrix=u_matrix1,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        client2 = BondOnlyClient(
            u_matrix=u_matrix2,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Force same bond
        client1.bond = 0.7
        client2.bond = 0.7

        payoffs1 = client1._calculate_expected_payoffs()
        payoffs2 = client2._calculate_expected_payoffs()

        # Payoffs should differ due to different U_matrices
        assert not np.array_equal(payoffs1, payoffs2)


# ==============================================================================
# TEST PERCENTILE SELECTION
# ==============================================================================

class TestPercentileSelection:
    """Test bond-based percentile interpolation."""

    def test_bond_zero_selects_minimum(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Bond=0 should select minimum utility for each action."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Force bond to 0
        client.bond = 0.0

        payoffs = client._calculate_expected_payoffs()

        # Each payoff should be the minimum utility for that client action
        for client_action in range(8):
            expected_min = fixed_u_matrix[client_action, :].min()
            np.testing.assert_almost_equal(payoffs[client_action], expected_min, decimal=6)

    def test_bond_one_selects_maximum(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Bond=1 should select maximum utility for each action."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Force bond to 1
        client.bond = 1.0

        payoffs = client._calculate_expected_payoffs()

        # Each payoff should be the maximum utility for that client action
        for client_action in range(8):
            expected_max = fixed_u_matrix[client_action, :].max()
            np.testing.assert_almost_equal(payoffs[client_action], expected_max, decimal=6)

    def test_bond_half_selects_median(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Bond=0.5 should select near-median utility for each action."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Force bond to 0.5
        client.bond = 0.5

        payoffs = client._calculate_expected_payoffs()

        # Each payoff should be near median
        for client_action in range(8):
            sorted_utils = np.sort(fixed_u_matrix[client_action, :])
            # Bond=0.5 → position = 0.5 * 7 = 3.5
            # Should interpolate between index 3 and 4
            expected_payoff = (sorted_utils[3] + sorted_utils[4]) / 2
            np.testing.assert_almost_equal(payoffs[client_action], expected_payoff, decimal=6)

    def test_bond_monotonicity(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Higher bond should give higher or equal expected payoffs."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Test multiple bond levels
        bond_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        payoffs_by_bond = []

        for bond in bond_levels:
            client.bond = bond
            payoffs = client._calculate_expected_payoffs()
            payoffs_by_bond.append(payoffs)

        # For each client action, payoffs should be monotonically increasing
        for client_action in range(8):
            payoffs_for_action = [p[client_action] for p in payoffs_by_bond]

            # Check monotonicity
            for i in range(1, len(payoffs_for_action)):
                assert payoffs_for_action[i] >= payoffs_for_action[i-1], \
                    f"Action {client_action}: payoff decreased from bond {bond_levels[i-1]} to {bond_levels[i]}"

    def test_interpolation_accuracy(self, low_entropy, complementary_memory, fixed_seed):
        """Test exact interpolation formula for specific bond values."""
        # Create simple U_matrix for easy verification
        u_matrix = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],  # Utilities already sorted for client action 0
            [-10, -5, 0, 5, 10, 15, 20, 25],  # Sorted for action 1
            [10, 10, 10, 10, 10, 10, 10, 10],  # All same for action 2
            [7, 6, 5, 4, 3, 2, 1, 0],  # Reverse sorted for action 3
            [0, 0, 0, 0, 100, 100, 100, 100],  # Step function for action 4
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],  # With decimals for action 5
            [-50, -40, -30, -20, -10, 0, 10, 20],  # Mix of negative/positive for action 6
            [0, 10, 20, 30, 40, 50, 60, 70],  # Linear spacing for action 7
        ])

        client = BondOnlyClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Test bond=0.25 (position = 1.75, interpolate between index 1 and 2)
        client.bond = 0.25
        payoffs = client._calculate_expected_payoffs()

        # For action 0 (already sorted): [0,1,2,3,4,5,6,7]
        # position=1.75 → lower_idx=1, upper_idx=2, weight=0.75
        # (1-0.75)*sorted[1] + 0.75*sorted[2] = 0.25*1 + 0.75*2 = 1.75
        expected_action_0 = 0.25 * 1 + 0.75 * 2
        np.testing.assert_almost_equal(payoffs[0], expected_action_0, decimal=6)

        # For action 2 (all same): should be 10 regardless
        assert payoffs[2] == 10.0


# ==============================================================================
# TEST HISTORY INDEPENDENCE
# ==============================================================================

class TestHistoryIndependence:
    """Test that BondOnlyClient truly ignores interaction history."""

    def test_memory_updates_dont_change_payoffs_at_same_bond(
        self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """Updating memory should not change payoffs if bond stays constant."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Record initial bond and payoffs
        initial_bond = client.bond

        # Force specific bond
        client.bond = 0.6
        initial_payoffs = client._calculate_expected_payoffs()

        # Update memory with various interactions
        for _ in range(10):
            client.update_memory(client_action=0, therapist_action=0)  # Conflictual

        # Force bond back to same value
        client.bond = 0.6
        after_payoffs = client._calculate_expected_payoffs()

        # Payoffs should be identical
        np.testing.assert_array_equal(initial_payoffs, after_payoffs)

    def test_no_frequency_calculation_method(self, bond_only_client):
        """BondOnlyClient should not have frequency calculation methods."""
        # Should NOT have these methods (used by amplifier mechanisms)
        assert not hasattr(bond_only_client, '_calculate_marginal_frequencies')
        assert not hasattr(bond_only_client, '_calculate_conditional_frequencies')

    def test_no_history_weight_parameter(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """BondOnlyClient should not have history_weight parameter."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Should NOT have history_weight attribute
        assert not hasattr(client, 'history_weight')


# ==============================================================================
# TEST INTEGRATION WITH BASE CLIENT
# ==============================================================================

class TestIntegrationWithBase:
    """Test that BondOnlyClient properly integrates with BaseClientAgent."""

    def test_inherits_all_base_functionality(self, bond_only_client):
        """Should have all BaseClientAgent methods and attributes."""
        # Core attributes
        assert hasattr(bond_only_client, 'u_matrix')
        assert hasattr(bond_only_client, 'entropy')
        assert hasattr(bond_only_client, 'memory')
        assert hasattr(bond_only_client, 'bond')
        assert hasattr(bond_only_client, 'relationship_satisfaction')

        # Core methods
        assert hasattr(bond_only_client, 'select_action')
        assert hasattr(bond_only_client, 'update_memory')
        assert hasattr(bond_only_client, 'check_dropout')
        assert hasattr(bond_only_client, '_calculate_relationship_satisfaction')
        assert hasattr(bond_only_client, '_calculate_bond')

    def test_full_action_cycle(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Test full cycle: calculate payoffs → select action → valid octant."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Calculate payoffs
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8

        # Select action based on payoffs
        action = client.select_action()
        assert_valid_octant(action)

        # With low entropy, should select action with highest payoff most of the time
        # (but not guaranteed due to softmax)
        assert isinstance(action, (int, np.integer))
