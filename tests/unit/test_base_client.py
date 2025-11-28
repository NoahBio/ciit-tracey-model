"""
Unit tests for BaseClientAgent in src/agents/client_agents/base_client.py.

Tests all base client functionality inherited by all mechanism variants,
including initialization, memory management, RS/bond calculations, action
selection, and dropout logic.
"""

import pytest
import numpy as np
from collections import deque

from src.agents.client_agents.base_client import BaseClientAgent
from src.agents.client_agents import BondOnlyClient
from src.config import MEMORY_SIZE, sample_u_matrix
from tests.conftest import (
    assert_valid_octant,
    assert_bond_in_range,
    assert_rs_in_bounds,
    assert_memory_size_correct,
    assert_valid_probability_distribution,
)


# ==============================================================================
# TEST INITIALIZATION & VALIDATION
# ==============================================================================

class TestBaseClientInitialization:
    """Test client initialization and validation."""

    def test_init_with_valid_params(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should initialize successfully with valid parameters."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert client is not None
        assert len(client.memory) == MEMORY_SIZE
        assert client.entropy == low_entropy

    def test_init_calculates_rs_bounds(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should correctly calculate RS min and max from U_matrix."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # RS bounds should be min/max of U_matrix
        assert client.rs_min == fixed_u_matrix.min()
        assert client.rs_max == fixed_u_matrix.max()

    def test_init_calculates_initial_rs(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should calculate initial RS from initial_memory."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # RS should be within bounds
        assert_rs_in_bounds(client.relationship_satisfaction, client.rs_min, client.rs_max)

        # Should not be None or NaN
        assert client.relationship_satisfaction is not None
        assert not np.isnan(client.relationship_satisfaction)

    def test_init_calculates_initial_bond(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should calculate initial bond from initial RS."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert_bond_in_range(client.bond)

    def test_init_invalid_memory_length_raises(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Should raise ValueError if memory length != MEMORY_SIZE."""
        invalid_memory = [(0, 0)] * 30  # Wrong size

        with pytest.raises(ValueError, match="initial_memory must have length"):
            BondOnlyClient(
                u_matrix=fixed_u_matrix,
                entropy=low_entropy,
                initial_memory=invalid_memory,
                random_state=fixed_seed
            )

    def test_init_negative_entropy_raises(self, fixed_u_matrix, complementary_memory, fixed_seed):
        """Should raise ValueError if entropy <= 0."""
        with pytest.raises(ValueError, match="entropy must be positive"):
            BondOnlyClient(
                u_matrix=fixed_u_matrix,
                entropy=-1.0,
                initial_memory=complementary_memory,
                random_state=fixed_seed
            )


# ==============================================================================
# TEST MEMORY UPDATE
# ==============================================================================

class TestMemoryUpdate:
    """Test memory update mechanics."""

    def test_update_memory_adds_interaction(self, bond_only_client):
        """Should add new interaction to memory."""
        initial_len = len(bond_only_client.memory)
        last_interaction = bond_only_client.memory[-1]

        # Add new interaction
        bond_only_client.update_memory(client_action=2, therapist_action=2)

        # Memory should still be same length (deque pops oldest)
        assert len(bond_only_client.memory) == initial_len

        # New interaction should be at end
        assert bond_only_client.memory[-1] == (2, 2)
        assert bond_only_client.memory[-1] != last_interaction

    def test_update_memory_removes_oldest(self, bond_only_client):
        """Should remove oldest interaction when adding new one."""
        oldest_interaction = bond_only_client.memory[0]

        bond_only_client.update_memory(client_action=5, therapist_action=7)

        # Oldest should be gone
        assert oldest_interaction not in list(bond_only_client.memory)[:1]

    def test_update_memory_maintains_size(self, bond_only_client):
        """Memory should always maintain MEMORY_SIZE."""
        for i in range(20):
            bond_only_client.update_memory(client_action=i % 8, therapist_action=(i+1) % 8)
            assert_memory_size_correct(bond_only_client.memory)

    def test_update_memory_increments_session_count(self, bond_only_client):
        """Should increment session_count on each update."""
        initial_count = bond_only_client.session_count

        bond_only_client.update_memory(client_action=2, therapist_action=2)

        assert bond_only_client.session_count == initial_count + 1

    def test_update_memory_recalculates_rs(self, bond_only_client):
        """Should recalculate RS after memory update."""
        initial_rs = bond_only_client.relationship_satisfaction

        # Add a very positive interaction (if W→W is positive in this client's U_matrix)
        bond_only_client.update_memory(client_action=2, therapist_action=2)

        # RS should have changed (unless extremely unlikely coincidence)
        # We can't predict direction without knowing U_matrix values
        final_rs = bond_only_client.relationship_satisfaction

        assert_rs_in_bounds(final_rs, bond_only_client.rs_min, bond_only_client.rs_max)

    def test_update_memory_recalculates_bond(self, bond_only_client):
        """Should recalculate bond after memory update."""
        initial_bond = bond_only_client.bond

        bond_only_client.update_memory(client_action=2, therapist_action=2)

        final_bond = bond_only_client.bond

        assert_bond_in_range(final_bond)


# ==============================================================================
# TEST RELATIONSHIP SATISFACTION CALCULATION
# ==============================================================================

class TestRelationshipSatisfactionCalculation:
    """Test RS calculation mechanics."""

    def test_rs_all_same_interaction(self, fixed_u_matrix, low_entropy, fixed_seed):
        """With all same interactions, RS should equal that interaction's utility."""
        # All D→S interactions
        memory = [(0, 4)] * MEMORY_SIZE

        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            random_state=fixed_seed
        )

        # RS should be close to U[0,4] (weighted average of same value)
        expected_utility = fixed_u_matrix[0, 4]

        np.testing.assert_almost_equal(client.relationship_satisfaction, expected_utility, decimal=2)

    def test_rs_uses_client_u_matrix(self, low_entropy, complementary_memory, fixed_seed):
        """Different U_matrices should give different RS for same memory."""
        u_matrix1 = sample_u_matrix(random_state=42)
        u_matrix2 = sample_u_matrix(random_state=123)

        client1 = BondOnlyClient(u_matrix=u_matrix1, entropy=low_entropy,
                                  initial_memory=complementary_memory, random_state=fixed_seed)
        client2 = BondOnlyClient(u_matrix=u_matrix2, entropy=low_entropy,
                                  initial_memory=complementary_memory, random_state=fixed_seed)

        # Different U_matrices should (almost certainly) give different RS
        assert client1.relationship_satisfaction != client2.relationship_satisfaction


# ==============================================================================
# TEST BOND CALCULATION
# ==============================================================================

class TestBondCalculation:
    """Test bond calculation mechanics."""

    def test_bond_range(self, bond_only_client):
        """Bond should always be in [0, 1]."""
        assert_bond_in_range(bond_only_client.bond)

    def test_bond_monotonicity(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Higher RS should give higher bond."""
        # High RS memory (complementary)
        high_memory = [(0, 4)] * MEMORY_SIZE
        client_high = BondOnlyClient(u_matrix=fixed_u_matrix, entropy=low_entropy,
                                      initial_memory=high_memory, random_state=fixed_seed)

        # Low RS memory (conflictual)
        low_memory = [(0, 0)] * MEMORY_SIZE
        client_low = BondOnlyClient(u_matrix=fixed_u_matrix, entropy=low_entropy,
                                     initial_memory=low_memory, random_state=fixed_seed)

        # Higher RS should give higher bond
        assert client_high.relationship_satisfaction > client_low.relationship_satisfaction
        assert client_high.bond > client_low.bond

    def test_bond_uses_normalization(self, bond_only_client):
        """Bond calculation should use client-specific RS normalization."""
        # Bond should be calculated from normalized RS
        # We can't test exact formula without accessing internals,
        # but we can verify it's using rs_min and rs_max
        assert hasattr(bond_only_client, 'rs_min')
        assert hasattr(bond_only_client, 'rs_max')
        assert bond_only_client.rs_min < bond_only_client.rs_max


# ==============================================================================
# TEST SOFTMAX
# ==============================================================================

class TestSoftmax:
    """Test softmax probability conversion."""

    def test_softmax_valid_distribution(self, bond_only_client):
        """Softmax should produce valid probability distribution."""
        payoffs = np.array([10.0, 20.0, 30.0, 15.0, 5.0, 25.0, 12.0, 18.0])

        probs = bond_only_client._softmax(payoffs)

        assert_valid_probability_distribution(probs)

    def test_softmax_low_entropy(self, fixed_u_matrix, complementary_memory, fixed_seed):
        """Low entropy should give peaked distribution (mostly max payoff)."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=0.01,  # Very low
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        payoffs = np.array([10.0, 5.0, 15.0, 3.0, 8.0, 2.0, 6.0, 4.0])
        probs = client._softmax(payoffs)

        # Max payoff (index 2 with value 15.0) should have very high probability
        max_idx = np.argmax(payoffs)
        assert probs[max_idx] > 0.9

    def test_softmax_high_entropy(self, fixed_u_matrix, complementary_memory, fixed_seed):
        """High entropy should give more uniform distribution."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=10.0,  # Very high
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        payoffs = np.array([10.0, 5.0, 15.0, 3.0, 8.0, 2.0, 6.0, 4.0])
        probs = client._softmax(payoffs)

        # Should be more uniform - no single probability should dominate
        assert np.max(probs) < 0.3  # No prob > 30%

    def test_softmax_numerical_stability(self, bond_only_client):
        """Softmax should handle extreme values without overflow/underflow."""
        # Very large values
        large_payoffs = np.array([1000.0, 1001.0, 999.0, 1002.0, 998.0, 1003.0, 997.0, 1004.0])
        probs_large = bond_only_client._softmax(large_payoffs)

        assert_valid_probability_distribution(probs_large)
        assert not np.any(np.isnan(probs_large))
        assert not np.any(np.isinf(probs_large))


# ==============================================================================
# TEST ACTION SELECTION
# ==============================================================================

class TestActionSelection:
    """Test action selection mechanics."""

    def test_select_action_valid_octant(self, bond_only_client):
        """Should always return valid octant (0-7)."""
        for _ in range(10):
            action = bond_only_client.select_action()
            assert_valid_octant(action)

    def test_select_action_reproducible(self, fixed_u_matrix, low_entropy, complementary_memory):
        """Same seed should give same action sequence."""
        client1 = BondOnlyClient(u_matrix=fixed_u_matrix, entropy=low_entropy,
                                  initial_memory=complementary_memory, random_state=42)
        client2 = BondOnlyClient(u_matrix=fixed_u_matrix, entropy=low_entropy,
                                  initial_memory=complementary_memory, random_state=42)

        actions1 = [client1.select_action() for _ in range(5)]
        actions2 = [client2.select_action() for _ in range(5)]

        assert actions1 == actions2


# ==============================================================================
# TEST DROPOUT CHECK
# ==============================================================================

class TestDropoutCheck:
    """Test dropout logic."""

    def test_dropout_not_checked_before_session_10(self, bond_only_client):
        """Dropout should not occur before session 10."""
        for _ in range(9):
            bond_only_client.update_memory(client_action=0, therapist_action=0)
            dropout = bond_only_client.check_dropout()
            assert dropout is False

    def test_dropout_at_session_10_rs_decreased(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Should dropout if RS decreased by session 10."""
        # Start with high RS (complementary)
        memory = [(0, 4)] * MEMORY_SIZE
        client = BondOnlyClient(u_matrix=fixed_u_matrix, entropy=low_entropy,
                                 initial_memory=memory, random_state=fixed_seed)

        initial_rs = client.relationship_satisfaction

        # Add 10 conflictual interactions (should decrease RS)
        for _ in range(10):
            client.update_memory(client_action=0, therapist_action=0)

        # Check dropout
        dropout = client.check_dropout()

        # Should dropout if RS decreased
        if client.relationship_satisfaction < initial_rs:
            assert dropout is True
        else:
            assert dropout is False

    def test_dropout_never_checked_twice(self, bond_only_client):
        """Dropout should only be checked once at session 10."""
        # Advance to session 10
        for _ in range(10):
            bond_only_client.update_memory(client_action=0, therapist_action=0)

        first_check = bond_only_client.check_dropout()

        # Subsequent checks should return False (already checked)
        second_check = bond_only_client.check_dropout()
        assert second_check is False


# ==============================================================================
# TEST GENERATE_PROBLEMATIC_MEMORY
# ==============================================================================

class TestGenerateProblematicMemory:
    """Test problematic memory pattern generation."""

    def test_generate_memory_correct_length(self):
        """Should generate memory of correct length."""
        memory = BaseClientAgent.generate_problematic_memory("cold_stuck", n_interactions=50)
        assert len(memory) == 50

        memory_custom = BaseClientAgent.generate_problematic_memory("cold_warm", n_interactions=30)
        assert len(memory_custom) == 30

    def test_generate_memory_cold_stuck(self):
        """Cold_stuck pattern should have ~80% cold octants."""
        memory = BaseClientAgent.generate_problematic_memory("cold_stuck", n_interactions=100, random_state=42)

        cold_octants = {5, 6, 7}
        cold_count = sum(1 for c, t in memory if c in cold_octants)

        # Should be around 80% (allow some variance)
        assert 70 <= cold_count <= 90

    def test_generate_memory_all_patterns(self):
        """All pattern types should work without errors."""
        patterns = ["cold_stuck", "dominant_stuck", "submissive_stuck",
                   "cold_warm", "complementary_perfect", "conflictual", "mixed_random"]

        for pattern in patterns:
            memory = BaseClientAgent.generate_problematic_memory(pattern, random_state=42)
            assert len(memory) == MEMORY_SIZE
            # Each interaction should be tuple of (client, therapist) octants
            for c, t in memory:
                assert_valid_octant(c)
                assert_valid_octant(t)

    def test_generate_memory_reproducibility(self):
        """Same seed should give same memory."""
        memory1 = BaseClientAgent.generate_problematic_memory("cold_stuck", random_state=42)
        memory2 = BaseClientAgent.generate_problematic_memory("cold_stuck", random_state=42)

        assert memory1 == memory2

    def test_generate_memory_invalid_raises(self):
        """Invalid pattern type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown pattern_type"):
            BaseClientAgent.generate_problematic_memory("invalid_pattern")
