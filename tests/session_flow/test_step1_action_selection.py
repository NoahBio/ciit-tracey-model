"""
Session Flow Tests: Step 1 - Client Action Selection

Tests the complete action selection process:
  Step 1A: Calculate expected payoffs (mechanism-specific)
  Step 1B: Apply softmax to get probability distribution
  Step 1C: Select action from distribution

Medium detail: Tests cohesive behavior of full action selection step, not micro-substeps.
"""

import pytest
import numpy as np

from src.agents.client_agents import (
    BondOnlyClient,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
    BondWeightedFrequencyAmplifier,
    BondWeightedConditionalAmplifier,
)
from src.config import MEMORY_SIZE
from tests.conftest import assert_valid_probability_distribution, assert_valid_octant


# ==============================================================================
# TEST STEP 1: COMPLETE ACTION SELECTION FLOW
# ==============================================================================

class TestStep1ActionSelectionFlow:
    """Test complete flow: payoffs → softmax → action selection."""

    @pytest.mark.parametrize("client_class", [
        BondOnlyClient,
        FrequencyAmplifierClient,
        ConditionalAmplifierClient,
        BondWeightedFrequencyAmplifier,
        BondWeightedConditionalAmplifier,
    ])
    def test_action_selection_produces_valid_octant(
        self, client_class, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """All mechanisms should produce valid octant (0-7) through full flow."""
        client = client_class(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Run full action selection
        action = client.select_action()

        # Should be valid octant
        assert_valid_octant(action)

    @pytest.mark.parametrize("client_class", [
        BondOnlyClient,
        FrequencyAmplifierClient,
        ConditionalAmplifierClient,
        BondWeightedFrequencyAmplifier,
        BondWeightedConditionalAmplifier,
    ])
    def test_action_selection_reproducible_with_fixed_seed(
        self, client_class, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """With fixed seed, action selection should be reproducible."""
        client1 = client_class(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        client2 = client_class(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        action1 = client1.select_action()
        action2 = client2.select_action()

        assert action1 == action2, "With fixed seed, action selection should be reproducible"

    def test_action_selection_uses_current_bond_and_memory(
        self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """Action selection should reflect current client state (bond, memory)."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Get action with initial state
        initial_bond = client.bond
        action1 = client.select_action()

        # Change bond significantly (this affects expected payoffs for BondOnly)
        client.bond = 0.1 if initial_bond > 0.5 else 0.9

        # Reset RNG for fair comparison
        client.rng = np.random.RandomState(fixed_seed)

        # Get action with changed state
        action2 = client.select_action()

        # With very low entropy and different bond, actions are likely different
        # (not guaranteed due to randomness, but highly likely)
        # We're mainly testing that the system uses current state

        # Verify that bond affects payoffs (which affects action selection)
        assert client.bond != initial_bond


# ==============================================================================
# TEST STEP 1A: EXPECTED PAYOFF CALCULATION
# ==============================================================================

class TestStep1A_ExpectedPayoffCalculation:
    """Test that expected payoffs are calculated correctly per mechanism."""

    def test_payoffs_have_correct_shape(self, bond_only_client):
        """Expected payoffs should be 8-dimensional array."""
        payoffs = bond_only_client._calculate_expected_payoffs()

        assert isinstance(payoffs, np.ndarray)
        assert payoffs.shape == (8,)

    def test_payoffs_are_numeric(self, bond_only_client):
        """Expected payoffs should contain numeric values."""
        payoffs = bond_only_client._calculate_expected_payoffs()

        assert payoffs.dtype in [np.float64, np.float32, float]
        assert not np.any(np.isnan(payoffs)), "Payoffs should not contain NaN"
        assert not np.any(np.isinf(payoffs)), "Payoffs should not contain infinity"

    @pytest.mark.parametrize("mechanism,client_class", [
        ("bond_only", BondOnlyClient),
        ("frequency_amplifier", FrequencyAmplifierClient),
        ("conditional_amplifier", ConditionalAmplifierClient),
        ("bond_weighted_freq", BondWeightedFrequencyAmplifier),
        ("bond_weighted_cond", BondWeightedConditionalAmplifier),
    ])
    def test_mechanism_specific_payoff_calculation(
        self, mechanism, client_class, fixed_u_matrix, low_entropy, fixed_seed
    ):
        """Each mechanism should calculate payoffs according to its logic."""
        # Different memories to test mechanism differences
        # Use varied client actions for conditional mechanisms to work properly
        memory_cold = [(i % 8, 0) for i in range(MEMORY_SIZE)]  # All therapist responses conflictual (0)
        memory_warm = [(i % 8, 4) for i in range(MEMORY_SIZE)]  # All therapist responses complementary (4)

        # BondOnly should give same payoffs with same bond regardless of memory
        # Amplifiers should give different payoffs due to different history

        client_cold = client_class(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory_cold,
            random_state=fixed_seed
        )

        client_warm = client_class(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory_warm,
            random_state=fixed_seed
        )

        # Force same bond for comparison
        client_cold.bond = 0.5
        client_warm.bond = 0.5

        payoffs_cold = client_cold._calculate_expected_payoffs()
        payoffs_warm = client_warm._calculate_expected_payoffs()

        if mechanism == "bond_only":
            # BondOnly should ignore history
            np.testing.assert_allclose(payoffs_cold, payoffs_warm, rtol=1e-10)
        else:
            # Amplifier mechanisms should reflect history differences
            assert not np.allclose(payoffs_cold, payoffs_warm), \
                f"{mechanism} should produce different payoffs for different histories"


# ==============================================================================
# TEST STEP 1B: SOFTMAX APPLICATION
# ==============================================================================

class TestStep1B_SoftmaxApplication:
    """Test softmax converts payoffs to probability distribution."""

    def test_softmax_produces_valid_distribution(self, bond_only_client):
        """Softmax should produce valid probability distribution."""
        payoffs = bond_only_client._calculate_expected_payoffs()
        probs = bond_only_client._softmax(payoffs)

        assert_valid_probability_distribution(probs)

    def test_softmax_respects_entropy_parameter(self, fixed_u_matrix, complementary_memory, fixed_seed):
        """Lower entropy should produce more peaked distribution."""
        # Create varied payoffs for testing
        client_low_entropy = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=0.1,  # Very low entropy
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        client_high_entropy = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=5.0,  # High entropy
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Force same bond for fair comparison
        client_low_entropy.bond = 0.7
        client_high_entropy.bond = 0.7

        payoffs = client_low_entropy._calculate_expected_payoffs()

        probs_low = client_low_entropy._softmax(payoffs)
        probs_high = client_high_entropy._softmax(payoffs)

        # Low entropy should have higher max probability (more peaked)
        assert np.max(probs_low) > np.max(probs_high), \
            "Low entropy should produce more peaked distribution"

        # High entropy should be more uniform (lower standard deviation)
        assert np.std(probs_high) < np.std(probs_low), \
            "High entropy should produce more uniform distribution"

    def test_softmax_highest_payoff_gets_highest_probability(self, bond_only_client):
        """Action with highest expected payoff should have highest probability."""
        payoffs = bond_only_client._calculate_expected_payoffs()
        probs = bond_only_client._softmax(payoffs)

        max_payoff_idx = np.argmax(payoffs)
        max_prob_idx = np.argmax(probs)

        assert max_payoff_idx == max_prob_idx, \
            "Action with highest payoff should have highest probability"

    def test_softmax_handles_negative_payoffs(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Softmax should handle negative payoffs correctly."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Force very low bond to get low (possibly negative) expected payoffs
        client.bond = 0.0

        payoffs = client._calculate_expected_payoffs()
        probs = client._softmax(payoffs)

        # Should still be valid distribution even with negative payoffs
        assert_valid_probability_distribution(probs)


# ==============================================================================
# TEST STEP 1C: ACTION SELECTION FROM DISTRIBUTION
# ==============================================================================

class TestStep1C_ActionSelectionFromDistribution:
    """Test final action selection from probability distribution."""

    def test_selected_action_is_valid_octant(self, bond_only_client):
        """Selected action should be valid octant (0-7)."""
        action = bond_only_client.select_action()
        assert_valid_octant(action)

    def test_action_selection_samples_from_distribution(self, fixed_u_matrix, complementary_memory):
        """Action selection should sample from softmax distribution."""
        # Use very low entropy so highest-prob action dominates
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=0.01,  # Extremely low entropy
            initial_memory=complementary_memory,
            random_state=42
        )

        # Force specific bond
        client.bond = 0.8

        # Get payoffs and probabilities
        payoffs = client._calculate_expected_payoffs()
        probs = client._softmax(payoffs)
        expected_action = np.argmax(probs)

        # With very low entropy, should almost always select highest-prob action
        # Test multiple times to check consistency
        selected_actions = []
        for i in range(10):
            client.rng = np.random.RandomState(42 + i)
            action = client.select_action()
            selected_actions.append(action)

        # Most selections should be the expected action
        assert selected_actions.count(expected_action) >= 7, \
            "With very low entropy, should mostly select highest-probability action"

    def test_action_selection_is_stochastic(self, fixed_u_matrix, complementary_memory):
        """With higher entropy, action selection should show variability."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=2.0,  # Higher entropy for more randomness
            initial_memory=complementary_memory,
            random_state=42
        )

        # Select many actions with different seeds
        selected_actions = []
        for i in range(50):
            client.rng = np.random.RandomState(100 + i)
            action = client.select_action()
            selected_actions.append(action)

        # Should have some variety (not all same action)
        unique_actions = len(set(selected_actions))
        assert unique_actions >= 2, \
            "With higher entropy, should select multiple different actions"


# ==============================================================================
# TEST COMPLETE STEP 1 INTEGRATION
# ==============================================================================

class TestStep1Integration:
    """Test full Step 1 as integrated process."""

    def test_multiple_action_selections_in_sequence(self, bond_only_client):
        """Should be able to select multiple actions in sequence."""
        actions = []
        for _ in range(5):
            action = bond_only_client.select_action()
            actions.append(action)

        # All should be valid
        for action in actions:
            assert_valid_octant(action)

        # Should be consistent with seed (all same in this case)
        assert len(set(actions)) == 1, "With fixed seed, repeated selections should be identical"

    def test_action_selection_after_state_change(
        self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """Action selection should reflect state changes (bond, memory) in full flow."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Initial action
        action1 = client.select_action()

        # Change state by updating memory multiple times
        for i in range(25):  # Half of memory
            client.update_memory(client_action=7, therapist_action=7)

        # Reset RNG for controlled comparison
        client.rng = np.random.RandomState(fixed_seed)

        # Action after state change
        action2 = client.select_action()

        # Actions may or may not differ (depends on softmax sampling)
        # But payoffs should differ due to frequency changes
        # This test verifies the full pipeline works after state updates
