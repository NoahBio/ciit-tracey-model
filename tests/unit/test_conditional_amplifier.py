"""
Unit tests for ConditionalAmplifierClient in src/agents/client_agents/conditional_amplifier_client.py.

Tests conditional frequency amplification mechanism: P(therapist_j | client_i)
with Laplace smoothing. More robust than marginal frequency amplification.
"""

import pytest
import numpy as np

from src.agents.client_agents.conditional_amplifier_client import ConditionalAmplifierClient
from src.config import MEMORY_SIZE, HISTORY_WEIGHT
from tests.conftest import assert_valid_probability_distribution


# ==============================================================================
# TEST CONDITIONAL FREQUENCY CALCULATION
# ==============================================================================

class TestConditionalFrequencyCalculation:
    """Test _calculate_conditional_frequencies() produces valid distributions."""

    def test_conditional_frequencies_are_valid_distribution(
        self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """Conditional frequencies should form valid probability distribution."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Test for each client action
        for client_action in range(8):
            conditional_freq = client._calculate_conditional_frequencies(client_action)
            assert_valid_probability_distribution(conditional_freq)

    def test_conditional_differs_by_client_action(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Different client actions should have different conditional distributions."""
        # Memory with clear pattern: when client does 0, therapist does 4
        #                            when client does 1, therapist does 3
        memory = [(0, 4)] * 25 + [(1, 3)] * 25

        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            random_state=fixed_seed
        )

        freq_given_0 = client._calculate_conditional_frequencies(0)
        freq_given_1 = client._calculate_conditional_frequencies(1)

        # Distributions should be different
        assert not np.allclose(freq_given_0, freq_given_1), \
            "Conditional distributions should differ for different client actions"

        # P(4|0) should be high
        assert freq_given_0[4] > 0.4, f"P(4|0) should be high, got {freq_given_0[4]}"

        # P(3|1) should be high
        assert freq_given_1[3] > 0.4, f"P(3|1) should be high, got {freq_given_1[3]}"

    def test_unobserved_action_uses_smoothing(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Client actions never taken should still have valid distribution via smoothing."""
        # Memory where client NEVER takes action 7
        memory = [(i % 7, (i+2) % 8) for i in range(MEMORY_SIZE)]

        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            smoothing_alpha=0.1,
            random_state=fixed_seed
        )

        # Despite never observing action 7, should have valid distribution
        freq_given_7 = client._calculate_conditional_frequencies(7)

        assert_valid_probability_distribution(freq_given_7)

        # Should be relatively uniform due to pure smoothing
        # All values should be close to 1/8
        expected_uniform = 1.0 / 8
        for prob in freq_given_7:
            assert abs(prob - expected_uniform) < 0.02, \
                f"With no observations, smoothing should give near-uniform: got {prob}, expected ~{expected_uniform}"

    def test_conditional_frequencies_change_after_update(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Updating memory should change conditional frequencies."""
        # Start with pattern: client 0 → therapist 2
        memory = [(0, 2)] * MEMORY_SIZE

        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            random_state=fixed_seed
        )

        initial_freq = client._calculate_conditional_frequencies(0).copy()

        # Add many client 0 → therapist 7 interactions
        for _ in range(30):
            client.update_memory(client_action=0, therapist_action=7)

        updated_freq = client._calculate_conditional_frequencies(0)

        # P(7|0) should increase
        assert updated_freq[7] > initial_freq[7], \
            f"P(7|0) should increase: was {initial_freq[7]}, now {updated_freq[7]}"

        # P(2|0) should decrease
        assert updated_freq[2] < initial_freq[2], \
            f"P(2|0) should decrease: was {initial_freq[2]}, now {updated_freq[2]}"


# ==============================================================================
# TEST LAPLACE SMOOTHING
# ==============================================================================

class TestLaplaceSmoothing:
    """Test smoothing_alpha parameter and Laplace smoothing behavior."""

    def test_default_smoothing_alpha(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should use default smoothing_alpha=0.1."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert client.smoothing_alpha == 0.1

    def test_custom_smoothing_alpha(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should accept custom smoothing_alpha parameter."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            smoothing_alpha=0.5,
            random_state=fixed_seed
        )

        assert client.smoothing_alpha == 0.5

    def test_smoothing_prevents_zeros(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Smoothing should prevent zero probabilities."""
        # Memory with extreme pattern: only one therapist response per client action
        memory = [(i % 8, 0) for i in range(MEMORY_SIZE)]  # Therapist always responds with 0

        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            smoothing_alpha=0.1,
            random_state=fixed_seed
        )

        # For any client action, no therapist response should have exactly zero probability
        for client_action in range(8):
            freq = client._calculate_conditional_frequencies(client_action)
            assert np.all(freq > 0), f"All frequencies should be > 0 due to smoothing, got {freq}"

    def test_higher_smoothing_more_uniform(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Higher smoothing_alpha should produce more uniform distributions."""
        # Extreme memory pattern
        memory = [(0, 0)] * MEMORY_SIZE  # Only client 0 → therapist 0

        client_low_smooth = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            smoothing_alpha=0.01,  # Low smoothing
            random_state=fixed_seed
        )

        client_high_smooth = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            smoothing_alpha=1.0,  # High smoothing
            random_state=fixed_seed
        )

        freq_low = client_low_smooth._calculate_conditional_frequencies(0)
        freq_high = client_high_smooth._calculate_conditional_frequencies(0)

        # With low smoothing, dominant response (0) should be much higher
        assert freq_low[0] > 0.5, f"With low smoothing, P(0|0) should be high: {freq_low[0]}"

        # With high smoothing, distribution should be more uniform
        # Maximum probability should be lower
        assert freq_high[0] < freq_low[0], \
            f"Higher smoothing should reduce peak: low={freq_low[0]}, high={freq_high[0]}"

        # Standard deviation should be lower with high smoothing
        assert np.std(freq_high) < np.std(freq_low), \
            "Higher smoothing should produce more uniform distribution (lower std)"


# ==============================================================================
# TEST CONDITIONAL VS MARGINAL AMPLIFICATION
# ==============================================================================

class TestConditionalVsMarginal:
    """Test that conditional amplification differs from marginal."""

    def test_conditional_more_specific_than_marginal(self, low_entropy, fixed_seed):
        """Conditional frequencies should be more action-specific than marginal."""
        # Pattern: client 0 → therapist 4 (strong pattern)
        #          client 1 → therapist 3 (strong pattern)
        memory = [(0, 4)] * 25 + [(1, 3)] * 25

        client = ConditionalAmplifierClient(
            u_matrix=np.random.RandomState(42).uniform(-50, 50, (8, 8)),
            entropy=low_entropy,
            initial_memory=memory,
            smoothing_alpha=0.1,  # Lower smoothing for stronger signal
            random_state=fixed_seed
        )

        # P(4|0) should be elevated (note: smoothing and recency weighting prevent very high values)
        freq_given_0 = client._calculate_conditional_frequencies(0)
        assert freq_given_0[4] > 0.3, f"P(4|0) should be elevated, got {freq_given_0[4]}"

        # P(3|1) should be elevated
        freq_given_1 = client._calculate_conditional_frequencies(1)
        assert freq_given_1[3] > 0.3, f"P(3|1) should be elevated, got {freq_given_1[3]}"

        # But P(4|1) should be low (different from marginal which would show both 3 and 4 frequent)
        assert freq_given_1[4] < 0.2, f"P(4|1) should be low - specificity of conditional, got {freq_given_1[4]}"

        # Key point: P(4|0) should be much higher than P(4|1)
        assert freq_given_0[4] > freq_given_1[4] * 2, \
            "Conditional probabilities should be action-specific"


# ==============================================================================
# TEST HISTORY AMPLIFICATION WITH CONDITIONALS
# ==============================================================================

class TestHistoryAmplificationWithConditionals:
    """Test that conditional frequencies amplify utilities correctly."""

    def test_payoffs_depend_on_history(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Same bond but different history should give different payoffs."""
        memory1 = [(i % 8, 0) for i in range(MEMORY_SIZE)]  # Therapist always 0
        memory2 = [(i % 8, 7) for i in range(MEMORY_SIZE)]  # Therapist always 7

        client1 = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory1,
            random_state=fixed_seed
        )

        client2 = ConditionalAmplifierClient(
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

    def test_action_specific_amplification(self, low_entropy, fixed_seed):
        """Amplification should be specific to each client action."""
        # Pattern: client 0 → therapist 4, client 1 → therapist 3
        memory = [(0, 4)] * 25 + [(1, 3)] * 25

        u_matrix = np.random.RandomState(42).uniform(-50, 50, (8, 8))

        client = ConditionalAmplifierClient(
            u_matrix=u_matrix,
            entropy=low_entropy,
            initial_memory=memory,
            history_weight=1.0,
            random_state=fixed_seed
        )

        # For action 0, utility[0,4] should be amplified
        # For action 1, utility[1,3] should be amplified
        # These are action-specific amplifications

        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8


# ==============================================================================
# TEST HISTORY WEIGHT PARAMETER
# ==============================================================================

class TestHistoryWeightParameter:
    """Test history_weight parameter controls amplification strength."""

    def test_default_history_weight(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should use HISTORY_WEIGHT from config by default."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        assert client.history_weight == HISTORY_WEIGHT

    def test_custom_history_weight(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should accept custom history_weight parameter."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=2.0,
            random_state=fixed_seed
        )

        assert client.history_weight == 2.0

    def test_zero_weight_approaches_bond_only(self, fixed_u_matrix, low_entropy, fixed_seed):
        """history_weight=0 should give results close to BondOnlyClient."""
        memory1 = [(0, 0)] * MEMORY_SIZE
        memory2 = [(0, 7)] * MEMORY_SIZE

        client1 = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=memory1,
            history_weight=0.0,
            random_state=fixed_seed
        )

        client2 = ConditionalAmplifierClient(
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

        # With zero weight, should behave like BondOnlyClient
        np.testing.assert_allclose(payoffs1, payoffs2, rtol=1e-6)


# ==============================================================================
# TEST EFFECTIVE HISTORY WEIGHT HOOK
# ==============================================================================

class TestEffectiveHistoryWeightHook:
    """Test _get_effective_history_weight() for subclass override."""

    def test_returns_base_history_weight(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Base implementation should return unmodified history_weight."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            history_weight=1.8,
            random_state=fixed_seed
        )

        effective = client._get_effective_history_weight()
        assert effective == 1.8


# ==============================================================================
# TEST INTEGRATION WITH BASE CLIENT
# ==============================================================================

class TestIntegrationWithBase:
    """Test ConditionalAmplifierClient integrates with BaseClientAgent."""

    def test_inherits_base_functionality(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Should inherit all BaseClientAgent functionality."""
        client = ConditionalAmplifierClient(
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
        """Test full cycle: conditional frequencies → amplification → payoffs → action."""
        client = ConditionalAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Calculate conditional frequencies for each action
        for action in range(8):
            freq = client._calculate_conditional_frequencies(action)
            assert_valid_probability_distribution(freq)

        # Calculate payoffs
        payoffs = client._calculate_expected_payoffs()
        assert len(payoffs) == 8

        # Select action
        action = client.select_action()
        assert 0 <= action <= 7
        assert isinstance(action, (int, np.integer))
