"""
Unit tests for OmniscientStrategicTherapistV2-specific features.

Tests the new functionality added to V2:
- Memory context separation (perception_window vs full memory)
- Forward projection methods for target selection
- Integrated cost-benefit analysis
- Seeding session estimation
- Bond projection during seeding

Run with: pytest tests/test_omniscient_therapist_v2_features.py -v
"""
# pyright: reportPrivateUsage=false

import pytest
import numpy as np
from collections import deque

from src.agents.therapist_agents import OmniscientStrategicTherapistV2
from src.agents.client_agents import FrequencyAmplifierClient, with_parataxic
from src.config import (
    sample_u_matrix,
    MEMORY_SIZE,
    PARATAXIC_WINDOW,
    MAX_SESSIONS,
    get_memory_weights,
)
from tests.conftest import assert_valid_octant


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def fixed_seed():
    """Deterministic seed for reproducible tests."""
    return 42


@pytest.fixture
def sample_u_matrix_fixture(fixed_seed):
    """Fixed utility matrix."""
    return sample_u_matrix(random_state=fixed_seed)


@pytest.fixture
def uniform_memory():
    """Memory with uniform distribution of therapist actions."""
    # Create memory where each therapist action appears roughly equally
    memory = []
    for i in range(MEMORY_SIZE):
        client_action = i % 8
        therapist_action = i % 8
        memory.append((client_action, therapist_action))
    return memory


@pytest.fixture
def cold_dominant_memory():
    """Memory dominated by Cold (6) therapist responses."""
    # 40 Cold (6) and 10 Warm (2) responses
    memory = [(6, 6)] * 40 + [(2, 2)] * 10
    return memory


@pytest.fixture
def mixed_memory():
    """Memory with mixed therapist actions, some more common than others."""
    # 20 Warm (2), 15 Cold (6), 10 WD (1), 5 others
    memory = (
        [(2, 2)] * 20 +
        [(6, 6)] * 15 +
        [(1, 1)] * 10 +
        [(0, 0)] * 5
    )
    return memory


@pytest.fixture
def sample_client(sample_u_matrix_fixture, cold_dominant_memory, fixed_seed):
    """Create a sample client for testing."""
    client = FrequencyAmplifierClient(
        u_matrix=sample_u_matrix_fixture,
        entropy=0.1,
        initial_memory=cold_dominant_memory,
        random_state=fixed_seed,
        history_weight=1.0
    )
    client.success_threshold = 0.8
    return client


@pytest.fixture
def parataxic_client(sample_u_matrix_fixture, cold_dominant_memory, fixed_seed):
    """Create a client with parataxic distortion enabled."""
    ClientClass = with_parataxic(FrequencyAmplifierClient)
    client = ClientClass(
        u_matrix=sample_u_matrix_fixture,
        entropy=0.1,
        initial_memory=cold_dominant_memory,
        random_state=fixed_seed,
        history_weight=1.0,
        baseline_accuracy=0.5,
        enable_parataxic=True
    )
    client.success_threshold = 0.8
    return client


@pytest.fixture
def v2_therapist(sample_client):
    """Create V2 therapist with default parameters."""
    return OmniscientStrategicTherapistV2(
        client_ref=sample_client,
        perception_window=15,
        baseline_accuracy=0.5
    )


@pytest.fixture
def v2_therapist_with_parataxic(parataxic_client):
    """Create V2 therapist for parataxic client."""
    return OmniscientStrategicTherapistV2(
        client_ref=parataxic_client,
        perception_window=15,
        baseline_accuracy=0.5
    )


# ==============================================================================
# TEST DEFAULT PARAMETERS FROM CONFIG
# ==============================================================================

class TestDefaultParametersFromConfig:
    """Test that V2 uses config values as defaults."""

    def test_perception_window_defaults_to_parataxic_window(self, sample_client):
        """perception_window should default to PARATAXIC_WINDOW from config."""
        therapist = OmniscientStrategicTherapistV2(client_ref=sample_client)
        assert therapist.perception_window == PARATAXIC_WINDOW

    def test_max_sessions_defaults_to_config_value(self, sample_client):
        """max_sessions should default to MAX_SESSIONS from config."""
        therapist = OmniscientStrategicTherapistV2(client_ref=sample_client)
        assert therapist.max_sessions == MAX_SESSIONS

    def test_can_override_perception_window(self, sample_client):
        """Should be able to override perception_window."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            perception_window=10
        )
        assert therapist.perception_window == 10

    def test_can_override_max_sessions(self, sample_client):
        """Should be able to override max_sessions."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            max_sessions=50
        )
        assert therapist.max_sessions == 50


# ==============================================================================
# TEST MEMORY CONTEXT SEPARATION
# ==============================================================================

class TestMemoryContextSeparation:
    """Test that therapist correctly uses different memory contexts."""

    def test_get_client_perceived_memory_uses_perception_window(self, v2_therapist):
        """_get_client_perceived_memory should only return perception_window items."""
        perceived = v2_therapist._get_client_perceived_memory()
        assert len(perceived) == v2_therapist.perception_window

    def test_get_current_weighted_frequencies_uses_full_memory(self, v2_therapist):
        """_get_current_weighted_frequencies should use full memory."""
        # Get frequencies
        frequencies = v2_therapist._get_current_weighted_frequencies()

        # Should be a valid probability distribution
        assert len(frequencies) == 8
        assert np.isclose(frequencies.sum(), 1.0)
        assert np.all(frequencies >= 0)

    def test_weighted_frequencies_match_client_calculation(self, sample_client, v2_therapist):
        """Weighted frequencies should match client's _calculate_marginal_frequencies."""
        # Get therapist's calculation
        therapist_freqs = v2_therapist._get_current_weighted_frequencies()

        # Get client's calculation (if method exists)
        if hasattr(sample_client, '_calculate_marginal_frequencies'):
            client_freqs = sample_client._calculate_marginal_frequencies()
            np.testing.assert_allclose(
                therapist_freqs,
                client_freqs,
                rtol=1e-10,
                err_msg="Therapist weighted frequencies don't match client's"
            )

    def test_estimate_seeding_sessions_uses_perception_window(
        self, sample_u_matrix_fixture, fixed_seed
    ):
        """_estimate_seeding_sessions should use perception_window for mode detection."""
        # Create client with specific memory pattern
        # Recent 15 actions are dominated by action 2, older actions by action 6
        memory = [(6, 6)] * 35 + [(2, 2)] * 15  # Last 15 are (2, 2)

        client = FrequencyAmplifierClient(
            u_matrix=sample_u_matrix_fixture,
            entropy=0.1,
            initial_memory=memory,
            random_state=fixed_seed,
            history_weight=1.0
        )
        client.success_threshold = 0.8

        therapist = OmniscientStrategicTherapistV2(
            client_ref=client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        # Action 2 is dominant in perception_window (last 15)
        # So seeding for action 2 should need 0 sessions
        sessions_for_2 = therapist._estimate_seeding_sessions(2)
        assert sessions_for_2 == 0, "Action 2 should be dominant in perception_window"

        # Action 6 is NOT dominant in perception_window (only in full memory)
        # So seeding for action 6 should need > 0 sessions
        sessions_for_6 = therapist._estimate_seeding_sessions(6)
        assert sessions_for_6 > 0, "Action 6 should need seeding in perception_window"


# ==============================================================================
# TEST FORWARD PROJECTION METHODS
# ==============================================================================

class TestForwardProjectionMethods:
    """Test methods that project future state after seeding."""

    def test_project_therapist_frequencies_returns_valid_distribution(self, v2_therapist):
        """_project_therapist_frequencies should return valid probability distribution."""
        projected = v2_therapist._project_therapist_frequencies(target_action=2)

        assert len(projected) == 8
        assert np.isclose(projected.sum(), 1.0)
        assert np.all(projected >= 0)

    def test_project_therapist_frequencies_increases_target(self, v2_therapist):
        """Projecting frequencies should increase target action's frequency."""
        current = v2_therapist._get_current_weighted_frequencies()

        # Project for an action that's not currently dominant
        target = 0  # Dominant - unlikely to be most common in cold-dominated memory
        projected = v2_therapist._project_therapist_frequencies(target)

        # If seeding is needed, target should increase
        sessions_needed = v2_therapist._estimate_seeding_sessions(target)
        if sessions_needed > 0:
            assert projected[target] >= current[target], \
                "Target frequency should increase after projection"

    def test_project_therapist_frequencies_handles_already_dominant(self, v2_therapist):
        """Should handle case where target is already dominant."""
        # Find the current mode in perception_window
        perceived = v2_therapist._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        current_mode = np.argmax(counts)

        # Project for the current mode (already dominant)
        current = v2_therapist._get_current_weighted_frequencies()
        projected = v2_therapist._project_therapist_frequencies(current_mode)

        # Should return current frequencies (no change needed)
        np.testing.assert_allclose(
            projected,
            current,
            rtol=1e-6,
            err_msg="Already dominant action should not change frequencies"
        )

    def test_project_client_expected_payoffs_returns_8_values(self, v2_therapist):
        """_project_client_expected_payoffs should return 8 expected payoffs."""
        payoffs = v2_therapist._project_client_expected_payoffs(target_therapist_action=2)

        assert len(payoffs) == 8
        assert all(np.isfinite(payoffs))

    def test_project_client_probabilities_returns_valid_distribution(self, v2_therapist):
        """_project_client_probabilities should return valid probability distribution."""
        probs = v2_therapist._project_client_probabilities(target_therapist_action=2)

        assert len(probs) == 8
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)


# ==============================================================================
# TEST BOND PROJECTION
# ==============================================================================

class TestBondProjection:
    """Test bond projection during seeding."""

    def test_project_bond_returns_valid_bond(self, v2_therapist):
        """_project_bond_after_seeding should return bond in [0, 1]."""
        projected_bond = v2_therapist._project_bond_after_seeding(target_therapist_action=2)

        assert 0.0 <= projected_bond <= 1.0

    def test_project_bond_handles_no_seeding_needed(self, v2_therapist):
        """Should return current bond if no seeding needed."""
        # Find current mode
        perceived = v2_therapist._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        current_mode = np.argmax(counts)

        current_bond = v2_therapist.client_ref.bond
        projected_bond = v2_therapist._project_bond_after_seeding(current_mode)

        assert projected_bond == current_bond, \
            "No seeding needed should return current bond"

    def test_project_bond_accounts_for_seeding_cost(self, v2_therapist):
        """Projected bond should account for potential RS decrease during seeding."""
        # Target an action that requires seeding
        target = 0  # Dominant - not common in cold memory

        sessions_needed = v2_therapist._estimate_seeding_sessions(target)

        if sessions_needed > 0:
            current_bond = v2_therapist.client_ref.bond
            projected_bond = v2_therapist._project_bond_after_seeding(target)

            # Bond might decrease during seeding (depends on utility)
            # Just verify the calculation completes without error
            assert 0.0 <= projected_bond <= 1.0


# ==============================================================================
# TEST SEEDING SESSION ESTIMATION
# ==============================================================================

class TestSeedingSessionEstimation:
    """Test estimation of sessions needed for seeding."""

    def test_estimate_seeding_sessions_returns_non_negative(self, v2_therapist):
        """Should return non-negative integer."""
        for target in range(8):
            sessions = v2_therapist._estimate_seeding_sessions(target)
            assert sessions >= 0
            assert isinstance(sessions, int)

    def test_estimate_seeding_zero_for_dominant_action(self, v2_therapist):
        """Should return 0 for already-dominant action."""
        # Find current mode in perception_window
        perceived = v2_therapist._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        current_mode = np.argmax(counts)

        sessions = v2_therapist._estimate_seeding_sessions(current_mode)
        assert sessions == 0

    def test_estimate_seeding_positive_for_non_dominant(self, v2_therapist):
        """Should return positive for non-dominant action."""
        # Find least common action
        perceived = v2_therapist._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        least_common = np.argmin(counts)

        # If least_common is not also the mode
        if counts[least_common] < max(counts):
            sessions = v2_therapist._estimate_seeding_sessions(least_common)
            assert sessions > 0

    def test_estimate_seeding_accounts_for_baseline_accuracy(self, sample_client):
        """Higher baseline_accuracy should need fewer sessions."""
        therapist_low_acc = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            baseline_accuracy=0.3
        )
        therapist_high_acc = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            baseline_accuracy=0.9
        )

        # Find a non-dominant action
        perceived = therapist_low_acc._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        non_dominant = np.argmin(counts)

        sessions_low = therapist_low_acc._estimate_seeding_sessions(non_dominant)
        sessions_high = therapist_high_acc._estimate_seeding_sessions(non_dominant)

        # Higher accuracy should need fewer or equal sessions
        if sessions_low > 0:
            assert sessions_high <= sessions_low

    def test_estimate_seeding_uses_simple_counts_not_recency_weighting(
        self, sample_u_matrix_fixture, fixed_seed
    ):
        """Seeding estimation should use simple counts in perception_window."""
        # Create memory where recency weighting would give different result
        # Older items in perception_window are action 2, newer are action 6
        # With simple counts, both have equal count
        # With recency weighting, action 6 would be weighted higher
        memory = [(6, 6)] * 35 + [(2, 2)] * 8 + [(6, 6)] * 7

        client = FrequencyAmplifierClient(
            u_matrix=sample_u_matrix_fixture,
            entropy=0.1,
            initial_memory=memory,
            random_state=fixed_seed,
            history_weight=1.0
        )

        therapist = OmniscientStrategicTherapistV2(
            client_ref=client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        # In last 15: 8 are (2,2) and 7 are (6,6)
        # With simple counts: action 2 has 8, action 6 has 7
        # So action 2 should be dominant and need 0 sessions
        sessions_for_2 = therapist._estimate_seeding_sessions(2)
        assert sessions_for_2 == 0


# ==============================================================================
# TEST INTEGRATED COST-BENEFIT ANALYSIS
# ==============================================================================

class TestCostBenefitAnalysis:
    """Test integrated cost-benefit analysis for target selection."""

    def test_calculate_target_net_value_returns_tuple(self, v2_therapist):
        """_calculate_target_net_value should return (value, metadata) tuple."""
        v2_therapist.session_count = 15  # Past dropout check

        net_value, metadata = v2_therapist._calculate_target_net_value(
            client_action=2,
            therapist_action=2
        )

        assert isinstance(net_value, (float, int, np.floating))
        assert isinstance(metadata, dict)

    def test_calculate_target_net_value_rejects_no_improvement(self, v2_therapist):
        """Should reject targets that don't improve over current RS."""
        v2_therapist.session_count = 15

        # Find a low-utility interaction
        u_matrix = v2_therapist.client_ref.u_matrix
        current_rs = v2_therapist.client_ref.relationship_satisfaction

        # Find interaction with utility <= current RS
        for client_oct in range(8):
            for therapist_oct in range(8):
                if u_matrix[client_oct, therapist_oct] <= current_rs:
                    net_value, metadata = v2_therapist._calculate_target_net_value(
                        client_oct, therapist_oct
                    )
                    assert net_value == float('-inf')
                    assert metadata.get('reason') == 'no_improvement'
                    return

    def test_calculate_target_net_value_rejects_insufficient_time(self, v2_therapist):
        """Should reject targets when not enough time remains."""
        # Set session count very high
        v2_therapist.session_count = v2_therapist.max_sessions - 2

        net_value, metadata = v2_therapist._calculate_target_net_value(
            client_action=2,
            therapist_action=0  # Likely needs seeding
        )

        # Should be rejected due to insufficient time
        if net_value == float('-inf'):
            assert metadata.get('reason') in ['insufficient_time', 'no_improvement']

    def test_calculate_target_net_value_metadata_has_expected_keys(self, v2_therapist):
        """Metadata should contain cost-benefit breakdown."""
        v2_therapist.session_count = 15

        # Find an improving target
        u_matrix = v2_therapist.client_ref.u_matrix
        current_rs = v2_therapist.client_ref.relationship_satisfaction

        for client_oct in range(8):
            for therapist_oct in range(8):
                if u_matrix[client_oct, therapist_oct] > current_rs:
                    net_value, metadata = v2_therapist._calculate_target_net_value(
                        client_oct, therapist_oct
                    )

                    if net_value != float('-inf'):
                        expected_keys = [
                            'target_utility',
                            'current_rs',
                            'utility_improvement',
                            'seeding_sessions',
                            'total_seeding_cost',
                            'total_benefit',
                            'net_value',
                        ]
                        for key in expected_keys:
                            assert key in metadata, f"Missing key: {key}"
                        return

    def test_identify_target_considers_all_64_pairs(self, v2_therapist):
        """_identify_target_interaction should consider all 64 (client, therapist) pairs."""
        v2_therapist.session_count = 15

        # This test verifies the method runs and potentially finds a target
        # The actual selection depends on the specific utility matrix
        found = v2_therapist._identify_target_interaction()

        if found:
            # If target found, both should be valid octants
            assert_valid_octant(v2_therapist.current_target_client_action)
            assert_valid_octant(v2_therapist.current_target_therapist_action)
        else:
            # If not found, both should be None
            assert v2_therapist.current_target_client_action is None
            assert v2_therapist.current_target_therapist_action is None


# ==============================================================================
# TEST SIMPLIFIED _is_seeding_beneficial
# ==============================================================================

class TestSimplifiedSeedingBeneficial:
    """Test the simplified _is_seeding_beneficial method."""

    def test_returns_false_when_no_target(self, v2_therapist):
        """Should return False when no target is set."""
        v2_therapist.current_target_therapist_action = None

        result = v2_therapist._is_seeding_beneficial(client_action=2)
        assert result is False

    def test_returns_true_when_complement_equals_target(self, v2_therapist):
        """Should return True for free seeding (complement == target)."""
        # Set target to Cold (6), which complements Cold
        v2_therapist.current_target_client_action = 6
        v2_therapist.current_target_therapist_action = 6

        # Client takes Cold (6), complement is Cold (6) which equals target
        result = v2_therapist._is_seeding_beneficial(client_action=6)
        assert result is True

    def test_returns_false_when_seeding_complete(self, v2_therapist):
        """Should return False when target is already dominant."""
        # Find current mode
        perceived = v2_therapist._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        current_mode = np.argmax(counts)

        v2_therapist.current_target_therapist_action = current_mode
        v2_therapist.current_target_client_action = 2

        # Target is already dominant, so seeding should not be beneficial
        # (unless complement equals target)
        complement = v2_therapist._get_complementary_action(0)  # D -> S
        if complement != current_mode:
            result = v2_therapist._is_seeding_beneficial(client_action=0)
            assert result is False

    def test_returns_true_when_seeding_needed(self, v2_therapist):
        """Should return True when target needs seeding and is committed."""
        # Find a non-dominant action
        perceived = v2_therapist._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived]
        counts = [therapist_actions.count(a) for a in range(8)]
        non_dominant = np.argmin(counts)

        v2_therapist.current_target_therapist_action = non_dominant
        v2_therapist.current_target_client_action = 2

        # Pick a client action whose complement is NOT the target
        for client_action in range(8):
            complement = v2_therapist._get_complementary_action(client_action)
            if complement != non_dominant:
                result = v2_therapist._is_seeding_beneficial(client_action)
                # Should return True since we're committed to seeding
                assert result is True
                break


# ==============================================================================
# TEST SHOULD_START_LADDER_CLIMBING SIMPLIFICATION
# ==============================================================================

class TestSimplifiedLadderClimbingStart:
    """Test the simplified _should_start_ladder_climbing method."""

    def test_returns_false_before_session_10(self, v2_therapist):
        """Should not start ladder climbing before session 10."""
        v2_therapist.session_count = 5
        assert v2_therapist._should_start_ladder_climbing() is False

    def test_returns_false_with_low_bond(self, v2_therapist):
        """Should not start ladder climbing with very low bond."""
        v2_therapist.session_count = 15

        # Manually set very low bond
        original_bond = v2_therapist.client_ref.bond
        v2_therapist.client_ref.bond = 0.05  # Below threshold

        result = v2_therapist._should_start_ladder_climbing()

        # Restore bond
        v2_therapist.client_ref.bond = original_bond

        assert result is False

    def test_delegates_to_identify_target(self, v2_therapist):
        """Should call _identify_target_interaction for decision."""
        v2_therapist.session_count = 15

        # Ensure bond is sufficient
        if v2_therapist.client_ref.bond < 0.1:
            v2_therapist.client_ref.bond = 0.5

        # Result depends on whether a valid target exists
        result = v2_therapist._should_start_ladder_climbing()
        assert isinstance(result, bool)


# ==============================================================================
# TEST INTEGRATION: FULL THERAPY SIMULATION
# ==============================================================================

class TestFullTherapySimulation:
    """Integration tests running full therapy simulations."""

    def test_simulation_with_all_new_features(self, sample_client):
        """Run simulation exercising all new V2 features."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            perception_window=15,
            baseline_accuracy=0.5,
            max_sessions=50
        )

        phases_seen = set()
        targets_seen = []

        for session in range(1, 51):
            client_action = sample_client.select_action()
            therapist_action, metadata = therapist.decide_action(client_action, session)

            phases_seen.add(metadata['phase'])

            if therapist.current_target_client_action is not None:
                targets_seen.append((
                    therapist.current_target_client_action,
                    therapist.current_target_therapist_action
                ))

            # Update client memory
            sample_client.update_memory(client_action, therapist_action)

            # Process feedback
            therapist.process_feedback_after_memory_update(session, client_action)

            # Verify valid state throughout
            assert_valid_octant(therapist_action)
            assert therapist.session_count == session

        # Should have seen relationship_building at minimum
        assert 'relationship_building' in phases_seen

    def test_simulation_with_parataxic_client(self, parataxic_client):
        """Run simulation with parataxic distortion enabled."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=parataxic_client,
            perception_window=15,
            baseline_accuracy=0.5,
            max_sessions=30
        )

        for session in range(1, 31):
            client_action = parataxic_client.select_action()
            therapist_action, metadata = therapist.decide_action(client_action, session)

            # Update client memory (this applies parataxic distortion)
            parataxic_client.update_memory(client_action, therapist_action)

            # Process feedback (checks parataxic_history)
            therapist.process_feedback_after_memory_update(session, client_action)

            assert_valid_octant(therapist_action)

        # Check parataxic history was used
        if parataxic_client.parataxic_history:
            assert len(parataxic_client.parataxic_history) == 30


# ==============================================================================
# TEST EDGE CASES
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_perception_window_handled(self, sample_u_matrix_fixture, fixed_seed):
        """Should handle case with minimal memory gracefully."""
        # Create client with minimum memory
        memory = [(2, 2)] * MEMORY_SIZE

        client = FrequencyAmplifierClient(
            u_matrix=sample_u_matrix_fixture,
            entropy=0.1,
            initial_memory=memory,
            random_state=fixed_seed,
            history_weight=1.0
        )

        therapist = OmniscientStrategicTherapistV2(
            client_ref=client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        # Should not crash
        perceived = therapist._get_client_perceived_memory()
        assert len(perceived) == 15

        frequencies = therapist._get_current_weighted_frequencies()
        assert np.isclose(frequencies.sum(), 1.0)

    def test_all_same_action_memory(self, sample_u_matrix_fixture, fixed_seed):
        """Should handle memory with all identical actions."""
        memory = [(3, 3)] * MEMORY_SIZE  # All WS

        client = FrequencyAmplifierClient(
            u_matrix=sample_u_matrix_fixture,
            entropy=0.1,
            initial_memory=memory,
            random_state=fixed_seed,
            history_weight=1.0
        )
        client.success_threshold = 0.8

        therapist = OmniscientStrategicTherapistV2(
            client_ref=client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        # Action 3 should be dominant, needing 0 sessions
        sessions = therapist._estimate_seeding_sessions(3)
        assert sessions == 0

        # Any other action should need seeding
        for action in [0, 1, 2, 4, 5, 6, 7]:
            sessions = therapist._estimate_seeding_sessions(action)
            assert sessions > 0

    def test_baseline_accuracy_near_one(self, sample_client):
        """Should handle baseline_accuracy near 1.0."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            baseline_accuracy=0.99
        )

        # Should not crash and should need fewer sessions
        for target in range(8):
            sessions = therapist._estimate_seeding_sessions(target)
            assert sessions >= 0

    def test_baseline_accuracy_near_zero(self, sample_client):
        """Should handle baseline_accuracy near 0 (but positive)."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            baseline_accuracy=0.1
        )

        # Should not crash and should need more sessions
        for target in range(8):
            sessions = therapist._estimate_seeding_sessions(target)
            assert sessions >= 0

    def test_very_small_perception_window(self, sample_client):
        """Should handle very small perception window."""
        therapist = OmniscientStrategicTherapistV2(
            client_ref=sample_client,
            perception_window=3
        )

        # Should work with tiny window
        perceived = therapist._get_client_perceived_memory()
        assert len(perceived) == 3

        # Seeding should still work
        sessions = therapist._estimate_seeding_sessions(0)
        assert sessions >= 0


# ==============================================================================
# TEST CONSISTENCY BETWEEN METHODS
# ==============================================================================

class TestMethodConsistency:
    """Test that related methods are internally consistent."""

    def test_seeding_estimate_consistent_with_calculate_requirement(self, v2_therapist):
        """_estimate_seeding_sessions should be consistent with calculate_seeding_requirement."""
        for target in range(8):
            estimate = v2_therapist._estimate_seeding_sessions(target)
            requirement = v2_therapist.calculate_seeding_requirement(target)

            # Both should agree on whether seeding is needed
            if estimate == 0:
                assert requirement['raw_seeding_needed'] == 0
            else:
                assert requirement['raw_seeding_needed'] >= 0

    def test_projected_frequencies_normalized(self, v2_therapist):
        """Projected frequencies should always sum to 1."""
        for target in range(8):
            projected = v2_therapist._project_therapist_frequencies(target)
            assert np.isclose(projected.sum(), 1.0), \
                f"Projected frequencies for target {target} don't sum to 1"

    def test_projected_probabilities_normalized(self, v2_therapist):
        """Projected client probabilities should always sum to 1."""
        for target in range(8):
            probs = v2_therapist._project_client_probabilities(target)
            assert np.isclose(probs.sum(), 1.0), \
                f"Projected probabilities for target {target} don't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
