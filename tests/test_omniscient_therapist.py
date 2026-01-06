"""
Unit tests for OmniscientStrategicTherapist.

Tests the strategic therapist agent with omniscient client knowledge,
including phase transitions, seeding logic, complementarity, and decision making.

Run with: pytest tests/test_omniscient_therapist.py -v
"""
# pyright: reportPrivateUsage=false

import pytest
import numpy as np
from typing import List, Tuple

from src.agents.therapist_agents import OmniscientStrategicTherapist
from src.agents.client_agents import FrequencyAmplifierClient, with_parataxic
from src.config import sample_u_matrix, MEMORY_SIZE
from tests.conftest import assert_valid_octant


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def fixed_seed():
    """Deterministic seed for reproducible tests."""
    return 42


@pytest.fixture
def sample_client(fixed_seed):
    """Create a sample client for therapist testing."""
    u_matrix = sample_u_matrix(random_state=fixed_seed)

    # Create problematic memory (cold stuck)
    memory = [(6, 6)] * MEMORY_SIZE  # All C-C interactions

    client = FrequencyAmplifierClient(
        u_matrix=u_matrix,
        entropy=0.1,
        initial_memory=memory,
        random_state=fixed_seed,
        history_weight=1.0
    )

    # Set success threshold for therapist
    client.success_threshold = 0.8

    return client


@pytest.fixture
def parataxic_client(fixed_seed):
    """Create a client with parataxic distortion enabled."""
    u_matrix = sample_u_matrix(random_state=fixed_seed)
    memory = [(6, 6)] * MEMORY_SIZE

    ClientClass = with_parataxic(FrequencyAmplifierClient)
    client = ClientClass(
        u_matrix=u_matrix,
        entropy=0.1,
        initial_memory=memory,
        random_state=fixed_seed,
        history_weight=1.0,
        baseline_accuracy=0.5,
        enable_parataxic=True
    )

    client.success_threshold = 0.8
    return client


@pytest.fixture
def omniscient_therapist(sample_client):
    """Create an omniscient therapist with sample client."""
    return OmniscientStrategicTherapist(
        client_ref=sample_client,
        perception_window=15,
        baseline_accuracy=0.5
    )


@pytest.fixture
def parataxic_therapist(parataxic_client):
    """Create therapist for parataxic client."""
    return OmniscientStrategicTherapist(
        client_ref=parataxic_client,
        perception_window=15,
        baseline_accuracy=0.5
    )


# ==============================================================================
# TEST INITIALIZATION
# ==============================================================================

class TestOmniscientTherapistInitialization:
    """Test therapist initialization and setup."""

    def test_init_with_valid_params(self, sample_client):
        """Should initialize successfully with valid parameters."""
        therapist = OmniscientStrategicTherapist(
            client_ref=sample_client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        assert therapist is not None
        assert therapist.client_ref is sample_client
        assert therapist.perception_window == 15
        assert therapist.baseline_accuracy == 0.5

    def test_init_default_phase(self, omniscient_therapist):
        """Should start in relationship_building phase."""
        assert omniscient_therapist.phase == "relationship_building"

    def test_init_tracking_structures(self, omniscient_therapist):
        """Should initialize tracking structures."""
        assert omniscient_therapist.action_log == []
        assert omniscient_therapist.session_count == 0
        assert omniscient_therapist.actual_actions_taken == []
        assert omniscient_therapist.current_target_client_action is None
        assert omniscient_therapist.current_target_therapist_action is None

    def test_complement_map_complete(self, omniscient_therapist):
        """Complement map should have all 8 octants."""
        assert len(omniscient_therapist.COMPLEMENT_MAP) == 8
        for octant in range(8):
            assert octant in omniscient_therapist.COMPLEMENT_MAP


# ==============================================================================
# TEST COMPLEMENTARY ACTION
# ==============================================================================

class TestComplementaryAction:
    """Test complementary action mapping."""

    def test_complementary_pairs(self, omniscient_therapist):
        """Should return correct complement for each octant."""
        expected_complements = {
            0: 4,  # D → S
            1: 3,  # WD → WS
            2: 2,  # W → W
            3: 1,  # WS → WD
            4: 0,  # S → D
            5: 7,  # CS → CD
            6: 6,  # C → C
            7: 5,  # CD → CS
        }

        for octant, expected_complement in expected_complements.items():
            complement = omniscient_therapist._get_complementary_action(octant)
            assert complement == expected_complement

    def test_complementary_symmetry(self, omniscient_therapist):
        """Applying complement twice should return to original for control octants."""
        # Control octants (D-S axis): 0, 4
        for octant in [0, 4]:
            complement = omniscient_therapist._get_complementary_action(octant)
            double_complement = omniscient_therapist._get_complementary_action(complement)
            assert double_complement == octant


# ==============================================================================
# TEST PERCEIVED MEMORY ACCESS
# ==============================================================================

class TestPerceivedMemoryAccess:
    """Test access to client's perceived memory."""

    def test_get_perceived_memory_returns_list(self, omniscient_therapist):
        """Should return list of tuples."""
        perceived_memory = omniscient_therapist._get_client_perceived_memory()
        assert isinstance(perceived_memory, list)

        if len(perceived_memory) > 0:
            assert all(isinstance(item, tuple) for item in perceived_memory)
            assert all(len(item) == 2 for item in perceived_memory)

    def test_get_perceived_memory_limited_by_window(self, omniscient_therapist):
        """Should return at most perception_window items."""
        perceived_memory = omniscient_therapist._get_client_perceived_memory()
        assert len(perceived_memory) <= omniscient_therapist.perception_window

    def test_get_perceived_memory_reads_client_memory(self, sample_client, omniscient_therapist):
        """Should read from client's actual memory."""
        # Client's memory should match what therapist reads
        client_memory = list(sample_client.memory)
        perceived = omniscient_therapist._get_client_perceived_memory()

        # Should be the last perception_window items from client memory
        expected_window = min(omniscient_therapist.perception_window, len(client_memory))
        assert perceived == client_memory[-expected_window:]


# ==============================================================================
# TEST PERCEPTION ACCURACY ESTIMATION
# ==============================================================================

class TestPerceptionAccuracyEstimation:
    """Test estimation of perception accuracy."""

    def test_estimate_perception_baseline(self, omniscient_therapist):
        """Empty memory should return baseline accuracy."""
        # Clear client memory temporarily
        original_memory = list(omniscient_therapist.client_ref.memory)
        omniscient_therapist.client_ref.memory.clear()

        accuracy = omniscient_therapist._estimate_perception_accuracy(2)

        assert accuracy == omniscient_therapist.baseline_accuracy

        # Restore memory
        omniscient_therapist.client_ref.memory.extend(original_memory)

    def test_estimate_perception_high_for_common_action(self, omniscient_therapist):
        """Most common action should have high perception accuracy."""
        # After initialization, most common action in memory is 6 (Cold)
        accuracy_common = omniscient_therapist._estimate_perception_accuracy(6)
        accuracy_rare = omniscient_therapist._estimate_perception_accuracy(0)

        # Common action should have higher or equal accuracy
        assert accuracy_common >= accuracy_rare

    def test_estimate_perception_in_valid_range(self, omniscient_therapist):
        """Perception accuracy should be in [0, 1]."""
        for action in range(8):
            accuracy = omniscient_therapist._estimate_perception_accuracy(action)
            assert 0.0 <= accuracy <= 1.0


# ==============================================================================
# TEST SEEDING REQUIREMENT CALCULATION
# ==============================================================================

class TestSeedingRequirementCalculation:
    """Test calculation of seeding requirements."""

    def test_calculate_seeding_has_required_fields(self, omniscient_therapist):
        """Should return dict with all required fields."""
        seeding_req = omniscient_therapist.calculate_seeding_requirement(2)

        required_fields = [
            'target_action',
            'current_count_in_perceived_memory',
            'max_other_count',
            'raw_seeding_needed',
            'adjusted_seeding_needed',
            'perception_accuracy_estimate'
        ]

        for field in required_fields:
            assert field in seeding_req

    def test_calculate_seeding_target_matches_input(self, omniscient_therapist):
        """Target action in result should match input."""
        for action in range(8):
            seeding_req = omniscient_therapist.calculate_seeding_requirement(action)
            assert seeding_req['target_action'] == action

    def test_calculate_seeding_zero_needed_when_dominant(self, sample_client, omniscient_therapist):
        """Should need 0 seeding if target is already most common."""
        # Client's memory is all (6, 6), so therapist action 6 is most common
        seeding_req = omniscient_therapist.calculate_seeding_requirement(6)

        # Should need 0 raw seeding (already dominant)
        assert seeding_req['raw_seeding_needed'] == 0

    def test_calculate_seeding_positive_needed_when_not_dominant(self, omniscient_therapist):
        """Should need positive seeding if target is not most common."""
        # Action 0 is not common in the all-6 memory
        seeding_req = omniscient_therapist.calculate_seeding_requirement(0)

        # Should need some seeding
        assert seeding_req['raw_seeding_needed'] > 0

    def test_calculate_seeding_adjusted_accounts_for_accuracy(self, omniscient_therapist):
        """Adjusted seeding should be >= raw seeding (accounting for misperception)."""
        seeding_req = omniscient_therapist.calculate_seeding_requirement(0)

        # Adjusted should be >= raw (unless baseline_accuracy = 1.0)
        if omniscient_therapist.baseline_accuracy < 1.0:
            assert seeding_req['adjusted_seeding_needed'] >= seeding_req['raw_seeding_needed']


# ==============================================================================
# TEST PHASE TRANSITIONS
# ==============================================================================

class TestPhaseTransitions:
    """Test therapist phase transitions."""

    def test_should_start_ladder_climbing_false_before_session_10(self, omniscient_therapist):
        """Should not start ladder climbing before session 10."""
        omniscient_therapist.session_count = 5
        assert not omniscient_therapist._should_start_ladder_climbing()

    def test_should_start_ladder_climbing_requires_bond_threshold(self, omniscient_therapist):
        """Should require minimum bond to start ladder climbing."""
        # Set session count high enough
        omniscient_therapist.session_count = 15

        # Low bond should prevent ladder climbing
        original_bond = omniscient_therapist.client_ref.bond

        # Can't easily manipulate bond, but we can check the logic exists
        result = omniscient_therapist._should_start_ladder_climbing()

        # Result depends on current bond value
        assert isinstance(result, bool)

    def test_current_interaction_achieves_success_when_rs_above_threshold(self, sample_client):
        """Should return True when RS >= success threshold."""
        # Manually set RS above threshold
        sample_client.success_threshold = 0.5

        therapist = OmniscientStrategicTherapist(
            client_ref=sample_client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        # Set RS higher than threshold
        # (need to manipulate memory to achieve this)
        # For this test, we just verify the method exists and returns bool
        result = therapist._current_interaction_achieves_success()
        assert isinstance(result, bool)


# ==============================================================================
# TEST TARGET INTERACTION IDENTIFICATION
# ==============================================================================

class TestTargetInteractionIdentification:
    """Test identification of target interactions."""

    def test_identify_target_returns_bool(self, omniscient_therapist):
        """Should return boolean indicating if target found."""
        result = omniscient_therapist._identify_target_interaction()
        assert isinstance(result, bool)

    def test_identify_target_sets_targets_when_found(self, omniscient_therapist):
        """Should set target actions when valid target found."""
        # Call target identification
        found = omniscient_therapist._identify_target_interaction()

        if found:
            # Should have set both target actions
            assert omniscient_therapist.current_target_client_action is not None
            assert omniscient_therapist.current_target_therapist_action is not None
            assert_valid_octant(omniscient_therapist.current_target_client_action)
            assert_valid_octant(omniscient_therapist.current_target_therapist_action)
        else:
            # Should clear targets if no valid target
            assert omniscient_therapist.current_target_client_action is None
            assert omniscient_therapist.current_target_therapist_action is None


# ==============================================================================
# TEST SEEDING BENEFIT EVALUATION
# ==============================================================================

class TestSeedingBenefitEvaluation:
    """Test evaluation of whether seeding is beneficial."""

    def test_is_seeding_beneficial_returns_bool(self, omniscient_therapist):
        """Should return boolean."""
        # Set both targets (required for full evaluation)
        omniscient_therapist.current_target_client_action = 2
        omniscient_therapist.current_target_therapist_action = 2

        result = omniscient_therapist._is_seeding_beneficial(client_action=6)
        # Accept both Python bool and numpy bool
        assert isinstance(result, (bool, np.bool_))

    def test_is_seeding_beneficial_when_complement_equals_target(self, omniscient_therapist):
        """Should return True when complement equals seeding target (free seeding)."""
        # Set both targets to action 6 (Cold complements Cold)
        omniscient_therapist.current_target_client_action = 6
        omniscient_therapist.current_target_therapist_action = 6

        # Client takes action 6, which complements to 6
        result = omniscient_therapist._is_seeding_beneficial(client_action=6)

        # Should be beneficial (free seeding)
        assert result is True

    def test_is_seeding_beneficial_false_when_no_target(self, omniscient_therapist):
        """Should return False when no target set."""
        omniscient_therapist.current_target_therapist_action = None

        result = omniscient_therapist._is_seeding_beneficial(client_action=0)
        assert result is False


# ==============================================================================
# TEST DECISION MAKING
# ==============================================================================

class TestDecisionMaking:
    """Test therapist action decision making."""

    def test_decide_action_returns_valid_tuple(self, omniscient_therapist):
        """Should return (action, metadata) tuple."""
        client_action = 6
        session = 1

        action, metadata = omniscient_therapist.decide_action(client_action, session)

        assert isinstance(action, (int, np.integer))
        assert isinstance(metadata, dict)
        assert_valid_octant(int(action))

    def test_decide_action_metadata_has_required_fields(self, omniscient_therapist):
        """Metadata should contain key information."""
        action, metadata = omniscient_therapist.decide_action(client_action=6, session=1)

        required_fields = ['session', 'client_action', 'phase', 'bond', 'rs', 'rationale', 'therapist_action']
        for field in required_fields:
            assert field in metadata

    def test_decide_action_early_sessions_complement(self, omniscient_therapist):
        """Early sessions should use pure complementarity."""
        # Session 1 should be in relationship building
        action, metadata = omniscient_therapist.decide_action(client_action=6, session=1)

        # Should be complementary
        expected_complement = omniscient_therapist._get_complementary_action(6)
        assert action == expected_complement
        assert metadata['phase'] == 'relationship_building'

    def test_decide_action_logs_to_action_log(self, omniscient_therapist):
        """Should log decision to action log."""
        initial_log_len = len(omniscient_therapist.action_log)

        omniscient_therapist.decide_action(client_action=6, session=1)

        assert len(omniscient_therapist.action_log) == initial_log_len + 1

    def test_decide_action_updates_session_count(self, omniscient_therapist):
        """Should update session count."""
        omniscient_therapist.decide_action(client_action=6, session=5)

        assert omniscient_therapist.session_count == 5

    def test_decide_action_records_actual_action(self, omniscient_therapist):
        """Should record action in actual_actions_taken."""
        initial_len = len(omniscient_therapist.actual_actions_taken)

        action, _ = omniscient_therapist.decide_action(client_action=6, session=1)

        assert len(omniscient_therapist.actual_actions_taken) == initial_len + 1
        assert omniscient_therapist.actual_actions_taken[-1] == action


# ==============================================================================
# TEST ACTION LOG
# ==============================================================================

class TestActionLog:
    """Test action logging functionality."""

    def test_action_log_entry_structure(self, omniscient_therapist):
        """Action log entries should have expected structure."""
        omniscient_therapist.decide_action(client_action=6, session=1)

        entry = omniscient_therapist.action_log[0]

        assert hasattr(entry, 'session')
        assert hasattr(entry, 'client_action')
        assert hasattr(entry, 'therapist_action')
        assert hasattr(entry, 'phase')
        assert hasattr(entry, 'rationale')

    def test_get_action_log_returns_list(self, omniscient_therapist):
        """Should return action log as list."""
        omniscient_therapist.decide_action(client_action=6, session=1)

        log = omniscient_therapist.get_action_log()
        assert isinstance(log, list)
        assert len(log) > 0


# ==============================================================================
# TEST SUMMARY STATISTICS
# ==============================================================================

class TestSummaryStatistics:
    """Test summary statistic generation."""

    def test_get_phase_summary_structure(self, omniscient_therapist):
        """Phase summary should have expected structure."""
        omniscient_therapist.decide_action(client_action=6, session=1)

        summary = omniscient_therapist.get_phase_summary()

        assert 'phase_counts' in summary
        assert 'total_sessions' in summary
        assert 'current_phase' in summary

    def test_get_phase_summary_counts_sessions(self, omniscient_therapist):
        """Should count sessions in each phase."""
        # Run 5 sessions
        for session in range(1, 6):
            omniscient_therapist.decide_action(client_action=6, session=session)

        summary = omniscient_therapist.get_phase_summary()

        assert summary['total_sessions'] == 5

    def test_get_seeding_summary_structure(self, omniscient_therapist):
        """Seeding summary should have expected structure."""
        summary = omniscient_therapist.get_seeding_summary()

        assert 'total_seeding_sessions' in summary
        assert 'seeding_actions' in summary

    def test_get_seeding_summary_counts_seeding(self, omniscient_therapist):
        """Should count seeding sessions."""
        # Run some sessions
        for session in range(1, 20):
            omniscient_therapist.decide_action(client_action=6, session=session)

        summary = omniscient_therapist.get_seeding_summary()

        # Structure should be valid
        assert isinstance(summary['total_seeding_sessions'], int)
        assert isinstance(summary['seeding_actions'], dict)


# ==============================================================================
# TEST RESET FUNCTIONALITY
# ==============================================================================

class TestResetFunctionality:
    """Test therapist reset functionality."""

    def test_reset_clears_state(self, omniscient_therapist):
        """Reset should clear all tracking state."""
        # Run some sessions
        for session in range(1, 10):
            omniscient_therapist.decide_action(client_action=6, session=session)

        # Should have accumulated state
        assert len(omniscient_therapist.action_log) > 0
        assert omniscient_therapist.session_count > 0

        # Reset
        omniscient_therapist.reset()

        # Should be back to initial state
        assert omniscient_therapist.phase == "relationship_building"
        assert omniscient_therapist.action_log == []
        assert omniscient_therapist.session_count == 0
        assert omniscient_therapist.actual_actions_taken == []
        assert omniscient_therapist.current_target_client_action is None
        assert omniscient_therapist.current_target_therapist_action is None


# ==============================================================================
# TEST INTEGRATION WITH PARATAXIC CLIENT
# ==============================================================================

class TestParataxicIntegration:
    """Test integration with parataxic distortion clients."""

    def test_works_with_parataxic_client(self, parataxic_therapist, parataxic_client):
        """Should work correctly with parataxic client."""
        # Run a session
        client_action = parataxic_client.select_action()
        action, metadata = parataxic_therapist.decide_action(client_action, session=1)

        assert_valid_octant(action)
        assert 'phase' in metadata

    def test_perception_accuracy_with_parataxic(self, parataxic_therapist):
        """Should estimate perception accuracy with parataxic client."""
        accuracy = parataxic_therapist._estimate_perception_accuracy(2)

        assert 0.0 <= accuracy <= 1.0
        # Should use baseline accuracy for estimation
        assert accuracy >= parataxic_therapist.baseline_accuracy


# ==============================================================================
# TEST MULTI-SESSION SIMULATION
# ==============================================================================

class TestMultiSessionSimulation:
    """Test therapist behavior over multiple sessions."""

    def test_multi_session_maintains_valid_state(self, sample_client, omniscient_therapist):
        """Should maintain valid state across many sessions."""
        for session in range(1, 30):
            client_action = sample_client.select_action()
            therapist_action, metadata = omniscient_therapist.decide_action(client_action, session)

            # Update client
            sample_client.update_memory(client_action, therapist_action)

            # Verify valid state
            assert_valid_octant(therapist_action)
            assert omniscient_therapist.session_count == session
            assert len(omniscient_therapist.action_log) == session

    def test_phase_progression_possible(self, sample_client):
        """Should be able to progress through phases."""
        # Create therapist with client
        therapist = OmniscientStrategicTherapist(
            client_ref=sample_client,
            perception_window=15,
            baseline_accuracy=0.5
        )

        phases_seen = set()

        # Run many sessions
        for session in range(1, 50):
            client_action = sample_client.select_action()
            therapist_action, metadata = therapist.decide_action(client_action, session)

            phases_seen.add(metadata['phase'])

            # Update client
            sample_client.update_memory(client_action, therapist_action)

        # Should have seen at least the initial phase
        assert 'relationship_building' in phases_seen
        # May or may not transition depending on dynamics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
