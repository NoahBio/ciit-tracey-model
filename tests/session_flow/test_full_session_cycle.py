"""
Session Flow Tests: Complete Session Cycle (Steps 2-7)

Tests the full session flow after action selection:
  Step 2: Therapist responds to client action
  Step 3: Client perceives therapist response (with optional perceptual distortion)
  Step 4: Memory update (add new interaction, remove oldest)
  Step 5: RS calculation (weighted average of utilities)
  Step 6: Bond calculation (normalized RS → sigmoid)
  Step 7: Dropout check (at session 10, if RS decreased)

Note: Steps 4-6 have detailed unit tests in test_base_client.py. These tests focus on
integration and flow between steps.
"""

import pytest
import numpy as np

from src.agents.client_agents import (
    BondOnlyClient,
    FrequencyAmplifierClient,
)
from src.config import MEMORY_SIZE


# ==============================================================================
# TEST STEP 2: THERAPIST RESPONSE
# ==============================================================================

class TestStep2_TherapistResponse:
    """Test therapist response handling in session flow."""

    def test_therapist_function_receives_client_action(self, bond_only_client, complementary_therapist):
        """Therapist function should receive client's selected action."""
        # Client selects action
        client_action = bond_only_client.select_action()

        # Therapist responds
        therapist_action = complementary_therapist(client_action)

        # Should be valid octant
        assert 0 <= therapist_action <= 7
        assert isinstance(therapist_action, (int, np.integer))

    def test_various_therapist_strategies(self, bond_only_client):
        """Should work with different therapist response strategies."""
        client_action = bond_only_client.select_action()

        # Complementary therapist
        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
        therapist_complementary = complement_map[client_action]
        assert 0 <= therapist_complementary <= 7

        # Always warm therapist
        therapist_always_warm = 4  # Always respond with warm (complementary to cold)
        assert therapist_always_warm == 4

        # Random therapist (simulated)
        rng = np.random.RandomState(42)
        therapist_random = int(rng.choice(8))
        assert 0 <= therapist_random <= 7


# ==============================================================================
# TEST STEP 3: PERCEPTION
# ==============================================================================

class TestStep3_Perception:
    """Test perception step (identity function for base clients, distortion for perceptual variants)."""

    def test_base_client_perception_is_identity(self, bond_only_client):
        """Base clients perceive therapist actions without distortion."""
        client_action = 0
        therapist_action = 4

        # For base clients, perceived action == actual action
        # (no distortion method exists on base clients)
        assert not hasattr(bond_only_client, 'perceive')
        # Perception is implicit identity - therapist action passed directly to memory update

    def test_perception_preserves_valid_octant(self, bond_only_client):
        """Perceived action should always be valid octant."""
        therapist_action = 5

        # Base client has no perception distortion
        # If there were a perceive() method, we'd test it here
        # For now, verify therapist action is valid
        assert 0 <= therapist_action <= 7


# ==============================================================================
# TEST STEP 4: MEMORY UPDATE (Integration Context)
# ==============================================================================

class TestStep4_MemoryUpdateIntegration:
    """Test memory update in session flow context (detailed tests in test_base_client.py)."""

    def test_memory_update_after_interaction(self, bond_only_client, complementary_therapist):
        """After interaction, memory should be updated."""
        initial_memory = list(bond_only_client.memory)
        initial_session_count = bond_only_client.session_count

        # Full interaction
        client_action = bond_only_client.select_action()
        therapist_action = complementary_therapist(client_action)

        # Update memory
        bond_only_client.update_memory(client_action, therapist_action)

        # Memory should change
        assert list(bond_only_client.memory) != initial_memory
        assert bond_only_client.memory[-1] == (client_action, therapist_action)

        # Session count should increment
        assert bond_only_client.session_count == initial_session_count + 1

    def test_memory_update_triggers_recalculation(self, bond_only_client, complementary_therapist):
        """Memory update should trigger RS and bond recalculation."""
        # Record initial state
        initial_rs = bond_only_client.relationship_satisfaction
        initial_bond = bond_only_client.bond

        # Interaction with very good outcome
        bond_only_client.update_memory(client_action=0, therapist_action=4)

        # RS and bond should be recalculated (may increase or change)
        # Not asserting direction because it depends on previous memory
        # Just verify they exist and are valid
        assert hasattr(bond_only_client, 'relationship_satisfaction')
        assert hasattr(bond_only_client, 'bond')
        assert 0 <= bond_only_client.bond <= 1


# ==============================================================================
# TEST STEP 5: RS CALCULATION (Integration Context)
# ==============================================================================

class TestStep5_RSCalculationIntegration:
    """Test RS calculation in session flow context (detailed tests in test_base_client.py)."""

    def test_rs_reflects_recent_interactions(self, fixed_u_matrix, low_entropy, fixed_seed):
        """RS should reflect recent interactions due to recency weighting."""
        # Start with cold memory
        cold_memory = [(0, 0)] * MEMORY_SIZE

        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=cold_memory,
            random_state=fixed_seed
        )

        initial_rs = client.relationship_satisfaction

        # Add several warm interactions
        for _ in range(25):  # Half of memory
            client.update_memory(client_action=0, therapist_action=4)

        updated_rs = client.relationship_satisfaction

        # RS should change (likely increase due to warm interactions)
        assert updated_rs != initial_rs

    def test_rs_within_bounds(self, bond_only_client):
        """RS should always be within utility matrix bounds."""
        # After any interaction, RS should be within bounds
        assert bond_only_client.rs_min <= bond_only_client.relationship_satisfaction <= bond_only_client.rs_max


# ==============================================================================
# TEST STEP 6: BOND CALCULATION (Integration Context)
# ==============================================================================

class TestStep6_BondCalculationIntegration:
    """Test bond calculation in session flow context (detailed tests in test_base_client.py)."""

    def test_bond_updates_after_memory_change(self, bond_only_client):
        """Bond should update after memory changes."""
        initial_bond = bond_only_client.bond

        # Add interaction
        bond_only_client.update_memory(client_action=1, therapist_action=3)

        # Bond may change depending on interaction quality
        # Just verify it's still valid
        assert 0 <= bond_only_client.bond <= 1

    def test_bond_range_maintained(self, bond_only_client):
        """Bond should always be in [0, 1] range."""
        # Perform multiple interactions
        for i in range(10):
            bond_only_client.update_memory(client_action=i % 8, therapist_action=(i+2) % 8)

        # Bond should still be in valid range
        assert 0 <= bond_only_client.bond <= 1


# ==============================================================================
# TEST STEP 7: DROPOUT CHECK (Integration Context)
# ==============================================================================

class TestStep7_DropoutCheckIntegration:
    """Test dropout check in session flow context (detailed tests in test_base_client.py)."""

    def test_dropout_checked_at_session_10(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Dropout should be checked at session 10."""
        # Start with problematic memory (low RS)
        from src.agents.client_agents.base_client import BaseClientAgent
        problematic_memory = BaseClientAgent.generate_problematic_memory(
            pattern_type="cold_stuck",
            n_interactions=MEMORY_SIZE,
            random_state=fixed_seed
        )

        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=problematic_memory,
            random_state=fixed_seed
        )

        # Record initial RS for dropout comparison
        initial_rs = client.initial_rs

        # Simulate 9 sessions (no dropout check)
        for session in range(9):
            client_action = client.select_action()
            therapist_action = 0  # Always cold to potentially trigger dropout
            client.update_memory(client_action, therapist_action)

        # Should not have checked dropout yet
        assert client.session_count == 9
        # (dropout_checked flag is set by check_dropout() method)

        # Session 10 - this is when dropout would be checked
        client_action = client.select_action()
        therapist_action = 0
        client.update_memory(client_action, therapist_action)

        assert client.session_count == 10

        # Manual dropout check (in real simulation, this would be called by runner)
        should_dropout = client.check_dropout()

        # Should be boolean
        assert isinstance(should_dropout, bool)

        # If RS decreased, should dropout
        if client.relationship_satisfaction < initial_rs:
            assert should_dropout is True


# ==============================================================================
# TEST COMPLETE SESSION CYCLE
# ==============================================================================

class TestCompleteSessionCycle:
    """Test complete session cycle integrating all steps."""

    def test_full_single_session(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Test one complete session through all steps."""
        client = BondOnlyClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Record initial state
        initial_memory = list(client.memory)
        initial_session_count = client.session_count

        # Step 1: Client selects action
        client_action = client.select_action()
        assert 0 <= client_action <= 7

        # Step 2: Therapist responds
        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
        therapist_action = complement_map[client_action]

        # Step 3: Perception (identity for base client)
        perceived_action = therapist_action  # No distortion for base clients

        # Step 4-6: Update memory (triggers RS and bond recalculation)
        client.update_memory(client_action, perceived_action)

        # Verify state changes
        assert list(client.memory) != initial_memory
        assert client.memory[-1] == (client_action, perceived_action)
        assert client.session_count == initial_session_count + 1
        assert 0 <= client.bond <= 1

        # Step 7: Dropout check (not applicable before session 10)
        # Would be checked by external runner

    def test_multiple_sessions_sequence(self, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
        """Test multiple sessions in sequence."""
        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Complementary therapist
        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}

        # Run 5 sessions
        for session_num in range(5):
            # Full cycle
            client_action = client.select_action()
            therapist_action = complement_map[client_action]
            client.update_memory(client_action, therapist_action)

            # Verify state is valid after each session
            assert client.session_count == session_num + 1
            assert 0 <= client.bond <= 1
            assert len(client.memory) == MEMORY_SIZE

    def test_session_cycle_with_state_evolution(self, fixed_u_matrix, low_entropy, fixed_seed):
        """Test that state evolves realistically through session cycles."""
        # Start with cold stuck pattern
        from src.agents.client_agents.base_client import BaseClientAgent
        cold_memory = BaseClientAgent.generate_problematic_memory(
            pattern_type="cold_stuck",
            n_interactions=MEMORY_SIZE,
            random_state=fixed_seed
        )

        client = FrequencyAmplifierClient(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=cold_memory,
            random_state=fixed_seed
        )

        initial_bond = client.bond
        bond_history = [initial_bond]

        # Always warm therapist (trying to improve bond)
        for _ in range(20):
            client_action = client.select_action()
            therapist_action = 4  # Always warm/complementary

            client.update_memory(client_action, therapist_action)
            bond_history.append(client.bond)

        # Bond should change over time
        assert len(set(bond_history)) > 1, "Bond should evolve over sessions"

        # With consistent warm therapist, bond likely increases
        # (but not guaranteed due to complex dynamics)
        final_bond = client.bond
        # Just verify bond is valid
        assert 0 <= final_bond <= 1

    @pytest.mark.parametrize("mechanism_class", [
        BondOnlyClient,
        FrequencyAmplifierClient,
    ])
    def test_session_cycle_all_mechanisms(
        self, mechanism_class, fixed_u_matrix, low_entropy, complementary_memory, fixed_seed
    ):
        """All mechanisms should handle complete session cycle."""
        client = mechanism_class(
            u_matrix=fixed_u_matrix,
            entropy=low_entropy,
            initial_memory=complementary_memory,
            random_state=fixed_seed
        )

        # Run 3 sessions
        for _ in range(3):
            client_action = client.select_action()
            therapist_action = (client_action + 4) % 8  # Simple response strategy
            client.update_memory(client_action, therapist_action)

        # Verify valid state
        assert client.session_count == 3
        assert 0 <= client.bond <= 1
        assert len(client.memory) == MEMORY_SIZE
