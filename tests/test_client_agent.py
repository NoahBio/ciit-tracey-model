"""
Unit tests for ClientAgent.

Run with: pytest tests/test_client_agent.py -v
"""
# pyright: reportPrivateUsage=false

import pytest
import numpy as np
from src.agents.client_agent import ClientAgent, create_client
from src.config import MEMORY_SIZE, U_MATRIX


class TestClientAgent:
    """Test suite for ClientAgent class."""
    
    def test_initialization(self):
        """Test client initialization with valid memory."""
        # Create simple memory
        memory = [(0, 4)] * MEMORY_SIZE  # D-S pairs (complementary)
        client = ClientAgent(memory, entropy=0.5, random_state=42)
        
        assert len(client.memory) == MEMORY_SIZE
        assert client.entropy == 0.5
        assert 0 <= client.bond <= 1
    
    def test_initialization_invalid_memory_length(self):
        """Test that invalid memory length raises error."""
        memory = [(0, 4)] * 10  # Too short
        
        with pytest.raises(ValueError, match="must have length"):
            ClientAgent(memory, entropy=0.5)
    
    def test_initialization_invalid_entropy(self):
        """Test that non-positive entropy raises error."""
        memory = [(0, 4)] * MEMORY_SIZE
        
        with pytest.raises(ValueError, match="entropy must be positive"):
            ClientAgent(memory, entropy=-0.5)
    
    def test_relationship_satisfaction_calculation(self):
        """Test RS calculation for known memory patterns."""
        # Perfect complementarity: D-S pairs
        memory = [(0, 4)] * MEMORY_SIZE  # All D-S (utility = +2.0)
        client = ClientAgent(memory, entropy=0.5)
        
        expected_rs = U_MATRIX[0, 4]  # Should be +2.0
        assert np.isclose(client.relationship_satisfaction, expected_rs)
    
    def test_bond_range(self):
        """Test that bond is always in [0, 1]."""
        # Test with very negative RS
        memory = [(0, 0)] * MEMORY_SIZE  # D-D pairs (conflictual, -2.0)
        client = ClientAgent(memory, entropy=0.5)
        assert 0 <= client.bond <= 1
        
        # Test with very positive RS
        memory = [(0, 4)] * MEMORY_SIZE  # D-S pairs (complementary, +2.0)
        client = ClientAgent(memory, entropy=0.5)
        assert 0 <= client.bond <= 1
    
    def test_bond_monotonicity(self):
        """Test that better RS leads to higher bond."""
        # Poor relationship
        poor_memory = [(0, 0)] * MEMORY_SIZE  # D-D (conflict)
        client_poor = ClientAgent(poor_memory, entropy=0.5)
        
        # Good relationship
        good_memory = [(0, 4)] * MEMORY_SIZE  # D-S (complement)
        client_good = ClientAgent(good_memory, entropy=0.5)
        
        assert client_poor.bond < client_good.bond
    
    def test_expected_payoffs_shape(self):
        """Test that expected payoffs has correct shape."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=0.5)
        
        payoffs = client._calculate_expected_payoffs()
        assert payoffs.shape == (8,)
    
    def test_expected_payoffs_bond_effect(self):
        """Test that high bond leads to more optimistic expectations."""
        # High bond client (good history)
        good_memory = [(2, 2)] * MEMORY_SIZE  # W-W (very positive)
        client_high_bond = ClientAgent(good_memory, entropy=0.5)
        
        # Low bond client (poor history)
        poor_memory = [(0, 0)] * MEMORY_SIZE  # D-D (very negative)
        client_low_bond = ClientAgent(poor_memory, entropy=0.5)
        
        payoffs_high = client_high_bond._calculate_expected_payoffs()
        payoffs_low = client_low_bond._calculate_expected_payoffs()
        
        # High bond should have higher expected payoffs overall
        assert np.mean(payoffs_high) > np.mean(payoffs_low)
    
    def test_softmax_probabilities(self):
        """Test that softmax produces valid probability distribution."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=0.5, random_state=42)
        
        payoffs = client._calculate_expected_payoffs()
        probs = client._softmax(payoffs)
        
        # Check properties of probability distribution
        assert np.all(probs >= 0)
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_softmax_temperature_effect(self):
        """Test that entropy affects action distribution."""
        memory = [(0, 4)] * MEMORY_SIZE
        
        # Low entropy (deterministic)
        client_low_ent = ClientAgent(memory, entropy=0.1, random_state=42)
        payoffs = client_low_ent._calculate_expected_payoffs()
        probs_low = client_low_ent._softmax(payoffs)
        
        # High entropy (random)
        client_high_ent = ClientAgent(memory, entropy=2.0, random_state=42)
        probs_high = client_high_ent._softmax(payoffs)
        
        # Low entropy should be more peaked (higher max probability)
        assert np.max(probs_low) > np.max(probs_high)
    
    def test_action_selection(self):
        """Test that action selection returns valid octant."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=0.5, random_state=42)
        
        action = client.select_action()
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action <= 7
    
    def test_action_selection_stochastic(self):
        """Test that action selection is stochastic."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=1.0, random_state=42)
        
        # Sample many actions
        actions = [client.select_action() for _ in range(100)]
        
        # Should have some variety (not all the same)
        assert len(set(actions)) > 1
    
    def test_dropout_probability_range(self):
        """Test that dropout returns boolean."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=0.5, random_state=42)
        
        dropout = client.check_dropout()
        assert isinstance(dropout, (bool, np.bool_))
    
    def test_dropout_bond_effect(self):
        """Test that low bond increases dropout probability."""
        # High bond (should rarely dropout)
        good_memory = [(2, 2)] * MEMORY_SIZE
        client_high_bond = ClientAgent(good_memory, entropy=0.5, random_state=42)
        
        # Low bond (should dropout more)
        poor_memory = [(0, 0)] * MEMORY_SIZE
        client_low_bond = ClientAgent(poor_memory, entropy=0.5, random_state=42)
        
        # Sample dropout many times
        n_samples = 1000
        dropout_high = sum(client_high_bond.check_dropout() for _ in range(n_samples))
        dropout_low = sum(client_low_bond.check_dropout() for _ in range(n_samples))
        
        # Low bond should dropout more frequently
        assert dropout_low > dropout_high
    
    def test_memory_update(self):
        """Test that memory updates correctly."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=0.5)
        
        old_rs = client.relationship_satisfaction
        
        # Add a new (different) interaction
        client.update_memory(2, 2)  # W-W (very positive)
        
        # Memory should have changed
        assert len(client.memory) == MEMORY_SIZE  # Still 50
        assert client.memory[-1] == (2, 2)  # New interaction at end
        
        # RS should have changed
        assert client.relationship_satisfaction != old_rs
    
    def test_get_state(self):
        """Test state retrieval."""
        memory = [(0, 4)] * MEMORY_SIZE
        client = ClientAgent(memory, entropy=0.5)
        
        state = client.get_state()
        
        assert "relationship_satisfaction" in state
        assert "bond" in state
        assert "entropy" in state
        assert state["memory_length"] == MEMORY_SIZE
    
    def test_generate_problematic_memory(self):
        """Test problematic memory generation."""
        for pattern in ["cold_stuck", "dominant_stuck", "submissive_stuck"]:
            memory = ClientAgent.generate_problematic_memory(
                pattern_type=pattern,
                random_state=42
            )
            
            assert len(memory) == MEMORY_SIZE
            assert all(isinstance(pair, tuple) for pair in memory)
            assert all(len(pair) == 2 for pair in memory)
            assert all(0 <= c <= 7 and 0 <= t <= 7 for c, t in memory)
    
    def test_create_client_convenience(self):
        """Test convenience function for client creation."""
        client = create_client(pattern_type="cold_stuck", random_state=42)
        
        assert isinstance(client, ClientAgent)
        assert len(client.memory) == MEMORY_SIZE
        assert client.entropy > 0
        
        # Client with cold pattern should have low bond
        assert client.bond < 0.5


class TestClientIntegration:
    """Integration tests simulating therapy sessions."""
    
    def test_therapy_session_simulation(self):
        """Simulate a few therapy cycles."""
        client = create_client(pattern_type="cold_stuck", random_state=42)
        
        initial_bond = client.bond
        
        # Simulate 10 sessions with complementary responses
        for _ in range(10):
            client_action = client.select_action()
            
            # Therapist responds complementarily (simplified)
            # For cold actions, respond with warmth
            if client_action in [5, 6, 7]:  # CS, C, CD
                therapist_action = 2  # W (warm)
            else:
                therapist_action = 4  # S (submissive)
            
            client.update_memory(client_action, therapist_action)
            
            # Check dropout
            if client.check_dropout():
                break
        
        # Bond should improve with good therapist responses
        assert client.bond > initial_bond
    
    def test_no_dropout_with_perfect_complementarity(self):
        """Client with perfect complementarity should rarely dropout."""
        # Start with perfect complementary history
        memory = [(0, 4)] * MEMORY_SIZE  # D-S (perfect)
        client = ClientAgent(memory, entropy=0.5, random_state=42)
        
        # Sample dropout 100 times
        dropouts = sum(client.check_dropout() for _ in range(100))
        
        # Should have very few dropouts (allow for some randomness)
        assert dropouts < 10


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])