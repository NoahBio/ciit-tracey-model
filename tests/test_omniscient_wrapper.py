"""
Unit tests for OmniscientObservationWrapper.

Tests cover:
- Extended observation space definition
- Normalization functions (u_matrix, RS, entropy)
- Mechanism type mapping
- Perception info extraction
- Edge cases (perception disabled, no perception history)
- Gymnasium compatibility
"""

import pytest
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from src.environment import TherapyEnv
from src.environment.omniscient_wrapper import OmniscientObservationWrapper


class TestWrapperCreation:
    """Tests for wrapper initialization and observation space definition."""

    def test_wrapper_creation(self):
        """Create wrapper and verify extended observation space."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        assert wrapped_env is not None
        assert isinstance(wrapped_env.observation_space, spaces.Dict)

    def test_observation_space_has_base_components(self):
        """Verify wrapper preserves base environment observations."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs_space = wrapped_env.observation_space.spaces

        # Check base components are preserved
        assert "client_action" in obs_space
        assert "session_number" in obs_space
        assert "history" in obs_space

        # Verify types match original
        assert isinstance(obs_space["client_action"], spaces.Discrete)
        assert obs_space["client_action"].n == 8

        assert isinstance(obs_space["session_number"], spaces.Box)
        assert obs_space["session_number"].shape == (1,)

        assert isinstance(obs_space["history"], spaces.MultiDiscrete)

    def test_observation_space_has_omniscient_components(self):
        """Verify wrapper adds all omniscient observation components."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs_space = wrapped_env.observation_space.spaces

        # Check new omniscient components
        assert "u_matrix" in obs_space
        assert "relationship_satisfaction" in obs_space
        assert "bond" in obs_space
        assert "entropy" in obs_space
        assert "mechanism_type" in obs_space
        assert "last_actual_action" in obs_space
        assert "last_perceived_action" in obs_space
        assert "misperception_rate" in obs_space
        assert "perception_enabled" in obs_space

    def test_omniscient_observation_space_shapes(self):
        """Verify shapes and dtypes of omniscient components."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs_space = wrapped_env.observation_space.spaces

        # U-matrix: flattened 8x8 = 64 dims
        assert isinstance(obs_space["u_matrix"], spaces.Box)
        assert obs_space["u_matrix"].shape == (64,)
        assert obs_space["u_matrix"].dtype == np.float32
        assert obs_space["u_matrix"].low[0] == -1.0
        assert obs_space["u_matrix"].high[0] == 1.0

        # RS: scalar normalized to [-1, 1]
        assert isinstance(obs_space["relationship_satisfaction"], spaces.Box)
        assert obs_space["relationship_satisfaction"].shape == (1,)
        assert obs_space["relationship_satisfaction"].dtype == np.float32
        assert obs_space["relationship_satisfaction"].low[0] == -1.0
        assert obs_space["relationship_satisfaction"].high[0] == 1.0

        # Bond: scalar in [0, 1]
        assert isinstance(obs_space["bond"], spaces.Box)
        assert obs_space["bond"].shape == (1,)
        assert obs_space["bond"].dtype == np.float32
        assert obs_space["bond"].low[0] == 0.0
        assert obs_space["bond"].high[0] == 1.0

        # Entropy: scalar normalized to [-1, 1]
        assert isinstance(obs_space["entropy"], spaces.Box)
        assert obs_space["entropy"].shape == (1,)
        assert obs_space["entropy"].dtype == np.float32
        assert obs_space["entropy"].low[0] == -1.0
        assert obs_space["entropy"].high[0] == 1.0

        # Mechanism type: discrete 0-4 (5 types)
        assert isinstance(obs_space["mechanism_type"], spaces.Discrete)
        assert obs_space["mechanism_type"].n == 5

        # Last actions: discrete 0-8 (8 octants + "none")
        assert isinstance(obs_space["last_actual_action"], spaces.Discrete)
        assert obs_space["last_actual_action"].n == 9

        assert isinstance(obs_space["last_perceived_action"], spaces.Discrete)
        assert obs_space["last_perceived_action"].n == 9

        # Misperception rate: scalar in [0, 1]
        assert isinstance(obs_space["misperception_rate"], spaces.Box)
        assert obs_space["misperception_rate"].shape == (1,)
        assert obs_space["misperception_rate"].dtype == np.float32
        assert obs_space["misperception_rate"].low[0] == 0.0
        assert obs_space["misperception_rate"].high[0] == 1.0

        # Perception enabled: discrete binary
        assert isinstance(obs_space["perception_enabled"], spaces.Discrete)
        assert obs_space["perception_enabled"].n == 2


class TestObservationExtraction:
    """Tests for extracting omniscient observations from environment."""

    def test_reset_returns_extended_observation(self):
        """Verify reset() returns all omniscient components."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        obs, info = wrapped_env.reset(seed=42)

        # Check all keys present
        expected_keys = {
            "client_action", "session_number", "history",
            "u_matrix", "relationship_satisfaction", "bond", "entropy",
            "mechanism_type", "last_actual_action", "last_perceived_action",
            "misperception_rate", "perception_enabled"
        }
        assert set(obs.keys()) == expected_keys

    def test_step_returns_extended_observation(self):
        """Verify step() returns all omniscient components."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        wrapped_env.reset(seed=42)
        obs, reward, terminated, truncated, info = wrapped_env.step(0)

        # Check all keys present
        expected_keys = {
            "client_action", "session_number", "history",
            "u_matrix", "relationship_satisfaction", "bond", "entropy",
            "mechanism_type", "last_actual_action", "last_perceived_action",
            "misperception_rate", "perception_enabled"
        }
        assert set(obs.keys()) == expected_keys

    def test_u_matrix_normalized(self):
        """Verify u_matrix is properly normalized to [-1, 1]."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)
        u_matrix = obs["u_matrix"]

        # Check shape
        assert u_matrix.shape == (64,)

        # Check normalization range
        assert np.all(u_matrix >= -1.0)
        assert np.all(u_matrix <= 1.0)

        # Check dtype
        assert u_matrix.dtype == np.float32

    def test_rs_normalized(self):
        """Verify RS is properly normalized to [-1, 1]."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)
        rs = obs["relationship_satisfaction"]

        # Check shape
        assert rs.shape == (1,)

        # Check normalization range
        assert rs[0] >= -1.0
        assert rs[0] <= 1.0

        # Check dtype
        assert rs.dtype == np.float32

    def test_bond_in_range(self):
        """Verify bond is in [0, 1]."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)
        bond = obs["bond"]

        # Check shape
        assert bond.shape == (1,)

        # Check range
        assert bond[0] >= 0.0
        assert bond[0] <= 1.0

        # Check dtype
        assert bond.dtype == np.float32

    def test_entropy_normalized(self):
        """Verify entropy is properly normalized to [-1, 1]."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)
        entropy = obs["entropy"]

        # Check shape
        assert entropy.shape == (1,)

        # Check normalization range (should be in [-1, 1])
        assert entropy[0] >= -1.0
        assert entropy[0] <= 1.0

        # Check dtype
        assert entropy.dtype == np.float32


class TestMechanismMapping:
    """Tests for mechanism type mapping to integers."""

    @pytest.mark.parametrize("mechanism,expected_idx", [
        ("bond_only", 0),
        ("frequency_amplifier", 1),
        ("conditional_amplifier", 2),
        ("bond_weighted_frequency_amplifier", 3),
        ("bond_weighted_conditional_amplifier", 4),
    ])
    def test_mechanism_type_mapping(self, mechanism, expected_idx):
        """Verify each mechanism type maps to correct integer."""
        env = TherapyEnv(mechanism=mechanism)
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)
        mechanism_type = obs["mechanism_type"]

        assert mechanism_type == expected_idx


class TestPerceptionTracking:
    """Tests for perception-related observations."""

    def test_perception_enabled_flag(self):
        """Verify perception_enabled flag is correct."""
        # Test with perception enabled
        env = TherapyEnv(enable_perception=True)
        wrapped_env = OmniscientObservationWrapper(env)
        obs, _ = wrapped_env.reset(seed=42)
        assert obs["perception_enabled"] == 1

        # Test with perception disabled
        env = TherapyEnv(enable_perception=False)
        wrapped_env = OmniscientObservationWrapper(env)
        obs, _ = wrapped_env.reset(seed=42)
        assert obs["perception_enabled"] == 0

    def test_perception_tracking_with_perception_enabled(self):
        """Verify actual/perceived actions tracked when perception enabled."""
        env = TherapyEnv(enable_perception=True, baseline_accuracy=0.2)
        wrapped_env = OmniscientObservationWrapper(env)

        wrapped_env.reset(seed=42)

        # Take a step
        action = 3
        obs, _, _, _, _ = wrapped_env.step(action)

        # Check that last actions are tracked (should be valid octants or 8)
        assert obs["last_actual_action"] in range(9)
        assert obs["last_perceived_action"] in range(9)

        # After first step, actual action should match what we sent
        assert obs["last_actual_action"] == action

    def test_perception_tracking_with_perception_disabled(self):
        """Verify actual/perceived actions match when perception disabled."""
        env = TherapyEnv(enable_perception=False)
        wrapped_env = OmniscientObservationWrapper(env)

        wrapped_env.reset(seed=42)

        # Take a step
        action = 5
        obs, _, _, _, _ = wrapped_env.step(action)

        # When perception disabled, actual and perceived should match
        assert obs["last_actual_action"] == action
        assert obs["last_perceived_action"] == action

    def test_misperception_rate_in_range(self):
        """Verify misperception_rate is in [0, 1]."""
        env = TherapyEnv(enable_perception=True)
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)
        misperception_rate = obs["misperception_rate"]

        # Check shape
        assert misperception_rate.shape == (1,)

        # Check range
        assert misperception_rate[0] >= 0.0
        assert misperception_rate[0] <= 1.0

        # Check dtype
        assert misperception_rate.dtype == np.float32


class TestNormalizationEdgeCases:
    """Tests for edge cases in normalization."""

    def test_u_matrix_normalization_with_identical_values(self):
        """Verify u_matrix normalization handles all-identical values."""
        # This is a tricky edge case: if all utilities are the same,
        # normalization should return zeros (no gradient)
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        # Reset and manually set all u_matrix values to be identical
        obs, _ = wrapped_env.reset(seed=42)
        wrapped_env.env._client.u_matrix[:] = 1.0

        # Take a step to get new observation with modified u_matrix
        obs_modified, _, _, _, _ = wrapped_env.step(0)

        u_matrix_norm = obs_modified["u_matrix"]

        # Should be all zeros (or very close)
        assert np.allclose(u_matrix_norm, 0.0, atol=1e-6)

    def test_rs_normalization_with_identical_bounds(self):
        """Verify RS normalization handles identical min/max bounds."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)

        # Manually set rs_min == rs_max (edge case)
        # Note: We directly call _add_omniscient_obs to avoid triggering
        # bond calculation which would also divide by zero
        client = wrapped_env.env._client
        client.rs_min = 10.0
        client.rs_max = 10.0
        client.relationship_satisfaction = 10.0

        # Create a dummy observation and add omniscient components
        base_obs = obs.copy()
        obs_modified = wrapped_env._add_omniscient_obs(base_obs)

        rs_norm = obs_modified["relationship_satisfaction"]

        # Should be 0.0 (no gradient when bounds are identical)
        assert np.isclose(rs_norm[0], 0.0, atol=1e-6)


class TestGymnasiumCompatibility:
    """Tests for Gymnasium environment checker compatibility."""

    def test_env_checker_passes(self):
        """Verify wrapped environment passes Gymnasium env checker."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        # This will raise an error if the environment is not compliant
        try:
            check_env(wrapped_env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium env checker failed: {e}")


class TestMultipleSteps:
    """Tests for wrapper behavior over multiple steps."""

    def test_wrapper_maintains_consistency_over_steps(self):
        """Verify observations remain consistent over multiple steps."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        wrapped_env.reset(seed=42)

        for _ in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # Verify all omniscient keys present at each step
            expected_keys = {
                "client_action", "session_number", "history",
                "u_matrix", "relationship_satisfaction", "bond", "entropy",
                "mechanism_type", "last_actual_action", "last_perceived_action",
                "misperception_rate", "perception_enabled"
            }
            assert set(obs.keys()) == expected_keys

            # Verify all values in valid ranges
            assert np.all(obs["u_matrix"] >= -1.0) and np.all(obs["u_matrix"] <= 1.0)
            assert -1.0 <= obs["relationship_satisfaction"][0] <= 1.0
            assert 0.0 <= obs["bond"][0] <= 1.0
            assert -1.0 <= obs["entropy"][0] <= 1.0
            assert 0 <= obs["mechanism_type"] <= 4
            assert 0 <= obs["last_actual_action"] <= 8
            assert 0 <= obs["last_perceived_action"] <= 8
            assert 0.0 <= obs["misperception_rate"][0] <= 1.0
            assert 0 <= obs["perception_enabled"] <= 1

            if terminated or truncated:
                break

    def test_perception_history_accumulates(self):
        """Verify perception info updates correctly over multiple steps."""
        env = TherapyEnv(enable_perception=True, baseline_accuracy=0.2)
        wrapped_env = OmniscientObservationWrapper(env)

        wrapped_env.reset(seed=42)

        for i in range(5):
            action = i % 8  # Cycle through actions
            obs, _, terminated, truncated, _ = wrapped_env.step(action)

            # After first step, last_actual_action should be a valid action (not 8=sentinel)
            if i > 0:
                assert obs["last_actual_action"] in range(8)

            # last_actual_action should match the therapist action we just took
            assert obs["last_actual_action"] == action

            if terminated or truncated:
                break


class TestDifferentMechanisms:
    """Tests wrapper with all mechanism types."""

    @pytest.mark.parametrize("mechanism", [
        "bond_only",
        "frequency_amplifier",
        "conditional_amplifier",
        "bond_weighted_frequency_amplifier",
        "bond_weighted_conditional_amplifier",
    ])
    def test_wrapper_works_with_all_mechanisms(self, mechanism):
        """Verify wrapper works correctly with each mechanism type."""
        env = TherapyEnv(mechanism=mechanism)
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)

        # Verify observation is valid
        assert "u_matrix" in obs
        assert "mechanism_type" in obs

        # Take a few steps
        for _ in range(3):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                break


class TestWrapperEquivalence:
    """Tests that wrapper doesn't change core environment behavior."""

    def test_rewards_match_unwrapped_env(self):
        """Verify wrapper doesn't affect reward calculation."""
        # Create two identical environments
        env1 = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        env2 = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env2)

        # Reset with same seed
        env1.reset(seed=42)
        wrapped_env.reset(seed=42)

        # Take same actions and compare rewards
        for _ in range(10):
            action = 3  # Fixed action for determinism

            _, reward1, term1, trunc1, _ = env1.step(action)
            _, reward2, term2, trunc2, _ = wrapped_env.step(action)

            assert np.isclose(reward1, reward2), \
                f"Rewards differ: unwrapped={reward1}, wrapped={reward2}"
            assert term1 == term2
            assert trunc1 == trunc2

            if term1 or trunc1:
                break

    def test_termination_matches_unwrapped_env(self):
        """Verify wrapper doesn't affect termination logic."""
        env1 = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier", max_sessions=10)
        env2 = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier", max_sessions=10)
        wrapped_env = OmniscientObservationWrapper(env2)

        env1.reset(seed=42)
        wrapped_env.reset(seed=42)

        # Run until termination
        for _ in range(20):
            action = 0

            _, _, term1, trunc1, _ = env1.step(action)
            _, _, term2, trunc2, _ = wrapped_env.step(action)

            assert term1 == term2
            assert trunc1 == trunc2

            if term1 or trunc1:
                break
