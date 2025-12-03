#type: ignore
"""
Tests for TherapyEnv Gymnasium environment.

Tests cover:
- Environment creation and configuration
- Reset functionality
- Step mechanics
- Episode termination (success, dropout, truncation)
- Pattern sampling
- Gymnasium compatibility
"""

import pytest
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from src.environment import TherapyEnv


class TestEnvironmentCreation:
    """Tests for environment initialization and configuration."""

    def test_env_creation(self):
        """Create env with default params, verify spaces."""
        env = TherapyEnv()

        # Check that environment was created
        assert env is not None

        # Verify observation space structure
        assert isinstance(env.observation_space, spaces.Dict)
        assert "client_action" in env.observation_space.spaces
        assert "session_number" in env.observation_space.spaces
        assert "history" in env.observation_space.spaces

        # Verify observation space types and shapes
        assert isinstance(env.observation_space["client_action"], spaces.Discrete)
        assert env.observation_space["client_action"].n == 8

        assert isinstance(env.observation_space["session_number"], spaces.Box)
        assert env.observation_space["session_number"].shape == (1,)

        assert isinstance(env.observation_space["history"], spaces.MultiDiscrete)
        assert len(env.observation_space["history"].nvec) == 50  # 25 interactions * 2
        assert np.all(env.observation_space["history"].nvec == 9)  # 0-7 octants + 8 padding

        # Verify action space
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 8

    def test_env_creation_with_params(self):
        """Create env with custom parameters."""
        env = TherapyEnv(
            mechanism="bond_only",
            pattern="cold_stuck",
            threshold=0.8,
            max_sessions=50,
            entropy=1.0,
            history_weight=0.5,
            bond_alpha=3.0,
            bond_offset=0.6,
            enable_perception=False,
            baseline_accuracy=0.3,
            random_state=42
        )

        assert env is not None
        assert env._mechanism == "bond_only"
        assert env._pattern == "cold_stuck"
        assert env._threshold_percentile == 0.8
        assert env._max_sessions == 50
        assert env._entropy == 1.0

    def test_env_invalid_params(self):
        """Test that invalid parameters raise errors."""
        # Empty pattern list
        with pytest.raises(ValueError, match="Pattern list cannot be empty"):
            TherapyEnv(pattern=[])

        # Invalid threshold
        with pytest.raises(ValueError, match="Threshold must be in"):
            TherapyEnv(threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be in"):
            TherapyEnv(threshold=0.0)

        # Invalid max_sessions
        with pytest.raises(ValueError, match="max_sessions must be positive"):
            TherapyEnv(max_sessions=0)

        # Invalid entropy
        with pytest.raises(ValueError, match="entropy must be positive"):
            TherapyEnv(entropy=-1.0)

    def test_env_invalid_mechanism(self):
        """Test that invalid mechanism raises error."""
        env = TherapyEnv(mechanism="invalid_mechanism")
        with pytest.raises(ValueError, match="Unknown mechanism"):
            env.reset()


class TestReset:
    """Tests for environment reset functionality."""

    def test_reset(self):
        """Call reset, verify observation shape and types."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        # Check observation structure
        assert isinstance(obs, dict)
        assert "client_action" in obs
        assert "session_number" in obs
        assert "history" in obs

        # Verify observation types
        assert isinstance(obs["client_action"], (int, np.integer))
        assert 0 <= obs["client_action"] < 8

        assert isinstance(obs["session_number"], np.ndarray)
        assert obs["session_number"].shape == (1,)
        assert 0.0 <= obs["session_number"][0] <= 1.0

        assert isinstance(obs["history"], np.ndarray)
        assert obs["history"].shape == (50,)
        assert np.all((obs["history"] >= 0) & (obs["history"] <= 8))

        # Verify initial history is padded with 8
        assert np.all(obs["history"] == 8), "Initial history should be all padding (8s)"

        # Verify info dict
        assert isinstance(info, dict)
        assert "pattern" in info
        assert "entropy" in info
        assert "rs_threshold" in info
        assert "initial_rs" in info
        assert "initial_bond" in info
        assert "episode_seed" in info

    def test_reset_reproducibility(self):
        """Same seed produces same episode."""
        env1 = TherapyEnv()
        env2 = TherapyEnv()

        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        # Same seed should produce same initial observation
        assert obs1["client_action"] == obs2["client_action"]
        np.testing.assert_array_equal(obs1["session_number"], obs2["session_number"])
        np.testing.assert_array_equal(obs1["history"], obs2["history"])

        # Same seed should produce same initial state
        assert info1["initial_rs"] == info2["initial_rs"]
        assert info1["initial_bond"] == info2["initial_bond"]
        assert info1["rs_threshold"] == info2["rs_threshold"]

    def test_reset_different_seeds(self):
        """Different seeds produce different episodes."""
        env = TherapyEnv()

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=123)

        # Different seeds should produce different results
        # (with high probability)
        assert (obs1["client_action"] != obs2["client_action"] or
                info1["initial_rs"] != info2["initial_rs"])

    def test_reset_session_count(self):
        """Reset should set session count to 0."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        # Take some steps
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        # Reset and verify session count is 0
        obs, info = env.reset(seed=43)
        assert obs["session_number"][0] == 0.0


class TestStep:
    """Tests for environment step mechanics."""

    def test_step_basic(self):
        """Take a few random steps, verify returns."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify return types
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            # Verify observation validity
            assert env.observation_space.contains(obs)

            # Verify info dict
            assert "session" in info
            assert "rs" in info
            assert "bond" in info
            assert "client_action" in info
            assert "therapist_action" in info
            assert "success" in info
            assert "dropped_out" in info
            assert "max_reached" in info

            if terminated or truncated:
                break

    def test_step_invalid_action(self):
        """Test that invalid actions raise errors."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        # Action outside valid range
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(8)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_step_session_count_increments(self):
        """Session count should increment with each step."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        for i in range(1, 6):
            obs, reward, terminated, truncated, info = env.step(0)

            # Session count should increment
            assert info["session"] == i
            assert obs["session_number"][0] == pytest.approx(i / env._max_sessions)

            if terminated or truncated:
                break

    def test_step_history_builds(self):
        """History should build up as steps are taken."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        # Initial history should be all 8s (padding)
        assert np.all(obs["history"] == 8)

        # Take first step
        client_action_1 = obs["client_action"]
        therapist_action_1 = 2
        obs, reward, terminated, truncated, info = env.step(therapist_action_1)

        # History should now have first interaction
        assert obs["history"][0] == client_action_1
        assert obs["history"][1] == therapist_action_1
        assert np.all(obs["history"][2:] == 8)  # Rest still padding

        # Take second step
        client_action_2 = obs["client_action"]
        therapist_action_2 = 3
        obs, reward, terminated, truncated, info = env.step(therapist_action_2)

        # History should now have two interactions
        assert obs["history"][0] == client_action_1
        assert obs["history"][1] == therapist_action_1
        assert obs["history"][2] == client_action_2
        assert obs["history"][3] == therapist_action_2
        assert np.all(obs["history"][4:] == 8)  # Rest still padding

    def test_step_non_terminal_reward(self):
        """Non-terminal steps should have zero reward."""
        env = TherapyEnv()
        obs, info = env.reset(seed=42)

        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)

            if not terminated and not truncated:
                assert reward == 0.0
            else:
                break


class TestEpisodeTermination:
    """Tests for episode termination conditions."""

    def test_episode_success(self):
        """Mock a scenario that should succeed."""
        # Use complementary policy which should lead to success
        env = TherapyEnv(
            pattern="complementary_perfect",  # Easy pattern
            threshold=0.7,  # Lower threshold
            entropy=0.3,  # Low entropy (less random)
            max_sessions=100
        )

        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}

        success_found = False
        for seed in range(10):  # Try multiple seeds
            obs, info = env.reset(seed=seed)

            for _ in range(100):
                therapist_action = complement_map[obs["client_action"]]
                obs, reward, terminated, truncated, info = env.step(therapist_action)

                if terminated:
                    if info["success"]:
                        success_found = True
                        # Verify success conditions
                        assert info["rs"] >= info.get("rs_threshold", env._rs_threshold)
                        assert reward > 0  # Should get positive reward
                        # Verify efficiency bonus: 100 + (100 - session) * 2
                        expected_reward = 100 + (100 - info["session"]) * 2
                        assert reward == pytest.approx(expected_reward)
                        break
                    break

                if truncated:
                    break

            if success_found:
                break

        assert success_found, "Should achieve success with complementary policy on easy pattern"

    def test_episode_dropout(self):
        """Verify dropout triggers correctly at session 10."""
        # Use a policy that should worsen RS (anticomplementary)
        env = TherapyEnv(
            pattern="cold_stuck",
            threshold=0.9,
            entropy=0.5,
            max_sessions=100
        )

        # Anticomplementary policy (should worsen RS)
        # Map: opposite octants
        anticomplement_map = {0: 0, 1: 1, 2: 6, 3: 7, 4: 4, 5: 5, 6: 2, 7: 3}

        dropout_found = False
        for seed in range(20):  # Try multiple seeds
            obs, info = env.reset(seed=seed)
            initial_rs = info["initial_rs"]

            for session in range(1, 101):
                therapist_action = anticomplement_map[obs["client_action"]]
                obs, reward, terminated, truncated, info = env.step(therapist_action)

                if terminated:
                    if info["dropped_out"]:
                        dropout_found = True
                        # Verify dropout conditions
                        assert info["session"] == 10, "Dropout should occur at session 10"
                        assert info["rs"] < initial_rs, "RS should have decreased"
                        assert reward == -150.0, "Dropout should give -150 reward"
                        assert not info["success"], "Dropout should not be success"
                        break
                    break

                if truncated:
                    break

            if dropout_found:
                break

        assert dropout_found, "Should find dropout with anticomplementary policy"

    def test_episode_max_sessions(self):
        """Verify truncation at max_sessions."""
        env = TherapyEnv(
            pattern="cold_stuck",
            threshold=0.95,  # Very high threshold (unlikely to reach)
            max_sessions=20,  # Short episode
            entropy=0.5
        )

        obs, info = env.reset(seed=42)

        for session in range(1, 25):  # Run longer than max_sessions
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated:
                # Should not terminate early (unless dropout)
                if info["dropped_out"]:
                    break
                pytest.fail("Should not terminate before max_sessions without success")

            if truncated:
                # Verify max_sessions truncation
                assert info["session"] == 20, "Should truncate at max_sessions"
                assert info["max_reached"], "max_reached should be True"
                assert not info["success"], "Should not succeed"
                assert not info["dropped_out"], "Should not dropout"
                assert reward == 0.0, "Max sessions should give 0 reward"
                break

        assert truncated, "Should truncate at max_sessions"

    def test_success_takes_precedence_over_max_sessions(self):
        """If success and max_sessions both occur, success wins."""
        # This test verifies that if RS >= threshold at exactly max_sessions,
        # it's treated as success (terminated) not truncation
        env = TherapyEnv(
            pattern="complementary_perfect",
            threshold=0.5,  # Low threshold
            max_sessions=10,
            entropy=0.3
        )

        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}

        obs, info = env.reset(seed=42)

        for _ in range(10):
            therapist_action = complement_map[obs["client_action"]]
            obs, reward, terminated, truncated, info = env.step(therapist_action)

            if terminated or truncated:
                if info["success"]:
                    # Success should terminate, not truncate
                    assert terminated
                    assert not truncated
                break


class TestPatternSampling:
    """Tests for pattern sampling functionality."""

    def test_single_pattern(self):
        """Single pattern should always be used."""
        env = TherapyEnv(pattern="cold_stuck")

        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert info["pattern"] == "cold_stuck"

    def test_multi_pattern_sampling(self):
        """When patterns is a list, verify different patterns are sampled."""
        patterns = ["cold_stuck", "dominant_stuck", "submissive_stuck"]
        env = TherapyEnv(pattern=patterns)

        patterns_seen = set()
        for seed in range(30):  # Run enough to likely see all patterns
            obs, info = env.reset(seed=seed)
            assert info["pattern"] in patterns
            patterns_seen.add(info["pattern"])

        # Should see multiple different patterns
        assert len(patterns_seen) >= 2, "Should sample different patterns across episodes"

    def test_pattern_sampling_randomness(self):
        """Different seeds should produce different pattern samples."""
        patterns = ["cold_stuck", "dominant_stuck"]
        env = TherapyEnv(pattern=patterns)

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=43)

        # Collect patterns from multiple resets
        pattern_sequences = []
        for base_seed in [100, 200, 300]:
            sequence = []
            for offset in range(10):
                _, info = env.reset(seed=base_seed + offset)
                sequence.append(info["pattern"])
            pattern_sequences.append(tuple(sequence))

        # Different seed sequences should produce different pattern sequences
        assert len(set(pattern_sequences)) > 1, "Different seeds should produce different pattern sequences"


class TestGymnasiumCompatibility:
    """Tests for Gymnasium compatibility."""

    def test_gymnasium_check_env(self):
        """Use gymnasium.utils.env_checker.check_env()."""
        env = TherapyEnv()

        # This will raise an error if environment is not compatible
        check_env(env, skip_render_check=True)

    def test_observation_space_sample(self):
        """Test that observation space sampling works."""
        env = TherapyEnv()

        # Sample from observation space
        sample = env.observation_space.sample()

        assert "client_action" in sample
        assert "session_number" in sample
        assert "history" in sample

        # Verify sample is valid
        assert env.observation_space.contains(sample)

    def test_action_space_sample(self):
        """Test that action space sampling works."""
        env = TherapyEnv()

        # Sample from action space
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
            assert 0 <= action < 8


class TestRewardStructure:
    """Tests for reward calculation."""

    def test_success_reward_formula(self):
        """Verify success reward formula: 100 + (max_sessions - session) * 2."""
        env = TherapyEnv(
            pattern="complementary_perfect",
            threshold=0.5,
            max_sessions=100,
            entropy=0.3
        )

        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}

        # Try to achieve success
        for seed in range(10):
            obs, info = env.reset(seed=seed)

            for _ in range(100):
                therapist_action = complement_map[obs["client_action"]]
                obs, reward, terminated, truncated, info = env.step(therapist_action)

                if terminated and info["success"]:
                    # Verify reward formula
                    expected_reward = 100 + (100 - info["session"]) * 2
                    assert reward == pytest.approx(expected_reward)
                    # Earlier success should give higher reward
                    assert reward >= 100
                    return  # Test passed

                if terminated or truncated:
                    break

        pytest.skip("Could not achieve success to test reward formula")

    def test_dropout_reward(self):
        """Verify dropout gives -150 reward."""
        env = TherapyEnv(
            pattern="cold_stuck",
            threshold=0.9,
            entropy=0.5
        )

        # Anticomplementary policy
        anticomplement_map = {0: 0, 1: 1, 2: 6, 3: 7, 4: 4, 5: 5, 6: 2, 7: 3}

        for seed in range(20):
            obs, info = env.reset(seed=seed)

            for _ in range(15):
                therapist_action = anticomplement_map[obs["client_action"]]
                obs, reward, terminated, truncated, info = env.step(therapist_action)

                if terminated and info["dropped_out"]:
                    assert reward == -150.0
                    return  # Test passed

                if terminated or truncated:
                    break

        pytest.skip("Could not trigger dropout to test reward")

    def test_max_sessions_reward(self):
        """Verify max_sessions gives 0 reward."""
        env = TherapyEnv(
            pattern="cold_stuck",
            threshold=0.95,
            max_sessions=10,
            entropy=0.5
        )

        obs, info = env.reset(seed=42)

        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(0)

            if truncated and info["max_reached"]:
                assert reward == 0.0
                return  # Test passed

            if terminated:
                break

        pytest.skip("Could not reach max_sessions to test reward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
