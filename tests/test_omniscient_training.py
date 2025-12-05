"""
Integration tests for omniscient RL training pipeline.

Tests cover:
- Environment creation with wrapper
- Vectorized environments with wrapper
- Network accepts omniscient observations
- Training loop with small number of steps
- Checkpoint save/load
- Config loading
- End-to-end training smoke test
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.environment import TherapyEnv
from src.environment.omniscient_wrapper import OmniscientObservationWrapper
from src.training.config import TrainingConfig, load_config
from src.training.omniscient_networks import make_omniscient_networks
from tianshou.data import Batch


class TestEnvironmentCreation:
    """Tests for creating environments with omniscient wrapper."""

    def test_create_wrapped_environment(self):
        """Create environment with omniscient wrapper."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        assert wrapped_env is not None
        obs, info = wrapped_env.reset(seed=42)

        # Verify omniscient observations present
        assert "u_matrix" in obs
        assert "mechanism_type" in obs

    def test_create_multiple_wrapped_environments(self):
        """Create multiple wrapped environments for vectorization."""
        envs = []
        for i in range(4):
            env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
            wrapped_env = OmniscientObservationWrapper(env)
            envs.append(wrapped_env)

        assert len(envs) == 4

        # Verify each env works independently
        for env in envs:
            obs, _ = env.reset(seed=42)
            assert "u_matrix" in obs


class TestConfigLoading:
    """Tests for loading omniscient experiment config."""

    def test_load_omniscient_config(self):
        """Load omniscient experiment config file."""
        config_path = Path("configs/omniscient_experiment.yaml")

        if not config_path.exists():
            pytest.skip("omniscient_experiment.yaml not found")

        config = load_config(config_path)

        # Verify omniscient-specific settings
        assert hasattr(config, "use_omniscient_wrapper")
        assert config.use_omniscient_wrapper is True

        # Verify environment settings
        assert "cold_stuck" in config.patterns
        assert config.mechanism == "frequency_amplifier"

    def test_config_has_correct_hyperparameters(self):
        """Verify omniscient config has appropriate hyperparameters."""
        config_path = Path("configs/omniscient_experiment.yaml")

        if not config_path.exists():
            pytest.skip("omniscient_experiment.yaml not found")

        config = load_config(config_path)

        # Check adjusted hyperparameters for richer observation space
        assert config.learning_rate <= 0.0001  # Lower LR
        assert config.batch_size >= 128  # Larger batch size
        assert config.hidden_size >= 256  # Adequate network capacity


class TestNetworkIntegration:
    """Tests for network integration with wrapped environments."""

    def test_networks_accept_wrapped_observations(self):
        """Verify networks accept observations from wrapped environment."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        # Get observation
        obs, _ = wrapped_env.reset(seed=42)

        # Convert to batch format
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        # Forward pass through both networks
        logits, _ = actor(obs_batch)
        value = critic(obs_batch)

        assert logits.shape == (1, 8)
        assert value.shape == (1, 1)

    def test_networks_with_tianshou_batch(self):
        """Verify networks work with Tianshou Batch objects."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        # Get multiple observations
        obs_list = []
        for i in range(4):
            wrapped_env.reset(seed=i)
            obs, _, _, _, _ = wrapped_env.step(i % 8)
            obs_list.append(obs)

        # Create Tianshou Batch
        batch_dict = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                batch_dict[key] = np.stack(values)
            else:
                batch_dict[key] = np.array(values)

        batch = Batch(**batch_dict)

        # Forward pass
        logits, _ = actor(batch)
        value = critic(batch)

        assert logits.shape == (4, 8)
        assert value.shape == (4, 1)


class TestTrainingLoop:
    """Tests for training loop integration."""

    def test_environment_step_with_network_action(self):
        """Test full step: network output → action → environment step."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, _ = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)

        # Convert to batch
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        # Get action from network
        with torch.no_grad():
            logits, _ = actor(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        # Take action in environment
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)

        # Verify observation is valid
        assert "u_matrix" in next_obs
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_multi_step_training_loop(self):
        """Test multiple training steps."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            hidden_sizes=(128, 128),
            device="cpu"
        )

        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=0.0001
        )

        obs, _ = wrapped_env.reset(seed=42)

        # Run a few training steps
        for _ in range(10):
            # Convert to batch
            obs_batch = {
                key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
                for key, value in obs.items()
            }

            # Forward pass
            logits, _ = actor(obs_batch)
            value = critic(obs_batch)

            # Compute dummy loss
            loss = -logits.sum() + value.sum()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Take action
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                obs, _ = wrapped_env.reset()

        # If we get here, training loop works


class TestCheckpointing:
    """Tests for saving and loading checkpoints."""

    def test_save_and_load_networks(self):
        """Test saving and loading network checkpoints."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        # Create networks
        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        # Create temporary directory for checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"

            # Save checkpoint
            checkpoint = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            # Create new networks
            actor2, critic2 = make_omniscient_networks(
                observation_space=wrapped_env.observation_space,
                action_space=wrapped_env.action_space,
                device="cpu"
            )

            # Load checkpoint
            loaded = torch.load(checkpoint_path)
            actor2.load_state_dict(loaded["actor_state_dict"])
            critic2.load_state_dict(loaded["critic_state_dict"])

            # Verify networks produce same output
            obs, _ = wrapped_env.reset(seed=42)
            obs_batch = {
                key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
                for key, value in obs.items()
            }

            with torch.no_grad():
                logits1, _ = actor(obs_batch)
                logits2, _ = actor2(obs_batch)

                value1 = critic(obs_batch)
                value2 = critic2(obs_batch)

            assert torch.allclose(logits1, logits2)
            assert torch.allclose(value1, value2)


class TestEndToEndSmoke:
    """Smoke test for end-to-end training."""

    def test_short_training_run(self):
        """Run short training to verify everything works together."""
        # This is a minimal smoke test - just verify no errors
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            hidden_sizes=(64, 64),  # Small for speed
            device="cpu"
        )

        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=0.001
        )

        obs, _ = wrapped_env.reset(seed=42)
        total_steps = 0
        num_episodes = 0

        # Run for 100 steps or 5 episodes
        while total_steps < 100 and num_episodes < 5:
            obs_batch = {
                key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
                for key, value in obs.items()
            }

            logits, _ = actor(obs_batch)
            value = critic(obs_batch)

            # Dummy loss
            loss = -logits.sum() + value.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Take action
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_steps += 1

            if terminated or truncated:
                obs, _ = wrapped_env.reset()
                num_episodes += 1

        # If we get here, training completed without errors
        assert total_steps > 0
        assert num_episodes > 0


class TestVectorizedEnvironments:
    """Tests for vectorized environment support."""

    def test_create_vectorized_wrapped_envs(self):
        """Create vectorized environments with wrapper."""
        from tianshou.env import DummyVectorEnv

        def make_env():
            env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
            wrapped_env = OmniscientObservationWrapper(env)
            return wrapped_env

        # Create vectorized environment
        vec_env = DummyVectorEnv([make_env for _ in range(4)])

        # Reset and verify
        obs = vec_env.reset()

        # Check that we have batch of observations
        assert obs.u_matrix.shape[0] == 4  # batch size
        assert obs.u_matrix.shape[1] == 64  # u_matrix dimension

    def test_vectorized_env_step(self):
        """Test stepping through vectorized wrapped environments."""
        from tianshou.env import DummyVectorEnv

        def make_env():
            env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
            wrapped_env = OmniscientObservationWrapper(env)
            return wrapped_env

        vec_env = DummyVectorEnv([make_env for _ in range(4)])
        vec_env.reset()

        # Take a step with batch of actions
        actions = np.array([0, 1, 2, 3])
        obs, rewards, dones, infos = vec_env.step(actions)

        # Verify batch sizes
        assert obs.u_matrix.shape[0] == 4
        assert len(rewards) == 4
        assert len(dones) == 4

    def test_networks_with_vectorized_observations(self):
        """Verify networks work with vectorized environment observations."""
        from tianshou.env import DummyVectorEnv

        def make_env():
            env = TherapyEnv()
            wrapped_env = OmniscientObservationWrapper(env)
            return wrapped_env

        vec_env = DummyVectorEnv([make_env for _ in range(8)])

        actor, critic = make_omniscient_networks(
            observation_space=vec_env.envs[0].observation_space,
            action_space=vec_env.envs[0].action_space,
            device="cpu"
        )

        # Get vectorized observations
        obs = vec_env.reset()

        # Forward pass
        with torch.no_grad():
            logits, _ = actor(obs)
            value = critic(obs)

        # Verify batch dimensions
        assert logits.shape == (8, 8)  # (batch_size=8, action_dim=8)
        assert value.shape == (8, 1)


class TestObservationConsistency:
    """Tests for observation consistency during training."""

    def test_observations_remain_valid_over_episode(self):
        """Verify omniscient observations remain valid throughout episode."""
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)

        obs, _ = wrapped_env.reset(seed=42)

        for step in range(50):
            # Verify observation validity
            assert "u_matrix" in obs
            assert obs["u_matrix"].shape == (64,)
            assert np.all(obs["u_matrix"] >= -1.0) and np.all(obs["u_matrix"] <= 1.0)

            assert "relationship_satisfaction" in obs
            assert -1.0 <= obs["relationship_satisfaction"][0] <= 1.0

            assert "bond" in obs
            assert 0.0 <= obs["bond"][0] <= 1.0

            assert "mechanism_type" in obs
            assert 0 <= obs["mechanism_type"] <= 4

            # Take action
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                break


class TestMemoryLeaks:
    """Tests for potential memory leaks."""

    def test_no_memory_leak_over_many_steps(self):
        """Verify no obvious memory leaks over many steps."""
        import gc

        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)

        # Run many steps
        for _ in range(1000):
            obs_batch = {
                key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
                for key, value in obs.items()
            }

            with torch.no_grad():
                logits, _ = actor(obs_batch)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                obs, _ = wrapped_env.reset()

        # Force garbage collection
        gc.collect()

        # If we get here without running out of memory, test passes


class TestDifferentConfigurations:
    """Tests with different environment configurations."""

    @pytest.mark.parametrize("mechanism", [
        "bond_only",
        "frequency_amplifier",
        "conditional_amplifier",
        "bond_weighted_frequency_amplifier",
        "bond_weighted_conditional_amplifier",
    ])
    def test_training_with_different_mechanisms(self, mechanism):
        """Verify training works with all mechanism types."""
        env = TherapyEnv(mechanism=mechanism)
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)

        # Run a few steps
        for _ in range(10):
            obs_batch = {
                key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
                for key, value in obs.items()
            }

            with torch.no_grad():
                logits, _ = actor(obs_batch)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                break

    @pytest.mark.parametrize("enable_perception", [True, False])
    def test_training_with_perception_settings(self, enable_perception):
        """Verify training works with perception enabled/disabled."""
        env = TherapyEnv(enable_perception=enable_perception)
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)

        # Verify perception_enabled flag matches
        assert obs["perception_enabled"] == (1 if enable_perception else 0)

        # Run a few steps
        for _ in range(10):
            obs_batch = {
                key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
                for key, value in obs.items()
            }

            with torch.no_grad():
                logits, _ = actor(obs_batch)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                break
