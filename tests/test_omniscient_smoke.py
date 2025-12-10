"""
Comprehensive smoke test for omniscient RL implementation.

This script runs a quick end-to-end test of the entire omniscient RL pipeline:
1. Environment creation with wrapper
2. Network initialization
3. Training loop (small number of steps)
4. Checkpoint save/load
5. Evaluation

Can be run standalone for quick validation:
    python tests/test_omniscient_smoke.py

Or via pytest:
    pytest tests/test_omniscient_smoke.py -v
"""

import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment import TherapyEnv
from src.environment.omniscient_wrapper import OmniscientObservationWrapper
from src.training.omniscient_networks import make_omniscient_networks


def obs_to_batch(obs):
    """Convert observation dict to batch format with correct dtypes."""
    # Keys that should be long tensors (for embeddings)
    discrete_keys = {"history", "client_action", "mechanism_type",
                     "last_actual_action", "last_perceived_action", "perception_enabled"}

    obs_batch = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            if key in discrete_keys:
                obs_batch[key] = torch.tensor([value], dtype=torch.long)
            else:
                obs_batch[key] = torch.tensor([value], dtype=torch.float32)
        else:
            # Scalar discrete values
            obs_batch[key] = torch.tensor([value], dtype=torch.long)

    return obs_batch


def test_smoke_omniscient_wrapper():
    """Smoke test: Verify wrapper works correctly."""
    print("\n=== Testing Omniscient Wrapper ===")

    env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
    wrapped_env = OmniscientObservationWrapper(env)

    # Test reset
    obs, info = wrapped_env.reset(seed=42)

    print(f"✓ Environment reset successfully")
    print(f"✓ Observation keys: {list(obs.keys())}")

    # Verify omniscient components present
    assert "u_matrix" in obs
    assert "relationship_satisfaction" in obs
    assert "bond" in obs
    assert "entropy" in obs
    assert "mechanism_type" in obs
    assert "last_actual_action" in obs
    assert "last_perceived_action" in obs
    assert "parataxic_distortion_rate" in obs
    assert "parataxic_enabled" in obs

    print(f"✓ All omniscient components present")

    # Test step
    action = 3
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

    print(f"✓ Step executed successfully (reward={reward:.3f})")

    # Verify shapes and ranges
    assert obs["u_matrix"].shape == (64,)
    assert np.all(obs["u_matrix"] >= -1.0) and np.all(obs["u_matrix"] <= 1.0)
    assert -1.0 <= obs["relationship_satisfaction"][0] <= 1.0
    assert 0.0 <= obs["bond"][0] <= 1.0
    assert -1.0 <= obs["entropy"][0] <= 1.0
    assert 0 <= obs["mechanism_type"] <= 4

    print(f"✓ Observation shapes and ranges valid")
    print(f"✓ Wrapper smoke test PASSED\n")


def test_smoke_omniscient_networks():
    """Smoke test: Verify networks initialize and forward pass works."""
    print("\n=== Testing Omniscient Networks ===")

    env = TherapyEnv()
    wrapped_env = OmniscientObservationWrapper(env)

    # Create networks
    actor, critic = make_omniscient_networks(
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        hidden_sizes=(128, 128),
        device="cpu"
    )

    print(f"✓ Networks created successfully")

    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())

    print(f"✓ Actor parameters: {actor_params:,}")
    print(f"✓ Critic parameters: {critic_params:,}")

    # Get observation
    obs, _ = wrapped_env.reset(seed=42)

    # Convert to batch
    obs_batch = obs_to_batch(obs)

    # Forward pass
    with torch.no_grad():
        logits, _ = actor(obs_batch)
        value = critic(obs_batch)

    assert logits.shape == (1, 8)
    assert value.shape == (1, 1)

    print(f"✓ Forward pass successful")
    print(f"✓ Actor output shape: {logits.shape}")
    print(f"✓ Critic output shape: {value.shape}")

    # Verify logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor([1.0]))

    print(f"✓ Action probabilities valid")
    print(f"✓ Networks smoke test PASSED\n")


def test_smoke_vectorized_environments():
    """Smoke test: Verify vectorized environments work."""
    print("\n=== Testing Vectorized Environments ===")

    # Note: Just test that we can create wrapped envs and step them individually
    # Full vectorization with tianshou is tested in integration tests

    def make_env():
        env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
        wrapped_env = OmniscientObservationWrapper(env)
        return wrapped_env

    # Create multiple wrapped environments
    n_envs = 4
    envs = [make_env() for _ in range(n_envs)]

    print(f"✓ Created {n_envs} wrapped environments")

    # Reset all
    obs_list = []
    for env in envs:
        obs, _ = env.reset(seed=42)
        obs_list.append(obs)

    print(f"✓ All environments reset successfully")

    # Verify observations have correct structure
    for obs in obs_list:
        assert obs["u_matrix"].shape == (64,)
        assert obs["relationship_satisfaction"].shape == (1,)

    print(f"✓ All observations have correct structure")

    # Take steps in all environments
    for i, env in enumerate(envs):
        obs, reward, terminated, truncated, info = env.step(i % 8)

    print(f"✓ All environments stepped successfully")

    # Test networks can process batch
    # Stack observations manually
    obs_dict = {}
    for key in obs_list[0].keys():
        values = [obs[key] for obs in obs_list]
        if isinstance(values[0], np.ndarray):
            obs_dict[key] = np.stack(values)
        else:
            obs_dict[key] = np.array(values)

    obs_batch = Batch(obs_dict)

    # Create networks
    actor, critic = make_omniscient_networks(
        observation_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        device="cpu"
    )

    with torch.no_grad():
        logits, _ = actor(obs_batch)
        value = critic(obs_batch)

    assert logits.shape == (n_envs, 8)
    assert value.shape == (n_envs, 1)

    print(f"✓ Networks handle batched observations")
    print(f"✓ Vectorized environments smoke test PASSED\n")


def test_smoke_training_loop():
    """Smoke test: Run short training loop."""
    print("\n=== Testing Training Loop ===")

    env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
    wrapped_env = OmniscientObservationWrapper(env)

    # Create networks
    actor, critic = make_omniscient_networks(
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        hidden_sizes=(64, 64),  # Small for speed
        device="cpu"
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=0.001
    )

    print(f"✓ Networks and optimizer created")

    # Run training steps
    obs, _ = wrapped_env.reset(seed=42)
    total_steps = 0
    total_reward = 0.0
    num_episodes = 0
    max_steps = 200

    print(f"✓ Starting training loop (max {max_steps} steps)...")

    while total_steps < max_steps:
        # Prepare batch
        obs_batch = obs_to_batch(obs)

        # Forward pass
        logits, _ = actor(obs_batch)
        value_pred = critic(obs_batch)

        # Compute dummy loss (just to test gradient flow)
        policy_loss = -logits.mean()
        value_loss = value_pred.mean()
        loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Select action
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        # Environment step
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        total_steps += 1
        total_reward += reward

        if terminated or truncated:
            obs, _ = wrapped_env.reset()
            num_episodes += 1

    print(f"✓ Training completed: {total_steps} steps, {num_episodes} episodes")
    print(f"✓ Average reward: {total_reward/total_steps:.4f}")
    print(f"✓ Training loop smoke test PASSED\n")


def test_smoke_checkpoint_save_load():
    """Smoke test: Save and load checkpoint."""
    print("\n=== Testing Checkpoint Save/Load ===")

    env = TherapyEnv()
    wrapped_env = OmniscientObservationWrapper(env)

    # Create networks
    actor, critic = make_omniscient_networks(
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device="cpu"
    )

    print(f"✓ Networks created")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"

        # Save checkpoint
        checkpoint = {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        print(f"✓ Checkpoint saved to {checkpoint_path}")

        # Create new networks
        actor2, critic2 = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        # Load checkpoint
        loaded = torch.load(checkpoint_path, weights_only=True)
        actor2.load_state_dict(loaded["actor_state_dict"])
        critic2.load_state_dict(loaded["critic_state_dict"])

        print(f"✓ Checkpoint loaded successfully")

        # Verify networks produce same output
        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = obs_to_batch(obs)

        with torch.no_grad():
            logits1, _ = actor(obs_batch)
            logits2, _ = actor2(obs_batch)

            value1 = critic(obs_batch)
            value2 = critic2(obs_batch)

        assert torch.allclose(logits1, logits2)
        assert torch.allclose(value1, value2)

        print(f"✓ Loaded networks produce identical outputs")
        print(f"✓ Checkpoint save/load smoke test PASSED\n")


def test_smoke_evaluation():
    """Smoke test: Run evaluation rollout."""
    print("\n=== Testing Evaluation Rollout ===")

    env = TherapyEnv(pattern=["cold_stuck"], mechanism="frequency_amplifier")
    wrapped_env = OmniscientObservationWrapper(env)

    # Create networks
    actor, _ = make_omniscient_networks(
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device="cpu"
    )

    print(f"✓ Networks created")

    # Run evaluation episode
    obs, _ = wrapped_env.reset(seed=42)
    episode_reward = 0.0
    episode_length = 0

    print(f"✓ Starting evaluation episode...")

    while episode_length < 100:
        # Convert to batch
        obs_batch = obs_to_batch(obs)

        # Get action from policy
        with torch.no_grad():
            logits, _ = actor(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()  # Greedy for evaluation

        # Take action
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        episode_reward += reward
        episode_length += 1

        if terminated or truncated:
            break

    print(f"✓ Episode completed: length={episode_length}, reward={episode_reward:.3f}")

    if terminated and not truncated:
        print(f"✓ Episode terminated naturally (success or dropout)")
    elif truncated:
        print(f"✓ Episode truncated (max sessions reached)")

    print(f"✓ Evaluation rollout smoke test PASSED\n")


def test_smoke_all_mechanisms():
    """Smoke test: Verify all mechanism types work."""
    print("\n=== Testing All Mechanism Types ===")

    mechanisms = [
        "bond_only",
        "frequency_amplifier",
        "conditional_amplifier",
        "bond_weighted_frequency_amplifier",
        "bond_weighted_conditional_amplifier",
    ]

    for mechanism in mechanisms:
        env = TherapyEnv(mechanism=mechanism)
        wrapped_env = OmniscientObservationWrapper(env)

        # Create networks
        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        # Run a few steps
        obs, _ = wrapped_env.reset(seed=42)

        for _ in range(5):
            obs_batch = obs_to_batch(obs)

            with torch.no_grad():
                logits, _ = actor(obs_batch)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            if terminated or truncated:
                break

        print(f"  ✓ {mechanism}")

    print(f"✓ All mechanisms smoke test PASSED\n")


def run_all_smoke_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("OMNISCIENT RL SMOKE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Wrapper", test_smoke_omniscient_wrapper),
        ("Networks", test_smoke_omniscient_networks),
        ("Vectorized Environments", test_smoke_vectorized_environments),
        ("Training Loop", test_smoke_training_loop),
        ("Checkpoint Save/Load", test_smoke_checkpoint_save_load),
        ("Evaluation", test_smoke_evaluation),
        ("All Mechanisms", test_smoke_all_mechanisms),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} smoke test FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"SMOKE TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_smoke_tests()
