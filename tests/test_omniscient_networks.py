"""
Unit tests for OmniscientTherapyNet neural networks.

Tests cover:
- Network initialization
- Forward pass with omniscient observations
- Input dimension calculation (471)
- U-matrix processor compression (64 → 32)
- Embedding dimensions
- Actor and critic output shapes
- Batch processing
- Device handling
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from src.environment import TherapyEnv
from src.environment.omniscient_wrapper import OmniscientObservationWrapper
from src.training.omniscient_networks import (
    OmniscientTherapyNet,
    Actor,
    Critic,
    make_omniscient_networks
)


class TestNetworkInitialization:
    """Tests for network initialization."""

    def test_network_creation(self):
        """Create network and verify it initializes correctly."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        assert net is not None
        assert isinstance(net, nn.Module)

    def test_network_has_correct_components(self):
        """Verify network has all required components."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        # Check embeddings exist
        assert hasattr(net, "client_action_embed")
        assert hasattr(net, "history_embed")
        assert hasattr(net, "mechanism_embed")
        assert hasattr(net, "last_actual_action_embed")
        assert hasattr(net, "last_perceived_action_embed")
        assert hasattr(net, "perception_enabled_embed")

        # Check u_matrix processor exists
        assert hasattr(net, "u_matrix_processor")

        # Check normalization and MLP exist
        assert hasattr(net, "feature_norm")
        assert hasattr(net, "mlp")

    def test_network_output_dim(self):
        """Verify network output dimension matches hidden size."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        hidden_size = 128
        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(hidden_size, hidden_size),
            device="cpu"
        )

        assert net.output_dim == hidden_size

    def test_embedding_dimensions(self):
        """Verify embedding dimensions are correct."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        # Check embedding dimensions
        assert net.client_action_embed.num_embeddings == 8
        assert net.client_action_embed.embedding_dim == 16

        assert net.history_embed.num_embeddings == 9
        assert net.history_embed.embedding_dim == 8

        assert net.mechanism_embed.num_embeddings == 5
        assert net.mechanism_embed.embedding_dim == 8

        assert net.last_actual_action_embed.num_embeddings == 9
        assert net.last_actual_action_embed.embedding_dim == 4

        assert net.last_perceived_action_embed.num_embeddings == 9
        assert net.last_perceived_action_embed.embedding_dim == 4

        assert net.perception_enabled_embed.num_embeddings == 2
        assert net.perception_enabled_embed.embedding_dim == 2

    def test_u_matrix_processor_architecture(self):
        """Verify u_matrix processor has correct architecture."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        # Check u_matrix processor layers
        processor = net.u_matrix_processor

        # Should have: Linear(64, 48) -> ReLU -> Linear(48, 32) -> ReLU
        assert len(processor) == 4

        assert isinstance(processor[0], nn.Linear)
        assert processor[0].in_features == 64
        assert processor[0].out_features == 48

        assert isinstance(processor[1], nn.ReLU)

        assert isinstance(processor[2], nn.Linear)
        assert processor[2].in_features == 48
        assert processor[2].out_features == 32

        assert isinstance(processor[3], nn.ReLU)


class TestForwardPass:
    """Tests for forward pass through network."""

    def test_forward_pass_single_observation(self):
        """Test forward pass with single observation."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        # Get observation
        obs, _ = wrapped_env.reset(seed=42)

        # Convert to batch format (add batch dimension)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        # Forward pass
        output, state = net(obs_batch)

        # Check output shape
        assert output.shape == (1, 256)  # (batch_size=1, hidden_size=256)
        assert state is None  # MLP has no state

    def test_forward_pass_batch_observations(self):
        """Test forward pass with batch of observations."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(128, 128),
            device="cpu"
        )

        # Get multiple observations
        batch_size = 8
        obs_list = []
        for i in range(batch_size):
            wrapped_env.reset(seed=i)
            obs, _, _, _, _ = wrapped_env.step(i % 8)
            obs_list.append(obs)

        # Stack into batch
        obs_batch = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                obs_batch[key] = torch.tensor(np.stack(values), dtype=torch.float32)
            else:
                obs_batch[key] = torch.tensor(values, dtype=torch.long)

        # Forward pass
        output, state = net(obs_batch)

        # Check output shape
        assert output.shape == (batch_size, 128)
        assert state is None

    def test_forward_pass_preserves_gradients(self):
        """Verify forward pass allows gradient computation."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)

        # Convert to batch format with gradients enabled
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        # Forward pass
        output, _ = net(obs_batch)

        # Compute dummy loss and backprop
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_u_matrix_compression(self):
        """Verify u_matrix is compressed from 64 to 32 dims."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        # Create dummy u_matrix (64 dims)
        u_matrix = torch.randn(4, 64)  # batch_size=4

        # Pass through processor
        compressed = net.u_matrix_processor(u_matrix)

        # Check output shape
        assert compressed.shape == (4, 32)


class TestActorCriticNetworks:
    """Tests for Actor and Critic networks."""

    def test_actor_creation(self):
        """Create actor network and verify initialization."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        preprocess_net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        actor = Actor(preprocess_net, action_dim=8)

        assert actor is not None
        assert isinstance(actor, nn.Module)
        assert hasattr(actor, "preprocess")
        assert hasattr(actor, "output_layer")

    def test_actor_forward_pass(self):
        """Test actor forward pass returns correct logits shape."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        preprocess_net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        actor = Actor(preprocess_net, action_dim=8)

        # Get observation
        obs, _ = wrapped_env.reset(seed=42)

        # Convert to batch format
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        # Forward pass
        logits, state = actor(obs_batch)

        # Check output shape: (batch_size, action_dim)
        assert logits.shape == (1, 8)
        assert state is None

    def test_actor_logits_to_probabilities(self):
        """Verify actor logits can be converted to valid probability distribution."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        preprocess_net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        actor = Actor(preprocess_net, action_dim=8)

        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        logits, _ = actor(obs_batch)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Check valid probability distribution
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))

    def test_critic_creation(self):
        """Create critic network and verify initialization."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        preprocess_net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        critic = Critic(preprocess_net)

        assert critic is not None
        assert isinstance(critic, nn.Module)
        assert hasattr(critic, "preprocess")
        assert hasattr(critic, "output_layer")

    def test_critic_forward_pass(self):
        """Test critic forward pass returns correct value shape."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        preprocess_net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        critic = Critic(preprocess_net)

        # Get observation
        obs, _ = wrapped_env.reset(seed=42)

        # Convert to batch format
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        # Forward pass
        value = critic(obs_batch)

        # Check output shape: (batch_size, 1)
        assert value.shape == (1, 1)


class TestNetworkFactory:
    """Tests for make_omniscient_networks factory function."""

    def test_factory_creates_networks(self):
        """Verify factory creates both actor and critic."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        assert actor is not None
        assert critic is not None
        assert isinstance(actor, nn.Module)
        assert isinstance(critic, nn.Module)

    def test_factory_networks_on_correct_device(self):
        """Verify factory places networks on specified device."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, critic = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        # Check that parameters are on CPU
        for param in actor.parameters():
            assert param.device.type == "cpu"

        for param in critic.parameters():
            assert param.device.type == "cpu"

    def test_factory_actor_has_correct_action_dim(self):
        """Verify factory creates actor with correct action dimension."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        actor, _ = make_omniscient_networks(
            observation_space=wrapped_env.observation_space,
            action_space=wrapped_env.action_space,
            device="cpu"
        )

        # Get observation
        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        logits, _ = actor(obs_batch)

        # Should have 8 action dimensions
        assert logits.shape[-1] == 8


class TestDeviceHandling:
    """Tests for device handling (CPU/GPU)."""

    def test_network_on_cpu(self):
        """Verify network works on CPU."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long, device="cpu")
            for key, value in obs.items()
        }

        output, _ = net(obs_batch)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_network_on_cuda(self):
        """Verify network works on CUDA if available."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cuda"
        )

        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long, device="cuda")
            for key, value in obs.items()
        }

        output, _ = net(obs_batch)

        assert output.device.type == "cuda"


class TestDifferentHiddenSizes:
    """Tests network with different hidden sizes."""

    @pytest.mark.parametrize("hidden_sizes", [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (128, 256),
        (256, 128, 64),
    ])
    def test_network_with_different_hidden_sizes(self, hidden_sizes):
        """Verify network works with various hidden size configurations."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=hidden_sizes,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        output, _ = net(obs_batch)

        # Output dimension should match last hidden size
        assert output.shape == (1, hidden_sizes[-1])


class TestBatchProcessing:
    """Tests for processing batches of different sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64, 128])
    def test_network_handles_different_batch_sizes(self, batch_size):
        """Verify network handles various batch sizes correctly."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            hidden_sizes=(256, 256),
            device="cpu"
        )

        # Get single observation and replicate
        obs, _ = wrapped_env.reset(seed=42)

        # Create batch
        obs_batch = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_batch[key] = torch.tensor(
                    np.repeat([value], batch_size, axis=0),
                    dtype=torch.float32
                )
            else:
                obs_batch[key] = torch.tensor(
                    [value] * batch_size,
                    dtype=torch.long
                )

        # Forward pass
        output, _ = net(obs_batch)

        # Check output shape
        assert output.shape == (batch_size, 256)


class TestInputDimensions:
    """Tests for verifying total input dimension is 471."""

    def test_total_input_dimension_is_471(self):
        """Verify concatenated features have 471 dimensions."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        # Check feature_norm expects 471 dims
        assert net.feature_norm.normalized_shape[0] == 471

    def test_feature_breakdown(self):
        """Verify feature dimension breakdown sums to 471."""
        # Expected dimensions:
        # client_action_embed: 16
        # session_number: 1
        # history_embed (flattened): 50 * 8 = 400
        # u_matrix (compressed): 32
        # relationship_satisfaction: 1
        # bond: 1
        # entropy: 1
        # mechanism_embed: 8
        # last_actual_action_embed: 4
        # last_perceived_action_embed: 4
        # misperception_rate: 1
        # perception_enabled_embed: 2
        # TOTAL: 16 + 1 + 400 + 32 + 1 + 1 + 1 + 8 + 4 + 4 + 1 + 2 = 471

        total = 16 + 1 + 400 + 32 + 1 + 1 + 1 + 8 + 4 + 4 + 1 + 2
        assert total == 471


class TestGradientFlow:
    """Tests for verifying gradients flow through all components."""

    def test_gradients_flow_to_all_embeddings(self):
        """Verify all embeddings receive gradients."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        output, _ = net(obs_batch)
        loss = output.sum()
        loss.backward()

        # Check all embeddings have gradients
        assert net.client_action_embed.weight.grad is not None
        assert net.history_embed.weight.grad is not None
        assert net.mechanism_embed.weight.grad is not None
        assert net.last_actual_action_embed.weight.grad is not None
        assert net.last_perceived_action_embed.weight.grad is not None
        assert net.perception_enabled_embed.weight.grad is not None

    def test_gradients_flow_to_u_matrix_processor(self):
        """Verify u_matrix processor receives gradients."""
        env = TherapyEnv()
        wrapped_env = OmniscientObservationWrapper(env)

        net = OmniscientTherapyNet(
            observation_space=wrapped_env.observation_space,
            device="cpu"
        )

        obs, _ = wrapped_env.reset(seed=42)
        obs_batch = {
            key: torch.tensor([value], dtype=torch.float32 if isinstance(value, np.ndarray) else torch.long)
            for key, value in obs.items()
        }

        output, _ = net(obs_batch)
        loss = output.sum()
        loss.backward()

        # Check u_matrix processor layers have gradients
        assert net.u_matrix_processor[0].weight.grad is not None  # First linear
        assert net.u_matrix_processor[2].weight.grad is not None  # Second linear
