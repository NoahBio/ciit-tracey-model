"""Neural network architectures for omniscient therapy RL agents.

These networks process extended observations that include client internal state
(u_matrix, RS, bond, entropy, mechanism type, perception info).
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class OmniscientTherapyNet(nn.Module):
    """
    Feature extraction network for omniscient therapy observations.

    Processes extended Dict observation space with:
    - client_action: Discrete(8) -> embedded
    - session_number: Box([0,1], shape=(1,))
    - history: MultiDiscrete([9]*50) -> embedded and flattened
    - u_matrix: Box([-1,1], shape=(64,)) -> compressed via MLP
    - relationship_satisfaction: Box([-1,1], shape=(1,))
    - bond: Box([0,1], shape=(1,))
    - entropy: Box([-1,1], shape=(1,))
    - mechanism_type: Discrete(5) -> embedded
    - last_actual_action: Discrete(9) -> embedded
    - last_perceived_action: Discrete(9) -> embedded
    - misperception_rate: Box([0,1], shape=(1,))
    - perception_enabled: Discrete(2) -> embedded

    Parameters
    ----------
    observation_space : spaces.Dict
        Extended gymnasium Dict space from OmniscientObservationWrapper
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes after concatenating all features
    device : str, default="cpu"
        Device to place network on
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_sizes: Sequence[int] = (256, 256),
        device: Union[str, torch.device] = "cpu"
    ):
        super().__init__()
        self.device = device

        # === Existing embeddings from base TherapyNet ===
        self.client_action_embed = nn.Embedding(8, 16)
        self.history_embed = nn.Embedding(9, 8)

        # === NEW: Omniscient component embeddings ===
        self.mechanism_embed = nn.Embedding(5, 8)  # 5 mechanism types
        self.last_actual_action_embed = nn.Embedding(9, 4)  # 9 includes "none" (8)
        self.last_perceived_action_embed = nn.Embedding(9, 4)  # 9 includes "none" (8)
        self.perception_enabled_embed = nn.Embedding(2, 2)  # Binary: enabled or not

        # === NEW: U-matrix processor (compress 64 -> 32 dims) ===
        self.u_matrix_processor = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU()
        )

        # === Calculate total input dimension ===
        # From base TherapyNet:
        #   client_action_embed: 16
        #   session_number: 1
        #   history_embed: 50 * 8 = 400
        # NEW omniscient components:
        #   u_matrix (compressed): 32
        #   relationship_satisfaction: 1
        #   bond: 1
        #   entropy: 1
        #   mechanism_type_embed: 8
        #   last_actual_action_embed: 4
        #   last_perceived_action_embed: 4
        #   misperception_rate: 1
        #   perception_enabled_embed: 2
        # TOTAL: 16 + 1 + 400 + 32 + 1 + 1 + 1 + 8 + 4 + 4 + 1 + 2 = 471
        input_dim = 471

        # === Feature normalization for stability ===
        self.feature_norm = nn.LayerNorm(input_dim)

        # === MLP layers ===
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size

        self.mlp = nn.Sequential(*layers)
        self.output_dim: int = prev_size

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass through omniscient feature extraction network.

        Parameters
        ----------
        obs : np.ndarray or torch.Tensor
            Batch of Dict observations. For Tianshou, this comes as a Batch object.
        state : Any, optional
            RNN hidden state (unused for MLP)
        info : dict, optional
            Additional info (unused)

        Returns
        -------
        features : torch.Tensor
            Extracted features of shape (batch_size, output_dim)
        state : Any
            Updated state (None for MLP)
        """
        # Handle Tianshou Batch objects
        if hasattr(obs, 'client_action'):
            # Extract base TherapyNet components
            client_action = obs.client_action  # type: ignore[attr-defined]
            session_number = obs.session_number  # type: ignore[attr-defined]
            history = obs.history  # type: ignore[attr-defined]

            # Extract omniscient components
            u_matrix = obs.u_matrix  # type: ignore[attr-defined]
            rs = obs.relationship_satisfaction  # type: ignore[attr-defined]
            bond = obs.bond  # type: ignore[attr-defined]
            entropy = obs.entropy  # type: ignore[attr-defined]
            mechanism_type = obs.mechanism_type  # type: ignore[attr-defined]
            last_actual = obs.last_actual_action  # type: ignore[attr-defined]
            last_perceived = obs.last_perceived_action  # type: ignore[attr-defined]
            misperception_rate = obs.misperception_rate  # type: ignore[attr-defined]
            perception_enabled = obs.perception_enabled  # type: ignore[attr-defined]
        else:
            # Fallback for dict-like objects
            client_action = obs['client_action']  # type: ignore[index]
            session_number = obs['session_number']  # type: ignore[index]
            history = obs['history']  # type: ignore[index]
            u_matrix = obs['u_matrix']  # type: ignore[index]
            rs = obs['relationship_satisfaction']  # type: ignore[index]
            bond = obs['bond']  # type: ignore[index]
            entropy = obs['entropy']  # type: ignore[index]
            mechanism_type = obs['mechanism_type']  # type: ignore[index]
            last_actual = obs['last_actual_action']  # type: ignore[index]
            last_perceived = obs['last_perceived_action']  # type: ignore[index]
            misperception_rate = obs['misperception_rate']  # type: ignore[index]
            perception_enabled = obs['perception_enabled']  # type: ignore[index]

        # === Convert to tensors if needed ===
        # Base components
        if not isinstance(client_action, torch.Tensor):
            client_action = torch.as_tensor(client_action, dtype=torch.long, device=self.device)
        if not isinstance(session_number, torch.Tensor):
            session_number = torch.as_tensor(session_number, dtype=torch.float32, device=self.device)
        if not isinstance(history, torch.Tensor):
            history = torch.as_tensor(history, dtype=torch.long, device=self.device)
        elif history.dtype != torch.long:
            # Convert existing tensor to correct dtype (embeddings require integer indices)
            history = history.to(dtype=torch.long, device=self.device)
        else:
            # Already correct dtype, just ensure correct device
            history = history.to(self.device)

        # Omniscient components
        if not isinstance(u_matrix, torch.Tensor):
            u_matrix = torch.as_tensor(u_matrix, dtype=torch.float32, device=self.device)
        if not isinstance(rs, torch.Tensor):
            rs = torch.as_tensor(rs, dtype=torch.float32, device=self.device)
        if not isinstance(bond, torch.Tensor):
            bond = torch.as_tensor(bond, dtype=torch.float32, device=self.device)
        if not isinstance(entropy, torch.Tensor):
            entropy = torch.as_tensor(entropy, dtype=torch.float32, device=self.device)
        if not isinstance(mechanism_type, torch.Tensor):
            mechanism_type = torch.as_tensor(mechanism_type, dtype=torch.long, device=self.device)
        if not isinstance(last_actual, torch.Tensor):
            last_actual = torch.as_tensor(last_actual, dtype=torch.long, device=self.device)
        if not isinstance(last_perceived, torch.Tensor):
            last_perceived = torch.as_tensor(last_perceived, dtype=torch.long, device=self.device)
        if not isinstance(misperception_rate, torch.Tensor):
            misperception_rate = torch.as_tensor(misperception_rate, dtype=torch.float32, device=self.device)
        if not isinstance(perception_enabled, torch.Tensor):
            perception_enabled = torch.as_tensor(perception_enabled, dtype=torch.long, device=self.device)

        # === Ensure correct device ===
        client_action = client_action.to(self.device)
        session_number = session_number.to(self.device)
        # history device transfer handled in dtype conversion above
        u_matrix = u_matrix.to(self.device)
        rs = rs.to(self.device)
        bond = bond.to(self.device)
        entropy = entropy.to(self.device)
        mechanism_type = mechanism_type.to(self.device)
        last_actual = last_actual.to(self.device)
        last_perceived = last_perceived.to(self.device)
        misperception_rate = misperception_rate.to(self.device)
        perception_enabled = perception_enabled.to(self.device)

        # === Process base TherapyNet components ===
        # Embed client action: (batch_size,) -> (batch_size, 16)
        client_embed = self.client_action_embed(client_action)

        # Embed history: (batch_size, 50) -> (batch_size, 50, 8) -> (batch_size, 400)
        history_embed = self.history_embed(history)
        history_flat = history_embed.flatten(start_dim=1)

        # Session number: ensure shape (batch_size, 1)
        if session_number.dim() == 1:
            session_number = session_number.unsqueeze(-1)

        # === Process omniscient components ===
        # U-matrix: (batch_size, 64) -> (batch_size, 32)
        u_matrix_features = self.u_matrix_processor(u_matrix)

        # RS: ensure shape (batch_size, 1)
        if rs.dim() == 1:
            rs = rs.unsqueeze(-1)

        # Bond: ensure shape (batch_size, 1)
        if bond.dim() == 1:
            bond = bond.unsqueeze(-1)

        # Entropy: ensure shape (batch_size, 1)
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(-1)

        # Mechanism type: (batch_size,) -> (batch_size, 8)
        mechanism_embed = self.mechanism_embed(mechanism_type)

        # Last actual action: (batch_size,) -> (batch_size, 4)
        last_actual_embed = self.last_actual_action_embed(last_actual)

        # Last perceived action: (batch_size,) -> (batch_size, 4)
        last_perceived_embed = self.last_perceived_action_embed(last_perceived)

        # Misperception rate: ensure shape (batch_size, 1)
        if misperception_rate.dim() == 1:
            misperception_rate = misperception_rate.unsqueeze(-1)

        # Perception enabled: (batch_size,) -> (batch_size, 2)
        perception_enabled_embed = self.perception_enabled_embed(perception_enabled)

        # === Concatenate all features ===
        features = torch.cat([
            client_embed,  # 16
            session_number,  # 1
            history_flat,  # 400
            u_matrix_features,  # 32
            rs,  # 1
            bond,  # 1
            entropy,  # 1
            mechanism_embed,  # 8
            last_actual_embed,  # 4
            last_perceived_embed,  # 4
            misperception_rate,  # 1
            perception_enabled_embed,  # 2
        ], dim=-1)  # Total: 471

        # === Apply layer normalization ===
        features = self.feature_norm(features)

        # === Pass through MLP ===
        output = self.mlp(features)

        return output, state


class Actor(nn.Module):
    """Actor network: maps features to action probabilities."""

    def __init__(self, preprocess_net: nn.Module, action_dim: int, softmax_output: bool = True):
        super().__init__()
        self.preprocess = preprocess_net
        self.softmax_output = softmax_output
        output_dim = int(getattr(preprocess_net, "output_dim", 0))  # runtime-safe
        self.output_layer = nn.Linear(output_dim, action_dim)

    def forward(self, obs, state=None, info={}) -> Tuple[torch.Tensor, Any]:
        features, state = self.preprocess(obs, state)
        logits = self.output_layer(features)
        if self.softmax_output:
            return torch.softmax(logits, dim=-1), state
        return logits, state


class Critic(nn.Module):
    """Critic network: maps features to state value."""

    def __init__(self, preprocess_net: nn.Module):
        super().__init__()
        self.preprocess = preprocess_net
        output_dim = int(getattr(preprocess_net, "output_dim", 0))
        self.output_layer = nn.Linear(output_dim, 1)

    def forward(self, obs, state=None, info={}) -> torch.Tensor:
        features, state = self.preprocess(obs, state)
        value = self.output_layer(features)
        return value


def make_omniscient_networks(
    observation_space: spaces.Dict,
    action_space: spaces.Discrete,
    hidden_sizes: Sequence[int] = (256, 256),
    device: Union[str, torch.device] = "cpu"
) -> Tuple[nn.Module, nn.Module]:
    """
    Create actor and critic networks for omniscient therapy environment.

    Parameters
    ----------
    observation_space : spaces.Dict
        Extended observation space from OmniscientObservationWrapper
    action_space : spaces.Discrete
        TherapyEnv action space (should be Discrete(8))
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes for feature extraction
    device : str or torch.device, default="cpu"
        Device to place networks on

    Returns
    -------
    actor : nn.Module
        Actor network (policy)
    critic : nn.Module
        Critic network (value function)
    """
    # Create separate feature extraction networks for actor and critic
    actor_preprocess = OmniscientTherapyNet(
        observation_space=observation_space,
        hidden_sizes=hidden_sizes,
        device=device
    )

    critic_preprocess = OmniscientTherapyNet(
        observation_space=observation_space,
        hidden_sizes=hidden_sizes,
        device=device
    )

    # Create actor and critic
    actor = Actor(actor_preprocess, action_dim=int(action_space.n))  # type: ignore[attr-defined]
    critic = Critic(critic_preprocess)

    # Move to device
    actor = actor.to(device)
    critic = critic.to(device)

    return actor, critic
