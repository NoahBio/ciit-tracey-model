"""Neural network architectures for therapy RL agents."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


class TherapyNet(nn.Module):
    """
    Feature extraction network for therapy environment observations.

    Processes Dict observation space with:
    - client_action: Discrete(8) -> embedded
    - session_number: Box([0,1], shape=(1,))
    - history: MultiDiscrete([9]*50) -> embedded and flattened

    Parameters
    ----------
    observation_space : spaces.Dict
        Gymnasium Dict space from TherapyEnv
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

        # Embedding for client_action (8 octants)
        self.client_action_embed = nn.Embedding(8, 16)

        # Embedding for history (9 possible values: 0-7 octants + 8 for padding)
        self.history_embed = nn.Embedding(9, 8)

        # Calculate total input dimension
        # client_action_embed: 16
        # session_number: 1
        # history_embed: 50 * 8 = 400
        input_dim = 16 + 1 + 400

        # MLP layers
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
        Forward pass through feature extraction network.

        Parameters
        ----------
        obs : np.ndarray or torch.Tensor
            Batch of Dict observations. For Tianshou, this comes as a Batch object
            that can be indexed with keys.
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
            client_action = obs.client_action  # type: ignore[attr-defined]
            session_number = obs.session_number  # type: ignore[attr-defined]
            history = obs.history  # type: ignore[attr-defined]
        else:
            # Fallback for dict-like objects
            client_action = obs['client_action']  # type: ignore[index]
            session_number = obs['session_number']  # type: ignore[index]
            history = obs['history']  # type: ignore[index]

        # Convert to tensors if needed
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

        # Ensure correct device
        client_action = client_action.to(self.device)
        session_number = session_number.to(self.device)
        # history device transfer handled in dtype conversion above

        # Embed client action: (batch_size,) -> (batch_size, 16)
        client_embed = self.client_action_embed(client_action)

        # Embed history: (batch_size, 50) -> (batch_size, 50, 8) -> (batch_size, 400)
        history_embed = self.history_embed(history)
        history_flat = history_embed.flatten(start_dim=1)

        # Session number: ensure shape (batch_size, 1)
        if session_number.dim() == 1:
            session_number = session_number.unsqueeze(-1)

        # Concatenate all features
        features = torch.cat([client_embed, session_number, history_flat], dim=-1)

        # Pass through MLP
        output = self.mlp(features)

        return output, state


def make_therapy_networks(
    observation_space: spaces.Dict,
    action_space: spaces.Discrete,
    hidden_sizes: Sequence[int] = (256, 256),
    device: Union[str, torch.device] = "cpu"
) -> Tuple[nn.Module, nn.Module]:
    """
    Create actor and critic networks for therapy environment.

    Parameters
    ----------
    observation_space : spaces.Dict
        TherapyEnv observation space
    action_space : spaces.Discrete
        TherapyEnv action space (should be Discrete(8))
    hidden_sizes : Sequence[int], default=(256, 256)
        Hidden layer sizes for feature extraction
    device : str or torch.device, default="cpu"
        Device to place networks on

    Returns
    -------
    actor : nn.Module
        Actor network (policy) that outputs action probabilities
    critic : nn.Module
        Critic network (value function) that outputs state values

    Examples
    --------
    >>> from src.environment import TherapyEnv
    >>> env = TherapyEnv()
    >>> actor, critic = make_therapy_networks(
    ...     env.observation_space,
    ...     env.action_space,
    ...     hidden_sizes=(256, 256)
    ... )
    """
    # Create feature extraction network
    net = TherapyNet(
        observation_space=observation_space,
        hidden_sizes=hidden_sizes,
        device=device
    )

    # Actor: outputs logits for action distribution
    actor = Actor(
        preprocess_net=net,
        action_shape=int(action_space.n),
        device=device
    ).to(device)

    # Critic: outputs single value estimate
    critic = Critic(
        preprocess_net=TherapyNet(
            observation_space=observation_space,
            hidden_sizes=hidden_sizes,
            device=device
        ),
        device=device
    ).to(device)

    return actor, critic


class Actor(nn.Module):
    """
    Actor network for discrete action spaces.

    Takes features from preprocess network and outputs action logits.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: int,
        device: Union[str, torch.device] = "cpu",
        softmax_output: bool = True
    ):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.action_shape = action_shape
        self.softmax_output = softmax_output

        # Output layer
        output_dim = int(preprocess_net.output_dim) # type: ignore[attr-defined]
        self.output_layer = nn.Linear(output_dim, action_shape)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass to get action logits/probabilities.

        Returns
        -------
        logits : torch.Tensor
            Action logits or probabilities (if softmax_output=True)
        state : Any
            Updated hidden state
        """
        features, state = self.preprocess(obs, state)
        logits = self.output_layer(features)

        if self.softmax_output:
            return torch.softmax(logits, dim=-1), state
        return logits, state


class Critic(nn.Module):
    """
    Critic network for state value estimation.

    Takes features from preprocess network and outputs a single value.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        device: Union[str, torch.device] = "cpu"
    ):
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net

        # Output layer
        output_dim = int(preprocess_net.output_dim) # type: ignore[attr-defined]
        self.output_layer = nn.Linear(output_dim, 1)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {}
    ) -> torch.Tensor:
        """
        Forward pass to get state value estimate.

        Returns
        -------
        value : torch.Tensor
            State value estimate of shape (batch_size, 1)
        """
        features, state = self.preprocess(obs, state)
        value = self.output_layer(features)
        return value
