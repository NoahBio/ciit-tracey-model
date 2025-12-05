"""Gymnasium wrapper to expose client internal state for omniscient RL agent.

This wrapper extends the observation space to include all client internal
dynamics, enabling "perfect information" RL training.

Example usage:
    from src.environment import TherapyEnv
    from src.environment.omniscient_wrapper import OmniscientObservationWrapper

    env = TherapyEnv(patterns=["cold_stuck"], mechanism="frequency_amplifier")
    env = OmniscientObservationWrapper(env)

    obs, info = env.reset()
    # obs now contains u_matrix, RS, bond, entropy, mechanism_type, etc.
"""

from typing import Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray
import gymnasium
from gymnasium import spaces

from src.environment.therapy_env import TherapyEnv


class OmniscientObservationWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds client internal state to observations.

    This wrapper exposes all client internal dynamics to the RL agent:
    - Client's utility matrix (u_matrix)
    - Relationship satisfaction (RS)
    - Bond level
    - Entropy (temperature parameter)
    - Mechanism type (which expectation mechanism the client uses)
    - Perception information (if enabled): actual vs perceived actions

    The agent has "perfect information" about client psychology.
    """

    def __init__(self, env: TherapyEnv):
        """
        Initialize the omniscient wrapper.

        Parameters
        ----------
        env : TherapyEnv
            Base therapy environment to wrap
        """
        super().__init__(env)

        # Mechanism type mapping to integers
        self.mechanism_map = {
            'bond_only': 0,
            'frequency_amplifier': 1,
            'conditional_amplifier': 2,
            'bond_weighted_frequency_amplifier': 3,
            'bond_weighted_conditional_amplifier': 4,
        }

        # Extend observation space to include omniscient components
        self.observation_space = spaces.Dict({
            # === Existing base environment observations ===
            "client_action": env.observation_space['client_action'],  # type: ignore[index]
            "session_number": env.observation_space['session_number'],  # type: ignore[index]
            "history": env.observation_space['history'],  # type: ignore[index]

            # === NEW: Client Internal State ===
            "u_matrix": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(64,),
                dtype=np.float32
            ),  # Flattened 8x8 matrix, normalized to [-1, 1]

            "relationship_satisfaction": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),  # Normalized RS

            "bond": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),  # Already [0, 1] from sigmoid

            "entropy": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),  # Normalized temperature parameter

            "mechanism_type": spaces.Discrete(5),  # 0-4 for 5 mechanism types

            # === NEW: Perception Info ===
            "last_actual_action": spaces.Discrete(9),  # 0-7 for octants, 8 for "none"

            "last_perceived_action": spaces.Discrete(9),  # 0-7 for octants, 8 for "none"

            "misperception_rate": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),  # Overall misperception rate

            "perception_enabled": spaces.Discrete(2),  # 0=disabled, 1=enabled
        })

    def reset(
        self,
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment and add omniscient observations.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments passed to base environment reset()

        Returns
        -------
        obs : dict
            Extended observation with omniscient components
        info : dict
            Environment info dict
        """
        obs, info = self.env.reset(**kwargs)
        omniscient_obs = self._add_omniscient_obs(obs)
        return omniscient_obs, info

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take environment step and add omniscient observations.

        Parameters
        ----------
        action : int
            Therapist action (0-7 octant)

        Returns
        -------
        obs : dict
            Extended observation with omniscient components
        reward : float
            Step reward
        terminated : bool
            Whether episode terminated (success or dropout)
        truncated : bool
            Whether episode truncated (max sessions)
        info : dict
            Environment info dict
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        omniscient_obs = self._add_omniscient_obs(obs)
        return omniscient_obs, reward, terminated, truncated, info

    def _add_omniscient_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract client internal state and add to observation.

        This method normalizes all continuous values to appropriate ranges
        for neural network input stability.

        Parameters
        ----------
        obs : dict
            Base observation from environment

        Returns
        -------
        dict
            Extended observation with omniscient components
        """
        client = self.env._client  # type: ignore[attr-defined]

        # === 1. U-Matrix (Utility Matrix) ===
        # Flatten 8x8 matrix and normalize to [-1, 1]
        u_matrix_flat = client.u_matrix.flatten()
        u_min, u_max = client.u_matrix.min(), client.u_matrix.max()

        # Avoid division by zero if all utilities are identical
        if u_max - u_min < 1e-8:
            u_matrix_norm = np.zeros_like(u_matrix_flat)
        else:
            # Normalize to [-1, 1]: 2 * ((x - min) / (max - min)) - 1
            u_matrix_norm = 2 * (u_matrix_flat - u_min) / (u_max - u_min) - 1

        # === 2. Relationship Satisfaction (RS) ===
        # Normalize to [-1, 1] using client-specific bounds
        rs_min = client.rs_min
        rs_max = client.rs_max
        rs_value = client.relationship_satisfaction

        if rs_max - rs_min < 1e-8:
            rs_norm = 0.0
        else:
            rs_norm = 2 * (rs_value - rs_min) / (rs_max - rs_min) - 1

        # === 3. Bond ===
        # Already in [0, 1] from sigmoid, no normalization needed
        bond_value = client.bond

        # === 4. Entropy (Temperature Parameter) ===
        # Normalize to [-1, 1] (range: 1.5 to 5.0 from config)
        # Clamp to handle values outside expected range
        entropy_value = np.clip(client.entropy, 1.5, 5.0)
        entropy_norm = 2 * (entropy_value - 1.5) / 3.5 - 1

        # === 5. Mechanism Type ===
        # Map mechanism string to integer
        mechanism_str = self.env._mechanism  # type: ignore[attr-defined]
        mechanism_idx = self.mechanism_map.get(mechanism_str, 0)

        # === 6. Perception Information ===
        # Get last actual and perceived actions (use sentinel 8 for "none")
        last_actual = getattr(self.env, '_last_actual_action', 8)
        last_perceived = getattr(self.env, '_last_perceived_action', 8)

        # Get perception statistics if available
        if hasattr(client, 'get_perception_stats'):
            perception_stats = client.get_perception_stats()
            misperception_rate = perception_stats.get('overall_misperception_rate', 0.0)
        else:
            misperception_rate = 0.0

        # Check if perception is enabled
        perception_enabled = 1 if self.env._enable_perception else 0  # type: ignore[attr-defined]

        # === Add omniscient components to observation ===
        obs['u_matrix'] = u_matrix_norm.astype(np.float32)
        obs['relationship_satisfaction'] = np.array([rs_norm], dtype=np.float32)
        obs['bond'] = np.array([bond_value], dtype=np.float32)
        obs['entropy'] = np.array([entropy_norm], dtype=np.float32)
        obs['mechanism_type'] = mechanism_idx
        obs['last_actual_action'] = last_actual
        obs['last_perceived_action'] = last_perceived
        obs['misperception_rate'] = np.array([misperception_rate], dtype=np.float32)
        obs['perception_enabled'] = perception_enabled

        return obs


def test_wrapper():
    """Quick test to verify wrapper functionality."""
    from src.environment import TherapyEnv

    print("Testing OmniscientObservationWrapper...")

    # Create base environment
    env = TherapyEnv(
        pattern=["cold_stuck"],
        mechanism="frequency_amplifier",
        enable_perception=True
    )

    # Wrap with omniscient wrapper
    env = OmniscientObservationWrapper(env)

    # Reset and check observation space
    obs, info = env.reset(seed=42)

    print("\nObservation keys:", list(obs.keys()))
    print("\nObservation space:")
    for key, space in env.observation_space.spaces.items():  # type: ignore[attr-defined]
        print(f"  {key}: {space}")

    print("\nSample observation shapes:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: value={value}, type={type(value).__name__}")

    # Take a few steps
    print("\nTaking 3 steps...")
    for i in range(3):
        action = env.action_space.sample()  # type: ignore[attr-defined]
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            break

    print("\nWrapper test completed successfully!")


if __name__ == "__main__":
    test_wrapper()
