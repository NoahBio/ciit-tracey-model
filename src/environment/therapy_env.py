"""
Gymnasium-compatible therapy simulation environment.

This environment wraps the CIIT-client-agent to enable reinforcement
learning training of therapist agents.

Key Design: The client acts first, selecting their behavior based on their
current internal state. The therapist observes this action and responds.
This reflects the realistic therapy dynamic where therapists respond to
client presentations.
"""

import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from src.agents.client_agents import (
    with_perception,
    BaseClientAgent,
    BondOnlyClient,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
    BondWeightedConditionalAmplifier,
    BondWeightedFrequencyAmplifier,
)
from src import config
from src.config import (
    sample_u_matrix,
    calculate_success_threshold,
    MEMORY_SIZE,
)


class TherapyEnv(gymnasium.Env):
    """
    Gymnasium environment for training therapist agents.

    The environment simulates a therapy session where:
    1. Client selects behavior based on relationship satisfaction and bond
    2. Therapist observes client behavior and responds
    3. Client updates internal state based on interaction
    4. Episode continues until success, dropout, or max sessions

    Observation Space
    -----------------
    Dict with:
        - client_action: Discrete(8) - Client's current octant behavior
        - session_number: Box([0,1]) - Normalized session count
        - history: MultiDiscrete([9]*50) - Last 25 client-therapist interactions
          (padded with 8 for early sessions; 0-7 are octant indices)

    Action Space
    ------------
    Discrete(8) - Therapist selects one of 8 interpersonal octants:
        0: D (Dominant), 1: WD (Warm-Dominant), 2: W (Warm),
        3: WS (Warm-Submissive), 4: S (Submissive), 5: CS (Cold-Submissive),
        6: C (Cold), 7: CD (Cold-Dominant)

    Rewards
    -------
    Terminal-only reward structure:
        - Success: +100 + (max_sessions - sessions_at_success) * 2
        - Dropout (session 10): -150
        - Max sessions reached: 0
        - Non-terminal steps: 0

    Termination
    -----------
    Episode terminates when:
        - Success: Client's RS reaches threshold (terminated=True)
        - Dropout: At session 10, RS decreased from initial (terminated=True)
        - Max sessions: Reached max_sessions without success (truncated=True)

    Parameters
    ----------
    mechanism : str, default="frequency_amplifier"
        Client expectation mechanism. Options: 'bond_only',
        'frequency_amplifier', 'conditional_amplifier',
        'bond_weighted_frequency_amplifier', 'bond_weighted_conditional_amplifier'
    pattern : str or List[str], default=["cold_stuck", "dominant_stuck", ...]
        Initial memory pattern(s). If list, randomly samples one per episode.
        Options: 'cold_stuck', 'dominant_stuck', 'submissive_stuck',
        'cold_warm', 'complementary_perfect', 'conflictual', 'mixed_random'
    threshold : float, default=0.9
        Success threshold as percentile of client's RS range (0-1)
    max_sessions : int, default=100
        Maximum therapy sessions per episode
    entropy : float, default=0.5
        Client's temperature parameter for action selection (higher = more random)
    history_weight : float, default=1.0
        Weight for history influence in amplifier mechanisms
    bond_alpha : float or None, default=None
        Sigmoid steepness for bond calculation. If None, uses config.BOND_ALPHA
    bond_offset : float, default=0.7
        Sigmoid inflection point for bond calculation (0.7 = 70th percentile)
    enable_perception : bool, default=True
        Whether to enable perceptual distortion in client
    baseline_accuracy : float, default=0.5
        Baseline perception accuracy if perception enabled
    random_state : int or None, default=None
        Random seed for reproducibility

    Examples
    --------
    >>> env = TherapyEnv(mechanism="frequency_amplifier", pattern="cold_stuck")
    >>> obs, info = env.reset(seed=42)
    >>> obs['client_action']  # Client's current behavior
    6  # Cold octant
    >>> action = 2  # Therapist responds with Warm
    >>> obs, reward, terminated, truncated, info = env.step(action)

    References
    ----------
    Based on Tracey's (1993) Three Step Model of therapeutic change and
    interpersonal complementarity theory.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mechanism: str = "frequency_amplifier",
        pattern: Union[str, List[str]] = ["cold_stuck", "dominant_stuck", "submissive_stuck", "mixed_random", "cold_warm"],
        threshold: float = 0.9,
        max_sessions: int = 100,
        entropy: float = 0.5,
        history_weight: float = 1.0,
        bond_alpha: Optional[float] = None,
        bond_offset: float = 0.7,
        enable_perception: bool = True,
        baseline_accuracy: float = 0.5,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the therapy environment."""
        super().__init__()

        # Validate inputs
        if isinstance(pattern, list) and len(pattern) == 0:
            raise ValueError("Pattern list cannot be empty")
        if not (0 < threshold <= 1):
            raise ValueError(f"Threshold must be in (0, 1], got {threshold}")
        if max_sessions <= 0:
            raise ValueError(f"max_sessions must be positive, got {max_sessions}")
        if entropy <= 0:
            raise ValueError(f"entropy must be positive, got {entropy}")

        # Store configuration
        self._mechanism = mechanism
        self._pattern = pattern
        self._threshold_percentile = threshold
        self._max_sessions = max_sessions
        self._entropy = entropy
        self._history_weight = history_weight
        self._bond_alpha = bond_alpha
        self._bond_offset = bond_offset
        self._enable_perception = enable_perception
        self._baseline_accuracy = baseline_accuracy

        # Initialize random state
        self._seed_value = random_state
        self._np_random = np.random.default_rng(random_state)

        # Episode state (initialized in reset())
        self._client: Optional[BaseClientAgent] = None
        self._session_count: int = 0
        self._pending_client_action: int = 0
        self._rs_threshold: float = 0.0
        self._interaction_history: deque = deque(maxlen=25)

        # Define observation space
        self.observation_space = spaces.Dict({
            "client_action": spaces.Discrete(8),
            "session_number": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "history": spaces.MultiDiscrete([9] * 50),  # 25 interactions * 2, padded with 8
        })

        # Define action space
        self.action_space = spaces.Discrete(8)

    def _get_client_class(self, mechanism: str) -> type:
        """
        Map mechanism string to client class.

        Parameters
        ----------
        mechanism : str
            Client expectation mechanism name

        Returns
        -------
        type
            Client class for the specified mechanism

        Raises
        ------
        ValueError
            If mechanism is not recognized
        """
        mechanisms = {
            'bond_only': BondOnlyClient,
            'frequency_amplifier': FrequencyAmplifierClient,
            'conditional_amplifier': ConditionalAmplifierClient,
            'bond_weighted_conditional_amplifier': BondWeightedConditionalAmplifier,
            'bond_weighted_frequency_amplifier': BondWeightedFrequencyAmplifier,
        }

        if mechanism not in mechanisms:
            raise ValueError(
                f"Unknown mechanism: {mechanism}. "
                f"Must be one of: {list(mechanisms.keys())}"
            )

        return mechanisms[mechanism]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Union[int, np.ndarray]], Dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Parameters
        ----------
        seed : int or None
            Random seed for this episode
        options : dict or None
            Additional options (unused)

        Returns
        -------
        observation : dict
            Initial observation with client's first action
        info : dict
            Episode information (pattern, entropy, thresholds, etc.)
        """
        # Add assertions for type checker
        assert self._np_random is not None

        # Call parent reset for proper gymnasium seeding
        super().reset(seed=seed)

        # Update random state if seed provided
        if seed is not None:
            self._seed_value = seed
            self._np_random = np.random.default_rng(seed)

        # Sample pattern (if list provided)
        if isinstance(self._pattern, list):
            selected_pattern = self._np_random.choice(self._pattern)
        else:
            selected_pattern = self._pattern

        # Generate episode-specific seed for client creation
        episode_seed = int(self._np_random.integers(0, 2**31))

        # Sample client-specific u_matrix
        u_matrix = sample_u_matrix(random_state=episode_seed)

        # Generate initial memory
        initial_memory = BaseClientAgent.generate_problematic_memory(
            pattern_type=selected_pattern,
            n_interactions=MEMORY_SIZE,
            random_state=episode_seed,
        )

        # Set global bond parameters
        if self._bond_alpha is not None:
            config.BOND_ALPHA = self._bond_alpha
        config.BOND_OFFSET = self._bond_offset

        # Build client kwargs
        client_kwargs = {
            'u_matrix': u_matrix,
            'entropy': self._entropy,
            'initial_memory': initial_memory,
            'random_state': episode_seed,
        }

        # Add mechanism-specific parameters
        if 'amplifier' in self._mechanism:
            client_kwargs['history_weight'] = self._history_weight

        if self._enable_perception:
            client_kwargs['baseline_accuracy'] = self._baseline_accuracy

        # Create client
        # Add assertions for type checker
        assert self._np_random is not None

        ClientClass = self._get_client_class(self._mechanism)
        if self._enable_perception:
            ClientClass = with_perception(ClientClass)

        self._client = ClientClass(**client_kwargs)

        # Add assertions for type checker
        assert self._client is not None

        # Calculate success threshold
        self._rs_threshold = calculate_success_threshold(
            u_matrix,
            percentile=self._threshold_percentile
        )

        # Initialize episode state
        self._session_count = 0
        self._interaction_history = deque(maxlen=25)

        # Client acts first - select initial action
        self._pending_client_action = self._client.select_action()

        # Construct observation
        obs = self._construct_observation()

        # Build info dict
        info = {
            'pattern': selected_pattern,
            'entropy': float(self._entropy),
            'rs_threshold': float(self._rs_threshold),
            'initial_rs': float(self._client.relationship_satisfaction),
            'initial_bond': float(self._client.bond),
            'episode_seed': int(episode_seed),
        }

        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, Union[int, np.ndarray]], float, bool, bool, Dict[str, Any]]:
        """
        Execute one therapy session step.

        Parameters
        ----------
        action : int
            Therapist's octant action (0-7)

        Returns
        -------
        observation : dict
            Next observation with client's next action
        reward : float
            Reward for this step (terminal-only)
        terminated : bool
            Whether episode terminated (success or dropout)
        truncated : bool
            Whether episode truncated (max sessions)
        info : dict
            Step information (session, RS, bond, actions, etc.)
        """
        # Assertions for type checker
        assert self._client is not None
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}. Must be in [0, 7]."
            )

        # Use stored client action (client acted BEFORE therapist observed)
        client_action = self._pending_client_action
        therapist_action = action

        # Update client memory (triggers RS and bond recalculation)
        self._client.update_memory(client_action, therapist_action)

        # Add interaction to history
        self._interaction_history.append((client_action, therapist_action))

        # Increment session counter
        self._session_count += 1

        # Get updated state
        current_rs = self._client.relationship_satisfaction
        current_bond = self._client.bond

        # Check termination conditions
        success = current_rs >= self._rs_threshold
        dropped_out = self._client.check_dropout()  # Only True at session 10 if RS decreased
        max_reached = self._session_count >= self._max_sessions

        terminated = success or dropped_out
        truncated = max_reached and not terminated

        # Calculate reward (terminal-only)
        reward = self._calculate_reward(success, dropped_out, max_reached)

        # Select next client action (if episode continues)
        if not terminated and not truncated:
            self._pending_client_action = self._client.select_action()

        # Construct observation
        obs = self._construct_observation()

        # Build info dict
        info = {
            'session': self._session_count,
            'rs': float(current_rs),
            'bond': float(current_bond),
            'client_action': int(client_action),
            'therapist_action': int(therapist_action),
            'success': bool(success),
            'dropped_out': bool(dropped_out),
            'max_reached': bool(max_reached),
        }

        # Add perception stats if available
        if self._enable_perception and hasattr(self._client, 'get_perception_stats'):
            info['perception_stats'] = self._client.get_perception_stats() # type: ignore[attr-defined]

        return obs, reward, terminated, truncated, info

    def _construct_observation(self) -> Dict[str, Union[int, np.ndarray]]:
        """
        Build observation dict from current state.

        Returns
        -------
        dict
            Observation with client_action, session_number, and history
        """
        # Session number (normalized)
        session_norm = np.array(
            [self._session_count / self._max_sessions],
            dtype=np.float32
        )

        # History: flat array [c0, t0, c1, t1, ...], padded with 8s for early sessions
        # Contains only shared therapy interactions (excludes initial memory)
        # Use 8 as padding sentinel to avoid confusion with Octant 0 (Dominant)
        history_array = np.full(50, 8, dtype=np.int64)  # 25 interactions × 2 = 50 elements
        for i, (c_act, t_act) in enumerate(self._interaction_history):
            history_array[i * 2] = c_act
            history_array[i * 2 + 1] = t_act

        return {
            "client_action": self._pending_client_action,
            "session_number": session_norm,
            "history": history_array,
        }

    def _calculate_reward(
        self,
        success: bool,
        dropped_out: bool,
        max_reached: bool
    ) -> float:
        """
        Calculate reward for current step.

        Terminal-only reward structure:
        - Success: +100 + efficiency bonus
        - Dropout: -150
        - Max sessions: 0
        - Non-terminal: 0

        Parameters
        ----------
        success : bool
            Whether client reached success threshold
        dropped_out : bool
            Whether client dropped out
        max_reached : bool
            Whether max sessions reached

        Returns
        -------
        float
            Reward value
        """
        # Non-terminal: no reward
        if not success and not dropped_out and not max_reached:
            return 0.0

        # Dropout: worst outcome
        if dropped_out:
            return -150.0

        # Success: base reward + efficiency bonus
        if success:
            sessions_saved = self._max_sessions - self._session_count
            efficiency_bonus = sessions_saved * 2.0  # 2 points per session saved
            return 100.0 + efficiency_bonus

        # Max sessions without success: no reward
        if max_reached:
            return 0.0

        return 0.0
