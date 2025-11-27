"""
Base Client Agent for CIIT-Tracey Computational Model.

The client agent is reactive, optimizing immediate interpersonal motives without
long-term planning. Behavior is driven by past interaction history (memory),
which shapes relationship satisfaction ('rs'), bond, and thereby expectations about therapist responses.

This base class contains all shared functionality across mechanism variants.
Subclasses implement different expectation formation mechanisms via _calculate_expected_payoffs().
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

from src.config import (
    U_MATRIX,
    sample_u_matrix,
    MEMORY_SIZE,
    get_memory_weights,
    rs_to_bond,
)
from src import config


class BaseClientAgent:
    """
    Base client agent with shared functionality across mechanism variants.

    The client maintains a memory of recent interactions, calculates relationship
    satisfaction and bond from this history, and selects actions to maximize
    immediate utility given expectations about therapist responses.

    Subclasses must implement _calculate_expected_payoffs() to define how
    the client forms expectations about therapist behavior.

    Parameters
    ----------
    u_matrix : NDArray[np.float64]
        Octant interaction utility matrix (8x8).
    entropy : float
        Temperature parameter for softmax action selection.
        Higher values = more exploration/randomness.
        Typically sampled from normal distribution across clients.
    initial_memory : List[Tuple[int, int]]
        Pre-filled memory representing interpersonal biography.
        List of octant pairs, mostly [client_octant, therapist_octant]
        Length must equal MEMORY_SIZE (50).
    random_state : int or np.random.RandomState, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        u_matrix: NDArray[np.float64],
        entropy: float,
        initial_memory: List[Tuple[int, int]],
        random_state: Optional[int] = None,
    ):
        # Validate inputs
        if len(initial_memory) != MEMORY_SIZE:
            raise ValueError(
                f"initial_memory must have length {MEMORY_SIZE}, got {len(initial_memory)}"
            )

        if entropy <= 0:
            raise ValueError(f"entropy must be positive, got {entropy}")

        # Initialize parameters
        self.entropy = entropy
        self.u_matrix = u_matrix

        # Set random state
        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

        # Initialize memory as deque for efficient append/pop
        self.memory = deque(initial_memory, maxlen=MEMORY_SIZE)

        # Store client-specific RS bounds for normalization
        self.rs_min = float(self.u_matrix.min())
        self.rs_max = float(self.u_matrix.max())

        # Calculate initial state
        self.relationship_satisfaction = self._calculate_relationship_satisfaction()
        self.bond = self._calculate_bond()

        # Store initial RS for dropout check
        self.initial_rs = self.relationship_satisfaction
        self.session_count = 0
        self.dropout_checked = False  # Track if we've already checked dropout

        # Calculate client-specific success threshold
        from src.config import calculate_success_threshold
        self.success_threshold = calculate_success_threshold(self.u_matrix)

    def _calculate_relationship_satisfaction(self) -> float:
        """
        Calculate relationship satisfaction as weighted average of interaction utilities.

        Returns
        -------
        float
            Relationship satisfaction value
        """
        # Get utilities from memory
        utilities = np.array([
            self.u_matrix[client_oct, therapist_oct]
            for client_oct, therapist_oct in self.memory
        ])

        # Get weights based on current scheme
        weights = get_memory_weights(len(utilities))

        # Weighted average
        rs = np.average(utilities, weights=weights)

        return float(rs)

    def _calculate_bond(self) -> float:
        """Calculate bond using client-specific normalization."""
        return rs_to_bond(
            rs=self.relationship_satisfaction,
            rs_min=self.rs_min,
            rs_max=self.rs_max,
            alpha=config.BOND_ALPHA,  # Read dynamically from config
            offset=config.BOND_OFFSET  # Read dynamically from config
        )

    def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
        """
        Calculate expected payoff for each possible client action.

        MUST be implemented by subclasses to define expectation mechanism.

        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        raise NotImplementedError("Subclass must implement _calculate_expected_payoffs()")

    def _softmax(self, payoffs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply softmax with temperature (entropy) to get action probabilities.

        Parameters
        ----------
        payoffs : NDArray[np.float64]
            Expected payoffs for each action

        Returns
        -------
        NDArray[np.float64]
            Probability distribution over actions
        """
        # Divide by temperature (entropy)
        scaled_payoffs = payoffs / self.entropy

        # Subtract max for numerical stability
        scaled_payoffs = scaled_payoffs - np.max(scaled_payoffs)

        # Compute softmax
        exp_payoffs = np.exp(scaled_payoffs)
        probabilities = exp_payoffs / np.sum(exp_payoffs)

        return probabilities

    def select_action(self) -> int:
        """
        Select client behavior (octant) based on expected payoffs.

        Returns
        -------
        int
            Selected octant (0-7)
        """
        payoffs = self._calculate_expected_payoffs()
        probabilities = self._softmax(payoffs)

        # Validate probabilities
        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError(
                f"Softmax probabilities do not sum to 1: {np.sum(probabilities)}"
            )
        if np.any(probabilities < 0):
            raise ValueError(f"Negative probabilities detected: {probabilities}")

        action = self.rng.choice(8, p=probabilities)
        return action

    def check_dropout(self) -> bool:
        """
        Determine if client drops out of therapy this session.

        Dropout is checked ONCE after 10 sessions. If RS has decreased
        compared to initial RS, client drops out. Otherwise, they continue
        and dropout is never checked again.

        Returns
        -------
        bool
            True if client drops out, False otherwise
        """
        # Only check dropout once, after 10 sessions
        if self.session_count != 10 or self.dropout_checked:
            return False

        # Mark that we've checked
        self.dropout_checked = True

        # Dropout if RS has decreased
        return self.relationship_satisfaction < self.initial_rs

    def update_memory(self, client_action: int, therapist_action: int) -> None:
        """
        Add new interaction to memory and update internal state.

        Parameters
        ----------
        client_action : int
            Client's octant in this interaction
        therapist_action : int
            Therapist's octant in this interaction
        """
        # Add to memory (automatically removes oldest if full)
        self.memory.append((client_action, therapist_action))

        # Increment session count
        self.session_count += 1

        # Recalculate relationship satisfaction and bond
        self.relationship_satisfaction = self._calculate_relationship_satisfaction()
        self.bond = self._calculate_bond()

    def get_state(self) -> Dict[str, Any]:
        """
        Get current client state for logging/analysis.

        Returns
        -------
        dict
            Dictionary containing current state variables
        """
        return {
            "relationship_satisfaction": self.relationship_satisfaction,
            "bond": self.bond,
            "entropy": self.entropy,
            "memory_length": len(self.memory),
            "initial_rs": self.initial_rs,
            "dropout_checked": self.dropout_checked,
        }

    def get_memory(self) -> List[Tuple[int, int]]:
        """
        Get current memory contents.

        Returns
        -------
        List[Tuple[int, int]]
            List of [client_octant, therapist_octant] pairs
        """
        return list(self.memory)

    @staticmethod
    def generate_problematic_memory(
        pattern_type: str = "cold_stuck",
        n_interactions: int = MEMORY_SIZE,
        random_state: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Generate initial memory representing problematic interpersonal patterns.

        Creates interaction history where:
        1. Problem client samples from problematic octants 80% of time, random 20%
        2. Other person responds anticomplementarily (maximum interpersonal tension)

        Parameters
        ----------
        pattern_type : str
            Type of problematic pattern. Options:
            - "cold_stuck": Client stuck in cold behaviors (CS, C, CD)
            - "dominant_stuck": Client stuck in dominant behaviors (D, WD, CD)
            - "submissive_stuck": Client stuck in submissive behaviors (S, WS, CS)
            - "cold_warm": Client always cold (C), therapist always warm (W)
            - "complementary_perfect": 70% warm, 20% D/S, 10% cold with complementary responses
            - "conflictual": D→D conflictual pattern (all identical)
            - "mixed_random": Fully random interactions
        n_interactions : int
            Number of interactions to generate
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        List[Tuple[int, int]]
            Memory of [client_octant, therapist_octant] pairs
        """
        rng = np.random.RandomState(random_state)

        # Define problematic octant pools and anticomplementary responses
        if pattern_type == "cold_stuck":
            problematic_octants = [5, 6, 7]  # CS, C, CD
            # Anticomplementary: Warm responses (maximum communion violation)
            anticomp_map = {
                5: [1, 2, 3],  # CS → WD, W, WS (warm + agency mismatch)
                6: [1, 2, 3],  # C → WD, W, WS (warm responses)
                7: [1, 2, 3]   # CD → WD, W, WS (warm + agency conflict)
            }

        elif pattern_type == "dominant_stuck":
            problematic_octants = [0, 1, 7]  # D, WD, CD
            # Anticomplementary: Dominant responses (agency anticomplementarity)
            anticomp_map = {
                0: [0, 1, 7],  # D → D, WD, CD (power struggle)
                1: [0, 1, 7],  # WD → D, WD, CD (power struggle)
                7: [0, 1, 7]   # CD → D, WD, CD (power struggle)
            }

        elif pattern_type == "submissive_stuck":
            problematic_octants = [3, 4, 5]  # WS, S, CS
            # Anticomplementary: Submissive responses (leadership vacuum)
            anticomp_map = {
                3: [3, 4, 5],  # WS → WS, S, CS (no leadership)
                4: [3, 4, 5],  # S → WS, S, CS (no leadership)
                5: [3, 4, 5]   # CS → WS, S, CS (no leadership)
            }

        elif pattern_type == "cold_warm":
            # Client always C (6), therapist always W (2)
            memory: List[tuple[int, int]] = [(6, 2)] * n_interactions
            return memory

        elif pattern_type == "complementary_perfect":
            # 70% warm side, 20% D/S, 10% cold side, all complementary
            # Complementary mapping
            complement_map = {
                0: 4,  # D → S
                1: 3,  # WD → WS
                2: 2,  # W → W
                3: 1,  # WS → WD
                4: 0,  # S → D
                5: 7,  # CS → CD
                6: 6,  # C → C
                7: 5,  # CD → CS
            }

            memory: List[Tuple[int, int]] = []
            for _ in range(n_interactions):
                rand_val = rng.random()
                if rand_val < 0.70:
                    # 70% warm side: WD, W, WS (octants 1, 2, 3)
                    client_action = rng.choice([1, 2, 3])
                elif rand_val < 0.90:
                    # 20% D or S (octants 0, 4)
                    client_action = rng.choice([0, 4])
                else:
                    # 10% cold side: CD, C, CS (octants 5, 6, 7)
                    client_action = rng.choice([5, 6, 7])

                therapist_action = complement_map[client_action]
                memory.append((client_action, therapist_action))

            return memory

        elif pattern_type == "conflictual":
            # Conflictual: D→D
            memory: List[Tuple[int, int]] = [(0, 0)] * n_interactions
            return memory

        elif pattern_type == "mixed_random":
            # Fully random interactions
            memory: List[Tuple[int, int]] = [
                (rng.randint(0, 8), rng.randint(0, 8))
                for _ in range(n_interactions)
            ]
            return memory

        else:
            raise ValueError(
                f"Unknown pattern_type: {pattern_type}. "
                f"Must be one of: cold_stuck, dominant_stuck, submissive_stuck, "
                f"cold_warm, complementary_perfect, conflictual, mixed_random"
            )

        # Generate interactions
        memory: List[Tuple[int, int]] = []

        for _ in range(n_interactions):
            # Client action: mostly from problematic octants (80%), some random (20%)
            if rng.random() < 0.8:
                client_action = rng.choice(problematic_octants)
            else:
                client_action = rng.choice(8)

            # Other's response: ANTICOMPLEMENTARY if client action is problematic
            if client_action in problematic_octants:
                # Strongly anticomplementary response
                other_action = rng.choice(anticomp_map[client_action])
            else:
                # Random response for non-problematic actions
                other_action = rng.choice(8)

            # Store interaction
            memory.append((client_action, other_action))

        return memory
