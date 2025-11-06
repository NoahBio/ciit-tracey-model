"""
Client Agent for CIIT-Tracey Computational Model.

The client agent is reactive, optimizing immediate interpersonal motives without
long-term planning. Behavior is driven by past interaction history (memory),
which shapes relationship satisfaction ('rs'), bond, and thereby expectations about therapist responses.

TODO:
  - create function to get client-specific u_matrix sampled from U_MIN and U_MAX
  - update OCTANT selection in generate_problematic_memory () -> random octants for therapist
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
    BOND_ALPHA,
    #OCTANTS,
)


class ClientAgent:
    """
    Client agent that selects interpersonal behaviors based on past interactions.
    
    The client maintains a memory of recent interactions, calculates relationship
    satisfaction and bond from this history, and selects actions to maximize
    immediate utility given expectations about therapist responses.
    
    Parameters
    ----------
    initial_memory : List[Tuple[int, int]]
        Pre-filled memory representing interpersonal biography.
        List of octant pairs, mostly [client_octant, therapist_octant]
        Length must equal MEMORY_SIZE (50).
    entropy : float
        Temperature parameter for softmax action selection.
        Higher values = more exploration/randomness.
        Typically sampled from normal distribution across clients.
    u_matrix : NDArray[np.float64], optional
        Octant interaction utility matrix (8x8).
        Defaults to global U_MATRIX from config.
    random_state : int or np.random.RandomState, optional
        Random state for reproducibility.
    """
    
    def __init__(
        self,
        initial_memory: List[Tuple[int, int]],
        entropy: float,
        u_matrix: Optional[NDArray[np.float64]] = None,
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
        # Sample client-specific U_MATRIX if not provided
        if u_matrix is None:
            self.u_matrix = sample_u_matrix(random_state=random_state)
        else:
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
            alpha=BOND_ALPHA  # Can still control steepness from config
        )
    
    def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
        """
        Calculate expected payoff for each possible client action.
        
        Uses trust-based percentile interpolation: clients with low bond expect
        poor therapist responses (low percentile), while high bond clients expect
        good responses (high percentile).
        
        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        expected_payoffs = np.zeros(8)
        
        for client_action in range(8):
            # Get all possible utilities for this client action
            utilities_row = self.u_matrix[client_action, :]
            
            # Sort from worst to best outcome
            sorted_utilities = np.sort(utilities_row)
            
            # Map bond (trust) to position in sorted list [0, 7]
            position = self.bond * 7
            
            # Interpolate between adjacent percentiles for smooth transitions
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, 7)
            interpolation_weight = position - lower_idx
            
            # Calculate expected payoff
            expected_payoff = (
                (1 - interpolation_weight) * sorted_utilities[lower_idx] +
                interpolation_weight * sorted_utilities[upper_idx]
            )
            
            expected_payoffs[client_action] = expected_payoff
        
        return expected_payoffs
    
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
        """Select client behavior (octant) based on expected payoffs..."""
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
        
        Dropout is checked ONCE after 5 sessions. If RS has decreased 
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
        2. Other person responds randomly (uniform across all octants)
        
        Parameters
        ----------
        pattern_type : str
            Type of problematic pattern. Options:
            - "cold_stuck": Client stuck in cold behaviors (CS, C, CD)
            - "dominant_stuck": Client stuck in dominant behaviors (D, WD, CD)
            - "submissive_stuck": Client stuck in submissive behaviors (S, WS, CS)
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
        
        # Define problematic octant pools
        if pattern_type == "cold_stuck":
            problematic_octants = [5, 6]  # CS, C, CD
            
        elif pattern_type == "dominant_stuck":
            problematic_octants = [0, 1, 7]  # D, WD, CD
            
        elif pattern_type == "submissive_stuck":
            problematic_octants = [3, 4, 5]  # WS, S, CS
            
        else:
            raise ValueError(
                f"Unknown pattern_type: {pattern_type}. "
                f"Must be one of: cold_stuck, dominant_stuck, submissive_stuck"
            )
        
        # Generate interactions
        memory: List[Tuple[int, int]] = []
        
        for _ in range(n_interactions):
            # Problem client selects action (80% problematic, 20% random)
            if rng.random() < 0.8:
                # Sample from problematic octants
                client_action = rng.choice(problematic_octants)
            else:
                # Random octant
                client_action = rng.choice(8)
            
            other_pool = [3, 4]  # Also cold → creates negative utilities
            # Other person responds randomly
            other_action = rng.choice(other_pool)  # TEST: Therapist uses cold octants
            
            # Store interaction
            memory.append((client_action, other_action))
        
        return memory


# =============================================================================
# HELPER FUNCTIONS FOR CLIENT INITIALIZATION
# =============================================================================

def create_client(
    pattern_type: str = "cold_stuck",
    entropy: Optional[float] = None,
    random_state: Optional[int] = None,
) -> ClientAgent:
    """
    Create a client with problematic interpersonal pattern.
    
    Each client gets:
    - Client-specific u_matrix sampled from U_MIN/U_MAX ranges
    - Entropy sampled from distribution (or custom value)
    - Problematic memory history generated realistically
    
    Parameters
    ----------
    pattern_type : str
        Type of problematic interpersonal pattern:
        - "cold_stuck": Stuck in cold behaviors
        - "dominant_stuck": Stuck in dominant behaviors  
        - "submissive_stuck": Stuck in submissive behaviors
    entropy : float, optional
        Temperature parameter. If None, samples from distribution.
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    ClientAgent
        Initialized client agent with client-specific U_MATRIX
    """
    from src.config import CLIENT_ENTROPY_MEAN, CLIENT_ENTROPY_STD
    from src.config import CLIENT_ENTROPY_MIN, CLIENT_ENTROPY_MAX
    
    rng = np.random.RandomState(random_state)
    
    # Sample entropy if not provided
    if entropy is None:
        entropy = float(rng.normal(CLIENT_ENTROPY_MEAN, CLIENT_ENTROPY_STD))
        entropy = float(np.clip(entropy, CLIENT_ENTROPY_MIN, CLIENT_ENTROPY_MAX))
    else:
        entropy = float(entropy)
    
    # Generate problematic memory using generate_problematic_memory()
    initial_memory = ClientAgent.generate_problematic_memory(
        pattern_type=pattern_type,
        random_state=random_state,
    )
    
    # Create client with client-specific u_matrix
    # (u_matrix=None triggers sampling from U_MIN/U_MAX in __init__)
    client = ClientAgent(
        initial_memory=initial_memory,
        entropy=entropy,
        u_matrix=None,  # Will sample client-specific matrix
        random_state=random_state,
    )
    
    return client