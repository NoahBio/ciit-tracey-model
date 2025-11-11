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
    HISTORY_WEIGHT,
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
            alpha=BOND_ALPHA  # Can still control steepness from config
        )
    
    def _calculate_marginal_frequencies(self) -> NDArray[np.float64]:
        """
        Calculate P(therapist_j) from memory with recency weighting.
        
        Computes simple frequency distribution of therapist behaviors,
        ignoring what client action preceded them (marginal probability).
        No Bayesian smoothing - raw empirical frequencies only.
        
        Returns
        -------
        NDArray[np.float64]
            Probability distribution P(therapist_j) for j=0..7
        """
        # Get memory weights (same recency scheme as RS calculation)
        memory_weights = get_memory_weights(len(self.memory))
        
        # Count therapist actions with recency weights
        weighted_counts = np.zeros(8)
        
        for idx, (client_oct, therapist_oct) in enumerate(self.memory):
            weighted_counts[therapist_oct] += memory_weights[idx]
        
        # Calculate frequencies (no smoothing)
        total_weight = sum(memory_weights)
        if total_weight == 0:
            return np.ones(8) / 8  # No memory = uniform
        
        frequencies = weighted_counts / total_weight
        
        return frequencies
    
    def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
        """
            Calculate expected payoff for each possible client action.
        
        Mechanism:
        1. History creates probability distribution over therapist actions: P(j)
        - Raw empirical frequencies (no Bayesian smoothing)
        2. Amplify utilities based on frequency: 
        adjusted[i,j] = U[i,j] + (U[i,j] * P(j) * k)
        - Observed behaviors (P(j) > 0): utilities amplified proportionally
        - Unobserved behaviors (P(j) = 0): utilities unchanged
        3. Bond selects within amplified utilities via percentile interpolation
        - High bond: expect good outcomes among likely responses (high percentiles)
        - Low bond: expect poor outcomes among likely responses (low percentiles)
        
        Softmax exploration handles uncertainty about unseen actions.
        
        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        # Calculate marginal therapist behavior frequencies
        # No smoothing - raw empirical frequencies
        therapist_frequencies = self._calculate_marginal_frequencies()
        
        expected_payoffs = np.zeros(8)
        
        for client_action in range(8):
            # Get raw utilities for this client action
            raw_utilities = self.u_matrix[client_action, :]
            
            # Weight utilities by probability of each therapist response
            # "Accentuate" likely columns, diminish unlikely ones
            adjusted_utilities = raw_utilities + (raw_utilities * therapist_frequencies * HISTORY_WEIGHT)
            
            # Sort probability-weighted utilities
            sorted_adjusted = np.sort(adjusted_utilities)
            
            # Apply bond-based percentile interpolation
            # High bond → expect good outcomes among likely responses
            # Low bond → expect poor outcomes among likely responses
            position = self.bond * 7
            
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, 7)
            interpolation_weight = position - lower_idx
            
            expected_payoff = (
                (1 - interpolation_weight) * sorted_adjusted[lower_idx] +
                interpolation_weight * sorted_adjusted[upper_idx]
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
            
        else:
            raise ValueError(
                f"Unknown pattern_type: {pattern_type}. "
                f"Must be one of: cold_stuck, dominant_stuck, submissive_stuck"
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