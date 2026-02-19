"""
Global configuration and parameters for the agent layer of this CIIT transactional cycle computational model.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List

# =============================================================================
# OCTANT DEFINITIONS
# =============================================================================

OCTANTS = {
    0: "D",   # Dominant (PA)
    1: "WD",  # Warm-Dominant (BC)
    2: "W",   # Warm (DE)
    3: "WS",  # Warm-Submissive (FG)
    4: "S",   # Submissive (HI)
    5: "CS",  # Cold-Submissive (JK)
    6: "C",   # Cold (LM)
    7: "CD",  # Cold-Dominant (NO)
}

OCTANT_NAMES = ["D", "WD", "W", "WS", "S", "CS", "C", "CD"]

# Reverse mapping: name -> index
OCTANT_INDEX = {name: idx for idx, name in OCTANTS.items()}

# =============================================================================
# UTILITY MATRIX BOUNDS - Version 3
# =============================================================================

"""
Utility matrix bounds define the range of possible utilities for each 
client-therapist octant interaction. Client-specific U_MATRIX values 
are sampled from these ranges.

Format: U_MIN[client, therapist] and U_MAX[client, therapist]
Rows = Client behavior (0-7: D, WD, w, WS, S, CS, C, CD)
Cols = Therapist behavior (0-7: D, WD, W, WS, S, CS, C, CD)
"""

# Minimum utility values
U_MIN = np.array([
    # Therapist:  D      WD      W      WS      S      CS      C      CD
    # Client D (Dominant)
    [           -40,    -50,   -10,    +20,    +30,    +20,   -10,    -50 ],  # D
    
    # Client WD (Warm-Dominant)
    [           -30,      0,    +30,    +50,    +10,    -30,   -50,    -70 ],  # WD
    
    # Client W (Warm)
    [           -10,    +30,    +40,    +30,    -10,    -70,   -60,    -70 ],  # W
    
    # Client WS (Warm-Submissive)
    [           +10,    +50,    +30,      0,    -30,    -70,   -50,    -30 ],  # WS
    
    # Client S (Submissive)
    [           +30,    +20,    -10,    -50,    -40,    -50,   -10,    +20 ],  # S
    
    # Client CS (Cold-Submissive)
    [           +10,    -10,    -30,    -40,    -20,    -10,   +10,    +20 ],  # CS
    
    # Client C (Cold)
    [           -10,    -40,    -30,    -40,    -10,    +10,   +20,    +10 ],  # C
    
    # Client CD (Cold-Dominant)
    [           -30,    -40,    -20,    -10,    +10,    +20,   +10,    -10 ],  # CD
])

# Maximum utility values
U_MAX = np.array([
    # Therapist:  D      WD      W      WS      S      CS      C      CD
    # Client D (Dominant)
    [           -20,    -30,    +10,    +40,    +50,    +40,   +10,    -30 ],  # D
    
    # Client WD (Warm-Dominant)
    [           -10,    +20,    +50,    +70,    +30,    -10,   -30,    -50 ],  # WD
    
    # Client W (Warm)
    [           +10,    +50,    +60,    +50,    +10,    -50,   -40,    -50 ],  # W
    
    # Client WS (Warm-Submissive)
    [           +30,    +70,    +50,    +20,    -10,    -50,   -30,    -10 ],  # WS
    
    # Client S (Submissive)
    [           +50,    +40,    +10,    -30,    -20,    -30,   +10,    +40 ],  # S
    
    # Client CS (Cold-Submissive)
    [           +30,    +10,    -10,    -20,      0,    +10,   +30,    +40 ],  # CS
    
    # Client C (Cold)
    [           +10,    -20,    -10,    -20,    +10,    +30,   +40,    +30 ],  # C
    
    # Client CD (Cold-Dominant)
    [           -10,    -20,      0,    +10,    +30,    +40,   +30,    +10 ],  # CD
])

# Verify shapes
assert U_MIN.shape == (8, 8), f"U_MIN must be 8x8, got {U_MIN.shape}"
assert U_MAX.shape == (8, 8), f"U_MAX must be 8x8, got {U_MAX.shape}"
assert np.all(U_MIN <= U_MAX), "U_MIN must be <= U_MAX for all entries"

def sample_u_matrix(random_state: Optional[int | np.random.RandomState] = None) -> NDArray[np.float64]:
    """
    Sample a client-specific utility matrix from the defined ranges.

    Parameters
    ----------
    random_state : int or np.random.RandomState, optional
        Random seed or RandomState instance for reproducibility

    Returns
    -------
    NDArray[np.float64]
        8x8 utility matrix sampled from [U_MIN, U_MAX] ranges
    """
    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)
    
    # Sample uniformly between min and max for each cell
    u_matrix = rng.uniform(low=U_MIN, high=U_MAX)
    
    return u_matrix

# Keep a default/mean matrix for reference
U_MATRIX = (U_MIN + U_MAX) / 2  # Midpoint of ranges

# =============================================================================
# NAMED U-MATRICES
# =============================================================================

"""
Named U-matrix configurations for controlled experiments.

To add a new U-matrix:
1. Define an 8x8 numpy array below
2. Add it to NAMED_U_MATRICES dictionary with a descriptive name
3. Use via CLI: --u-matrix <name>

Format: rows = client behavior (D, WD, W, WS, S, CS, C, CD)
        cols = therapist behavior (D, WD, W, WS, S, CS, C, CD)
"""

# Example 1: Average/midpoint of bounds (same as default U_MATRIX)
U_MATRIX_AVERAGE = (U_MIN + U_MAX) / 2

# Example 2: Highly complementary - strong preference for complementary actions
U_MATRIX_HIGHLY_COMPLEMENTARY = np.array([
    # Therapist:  D      WD      W      WS      S      CS      C      CD
    # Client D (Dominant)
    [           -30,    -40,      0,    +30,    +40,    +30,      0,    -40 ],
    # Client WD (Warm-Dominant)
    [           -20,    +10,    +40,    +60,    +20,    -20,   -40,    -60 ],
    # Client W (Warm)
    [             0,    +40,    +50,    +40,      0,    -60,   -50,    -60 ],
    # Client WS (Warm-Submissive)
    [           +20,    +60,    +40,    +10,    -20,    -60,   -40,    -20 ],
    # Client S (Submissive)
    [           +40,    +30,      0,    -40,    -30,    -40,      0,    +30 ],
    # Client CS (Cold-Submissive)
    [           +20,      0,    -20,    -30,    -10,      0,   +20,    +30 ],
    # Client C (Cold)
    [             0,    -30,    -20,    -30,      0,    +20,   +30,    +20 ],
    # Client CD (Cold-Dominant)
    [           -20,    -30,    -10,      0,    +20,    +30,   +20,      0 ],
])

# Cold-Trap Matrix: Frequency amplifier creates inescapable local optimum
# Trapped at RS≈+22 (bond≈0.42), need RS≥+56 for success
U_MATRIX_COLD_TRAP = np.array([
    # Therapist:  D      WD      W      WS      S      CS      C      CD
    # Client D (Dominant)
    [           -20,    -30,     -5,    +15,    +35,    +15,    -5,    -30 ],
    
    # Client WD (Warm-Dominant) - high complementary but unreachable
    [           -15,    +10,    +45,    +65,    +25,    -10,   -30,    -45 ],
    
    # Client W (Warm) - Very high complementary but unreachable
    # KEY: Cold responses (C/CS/CD) must be severely punishing at indices 0-3
    [            -5,    +45,    +68,    +45,     -5,    -50,   -45,    -52 ],
    
    # Client WS (Warm-Submissive) - high complementary but unreachable
    [           +15,    +65,    +45,    +10,    -10,    -48,   -30,    -10 ],
    
    # Client S (Submissive)
    [           +35,    +15,     -5,    -30,    -20,    -30,    -5,    +15 ],
    
    # Client CS (Cold-Submissive) - TRAP zone
    [           +10,     -5,    -18,    -28,     -5,     +8,   +15,    +20 ],
    
    # Client C (Cold) - TRAP: narrow spread, moderate complementary
    # KEY: Warm responses punishing but not catastrophic (prevents early escape)
    [            +5,    -12,    -18,    -28,     -5,    +15,   +23,    +18 ],
    
    # Client CD (Cold-Dominant) - TRAP zone
    [            -5,    -22,    -15,     -8,    +10,    +20,   +18,     +5 ],
])

# Registry of all named U-matrices
NAMED_U_MATRICES = {
    'average': U_MATRIX_AVERAGE,
    'highly_complementary': U_MATRIX_HIGHLY_COMPLEMENTARY,
    'cold_trap': U_MATRIX_COLD_TRAP,
    # Add more matrices here as needed
}


def get_u_matrix_by_name(name: str) -> Optional[NDArray[np.float64]]:
    """
    Retrieve a named U-matrix from the registry.

    Parameters
    ----------
    name : str
        Name of the U-matrix to retrieve

    Returns
    -------
    NDArray[np.float64] or None
        The requested U-matrix, or None if name not found
    """
    return NAMED_U_MATRICES.get(name)


def list_available_u_matrices() -> List[str]:
    """Return a list of all available named U-matrix names."""
    return sorted(NAMED_U_MATRICES.keys())

# =============================================================================
# MEMORY AND SATISFACTION PARAMETERS
# =============================================================================

MEMORY_SIZE = 50  # Number of interaction pairs stored

# Relationship Satisfaction weighting scheme: square root recency bias
def get_memory_weights(
    n_interactions: int = MEMORY_SIZE,
    recency_weighting_factor: float | None = None,
) -> NDArray[np.float64]:
    """
    Recency-weighted memory with configurable newest:oldest ratio.

    - Preserves the previous sqrt-shaped recency curve.
    - `recency_weighting_factor` is the newest:oldest weight ratio directly.
      E.g., 2.0 means the newest interaction is weighted 2x the oldest.
    - If `recency_weighting_factor` is None, uses global config.RECENCY_WEIGHTING_FACTOR

    Parameters
    ----------
    n_interactions : int
        Number of interactions to weight
    recency_weighting_factor : float or None
        Newest:oldest weight ratio (must be >= 1.0).
        If None, uses global config.RECENCY_WEIGHTING_FACTOR (default: 2.0)

    Returns
    -------
    NDArray[np.float64]
        Normalized weights that sum to 1.0
    """
    if n_interactions <= 0:
        return np.array([], dtype=float)
    if n_interactions == 1:
        return np.array([1.0], dtype=float)

    # Use global config if not specified
    if recency_weighting_factor is None:
        recency_weighting_factor = RECENCY_WEIGHTING_FACTOR

    if recency_weighting_factor < 1.0:
        raise ValueError(f"recency_weighting_factor must be >= 1.0, got {recency_weighting_factor}")

    ratio = float(recency_weighting_factor)
    # sqrt-shaped recency (oldest=1.0, newest=ratio)
    t = np.arange(n_interactions) / (n_interactions - 1)
    shape = np.sqrt(t)
    weights = 1.0 + (ratio - 1.0) * shape

    return weights / weights.sum()

# =============================================================================
# BOND CALCULATION PARAMETERS
# =============================================================================

BOND_ALPHA = 5  # Steepness of sigmoid transformation; chosen based on plotting the sigmoid curve for various alpha values and selecting a values
BOND_OFFSET = 0.8  # Offset for sigmoid inflection point (0.7 = inflection at 70th percentile of RS range)

def rs_to_bond(
    rs: float,
    rs_min: float,
    rs_max: float,
    alpha: float = BOND_ALPHA,
    offset: float = BOND_OFFSET
) -> float:
    """
    Sigmoid transformation of RS for bond calculation with normalization.

    Normalizes RS to [0, 1] range based on client-specific matrix bounds,
    applies offset shift, then applies sigmoid.

    Parameters
    ----------
    rs : float
        Current relationship satisfaction
    rs_min : float
        Minimum possible RS for this client (min of their U_MATRIX)
    rs_max : float
        Maximum possible RS for this client (max of their U_MATRIX)
    alpha : float
        Steepness parameter (default 5)
        Higher values = steeper transition around inflection point
    offset : float
        Percentile of RS range where bond = 0.5 (default 0.7)
        Higher values = lower initial bond for negative RS
        offset=0.5 centers sigmoid at midpoint
        offset=0.7 shifts inflection to 70th percentile

    Returns
    -------
    float
        Bond value in range [0, 1]
    """
    rs_range = rs_max - rs_min

    # Normalize RS to [0, 1] range
    rs_normalized = (rs - rs_min) / rs_range

    # Apply offset shift to center sigmoid at desired percentile
    rs_shifted = 2 * (rs_normalized - offset)

    # Apply sigmoid
    bond = 1.0 / (1.0 + np.exp(-alpha * rs_shifted))

    return float(bond)

# =============================================================================
# CLIENT PARAMETERS
# =============================================================================

# Entropy (temperature) for client action selection
# Higher entropy = more exploration/randomness
CLIENT_ENTROPY_MEAN = 3
CLIENT_ENTROPY_STD = 0.5
CLIENT_ENTROPY_MIN = 1.5
CLIENT_ENTROPY_MAX = 5

HISTORY_WEIGHT = 1.0  # Weighting factor for client history in utility calculation (used in amplifier mechanisms)

# Recency weighting for memory
RECENCY_WEIGHTING_FACTOR = 2.0  # Default: 2.0x newest:oldest ratio

# =============================================================================
# PARATAXIC DISTORTION PARAMETERS
# =============================================================================

# Parataxic distortion system for client agents
# Models Sullivan's concept of parataxic distortion - tendency to perceive present
# relationships through the lens of past experiences
PARATAXIC_WINDOW = 15                    # Number of recent interactions to consider for distortion
PARATAXIC_BASELINE_ACCURACY = 0.5        # Base probability of correct perception (50%)

# Backward compatibility aliases (deprecated)
PERCEPTION_WINDOW = PARATAXIC_WINDOW
PERCEPTION_BASELINE_ACCURACY = PARATAXIC_BASELINE_ACCURACY

# =============================================================================
# EPISODE PARAMETERS
# =============================================================================

MAX_SESSIONS = 100  # Maximum therapy sessions per episode

# Success threshold: 90th percentile of possible relationship satisfaction values
def calculate_success_threshold(
    u_matrix: NDArray[np.float64],
    percentile: float = 0.9
) -> float:
    """
    Calculate success threshold as a percentile of theoretically possible
    relationship satisfaction values for a specific client's utility matrix.
    
    Parameters
    ----------
    u_matrix : NDArray[np.float64]
        The client's 8x8 utility matrix
    percentile : float, default=0.9
        Percentile threshold for success (must be between 0 and 1)
        Higher values = more stringent success criteria
        
    Returns
    -------
    float
        The RS value that represents successful therapy for this client
        
    Raises
    ------
    ValueError
        If percentile is not between 0 and 1
    """
    if not 0 <= percentile <= 1:
        raise ValueError(f"percentile must be between 0 and 1, got {percentile}")
    
    # All possible RS values are bounded by this client's matrix values
    max_possible_rs = u_matrix.max()
    min_possible_rs = u_matrix.min()

    # Calculate threshold at specified percentile
    threshold = min_possible_rs + percentile * (max_possible_rs - min_possible_rs)
    
    return threshold

# =============================================================================
# REWARD FUNCTION PARAMETERS
# =============================================================================

# Reward structure: Terminal reward with efficiency incentive
REWARD_SUCCESS_BASE = 100.0       # Base reward for successful therapy
REWARD_EFFICIENCY_BONUS = 1.0     # Bonus per session saved (100 - num_sessions)
REWARD_DROPOUT_PENALTY = -150.0   # Penalty for client dropout
REWARD_MAX_LENGTH = 20.0          # Partial reward if max sessions reached

# =============================================================================
# THERAPIST AGENT PARAMETERS
# =============================================================================

# Model-free (PPO+LSTM) parameters
PPO_LEARNING_RATE = 3e-4
PPO_CLIP_RATIO = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_ENTROPY_COEF = 0.01
PPO_LSTM_HIDDEN_SIZE = 256

# Model-based (MBPO) parameters
MBPO_ENSEMBLE_SIZE = 5
MBPO_ROLLOUT_LENGTH = 5
MBPO_REAL_DATA_RATIO = 0.1

# Bond estimation parameters (for model-based agent)
BOND_PRIOR_MEAN = 0.3     # Initial belief about bond
BOND_PRIOR_WEIGHT = 1.0   # Weight of prior (decays over sessions)

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

RANDOM_SEED = 42
NUM_PARALLEL_ENVS = 8
TOTAL_TIMESTEPS = 500_000  # Total environment steps for model-free training

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

OCTANT_COLORS = {
    "D": "#FF6B6B",   # Red
    "WD": "#FFA07A",  # Light red-orange
    "W": "#FFD93D",   # Yellow
    "WS": "#95E1D3",  # Light teal
    "S": "#6BCB77",   # Green
    "CS": "#4D96FF",  # Blue
    "C": "#6C5CE7",   # Purple
    "CD": "#A78BFA",  # Light purple
}