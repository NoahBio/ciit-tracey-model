"""
Global configuration and parameters for CIIT-Tracey computational model.

⚠️ PROVISIONAL UTILITY MATRIX ⚠️
The U_MATRIX values below are placeholder values for initial development and should be replaced with empirically-grounded values when available.

TODO: Replace U_MATRIX with validated values from:
  - Expert consensus
  - Empirical interpersonal research
  - Calibration studies
"""

import numpy as np

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
# UTILITY MATRIX (U) - ⚠️ PROVISIONAL VALUES ⚠️
# =============================================================================

"""
Octant Interaction Utility Matrix U[client_behavior, therapist_behavior]
Rows = Client behavior (0-7: D, WD, W, WS, S, CS, C, CD)
Cols = Therapist behavior (0-7: D, WD, W, WS, S, CS, C, CD)
"""

U_MATRIX = np.array([
    # Therapist:  D     WD     W     WS     S     CS     C     CD
    # Client D (Dominant)
    [  -2.0,  -1.0,  +0.5,  +1.5,  +2.0,  +1.5,   0.0,  -2.0 ],  # D
    
    # Client WD (Warm-Dominant)
    [  -1.0,  +1.0,  +1.5,  +2.0,  +1.5,   0.0,  -1.0,  -2.0 ],  # WD
    
    # Client W (Warm)
    [  +0.5,  +1.5,  +2.0,  +1.5,  +0.5,  -0.5,  -2.0,  -1.5 ],  # W
    
    # Client WS (Warm-Submissive)
    [  +1.5,  +2.0,  +1.5,  +1.0,  -1.0,  -2.0,  -1.0,   0.0 ],  # WS
    
    # Client S (Submissive)
    [  +2.0,  +1.5,  +0.5,  -1.0,  -2.0,  -2.0,   0.0,  +1.5 ],  # S
    
    # Client CS (Cold-Submissive)
    [  +1.5,   0.0,  -1.0,  -2.0,  -1.0,  +1.0,  +1.5,  +2.0 ],  # CS
    
    # Client C (Cold)
    [   0.0,  -1.5,  -2.0,  -1.5,   0.0,  +1.5,  +2.0,  +1.5 ],  # C
    
    # Client CD (Cold-Dominant)
    [  -1.0,  -2.0,  -1.0,   0.0,  +1.5,  +2.0,  +1.5,  +1.0 ],  # CD
])

# Verify matrix shape
assert U_MATRIX.shape == (8, 8), f"U_MATRIX must be 8x8, got {U_MATRIX.shape}"

# =============================================================================
# MEMORY AND SATISFACTION PARAMETERS
# =============================================================================

MEMORY_SIZE = 50  # Number of interaction pairs stored

# Relationship Satisfaction weighting scheme
MEMORY_WEIGHTING = "equal"  # Options: "equal", "recency_bias", "hybrid"

def get_memory_weights(n_interactions: int = MEMORY_SIZE) -> np.ndarray:
    """
    Generate weights for relationship satisfaction calculation.
    
    Parameters
    ----------
    n_interactions : int
        Number of interactions to weight
        
    Returns
    -------
    weights : np.ndarray
        Normalized weights that sum to 1.0
    """
    if MEMORY_WEIGHTING == "equal":
        return np.ones(n_interactions) / n_interactions
    
    elif MEMORY_WEIGHTING == "recency_bias":
        # Exponential decay: recent interactions weighted more
        decay = 0.95
        weights = np.array([decay ** i for i in range(n_interactions-1, -1, -1)])
        return weights / weights.sum()
    
    elif MEMORY_WEIGHTING == "hybrid":
        # Equal for first half, recency bias for second half
        half = n_interactions // 2
        first_half = np.ones(half) / n_interactions
        decay = 0.95
        second_half = np.array([decay ** i for i in range(half-1, -1, -1)])
        second_half = second_half / second_half.sum() * 0.5
        return np.concatenate([first_half, second_half])
    
    else:
        raise ValueError(f"Unknown weighting scheme: {MEMORY_WEIGHTING}")

# =============================================================================
# BOND CALCULATION PARAMETERS
# =============================================================================

def sigmoid(x: float, alpha: float = 1.0) -> float:
    """
    Sigmoid transformation for bond calculation.
    
    Parameters
    ----------
    x : float
        Input value (typically relationship satisfaction)
    alpha : float
        Steepness parameter (higher = steeper transition)
        
    Returns
    -------
    float
        Value in range [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-alpha * x))

BOND_ALPHA = 1.0  # Steepness of sigmoid transformation

# =============================================================================
# CLIENT PARAMETERS
# =============================================================================

# Entropy (temperature) for client action selection
# Higher entropy = more exploration/randomness
CLIENT_ENTROPY_MEAN = 0.5
CLIENT_ENTROPY_STD = 0.15
CLIENT_ENTROPY_MIN = 0.1
CLIENT_ENTROPY_MAX = 2.0

# Dropout probability parameters
DROPOUT_ALPHA = 2.0  # Steepness: P(dropout) = sigmoid(-alpha * bond)

# =============================================================================
# EPISODE PARAMETERS
# =============================================================================

MAX_SESSIONS = 100  # Maximum therapy sessions per episode

# Success threshold: 90th percentile of possible relationship satisfaction values
def calculate_success_threshold() -> float:
    """
    Calculate success threshold as 90th percentile of theoretically possible
    relationship satisfaction values.
    
    Returns
    -------
    float
        The RS value that represents successful therapy
    """
    # All possible RS values are bounded by U_MATRIX values
    # Best case: all interactions at max utility
    # Worst case: all interactions at min utility
    max_possible_rs = U_MATRIX.max()
    min_possible_rs = U_MATRIX.min()
    
    # 90th percentile: 90% of the way from min to max
    threshold = min_possible_rs + 0.9 * (max_possible_rs - min_possible_rs)
    
    return threshold

SUCCESS_THRESHOLD = calculate_success_threshold()

print(f"Success threshold (90th percentile): {SUCCESS_THRESHOLD:.3f}")
print(f"  (Range: [{U_MATRIX.min():.1f}, {U_MATRIX.max():.1f}])")

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