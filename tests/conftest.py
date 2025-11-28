"""
Shared fixtures and utilities for pytest testing.

Provides controlled randomness, memory patterns, client fixtures,
therapist strategies, and assertion helpers for comprehensive testing.
"""

import pytest
import numpy as np
from collections.abc import Callable

from src.config import (
    sample_u_matrix,
    U_MATRIX,
    MEMORY_SIZE,
)
from src.agents.client_agents import (
    BondOnlyClient,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
    BondWeightedFrequencyAmplifier,
    BondWeightedConditionalAmplifier,
)


# ============================================================================
# CONTROLLED RANDOMNESS FIXTURES
# ============================================================================

@pytest.fixture
def fixed_seed():
    """Deterministic seed for reproducible tests."""
    return 42


@pytest.fixture
def fixed_u_matrix(fixed_seed):
    """Fixed utility matrix sampled with deterministic seed."""
    return sample_u_matrix(random_state=fixed_seed)


@pytest.fixture
def deterministic_u_matrix():
    """Use mean U_MATRIX for fully deterministic tests."""
    return U_MATRIX.copy()


@pytest.fixture
def low_entropy():
    """Very low entropy (0.1) for near-deterministic action selection."""
    return 0.1


@pytest.fixture
def medium_entropy():
    """Medium entropy (3.0) for balanced exploration."""
    return 3.0


@pytest.fixture
def high_entropy():
    """High entropy (5.0) for highly random action selection."""
    return 5.0


# ============================================================================
# MEMORY PATTERN FIXTURES
# ============================================================================

@pytest.fixture
def complementary_memory():
    """
    Perfect complementarity memory pattern.

    Each client action gets perfectly complementary therapist response:
    D(0)→S(4), WD(1)→WS(3), W(2)→W(2), etc.
    Results in high RS baseline.
    """
    complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
    return [(i % 8, complement_map[i % 8]) for i in range(MEMORY_SIZE)]


@pytest.fixture
def conflictual_memory():
    """
    All D→D conflictual interactions.
    Results in low RS baseline.
    """
    return [(0, 0)] * MEMORY_SIZE


@pytest.fixture
def warm_consistent_memory():
    """
    Therapist consistently responds with Warm (action 2).
    Useful for testing frequency-based mechanisms.
    """
    return [(i % 8, 2) for i in range(MEMORY_SIZE)]


@pytest.fixture
def cold_warm_memory():
    """
    Client always Cold (6), therapist always Warm (2).
    Anticomplementary pattern - creates tension.
    """
    return [(6, 2)] * MEMORY_SIZE


@pytest.fixture
def empty_memory():
    """
    Random filler memory (all zeros) for minimal baseline.
    Useful when testing memory update mechanics.
    """
    return [(0, 0)] * MEMORY_SIZE


# ============================================================================
# CLIENT FIXTURES
# ============================================================================

@pytest.fixture
def bond_only_client(fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
    """BondOnlyClient with controlled parameters for testing."""
    return BondOnlyClient(
        u_matrix=fixed_u_matrix,
        entropy=low_entropy,
        initial_memory=complementary_memory,
        random_state=fixed_seed
    )


@pytest.fixture
def frequency_amplifier_client(fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
    """FrequencyAmplifierClient with controlled parameters."""
    return FrequencyAmplifierClient(
        u_matrix=fixed_u_matrix,
        entropy=low_entropy,
        initial_memory=complementary_memory,
        random_state=fixed_seed,
        history_weight=1.0
    )


@pytest.fixture
def conditional_amplifier_client(fixed_u_matrix, low_entropy, complementary_memory, fixed_seed):
    """ConditionalAmplifierClient with controlled parameters."""
    return ConditionalAmplifierClient(
        u_matrix=fixed_u_matrix,
        entropy=low_entropy,
        initial_memory=complementary_memory,
        random_state=fixed_seed,
        history_weight=1.0,
        smoothing_alpha=0.1
    )


@pytest.fixture
def all_client_classes():
    """All 5 client mechanism classes for parametrized tests."""
    return [
        BondOnlyClient,
        FrequencyAmplifierClient,
        ConditionalAmplifierClient,
        BondWeightedFrequencyAmplifier,
        BondWeightedConditionalAmplifier,
    ]


# ============================================================================
# THERAPIST STRATEGY FIXTURES
# ============================================================================

@pytest.fixture
def complementary_therapist() -> Callable[[int], int]:
    """
    Therapist that always responds complementarily.

    Returns:
        Function that maps client_action → complementary therapist_action
    """
    def therapist(client_action: int) -> int:
        complement_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
        return complement_map[client_action]
    return therapist


@pytest.fixture
def always_warm_therapist() -> Callable[[int], int]:
    """
    Therapist that always responds with Warm (action 2).

    Returns:
        Function that always returns action 2 (Warm)
    """
    return lambda client_action: 2


@pytest.fixture
def always_dominant_therapist() -> Callable[[int], int]:
    """
    Therapist that always responds with Dominant (action 0).

    Returns:
        Function that always returns action 0 (Dominant)
    """
    return lambda client_action: 0


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def assert_valid_probability_distribution(probs: np.ndarray, tolerance: float = 1e-6):
    """
    Verify array is a valid probability distribution.

    Args:
        probs: Array of probabilities
        tolerance: Numerical tolerance for sum check

    Raises:
        AssertionError: If not a valid probability distribution
    """
    assert len(probs) == 8, f"Expected 8 probabilities, got {len(probs)}"
    assert np.all(probs >= 0), f"Negative probabilities found: {probs[probs < 0]}"
    assert np.isclose(np.sum(probs), 1.0, atol=tolerance), \
        f"Probabilities sum to {np.sum(probs):.6f}, expected 1.0"


def assert_valid_octant(action: int):
    """
    Verify action is a valid octant (0-7).

    Args:
        action: Octant action to validate

    Raises:
        AssertionError: If not a valid octant
    """
    assert isinstance(action, (int, np.integer)), \
        f"Action must be int, got {type(action)}"
    assert 0 <= action <= 7, \
        f"Action must be in range [0, 7], got {action}"


def assert_bond_in_range(bond: float):
    """
    Verify bond value is in valid range [0, 1].

    Args:
        bond: Bond value to validate

    Raises:
        AssertionError: If bond not in [0, 1]
    """
    assert 0.0 <= bond <= 1.0, \
        f"Bond must be in range [0, 1], got {bond:.4f}"


def assert_rs_in_bounds(rs: float, rs_min: float, rs_max: float):
    """
    Verify RS is within client-specific bounds.

    Args:
        rs: Relationship satisfaction value
        rs_min: Minimum possible RS for client
        rs_max: Maximum possible RS for client

    Raises:
        AssertionError: If RS out of bounds
    """
    # Allow small numerical tolerance
    tolerance = 1e-6
    assert rs_min - tolerance <= rs <= rs_max + tolerance, \
        f"RS {rs:.4f} outside bounds [{rs_min:.4f}, {rs_max:.4f}]"


def assert_memory_size_correct(memory, expected_size: int = MEMORY_SIZE):
    """
    Verify memory has correct size.

    Args:
        memory: Memory deque to check
        expected_size: Expected memory size (default: MEMORY_SIZE=50)

    Raises:
        AssertionError: If memory size incorrect
    """
    assert len(memory) == expected_size, \
        f"Memory size is {len(memory)}, expected {expected_size}"


def assert_monotonic_increasing(values: np.ndarray, strict: bool = False):
    """
    Verify array is monotonically increasing.

    Args:
        values: Array to check
        strict: If True, requires strict inequality (no equal consecutive values)

    Raises:
        AssertionError: If not monotonically increasing
    """
    diffs = np.diff(values)
    if strict:
        assert np.all(diffs > 0), "Array not strictly increasing"
    else:
        assert np.all(diffs >= 0), "Array not monotonically increasing"
