"""Client agent variants for testing different expectation mechanisms."""

import numpy as np
from typing import Optional

from .base_client import BaseClientAgent
from .bond_only_client import BondOnlyClient
from .frequency_filter_client import FrequencyFilterClient
from .frequency_amplifier_client import FrequencyAmplifierClient

__all__ = [
    'BaseClientAgent',
    'BondOnlyClient',
    'FrequencyFilterClient',
    'FrequencyAmplifierClient',
    'create_client',
    'create_problematic_client',
]


def create_client(mechanism='bond_only', **kwargs):
    """
    Factory function to create client agent with specified mechanism.

    Parameters
    ----------
    mechanism : str
        Expectation mechanism to use:
        - 'bond_only': Original (no frequency tracking)
        - 'frequency_filter': History filters via multiplication
        - 'frequency_amplifier': History amplifies via addition
    **kwargs : dict
        Arguments passed to client constructor:
        - u_matrix : ndarray (required)
        - entropy : float (required)
        - initial_memory : list (required)
        - history_weight : float (optional, only for amplifier)
        - random_state : int or RandomState (optional)

    Returns
    -------
    BaseClientAgent subclass instance

    Examples
    --------
    >>> from src.config import sample_u_matrix
    >>> u = sample_u_matrix(random_state=42)
    >>> mem = [(0, 0)] * 50
    >>> client = create_client('bond_only', u_matrix=u, entropy=1.0, initial_memory=mem)
    >>> client = create_client('frequency_filter', u_matrix=u, entropy=1.0, initial_memory=mem)
    >>> client = create_client('frequency_amplifier', u_matrix=u, entropy=1.0,
    ...                        initial_memory=mem, history_weight=1.5)
    """
    mechanisms = {
        'bond_only': BondOnlyClient,
        'frequency_filter': FrequencyFilterClient,
        'frequency_amplifier': FrequencyAmplifierClient,
    }

    if mechanism not in mechanisms:
        valid = ', '.join(mechanisms.keys())
        raise ValueError(
            f"Unknown mechanism: '{mechanism}'. "
            f"Valid options: {valid}"
        )

    return mechanisms[mechanism](**kwargs)


def create_problematic_client(
    pattern_type: str = "cold_stuck",
    entropy: Optional[float] = None,
    mechanism: str = 'frequency_amplifier',
    random_state: Optional[int] = None,
) -> BaseClientAgent:
    """
    Create a client with problematic interpersonal pattern (backward compatible helper).

    This function maintains backward compatibility with the old create_client() API
    that took pattern_type and entropy parameters. It automatically generates the
    problematic memory, samples the u_matrix, and handles entropy sampling.

    Each client gets:
    - Client-specific u_matrix sampled from U_MIN/U_MAX ranges
    - Entropy sampled from distribution (or custom value)
    - Problematic memory history generated realistically

    Parameters
    ----------
    pattern_type : str, default='cold_stuck'
        Type of problematic interpersonal pattern:
        - "cold_stuck": Stuck in cold behaviors
        - "dominant_stuck": Stuck in dominant behaviors
        - "submissive_stuck": Stuck in submissive behaviors
    entropy : float, optional
        Temperature parameter. If None, samples from distribution.
    mechanism : str, default='frequency_amplifier'
        Expectation mechanism to use:
        - 'bond_only': Original (no frequency tracking)
        - 'frequency_filter': History filters via multiplication
        - 'frequency_amplifier': History amplifies via addition (default, matches legacy)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    BaseClientAgent subclass instance
        Initialized client agent with client-specific U_MATRIX

    Examples
    --------
    >>> client = create_problematic_client('cold_stuck', random_state=42)
    >>> client = create_problematic_client('dominant_stuck', entropy=0.8, mechanism='bond_only')
    """
    from src.config import (
        CLIENT_ENTROPY_MEAN, CLIENT_ENTROPY_STD,
        CLIENT_ENTROPY_MIN, CLIENT_ENTROPY_MAX,
        sample_u_matrix, MEMORY_SIZE
    )

    rng = np.random.RandomState(random_state)

    # Sample entropy if not provided
    if entropy is None:
        entropy = float(rng.normal(CLIENT_ENTROPY_MEAN, CLIENT_ENTROPY_STD))
        entropy = float(np.clip(entropy, CLIENT_ENTROPY_MIN, CLIENT_ENTROPY_MAX))
    else:
        entropy = float(entropy)

    # Generate problematic memory
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type=pattern_type,
        n_interactions=MEMORY_SIZE,
        random_state=random_state,
    )

    # Sample client-specific u_matrix
    u_matrix = sample_u_matrix(random_state=random_state)

    # Create client with specified mechanism
    client = create_client(
        mechanism=mechanism,
        u_matrix=u_matrix,
        entropy=entropy,
        initial_memory=initial_memory,
        random_state=random_state,
    )

    return client
