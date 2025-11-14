"""Bond-weighted frequency-amplifier expectation mechanism."""

from .frequency_amplifier_client import FrequencyAmplifierClient
import numpy as np
from numpy.typing import NDArray


class BondWeightedFrequencyAmplifier(FrequencyAmplifierClient):
    """
    Marginal frequency amplifier with bond-modulated history influence.

    Extends FrequencyAmplifierClient by scaling history_weight by bond level:
        effective_weight = (bond ** bond_power) × history_weight

    This creates a mechanism where:
    - Low bond (low trust): History has minimal influence on expectations
    - High bond (high trust): History has full influence on expectations

    The bond_power parameter controls the steepness of this relationship:
    - bond_power = 1.0: Linear scaling (bond × history_weight)
    - bond_power = 2.0: Quadratic scaling (bond² × history_weight)
    - bond_power > 2.0: Even steeper drop-off at low bond

    Mathematical formulation:
    1. Calculate effective_weight = (bond ** bond_power) × history_weight
    2. Calculate marginal P(therapist_j) from memory
    3. Amplify: adjusted[i,j] = U[i,j] + (U[i,j] × P(j) × effective_weight)
    4. Sort and select percentile based on bond

    Example behaviors (with history_weight=1.0):
    - bond=0.3, power=1.0 → effective_weight=0.30 (30% history influence)
    - bond=0.3, power=2.0 → effective_weight=0.09 (9% history influence)
    - bond=0.8, power=1.0 → effective_weight=0.80 (80% history influence)
    - bond=0.8, power=2.0 → effective_weight=0.64 (64% history influence)

    Parameters
    ----------
    u_matrix : NDArray[np.float64]
        8×8 utility matrix
    entropy : float
        Temperature parameter for softmax action selection
    initial_memory : list of tuple
        Initial interaction history [(client_oct, therapist_oct), ...]
    history_weight : float, optional
        Base history weight before bond scaling (default: HISTORY_WEIGHT from config)
    bond_power : float, default=1.0
        Exponent for bond scaling. Higher values = steeper drop-off at low bond.
    random_state : int or RandomState, optional
        Random seed for reproducibility

    Examples
    --------
    >>> client = BondWeightedFrequencyAmplifier(
    ...     u_matrix=u, entropy=1.0, initial_memory=memory,
    ...     history_weight=1.0, bond_power=2.0
    ... )
    >>> # With bond=0.3: effective_weight = 0.3**2.0 * 1.0 = 0.09
    """

    def __init__(
        self,
        u_matrix: NDArray[np.float64],
        entropy: float,
        initial_memory,
        history_weight: float | None = None,
        bond_power: float = 1.0,
        random_state=None,
    ):
        super().__init__(
            u_matrix=u_matrix,
            entropy=entropy,
            initial_memory=initial_memory,
            history_weight=history_weight,
            random_state=random_state,
        )
        self.bond_power = bond_power

    def _get_effective_history_weight(self) -> float:
        """
        Apply bond-based scaling to history weight.

        Returns
        -------
        float
            Effective history weight = (bond ** bond_power) × history_weight
        """
        return (self.bond ** self.bond_power) * self.history_weight
