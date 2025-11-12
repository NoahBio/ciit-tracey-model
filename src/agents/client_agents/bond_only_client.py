"""Bond-only expectation mechanism (original)."""

from .base_client import BaseClientAgent
import numpy as np
from numpy.typing import NDArray


class BondOnlyClient(BaseClientAgent):
    """
    Original mechanism: Bond determines percentile of raw utilities.

    Expectations based solely on bond (trust/optimism level).
    No history/frequency tracking of therapist behavior.

    The client expects a percentile of possible outcomes based on their bond level:
    - High bond (approaching 1.0): Expects best possible outcomes (high percentiles)
    - Low bond (approaching 0.0): Expects worst possible outcomes (low percentiles)
    - Medium bond (around 0.5): Expects median outcomes

    Expected payoff = percentile(U[i,:], bond)

    This is the baseline mechanism against which frequency-based mechanisms are compared.
    """

    def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
        """
        Calculate expected payoffs using bond-based percentile of raw utilities.

        For each client action, sorts the possible utilities (across all therapist
        responses) and selects the percentile corresponding to the client's bond level.

        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        expected_payoffs = np.zeros(8)

        for client_action in range(8):
            # Get raw utilities for this action
            utilities_row = self.u_matrix[client_action, :]
            sorted_utilities = np.sort(utilities_row)

            # Bond-based percentile interpolation
            # bond=0 → index 0 (worst outcome)
            # bond=1 → index 7 (best outcome)
            position = self.bond * 7

            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, 7)
            interpolation_weight = position - lower_idx

            expected_payoffs[client_action] = (
                (1 - interpolation_weight) * sorted_utilities[lower_idx] +
                interpolation_weight * sorted_utilities[upper_idx]
            )

        return expected_payoffs
