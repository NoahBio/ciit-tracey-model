"""Frequency-filter expectation mechanism."""

from .base_client import BaseClientAgent
import numpy as np
from numpy.typing import NDArray
from src.config import get_memory_weights


class FrequencyFilterClient(BaseClientAgent):
    """
    Frequency as filter: adjusted[i,j] = U[i,j] * P(j)

    History creates probability distribution over therapist responses.
    Frequencies "filter" the utility space by multiplying raw utilities.
    Likely responses get full weight, unlikely responses get diminished.
    Bond then selects percentile within probability-weighted distribution.

    Mathematical formulation:
    1. Calculate P(therapist_j) from memory with recency weighting
    2. Adjust utilities: adjusted[i,j] = U[i,j] * P(j)
    3. Sort adjusted utilities for each client action
    4. Select percentile based on bond level

    Properties:
    - Zero-frequency responses get zero weight (completely filtered out)
    - High-frequency responses retain most of their utility value
    - Bond operates on filtered distribution (optimism about likely outcomes)

    No Bayesian smoothing - raw empirical frequencies only.
    Softmax exploration handles uncertainty about unseen actions.
    """

    def _calculate_marginal_frequencies(self) -> NDArray[np.float64]:
        """
        Calculate P(therapist_j) from memory with recency weighting.

        Computes simple frequency distribution of therapist behaviors,
        ignoring what client action preceded them (marginal probability).

        Returns
        -------
        NDArray[np.float64]
            Probability distribution P(therapist_j) for j=0..7
        """
        memory_weights = get_memory_weights(len(self.memory))
        weighted_counts = np.zeros(8)

        for idx, (client_oct, therapist_oct) in enumerate(self.memory):
            weighted_counts[therapist_oct] += memory_weights[idx]

        total_weight = sum(memory_weights)
        if total_weight == 0:
            return np.ones(8) / 8  # No memory = uniform

        return weighted_counts / total_weight

    def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
        """
        Calculate expected payoffs using frequency-filtered utilities.

        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        # Calculate marginal therapist behavior frequencies
        therapist_frequencies = self._calculate_marginal_frequencies()

        expected_payoffs = np.zeros(8)

        for client_action in range(8):
            raw_utilities = self.u_matrix[client_action, :]

            # Filter: multiply utilities by probability of each therapist response
            # Likely responses keep full utility, unlikely responses get diminished
            adjusted_utilities = raw_utilities * therapist_frequencies

            # Sort probability-weighted utilities
            sorted_adjusted = np.sort(adjusted_utilities)

            # Apply bond-based percentile interpolation
            position = self.bond * 7
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, 7)
            interpolation_weight = position - lower_idx

            expected_payoffs[client_action] = (
                (1 - interpolation_weight) * sorted_adjusted[lower_idx] +
                interpolation_weight * sorted_adjusted[upper_idx]
            )

        return expected_payoffs
