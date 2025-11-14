"""Frequency-amplifier expectation mechanism."""

from .base_client import BaseClientAgent
import numpy as np
from numpy.typing import NDArray
from src.config import get_memory_weights, HISTORY_WEIGHT


class FrequencyAmplifierClient(BaseClientAgent):
    """
    Frequency as amplifier: adjusted = raw + (raw * freq * history_weight)

    History amplifies utilities for frequently-observed therapist responses.
    Likely responses get boosted, unlikely responses remain near baseline.
    Bond still operates on the amplified utility distribution.

    Mathematical formulation:
    1. Calculate P(therapist_j) from memory with recency weighting
    2. Amplify utilities: adjusted[i,j] = U[i,j] + (U[i,j] * P(j) * k)
       where k is the history_weight parameter
    3. Sort amplified utilities for each client action
    4. Select percentile based on bond level

    Properties:
    - Unobserved responses (P(j)=0): utilities unchanged (baseline)
    - High-frequency responses: utilities amplified proportionally
    - Negative utilities: amplification makes them MORE negative (penalty amplified)
    - Positive utilities: amplification makes them MORE positive (reward amplified)

    Parameters
    ----------
    history_weight : float, default=HISTORY_WEIGHT from config
        Controls strength of frequency amplification.
        Higher values = stronger history influence.
        Range: 0.5 - 2.0 recommended for testing.
    """

    def __init__(
        self,
        u_matrix: NDArray[np.float64],
        entropy: float,
        initial_memory,
        history_weight: float | None = None,
        random_state=None,
    ):
        super().__init__(u_matrix, entropy, initial_memory, random_state)
        self.history_weight = history_weight if history_weight is not None else HISTORY_WEIGHT

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

    def _get_effective_history_weight(self) -> float:
        """
        Hook for subclasses to modify history weight based on bond.

        Default implementation returns unmodified history_weight.
        Subclasses can override to implement bond-weighted history influence.

        Returns
        -------
        float
            Effective history weight to use in calculations
        """
        return self.history_weight

    def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
        """
        Calculate expected payoffs using frequency-amplified utilities.

        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        # Calculate marginal therapist behavior frequencies
        therapist_frequencies = self._calculate_marginal_frequencies()
        effective_weight = self._get_effective_history_weight()

        expected_payoffs = np.zeros(8)

        for client_action in range(8):
            raw_utilities = self.u_matrix[client_action, :]

            # Amplify: add frequency-weighted boost to raw utilities
            # Frequently observed responses get amplified (both positive and negative utilities)
            adjusted_utilities = raw_utilities + (
                raw_utilities * therapist_frequencies * effective_weight
            )

            # Sort amplified utilities
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
