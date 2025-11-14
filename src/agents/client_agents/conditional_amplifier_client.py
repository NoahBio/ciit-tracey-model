"""Conditional frequency-amplifier expectation mechanism."""

from .base_client import BaseClientAgent
import numpy as np
from numpy.typing import NDArray
from src.config import get_memory_weights, HISTORY_WEIGHT


class ConditionalAmplifierClient(BaseClientAgent):
    """
    Conditional frequency as amplifier: adjusted = raw + (raw × P(j|i) × k)

    History creates conditional probability distribution P(therapist_j | client_i).
    Unlike marginal frequency_amplifier, this tracks how the therapist responds to
    EACH specific client action, providing more accurate expectations.

    Mathematical formulation:
    1. Calculate P(therapist_j | client_i) from memory with recency weighting
    2. Apply Laplace smoothing to prevent zeros: (count + α) / (total + α×8)
    3. Amplify utilities: adjusted[i,j] = U[i,j] + (U[i,j] × P(j|i) × k)
       where k is the history_weight parameter
    4. Sort amplified utilities for client action i
    5. Select percentile based on bond level

    Properties:
    - More robust than marginal frequency_amplifier
    - Smoothing provides baseline for unseen responses
    - Each client action has its own conditional distribution
    - Observed responses get amplified proportionally
    - Unobserved responses remain near (slightly amplified) baseline

    Parameters
    ----------
    history_weight : float, default=HISTORY_WEIGHT from config
        Controls strength of frequency amplification.
        Higher values = stronger history influence.
        Range: 0.5 - 2.0 recommended for testing.
    smoothing_alpha : float, default=0.1
        Laplace smoothing parameter (pseudo-count).
        Higher values = more uniform prior.
        Recommended range: 0.01 - 1.0
    """

    def __init__(
        self,
        u_matrix: NDArray[np.float64],
        entropy: float,
        initial_memory,
        history_weight: float | None = None,
        smoothing_alpha: float = 0.1,
        random_state=None,
    ):
        super().__init__(u_matrix, entropy, initial_memory, random_state)
        self.history_weight = history_weight if history_weight is not None else HISTORY_WEIGHT
        self.smoothing_alpha = smoothing_alpha

    def _calculate_conditional_frequencies(self, client_action: int) -> NDArray[np.float64]:
        """
        Calculate P(therapist_j | client_i) for specific client action.

        Uses Laplace smoothing to prevent zeros while staying data-driven.

        Parameters
        ----------
        client_action : int
            Client octant (0-7) to condition on

        Returns
        -------
        NDArray[np.float64]
            Conditional probability distribution P(therapist_j | client_action)
        """
        memory_weights = get_memory_weights(len(self.memory))
        weighted_counts = np.zeros(8)

        # Count weighted occurrences of each therapist response to this client action
        for idx, (client_oct, therapist_oct) in enumerate(self.memory):
            if client_oct == client_action:
                weighted_counts[therapist_oct] += memory_weights[idx]

        # Apply Laplace smoothing
        # Add pseudo-count α to each therapist response
        smoothed_counts = weighted_counts + self.smoothing_alpha
        total_weight = np.sum(smoothed_counts)

        if total_weight == 0:
            # No observations of this client action, use uniform
            return np.ones(8) / 8

        # Normalize to get conditional probabilities
        conditional_probs = smoothed_counts / total_weight

        return conditional_probs

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
        Calculate expected payoffs using conditional frequency amplification.

        Returns
        -------
        NDArray[np.float64]
            8-dimensional array of expected payoffs, one per octant
        """
        expected_payoffs = np.zeros(8)
        effective_weight = self._get_effective_history_weight()

        for client_action in range(8):
            # Get conditional probabilities for this client action
            conditional_freq = self._calculate_conditional_frequencies(client_action)

            # Get raw utilities for this client action
            raw_utilities = self.u_matrix[client_action, :]

            # Amplify: add frequency-weighted boost to raw utilities
            # Frequently observed responses get amplified (both positive and negative utilities)
            adjusted_utilities = raw_utilities + (
                raw_utilities * conditional_freq * effective_weight
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
