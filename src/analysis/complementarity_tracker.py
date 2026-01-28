"""Complementarity tracking for therapy simulations.

This module provides utilities for tracking complementarity between client and
therapist actions over time using a sliding window approach.
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np


# Complementary action mapping based on Interpersonal Circumplex
# Control dimension: opposite (D↔S), Affiliation dimension: same (W↔W, C↔C)
COMPLEMENT_MAP = {
    0: 4,  # D → S (Dominant → Submissive)
    1: 3,  # WD → WS (Warm-Dominant → Warm-Submissive)
    2: 2,  # W → W (Warm → Warm)
    3: 1,  # WS → WD (Warm-Submissive → Warm-Dominant)
    4: 0,  # S → D (Submissive → Dominant)
    5: 7,  # CS → CD (Cold-Submissive → Cold-Dominant)
    6: 6,  # C → C (Cold → Cold)
    7: 5,  # CD → CS (Cold-Dominant → Cold-Submissive)
}

# Octant categories for warm/cold filtering
WARM_OCTANTS = {1, 2, 3}  # WD, W, WS
COLD_OCTANTS = {5, 6, 7}  # CS, C, CD


def octant_distance(action1: int, action2: int) -> int:
    """Calculate minimum distance between two octants on the circular graph.

    The 8 octants are arranged in a circle, so distance is the minimum of
    clockwise or counterclockwise steps.

    Args:
        action1: First octant (0-7)
        action2: Second octant (0-7)

    Returns:
        Distance between octants (0-4, where 4 is maximally opposite)
    """
    raw_dist = abs(action1 - action2)
    return min(raw_dist, 8 - raw_dist)


def complementarity_distance(client_action: int, therapist_action: int) -> int:
    """Calculate distance from therapist action to the complementary action.

    Args:
        client_action: Client's octant action (0-7)
        therapist_action: Therapist's octant action (0-7)

    Returns:
        Distance from therapist action to complementary action (0-4).
        0 = perfectly complementary, 4 = maximally anti-complementary.
    """
    complementary = COMPLEMENT_MAP[client_action]
    return octant_distance(therapist_action, complementary)


class ComplementarityTracker:
    """Track complementarity between client and therapist actions over time.

    Uses a sliding window to calculate the proportion of therapist actions
    that are complementary to client actions. Supports filtering by warm/cold
    octants and tracking both enacted (actual) and perceived complementarity.

    Attributes:
        window_size: Number of recent interactions to consider
        recent_interactions: Deque of (client_action, therapist_action) tuples
        recent_enacted: Deque of (client_action, enacted_therapist_action) tuples
        recent_perceived: Deque of (client_action, perceived_therapist_action) tuples
        complement_map: Mapping from client action to complementary therapist action
    """

    def __init__(self, window_size: int = 10):
        """Initialize the complementarity tracker.

        Args:
            window_size: Number of recent interactions to include in calculations
                (default: 10 sessions)
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.window_size = window_size
        self.complement_map = COMPLEMENT_MAP

        # For tracking single perspective (when enacted = perceived)
        self.recent_interactions: deque = deque(maxlen=window_size)

        # For tracking dual perspectives (when parataxic distortion is enabled)
        self.recent_enacted: deque = deque(maxlen=window_size)
        self.recent_perceived: deque = deque(maxlen=window_size)

        self._dual_tracking = False

    def add_interaction(
        self,
        client_action: int,
        therapist_action: int,
        enacted_action: Optional[int] = None,
        perceived_action: Optional[int] = None
    ) -> None:
        """Add an interaction to the tracking window.

        Args:
            client_action: Client's octant action (0-7)
            therapist_action: Therapist's action (used if enacted/perceived not provided)
            enacted_action: Therapist's actual action (if parataxic distortion enabled)
            perceived_action: Client's perceived therapist action (if parataxic distortion enabled)
        """
        if enacted_action is not None or perceived_action is not None:
            # Dual tracking mode
            self._dual_tracking = True
            enacted = enacted_action if enacted_action is not None else therapist_action
            perceived = perceived_action if perceived_action is not None else therapist_action
            self.recent_enacted.append((client_action, enacted))
            self.recent_perceived.append((client_action, perceived))
        else:
            # Single tracking mode
            self.recent_interactions.append((client_action, therapist_action))

    def _calculate_complementarity_rate(
        self,
        interactions: deque,
        filter_octants: Optional[set] = None
    ) -> float:
        """Calculate complementarity rate for a set of interactions.

        Args:
            interactions: Deque of (client_action, therapist_action) tuples
            filter_octants: If provided, only consider interactions where
                client_action is in this set

        Returns:
            Complementarity rate as percentage (0-100), or np.nan if no
            interactions match the filter
        """
        if not interactions:
            # If filtering for warm/cold, return NaN when empty
            # If overall (no filter), return 0.0
            return np.nan if filter_octants is not None else 0.0

        # Filter interactions if requested
        if filter_octants is not None:
            filtered = [(c, t) for c, t in interactions if c in filter_octants]
            if not filtered:
                return np.nan
            interactions_to_check = filtered
        else:
            interactions_to_check = list(interactions)

        # Count complementary interactions
        matches = sum(
            1 for c, t in interactions_to_check
            if t == self.complement_map[c]
        )

        return 100.0 * matches / len(interactions_to_check)

    def get_overall_rate(self, perspective: str = 'enacted') -> float:
        """Get overall complementarity rate.

        Args:
            perspective: 'enacted', 'perceived', or 'both' (default: 'enacted')

        Returns:
            Complementarity rate as percentage (0-100)
        """
        if self._dual_tracking:
            if perspective == 'enacted':
                return self._calculate_complementarity_rate(self.recent_enacted)
            elif perspective == 'perceived':
                return self._calculate_complementarity_rate(self.recent_perceived)
            else:
                raise ValueError(f"Invalid perspective: {perspective}. Use 'enacted' or 'perceived'")
        else:
            return self._calculate_complementarity_rate(self.recent_interactions)

    def get_warm_rate(self, perspective: str = 'enacted') -> float:
        """Get complementarity rate for warm interactions only.

        Only considers interactions where client_action is in {1, 2, 3}.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Complementarity rate as percentage (0-100), or np.nan if no
            warm interactions in the current window
        """
        if self._dual_tracking:
            if perspective == 'enacted':
                return self._calculate_complementarity_rate(
                    self.recent_enacted, WARM_OCTANTS
                )
            elif perspective == 'perceived':
                return self._calculate_complementarity_rate(
                    self.recent_perceived, WARM_OCTANTS
                )
            else:
                raise ValueError(f"Invalid perspective: {perspective}")
        else:
            return self._calculate_complementarity_rate(
                self.recent_interactions, WARM_OCTANTS
            )

    def get_cold_rate(self, perspective: str = 'enacted') -> float:
        """Get complementarity rate for cold interactions only.

        Only considers interactions where client_action is in {5, 6, 7}.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Complementarity rate as percentage (0-100), or np.nan if no
            cold interactions in the current window
        """
        if self._dual_tracking:
            if perspective == 'enacted':
                return self._calculate_complementarity_rate(
                    self.recent_enacted, COLD_OCTANTS
                )
            elif perspective == 'perceived':
                return self._calculate_complementarity_rate(
                    self.recent_perceived, COLD_OCTANTS
                )
            else:
                raise ValueError(f"Invalid perspective: {perspective}")
        else:
            return self._calculate_complementarity_rate(
                self.recent_interactions, COLD_OCTANTS
            )

    def get_all_rates(self, perspective: str = 'enacted') -> Tuple[float, float, float]:
        """Get overall, warm, and cold complementarity rates.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Tuple of (overall_rate, warm_rate, cold_rate)
        """
        return (
            self.get_overall_rate(perspective),
            self.get_warm_rate(perspective),
            self.get_cold_rate(perspective)
        )

    def _calculate_mean_distance(
        self,
        interactions: deque,
        filter_octants: Optional[set] = None
    ) -> float:
        """Calculate mean octant distance from complementary action.

        Args:
            interactions: Deque of (client_action, therapist_action) tuples
            filter_octants: If provided, only consider interactions where
                client_action is in this set

        Returns:
            Mean distance (0-4), or np.nan if no interactions match the filter
        """
        if not interactions:
            return np.nan if filter_octants is not None else np.nan

        # Filter interactions if requested
        if filter_octants is not None:
            filtered = [(c, t) for c, t in interactions if c in filter_octants]
            if not filtered:
                return np.nan
            interactions_to_check = filtered
        else:
            interactions_to_check = list(interactions)

        # Calculate mean distance
        distances = [
            complementarity_distance(c, t)
            for c, t in interactions_to_check
        ]

        return float(np.mean(distances))

    def get_overall_distance(self, perspective: str = 'enacted') -> float:
        """Get overall mean octant distance from complementary action.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Mean distance (0-4), where 0 is perfectly complementary
        """
        if self._dual_tracking:
            if perspective == 'enacted':
                return self._calculate_mean_distance(self.recent_enacted)
            elif perspective == 'perceived':
                return self._calculate_mean_distance(self.recent_perceived)
            else:
                raise ValueError(f"Invalid perspective: {perspective}. Use 'enacted' or 'perceived'")
        else:
            return self._calculate_mean_distance(self.recent_interactions)

    def get_warm_distance(self, perspective: str = 'enacted') -> float:
        """Get mean octant distance for warm interactions only.

        Only considers interactions where client_action is in {1, 2, 3}.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Mean distance (0-4), or np.nan if no warm interactions
        """
        if self._dual_tracking:
            if perspective == 'enacted':
                return self._calculate_mean_distance(self.recent_enacted, WARM_OCTANTS)
            elif perspective == 'perceived':
                return self._calculate_mean_distance(self.recent_perceived, WARM_OCTANTS)
            else:
                raise ValueError(f"Invalid perspective: {perspective}")
        else:
            return self._calculate_mean_distance(self.recent_interactions, WARM_OCTANTS)

    def get_cold_distance(self, perspective: str = 'enacted') -> float:
        """Get mean octant distance for cold interactions only.

        Only considers interactions where client_action is in {5, 6, 7}.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Mean distance (0-4), or np.nan if no cold interactions
        """
        if self._dual_tracking:
            if perspective == 'enacted':
                return self._calculate_mean_distance(self.recent_enacted, COLD_OCTANTS)
            elif perspective == 'perceived':
                return self._calculate_mean_distance(self.recent_perceived, COLD_OCTANTS)
            else:
                raise ValueError(f"Invalid perspective: {perspective}")
        else:
            return self._calculate_mean_distance(self.recent_interactions, COLD_OCTANTS)

    def get_all_distances(self, perspective: str = 'enacted') -> Tuple[float, float, float]:
        """Get overall, warm, and cold mean octant distances.

        Args:
            perspective: 'enacted' or 'perceived' (default: 'enacted')

        Returns:
            Tuple of (overall_distance, warm_distance, cold_distance)
        """
        return (
            self.get_overall_distance(perspective),
            self.get_warm_distance(perspective),
            self.get_cold_distance(perspective)
        )

    def reset(self) -> None:
        """Clear all tracked interactions."""
        self.recent_interactions.clear()
        self.recent_enacted.clear()
        self.recent_perceived.clear()
        self._dual_tracking = False
