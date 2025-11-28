"""
Imperfect perception system for CIIT-Tracey client agents.

Implements a two-stage perceptual distortion model where clients may misperceive
therapist actions based on their interaction history.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Protocol
import numpy as np
from numpy.typing import NDArray
from collections import deque

from src.config import (
    PERCEPTION_WINDOW,
    PERCEPTION_BASELINE_ACCURACY,
    PERCEPTION_HISTORY_WEIGHT,
    PERCEPTION_ADJACENCY_NOISE,
)


@dataclass
class PerceptionRecord:
    """
    Record of a single perception event for analysis and debugging.

    Attributes
    ----------
    client_action : int
        The action (octant 0-7) the client performed
    actual_therapist_action : int
        The ground truth therapist action (octant 0-7)
    perceived_therapist_action : int
        The final perceived action after both stages (octant 0-7)
    stage1_result : int
        The perceived action after Stage 1 but before Stage 2
    baseline_path_succeeded : bool
        Whether the baseline accuracy path (20%) was successful
    stage1_changed_from_actual : bool
        Whether Stage 1 changed the perception from actual
    stage2_shifted : bool
        Whether Stage 2 applied adjacency noise (±1 shift)
    computed_accuracy : float
        The frequency-based accuracy value used in Stage 1
    """
    client_action: int
    actual_therapist_action: int
    perceived_therapist_action: int
    stage1_result: int
    baseline_path_succeeded: bool
    stage1_changed_from_actual: bool
    stage2_shifted: bool
    computed_accuracy: float


class ClientAgentProtocol(Protocol):
    """Protocol defining the interface expected from base client classes."""
    memory: deque
    rng: np.random.RandomState
    
    def update_memory(self, client_action: int, therapist_action: int) -> None:
        ...


class PerceptualClientMixin:
    """
    Mixin class that adds imperfect perception to any client agent.

    This mixin overrides the `update_memory()` method to filter therapist
    actions through a two-stage perceptual distortion process before storing
    them in memory.

    Stage 1: History-based perception
    - baseline_accuracy chance: perceive correctly (baseline path)
    - (1 - baseline_accuracy) chance: perceive the most common therapist action
      in recent history. If multiple actions tie for most common, choose the
      most recently enacted one.

    Stage 2: Adjacency noise
    - PERCEPTION_ADJACENCY_NOISE chance: shift perceived octant by ±1

    Parameters
    ----------
    baseline_accuracy : float, default=PERCEPTION_BASELINE_ACCURACY
        Probability of correct perception via baseline path (typically 0.2)
    enable_perception : bool, default=True
        If False, perception is perfect (for control experiments)
    **kwargs
        Additional arguments passed to parent class

    Attributes
    ----------
    perception_history : List[PerceptionRecord]
        Complete history of all perception events for analysis
    """

    # Type hints for attributes provided by base client class
    memory: deque
    rng: np.random.RandomState

    def __init__(
        self,
        baseline_accuracy: float = PERCEPTION_BASELINE_ACCURACY,
        enable_perception: bool = True,
        **kwargs
    ):
        """Initialize the perceptual client mixin."""
        super().__init__(**kwargs)
        self.baseline_accuracy = baseline_accuracy
        self.enable_perception = enable_perception
        self.perception_history: List[PerceptionRecord] = []

    def _perceive_therapist_action(self, actual_action: int) -> Tuple[int, PerceptionRecord]:
        """
        Apply two-stage perceptual distortion to therapist action.

        Stage 1: History-based perception
        - baseline_accuracy chance: perceive correctly (baseline path)
        - (1 - baseline_accuracy) chance: perceive the most common therapist action
          in recent history. If multiple actions tie for most common, choose the
          most recently enacted one.

        Stage 2: Adjacency noise
        - PERCEPTION_ADJACENCY_NOISE chance: shift perceived octant by ±1

        Parameters
        ----------
        actual_action : int
            The ground truth therapist action (octant 0-7)

        Returns
        -------
        perceived_action : int
            The final perceived action after both stages
        record : PerceptionRecord
            Metadata about the perception process for this interaction
        """
        # Get last PERCEPTION_WINDOW interactions from memory
        # Memory always has at least 50 interactions (pre-populated)
        recent_memory = list(self.memory)[-PERCEPTION_WINDOW:]

        # Extract therapist actions and calculate frequency distribution
        therapist_actions = [interaction[1] for interaction in recent_memory]
        frequency = np.zeros(8)
        for action in therapist_actions:
            frequency[action] += 1
        frequency /= len(therapist_actions)  # Normalize to probabilities

        # Stage 1: History-based perception
        baseline_path_succeeded = False
        stage1_changed_from_actual = False

        # First check: baseline accuracy path (20% by default)
        if self.rng.random() < self.baseline_accuracy:
            # Baseline path: perceive correctly
            stage1_result = actual_action
            baseline_path_succeeded = True
            computed_accuracy = self.baseline_accuracy
        else:
            # History-based path: perceive most common action
            computed_accuracy = frequency[actual_action]  # For record-keeping only
            
            # Find the maximum frequency
            max_freq = frequency.max()
            
            # Find all actions with maximum frequency
            most_common_actions = np.where(frequency == max_freq)[0]
            
            # If tie, choose the most recently enacted one
            if len(most_common_actions) > 1:
                # Search backwards through recent memory to find most recent
                for action in reversed(therapist_actions):
                    if action in most_common_actions:
                        stage1_result = action
                        break
            else:
                stage1_result = most_common_actions[0]
            
            # Only mark as changed if we actually perceived something different
            stage1_changed_from_actual = (stage1_result != actual_action)

        # Stage 2: Adjacency noise (applies to ALL perceptions)
        stage2_shifted = False
        if self.rng.random() < PERCEPTION_ADJACENCY_NOISE:
            # Apply ±1 shift with equal probability
            shift = self.rng.choice([-1, 1])
            perceived_action = (stage1_result + shift) % 8  # Wrap around (0↔7)
            stage2_shifted = True
        else:
            perceived_action = stage1_result

        # Create perception record
        record = PerceptionRecord(
            client_action=self.memory[-1][0] if self.memory else -1,  # Most recent client action
            actual_therapist_action=actual_action,
            perceived_therapist_action=perceived_action,
            stage1_result=stage1_result,
            baseline_path_succeeded=baseline_path_succeeded,
            stage1_changed_from_actual=stage1_changed_from_actual,
            stage2_shifted=stage2_shifted,
            computed_accuracy=computed_accuracy,
        )

        return perceived_action, record

    def update_memory(self, client_action: int, therapist_action: int) -> None:
        """
        Update memory with perceived therapist action (or actual if perception disabled).

        Overrides the base class method to apply perceptual distortion before
        storing the interaction in memory.

        Parameters
        ----------
        client_action : int
            The action (octant 0-7) the client performed
        therapist_action : int
            The actual therapist action (octant 0-7)
        """
        if not self.enable_perception:
            # Perfect perception: store actual action
            super().update_memory(client_action, therapist_action)  # type: ignore[misc]
        else:
            # Imperfect perception: apply distortion
            perceived_action, record = self._perceive_therapist_action(therapist_action)

            # Update record with current client action (now that we have it)
            record.client_action = client_action

            # Store perception record for analysis
            self.perception_history.append(record)

            # Update memory with PERCEIVED action (client's subjective reality)
            super().update_memory(client_action, perceived_action)  # type: ignore[misc]

    def get_perception_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics about perception accuracy.

        Returns
        -------
        dict
            Dictionary containing perception statistics:
            - total_interactions: Number of perception events
            - total_misperceptions: Count of final misperceptions
            - overall_misperception_rate: Proportion of misperceptions
            - stage1_overridden_count: Times Stage 1 changed perception
            - stage1_override_rate: Proportion of Stage 1 changes
            - stage2_shifted_count: Times Stage 2 applied shift
            - stage2_shift_rate: Proportion of Stage 2 shifts
            - mean_computed_accuracy: Average frequency-based accuracy
            - baseline_correct_count: Times baseline path succeeded
        """
        if not self.perception_history:
            return {
                'total_interactions': 0,
                'total_misperceptions': 0,
                'overall_misperception_rate': 0.0,
                'stage1_overridden_count': 0,
                'stage1_override_rate': 0.0,
                'stage2_shifted_count': 0,
                'stage2_shift_rate': 0.0,
                'mean_computed_accuracy': 0.0,
                'baseline_correct_count': 0,
            }

        total = len(self.perception_history)

        # Count misperceptions (final perceived != actual)
        total_misperceptions = sum(
            1 for r in self.perception_history
            if r.perceived_therapist_action != r.actual_therapist_action
        )

        # Count Stage 1 changes
        stage1_overridden_count = sum(
            1 for r in self.perception_history
            if r.stage1_changed_from_actual
        )

        # Count Stage 2 shifts
        stage2_shifted_count = sum(
            1 for r in self.perception_history
            if r.stage2_shifted
        )

        # Count baseline path successes
        baseline_correct_count = sum(
            1 for r in self.perception_history
            if r.baseline_path_succeeded
        )

        # Calculate mean computed accuracy
        mean_computed_accuracy = np.mean([
            r.computed_accuracy for r in self.perception_history
        ])

        return {
            'total_interactions': total,
            'total_misperceptions': total_misperceptions,
            'overall_misperception_rate': total_misperceptions / total if total > 0 else 0.0,
            'stage1_overridden_count': stage1_overridden_count,
            'stage1_override_rate': stage1_overridden_count / total if total > 0 else 0.0,
            'stage2_shifted_count': stage2_shifted_count,
            'stage2_shift_rate': stage2_shifted_count / total if total > 0 else 0.0,
            'mean_computed_accuracy': float(mean_computed_accuracy),
            'baseline_correct_count': baseline_correct_count,
        }


def with_perception(client_class):
    """
    Factory function to create a perceptual variant of any client class.

    Creates a new class that inherits from both PerceptualClientMixin and
    the provided client class, enabling perception capabilities for any
    client mechanism (BondOnly, FrequencyAmplifier, ConditionalAmplifier, etc.).

    Parameters
    ----------
    client_class : type
        The base client class to add perception to

    Returns
    -------
    type
        A new class with perception capabilities

    Examples
    --------
    >>> from src.agents.client_agents import BondOnlyClient
    >>> PerceptualBondOnly = with_perception(BondOnlyClient)
    >>> client = PerceptualBondOnly(
    ...     u_matrix=my_matrix,
    ...     entropy=3.0,
    ...     initial_memory=my_memory,
    ...     baseline_accuracy=0.2,
    ...     enable_perception=True,
    ...     random_state=42
    ... )
    """
    class PerceptualClient(PerceptualClientMixin, client_class):
        pass

    # Set a descriptive name for the new class
    PerceptualClient.__name__ = f"Perceptual{client_class.__name__}"
    PerceptualClient.__qualname__ = f"Perceptual{client_class.__qualname__}"

    return PerceptualClient
