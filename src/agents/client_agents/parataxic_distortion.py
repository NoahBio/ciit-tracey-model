"""
Imperfect perception system for CIIT-Tracey client agents.

Implements a history-based parataxic distortion model where clients may misperceive
therapist actions based on their interaction history. This models Sullivan's concept
of parataxic distortion - the tendency to perceive and react to present relationships
through the lens of past experiences.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Protocol
import numpy as np
from numpy.typing import NDArray
from collections import deque

from src.config import (
    PARATAXIC_WINDOW,
    PARATAXIC_BASELINE_ACCURACY,
)


@dataclass
class ParataxicRecord:
    """
    Record of a single parataxic distortion event for analysis and debugging.

    Attributes
    ----------
    client_action : int
        The action (octant 0-7) the client performed
    actual_therapist_action : int
        The ground truth therapist action (octant 0-7)
    perceived_therapist_action : int
        The final perceived action after distortion (octant 0-7)
    stage1_result : int
        The perceived action after Stage 1 (history-based distortion)
    baseline_path_succeeded : bool
        Whether the baseline accuracy path (e.g., 20%) was successful
    stage1_changed_from_actual : bool
        Whether Stage 1 changed the perception from actual
    computed_accuracy : float
        The frequency-based accuracy value used in Stage 1
    """
    client_action: int
    actual_therapist_action: int
    perceived_therapist_action: int
    stage1_result: int
    baseline_path_succeeded: bool
    stage1_changed_from_actual: bool
    computed_accuracy: float


class ClientAgentProtocol(Protocol):
    """Protocol defining the interface expected from base client classes."""
    memory: deque
    rng: np.random.RandomState

    def update_memory(self, client_action: int, therapist_action: int) -> None:
        ...


class ParataxicClientMixin:
    """
    Mixin class that adds parataxic distortion to any client agent.

    This mixin overrides the `update_memory()` method to filter therapist
    actions through a history-based parataxic distortion process before storing
    them in memory. This models Sullivan's parataxic distortion - where clients
    perceive current therapist behaviors through the lens of past experiences.

    Stage 1: History-based parataxic distortion
    - baseline_accuracy chance: perceive correctly (baseline path)
    - (1 - baseline_accuracy) chance: perceive the most common therapist action
      in recent history. If multiple actions tie for most common, choose the
      most recently enacted one.

    Parameters
    ----------
    baseline_accuracy : float, default=PARATAXIC_BASELINE_ACCURACY
        Probability of correct perception via baseline path (typically 0.2-0.5)
    enable_parataxic : bool, default=True
        If False, perception is perfect (for control experiments)
    **kwargs
        Additional arguments passed to parent class

    Attributes
    ----------
    parataxic_history : List[ParataxicRecord]
        Complete history of all parataxic distortion events for analysis
    """

    # Type hints for attributes provided by base client class
    memory: deque
    rng: np.random.RandomState

    def __init__(
        self,
        baseline_accuracy: float = PARATAXIC_BASELINE_ACCURACY,
        enable_parataxic: bool = True,
        **kwargs
    ):
        """Initialize the parataxic distortion client mixin."""
        # Backward compatibility: map old parameter name to new one
        if 'enable_perception' in kwargs:
            enable_parataxic = kwargs.pop('enable_perception')

        super().__init__(**kwargs)
        self.baseline_accuracy = baseline_accuracy
        self.enable_parataxic = enable_parataxic
        self.parataxic_history: List[ParataxicRecord] = []

    def _apply_parataxic_distortion(self, actual_action: int) -> Tuple[int, ParataxicRecord]:
        """
        Apply history-based parataxic distortion to therapist action.

        Stage 1: History-based parataxic distortion
        - baseline_accuracy chance: perceive correctly (baseline path)
        - (1 - baseline_accuracy) chance: perceive the most common therapist action
          in recent history. If multiple actions tie for most common, choose the
          most recently enacted one.

        Parameters
        ----------
        actual_action : int
            The ground truth therapist action (octant 0-7)

        Returns
        -------
        perceived_action : int
            The final perceived action after distortion
        record : ParataxicRecord
            Metadata about the distortion process for this interaction
        """
        # Get last PARATAXIC_WINDOW interactions from memory
        # Memory always has at least 50 interactions (pre-populated)
        recent_memory = list(self.memory)[-PARATAXIC_WINDOW:]

        # Extract therapist actions and calculate frequency distribution
        therapist_actions = [interaction[1] for interaction in recent_memory]
        frequency = np.zeros(8)
        for action in therapist_actions:
            frequency[action] += 1
        frequency /= len(therapist_actions)  # Normalize to probabilities

        # Stage 1: History-based parataxic distortion
        baseline_path_succeeded = False
        stage1_changed_from_actual = False

        # First check: baseline accuracy path (20-50% by default)
        if self.rng.random() < self.baseline_accuracy:
            # Baseline path: perceive correctly
            stage1_result = actual_action
            baseline_path_succeeded = True
            computed_accuracy = self.baseline_accuracy
        else:
            # History-based path: perceive most common action (parataxic distortion)
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

        # Final perceived action (only Stage 1 distortion, no adjacency noise)
        perceived_action = stage1_result

        # Create parataxic distortion record
        record = ParataxicRecord(
            client_action=self.memory[-1][0] if self.memory else -1,  # Most recent client action
            actual_therapist_action=actual_action,
            perceived_therapist_action=perceived_action,
            stage1_result=stage1_result,
            baseline_path_succeeded=baseline_path_succeeded,
            stage1_changed_from_actual=stage1_changed_from_actual,
            computed_accuracy=computed_accuracy,
        )

        return perceived_action, record

    def update_memory(self, client_action: int, therapist_action: int) -> None:
        """
        Update memory with perceived therapist action (or actual if distortion disabled).

        Overrides the base class method to apply parataxic distortion before
        storing the interaction in memory.

        Parameters
        ----------
        client_action : int
            The action (octant 0-7) the client performed
        therapist_action : int
            The actual therapist action (octant 0-7)
        """
        if not self.enable_parataxic:
            # Perfect perception: store actual action
            super().update_memory(client_action, therapist_action)  # type: ignore[misc]
        else:
            # Parataxic distortion: apply history-based distortion
            perceived_action, record = self._apply_parataxic_distortion(therapist_action)

            # Update record with current client action (now that we have it)
            record.client_action = client_action

            # Store parataxic distortion record for analysis
            self.parataxic_history.append(record)

            # Update memory with PERCEIVED action (client's subjective reality)
            super().update_memory(client_action, perceived_action)  # type: ignore[misc]

    def get_parataxic_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics about parataxic distortion accuracy.

        Returns
        -------
        dict
            Dictionary containing parataxic distortion statistics:
            - total_interactions: Number of distortion events
            - total_misperceptions: Count of final misperceptions
            - overall_misperception_rate: Proportion of misperceptions
            - stage1_overridden_count: Times Stage 1 changed perception
            - stage1_override_rate: Proportion of Stage 1 changes
            - mean_computed_accuracy: Average frequency-based accuracy
            - baseline_correct_count: Times baseline path succeeded
        """
        if not self.parataxic_history:
            return {
                'total_interactions': 0,
                'total_misperceptions': 0,
                'overall_misperception_rate': 0.0,
                'stage1_overridden_count': 0,
                'stage1_override_rate': 0.0,
                'mean_computed_accuracy': 0.0,
                'baseline_correct_count': 0,
            }

        total = len(self.parataxic_history)

        # Count misperceptions (final perceived != actual)
        total_misperceptions = sum(
            1 for r in self.parataxic_history
            if r.perceived_therapist_action != r.actual_therapist_action
        )

        # Count Stage 1 changes
        stage1_overridden_count = sum(
            1 for r in self.parataxic_history
            if r.stage1_changed_from_actual
        )

        # Count baseline path successes
        baseline_correct_count = sum(
            1 for r in self.parataxic_history
            if r.baseline_path_succeeded
        )

        # Calculate mean computed accuracy
        mean_computed_accuracy = np.mean([
            r.computed_accuracy for r in self.parataxic_history
        ])

        return {
            'total_interactions': total,
            'total_misperceptions': total_misperceptions,
            'overall_misperception_rate': total_misperceptions / total if total > 0 else 0.0,
            'stage1_overridden_count': stage1_overridden_count,
            'stage1_override_rate': stage1_overridden_count / total if total > 0 else 0.0,
            'mean_computed_accuracy': float(mean_computed_accuracy),
            'baseline_correct_count': baseline_correct_count,
        }

    # Backward compatibility property aliases (deprecated)
    @property
    def perception_history(self):
        """Backward compatibility alias for parataxic_history."""
        return self.parataxic_history

    def get_perception_stats(self):
        """Backward compatibility alias for get_parataxic_stats()."""
        return self.get_parataxic_stats()


def with_parataxic(client_class):
    """
    Factory function to create a parataxic variant of any client class.

    Creates a new class that inherits from both ParataxicClientMixin and
    the provided client class, enabling parataxic distortion capabilities for any
    client mechanism (BondOnly, FrequencyAmplifier, ConditionalAmplifier, etc.).

    Parameters
    ----------
    client_class : type
        The base client class to add parataxic distortion to

    Returns
    -------
    type
        A new class with parataxic distortion capabilities

    Examples
    --------
    >>> from src.agents.client_agents import BondOnlyClient
    >>> ParataxicBondOnly = with_parataxic(BondOnlyClient)
    >>> client = ParataxicBondOnly(
    ...     u_matrix=my_matrix,
    ...     entropy=3.0,
    ...     initial_memory=my_memory,
    ...     baseline_accuracy=0.2,
    ...     enable_parataxic=True,
    ...     random_state=42
    ... )
    """
    class ParataxicClient(ParataxicClientMixin, client_class):
        pass

    # Set a descriptive name for the new class
    ParataxicClient.__name__ = f"Parataxic{client_class.__name__}"
    ParataxicClient.__qualname__ = f"Parataxic{client_class.__qualname__}"

    return ParataxicClient


# Backward compatibility aliases (deprecated - use parataxic versions)
with_perception = with_parataxic
PerceptionRecord = ParataxicRecord  # Alias for backward compatibility
PerceptualClientMixin = ParataxicClientMixin  # Alias for backward compatibility
