"""Omniscient Strategic Therapist with perfect client knowledge.

This therapist agent has access to client internals and uses strategic
"perceptual seeding" to help clients escape maladaptive patterns.

The strategy operates in three phases:
1. Relationship Building: Pure complementarity to build bond and prevent dropout
2. Ladder-Climbing: Strategic seeding to enable higher-utility interactions
3. Consolidation: Complement at new level until next step opens

The therapist reads the client's PERCEIVED memory (after parataxic distortion)
to understand what the client actually stored, not just what the therapist did.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ActionLogEntry:
    """Record of therapist decision for logging/analysis."""
    session: int
    client_action: int
    therapist_action: int
    phase: str
    rationale: str
    target_client_action: Optional[int] = None
    target_therapist_action: Optional[int] = None
    seeding_progress: Optional[Dict[str, Any]] = None
    perception_accuracy: Optional[float] = None


class OmniscientStrategicTherapist:
    """Strategic therapist with perfect client knowledge.

    This therapist has omniscient access to client internals and uses this
    knowledge to strategically seed perceptions, enabling clients to escape
    maladaptive patterns.

    Attributes:
        COMPLEMENT_MAP: Mapping from client octant to complementary therapist octant
        client_ref: Reference to client agent (for reading state)
        perception_window: Size of perception window for parataxic distortion
        baseline_accuracy: Baseline perception accuracy parameter
        phase: Current strategy phase ('relationship_building', 'ladder_climbing', 'consolidation')
        action_log: History of decisions with rationale
    """

    # Complementary action mapping (IPC octant system)
    # Control: D↔S (dominant ↔ submissive)
    # Affiliation: W↔W, C↔C (warm ↔ warm, cold ↔ cold)
    COMPLEMENT_MAP = {
        0: 4,  # D → S
        1: 3,  # WD → WS
        2: 2,  # W → W
        3: 1,  # WS → WD
        4: 0,  # S → D
        5: 7,  # CS → CD
        6: 6,  # C → C
        7: 5,  # CD → CS
    }

    def __init__(
        self,
        client_ref,
        perception_window: int = 15,
        baseline_accuracy: float = 0.2,
    ):
        """Initialize the omniscient strategic therapist.

        Args:
            client_ref: Reference to client agent (for reading state)
            perception_window: Size of perception window for parataxic distortion
            baseline_accuracy: Baseline perception accuracy parameter
        """
        self.client_ref = client_ref
        self.perception_window = perception_window
        self.baseline_accuracy = baseline_accuracy

        # Phase management
        self.phase = "relationship_building"

        # Current target (set when entering ladder_climbing phase)
        self.current_target_client_action: Optional[int] = None
        self.current_target_therapist_action: Optional[int] = None

        # Tracking
        self.action_log: List[ActionLogEntry] = []
        self.session_count = 0

        # Track actual actions taken (for comparing with perceived)
        self.actual_actions_taken: List[int] = []

    def _get_complementary_action(self, client_action: int) -> int:
        """Return the complementary action for client_action."""
        return self.COMPLEMENT_MAP[client_action]

    def _get_client_perceived_memory(self) -> List[Tuple[int, int]]:
        """Get client's memory (which contains PERCEIVED therapist actions).

        The client stores perceived actions, not actual ones.
        This IS what shapes their expectations.

        Returns:
            List of (client_action, perceived_therapist_action) tuples
            from recent memory (limited to perception_window).
        """
        return list(self.client_ref.memory)[-self.perception_window:]

    def _estimate_perception_accuracy(self, actual_action: int) -> float:
        """Estimate probability client perceives `actual_action` correctly.

        The parataxic distortion model works as:
        - baseline_accuracy chance: Client perceives the ACTUAL action correctly
        - (1 - baseline_accuracy) chance: Client perceives the MOST COMMON
          action in their perceived memory window

        Args:
            actual_action: The action the therapist will take

        Returns:
            Estimated probability client perceives actual_action correctly
        """
        perceived_memory = self._get_client_perceived_memory()
        if not perceived_memory:
            return self.baseline_accuracy

        therapist_actions = [t for c, t in perceived_memory]

        # Count occurrences in PERCEIVED memory
        counts = {a: therapist_actions.count(a) for a in set(therapist_actions)}
        max_count = max(counts.values()) if counts else 0

        # Baseline accuracy always applies
        # If actual_action is (tied for) most common → also gets history path
        if counts.get(actual_action, 0) == max_count:
            # Both paths lead to correct perception
            return self.baseline_accuracy + (1 - self.baseline_accuracy) * 1.0
        else:
            # Only baseline path leads to correct perception
            return self.baseline_accuracy

    def calculate_seeding_requirement(self, target_action: int) -> Dict[str, Any]:
        """Calculate actions needed to make `target_action` dominant in perceived memory.

        Note: This operates on PERCEIVED memory. Seeding actions may themselves
        be misperceived, so actual seeding progress depends on baseline_accuracy.

        Args:
            target_action: The therapist action we want to make dominant

        Returns:
            Dictionary with seeding metrics:
            - target_action: The target octant
            - current_count_in_perceived_memory: Current count of target in memory
            - max_other_count: Max count of any other action
            - raw_seeding_needed: Raw count needed (ignoring misperception)
            - adjusted_seeding_needed: Adjusted for expected misperception
            - perception_accuracy_estimate: Current accuracy estimate
        """
        perceived_memory = self._get_client_perceived_memory()
        if not perceived_memory:
            return {
                'target_action': target_action,
                'current_count_in_perceived_memory': 0,
                'max_other_count': 0,
                'raw_seeding_needed': self.perception_window // 2 + 1,
                'adjusted_seeding_needed': self.perception_window // 2 + 1,
                'perception_accuracy_estimate': self.baseline_accuracy,
            }

        therapist_actions = [t for c, t in perceived_memory]

        current_count = therapist_actions.count(target_action)

        # Find max count of any other action in perceived memory
        other_counts = [therapist_actions.count(a) for a in range(8) if a != target_action]
        max_other_count = max(other_counts) if other_counts else 0

        # Need target to exceed max_other_count to be "most common"
        required_count = max_other_count + 1

        # Raw needed (ignoring misperception)
        raw_needed = max(0, required_count - current_count)

        # Adjust for expected misperception (conservative estimate)
        # If baseline_accuracy = 0.5, roughly half our seeding actions succeed
        expected_success_rate = self.baseline_accuracy
        if expected_success_rate < 1.0 and raw_needed > 0:
            adjusted_needed = int(np.ceil(raw_needed / expected_success_rate))
        else:
            adjusted_needed = raw_needed

        return {
            'target_action': target_action,
            'current_count_in_perceived_memory': current_count,
            'max_other_count': max_other_count,
            'raw_seeding_needed': raw_needed,
            'adjusted_seeding_needed': adjusted_needed,
            'perception_accuracy_estimate': self._estimate_perception_accuracy(target_action),
        }

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        # Subtract max for numerical stability
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    def _identify_target_interaction(self) -> bool:
        """Find best (client_action, therapist_action) pair to work toward.

        This is called when entering or continuing ladder_climbing phase.
        Scores each possible interaction by:
        - utility_improvement: How much better than current RS
        - probability: How likely client is to take that action

        Returns:
            True if a valid target was found, False if no improving target exists
        """
        u_matrix = self.client_ref.u_matrix
        current_rs = self.client_ref.relationship_satisfaction

        # Get client action probabilities at current bond
        payoffs = self.client_ref._calculate_expected_payoffs()
        probs = self._softmax(payoffs / self.client_ref.entropy)

        best_score = float('-inf')
        best_client_action = None
        best_therapist_action = None

        for client_oct in range(8):
            complement = self.COMPLEMENT_MAP[client_oct]
            utility = u_matrix[client_oct, complement]

            # Skip if utility doesn't improve over current RS
            if utility <= current_rs:
                continue

            # Score = utility_improvement × P(client_action)
            improvement = utility - current_rs
            prob = probs[client_oct]

            score = improvement * prob

            if score > best_score:
                best_score = score
                best_client_action = client_oct
                best_therapist_action = complement

        if best_client_action is not None:
            self.current_target_client_action = best_client_action
            self.current_target_therapist_action = best_therapist_action
            return True
        else:
            # No improving target found - stay in current phase
            self.current_target_client_action = None
            self.current_target_therapist_action = None
            return False

    def _should_start_ladder_climbing(self) -> bool:
        """Check if we should transition from relationship_building to ladder_climbing.

        Conditions:
        - Bond is above a safety threshold (to avoid dropout risk)
        - We're past the critical dropout check session (session 10)
        - There exists a reachable higher-utility interaction
        - Seeding would actually be beneficial (not just always complement)

        Returns:
            True if we should start ladder climbing
        """
        # Safety: Don't start ladder-climbing before dropout check (session 10)
        if self.session_count < 10:
            return False

        # Check if bond is high enough (prevent risk of retreat)
        # Lower threshold to start seeding earlier (before cold history dominates)
        bond_threshold = 0.1  # Start earlier
        if self.client_ref.bond < bond_threshold:
            return False

        # Check if there's a beneficial target to climb toward
        if not self._identify_target_interaction():
            return False

        # Check if seeding would actually help
        # If perception accuracy is already high, we don't need ladder-climbing
        if self.current_target_therapist_action is not None:
            seeding_req = self.calculate_seeding_requirement(self.current_target_therapist_action)
            if seeding_req['perception_accuracy_estimate'] > 0.9:
                return False

            # If no seeding is needed, stay in relationship_building
            if seeding_req['adjusted_seeding_needed'] == 0:
                return False

        return True

    def _new_ladder_step_available(self) -> bool:
        """Check if a new, higher ladder step has become available.

        This is called during consolidation phase to see if we should
        re-enter ladder_climbing.

        Returns:
            True if a new target is found that improves over current state
        """
        return self._identify_target_interaction()

    def _should_update_target(self, client_action: int) -> bool:
        """Check if client's unexpected action should trigger target update.

        If client took an action that leads to better utility than current
        target, we should adapt our strategy.

        Args:
            client_action: The action client just took

        Returns:
            True if we should reconsider our target
        """
        if self.current_target_client_action is None:
            return True

        # Check if client's action + complement yields better utility
        complement = self._get_complementary_action(client_action)
        utility = self.client_ref.u_matrix[client_action, complement]

        # Compare to current target utility
        current_target_utility = self.client_ref.u_matrix[
            self.current_target_client_action,
            self.current_target_therapist_action
        ]

        return utility > current_target_utility

    def _current_interaction_achieves_success(self) -> bool:
        """Check if current complementary interaction achieves therapy success.

        If the most likely current interaction already leads to success,
        there's no need for ladder-climbing.

        Returns:
            True if staying complementary will likely achieve success
        """
        # Get success threshold
        success_threshold = getattr(self.client_ref, 'success_threshold', None)
        if success_threshold is None:
            return False

        # Check if current RS is already above threshold
        if self.client_ref.relationship_satisfaction >= success_threshold:
            return True

        return False

    def _is_seeding_beneficial(self, client_action: int) -> bool:
        """Determine if seeding is worth the cost at this moment.

        Seeding has a cost: we don't complement, so we may get lower utility.
        It's only worth it if:
        1. The complement equals the seeding target (free seeding!)
        2. The expected future value of seeding exceeds current cost
        3. We haven't reached a point where seeding is futile

        The key insight is that seeding changes what gets stored in memory,
        which affects future expectations, which affects future action selection.

        Args:
            client_action: Current client action

        Returns:
            True if we should seed, False if we should complement
        """
        if self.current_target_therapist_action is None:
            return False

        complement = self._get_complementary_action(client_action)

        # Case 1: Complement IS the seeding target - always do it!
        if complement == self.current_target_therapist_action:
            return True

        # Case 2: Calculate costs and benefits
        u_matrix = self.client_ref.u_matrix

        # Utility if we complement
        utility_complement = u_matrix[client_action, complement]

        # Utility if we seed
        utility_seed = u_matrix[client_action, self.current_target_therapist_action]

        # Check seeding progress
        seeding_req = self.calculate_seeding_requirement(self.current_target_therapist_action)
        perception_accuracy = seeding_req['perception_accuracy_estimate']
        actions_needed = seeding_req['adjusted_seeding_needed']

        # If perception is already high, no need to seed
        if perception_accuracy > 0.9:
            return False

        # If very few actions needed, just do it
        if actions_needed <= 3:
            return True

        # Calculate expected value of completing seeding
        target_utility = u_matrix[
            self.current_target_client_action,
            self.current_target_therapist_action
        ]
        current_rs = self.client_ref.relationship_satisfaction

        # Expected improvement per session once seeding is complete
        expected_improvement = target_utility - current_rs

        # If target doesn't improve RS, don't seed
        if expected_improvement <= 0:
            return False

        # How many sessions until we benefit?
        sessions_until_benefit = actions_needed + 3

        # Remaining sessions in therapy
        remaining_sessions = 100 - self.session_count

        # If not enough time to benefit, don't seed
        if remaining_sessions < sessions_until_benefit:
            return False

        # Calculate total expected value (more aggressive), scaling factor 0.3 determined through tuning
        benefit_sessions = remaining_sessions - sessions_until_benefit
        total_expected_value = expected_improvement * benefit_sessions * 0.3

        # Calculate total cost of seeding
        utility_cost_per_action = max(0, utility_complement - utility_seed)
        total_seeding_cost = utility_cost_per_action * actions_needed

        # More aggressive: seed if expected value exceeds cost
        return total_expected_value > total_seeding_cost

    def decide_action(self, client_action: int, session: int) -> Tuple[int, Dict[str, Any]]:
        """Decide therapist action with full omniscience.

        This is the main decision method that implements the phased strategy:
        1. Relationship Building: Pure complementarity
        2. Ladder-Climbing: Strategic seeding (smart - considers cost/benefit)
        3. Consolidation: Complement at current level

        The key insight is that seeding should only happen when it doesn't hurt
        too much. We prefer to complement when possible, especially when the
        complement action helps seeding (e.g., W-W interactions seed for Warm).

        Args:
            client_action: The action the client just took (octant 0-7)
            session: Current session number

        Returns:
            Tuple of (action, metadata) where:
            - action: int (octant 0-7)
            - metadata: dict with reasoning for logging/analysis
        """
        self.session_count = session
        complement = self._get_complementary_action(client_action)

        # Build metadata for logging
        metadata: Dict[str, Any] = {
            'session': session,
            'client_action': client_action,
            'phase': self.phase,
            'bond': self.client_ref.bond,
            'rs': self.client_ref.relationship_satisfaction,
        }

        # Check stop condition: current interaction achieves success
        if self._current_interaction_achieves_success():
            self.phase = "consolidation"
            metadata['rationale'] = "Current interaction achieves success - pure complementarity"
            metadata['therapist_action'] = complement
            self._log_action(session, client_action, complement, "consolidation",
                           "Success threshold reachable - pure complementarity")
            self.actual_actions_taken.append(complement)
            return complement, metadata

        # Phase 1: Relationship Building (Pure Complementarity)
        if self.phase == "relationship_building":
            if self._should_start_ladder_climbing():
                self.phase = "ladder_climbing"
                metadata['phase'] = "ladder_climbing"
                metadata['rationale'] = "Bond sufficient, starting ladder-climbing"
                metadata['target_client_action'] = self.current_target_client_action
                metadata['target_therapist_action'] = self.current_target_therapist_action
                # Continue to ladder_climbing logic below
            else:
                metadata['rationale'] = "Building relationship - pure complementarity"
                metadata['therapist_action'] = complement
                self._log_action(session, client_action, complement, "relationship_building",
                               "Building bond via complementarity")
                self.actual_actions_taken.append(complement)
                return complement, metadata

        # Phase 3: Consolidation (Complement at Current Level)
        if self.phase == "consolidation":
            if self._new_ladder_step_available():
                self.phase = "ladder_climbing"
                metadata['phase'] = "ladder_climbing"
                metadata['rationale'] = "New ladder step available"
                metadata['target_client_action'] = self.current_target_client_action
                metadata['target_therapist_action'] = self.current_target_therapist_action
                # Continue to ladder_climbing logic below
            else:
                metadata['rationale'] = "Consolidating at current level - complementarity"
                metadata['therapist_action'] = complement
                self._log_action(session, client_action, complement, "consolidation",
                               "Maximizing RS/Bond at current level")
                self.actual_actions_taken.append(complement)
                return complement, metadata

        # Phase 2: Ladder-Climbing (Strategic Seeding)
        if self.phase == "ladder_climbing":
            # If client took target action → complement! (goal achieved)
            if client_action == self.current_target_client_action:
                self.phase = "consolidation"
                metadata['phase'] = "consolidation"
                metadata['rationale'] = "Client took target action - complementing and consolidating"
                metadata['therapist_action'] = complement
                self._log_action(session, client_action, complement, "ladder_climbing",
                               f"Target {self.current_target_client_action} achieved - complementing")
                self.actual_actions_taken.append(complement)
                return complement, metadata

            # If client took action that could lead to BETTER utility, adapt
            if self._should_update_target(client_action):
                old_target = self.current_target_client_action
                if self._identify_target_interaction():
                    metadata['rationale'] = f"Adapted target from {old_target} to {self.current_target_client_action}"
                    # If new target matches current action, complement
                    if client_action == self.current_target_client_action:
                        self.phase = "consolidation"
                        metadata['phase'] = "consolidation"
                        metadata['therapist_action'] = complement
                        self._log_action(session, client_action, complement, "ladder_climbing",
                                       "New target matches current action - complementing")
                        self.actual_actions_taken.append(complement)
                        return complement, metadata

            # Smart seeding decision: seed only when beneficial
            if self.current_target_therapist_action is not None:
                seeding_req = self.calculate_seeding_requirement(self.current_target_therapist_action)
                metadata['seeding_progress'] = seeding_req
                metadata['target_client_action'] = self.current_target_client_action
                metadata['target_therapist_action'] = self.current_target_therapist_action

                # Decide: seed or complement?
                if self._is_seeding_beneficial(client_action):
                    # Seed: play target therapist action
                    metadata['rationale'] = f"Strategic seeding for {self.current_target_therapist_action}"
                    metadata['therapist_action'] = self.current_target_therapist_action
                    self._log_action(
                        session, client_action, self.current_target_therapist_action,
                        "ladder_climbing",
                        f"Seeding: {seeding_req['adjusted_seeding_needed']} actions needed",
                        seeding_progress=seeding_req
                    )
                    self.actual_actions_taken.append(self.current_target_therapist_action)
                    return self.current_target_therapist_action, metadata
                else:
                    # Complement: seeding cost too high, maintain RS instead
                    metadata['rationale'] = f"Complementing (seeding cost too high)"
                    metadata['therapist_action'] = complement
                    self._log_action(
                        session, client_action, complement,
                        "ladder_climbing",
                        "Complementing - seeding cost exceeds benefit",
                        seeding_progress=seeding_req
                    )
                    self.actual_actions_taken.append(complement)
                    return complement, metadata

        # Default: complement (fallback)
        metadata['rationale'] = "Default fallback - complementarity"
        metadata['therapist_action'] = complement
        self._log_action(session, client_action, complement, self.phase,
                       "Default complementarity")
        self.actual_actions_taken.append(complement)
        return complement, metadata

    def _log_action(
        self,
        session: int,
        client_action: int,
        therapist_action: int,
        phase: str,
        rationale: str,
        seeding_progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log action decision for later analysis."""
        entry = ActionLogEntry(
            session=session,
            client_action=client_action,
            therapist_action=therapist_action,
            phase=phase,
            rationale=rationale,
            target_client_action=self.current_target_client_action,
            target_therapist_action=self.current_target_therapist_action,
            seeding_progress=seeding_progress,
            perception_accuracy=self._estimate_perception_accuracy(therapist_action),
        )
        self.action_log.append(entry)

    def get_action_log(self) -> List[ActionLogEntry]:
        """Return the action log for analysis."""
        return self.action_log

    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of time spent in each phase."""
        phase_counts = {
            'relationship_building': 0,
            'ladder_climbing': 0,
            'consolidation': 0,
        }

        for entry in self.action_log:
            if entry.phase in phase_counts:
                phase_counts[entry.phase] += 1

        return {
            'phase_counts': phase_counts,
            'total_sessions': len(self.action_log),
            'current_phase': self.phase,
        }

    def get_seeding_summary(self) -> Dict[str, Any]:
        """Get summary of seeding activity."""
        seeding_sessions = [e for e in self.action_log if e.seeding_progress is not None]

        if not seeding_sessions:
            return {
                'total_seeding_sessions': 0,
                'seeding_actions': {},
            }

        seeding_actions: Dict[int, int] = {}
        for entry in seeding_sessions:
            target = entry.target_therapist_action
            if target is not None:
                seeding_actions[target] = seeding_actions.get(target, 0) + 1

        return {
            'total_seeding_sessions': len(seeding_sessions),
            'seeding_actions': seeding_actions,
        }

    def reset(self) -> None:
        """Reset therapist state for new episode."""
        self.phase = "relationship_building"
        self.current_target_client_action = None
        self.current_target_therapist_action = None
        self.action_log = []
        self.session_count = 0
        self.actual_actions_taken = []
