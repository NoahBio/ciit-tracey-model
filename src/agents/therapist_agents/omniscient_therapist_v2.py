"""Omniscient Strategic Therapist with perfect client knowledge.

This therapist agent has access to client internals and uses strategic
"perceptual seeding" to help clients escape maladaptive patterns.

The strategy operates in three phases:
1. Relationship Building: Pure complementarity to build bond and prevent dropout
2. Ladder-Climbing: Strategic seeding to enable higher-utility interactions
3. Consolidation: Complement at new level until next step opens

The therapist reads the client's PERCEIVED memory (after parataxic distortion)
to understand what the client actually stored, not just what the therapist did.

Forward Projection for Target Selection:
----------------------------------------
When selecting which (client_action, therapist_action) target to pursue, the
therapist uses PROJECTED probabilities rather than current probabilities. This
accounts for how seeding will shift client behavior:

1. Seeding changes the client's perceived memory
2. Changed memory changes the frequency distribution of therapist actions
3. Changed frequencies affect the client's expected payoffs (via amplification)
4. Changed expected payoffs change which client actions become more/less likely

The projection simulates what the client's action distribution would look like
AFTER successful seeding, enabling the therapist to select targets that will
actually be reachable once seeding is complete.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque, Counter
import numpy as np

from src.config import get_memory_weights, MAX_SESSIONS, PARATAXIC_WINDOW


@dataclass
class FeedbackRecord:
    """Record of a single seeding attempt and its outcome."""
    session: int
    intended_action: int                    # What therapist tried to seed
    actual_perceived: int                   # What client actually perceived
    success: bool                           # Did client perceive correctly?
    client_action: int                      # What client did
    target_count_before: int                # Count of target in memory before
    target_count_after: int                 # Count of target in memory after
    competitor_counts_after: Dict[int, int] # Frequency of each competitor after
    competitor_gained_most: Optional[int] = None   # Which competitor gained most
    competitor_gain_amount: int = 0         # How much it gained


@dataclass
class SeedingMonitor:
    """Real-time feedback monitoring for seeding effectiveness."""
    target_action: int                      # Target therapist action
    target_client_action: int               # Target client action

    # Initial snapshot
    initial_target_count: int
    initial_competitor_counts: Dict[int, int]
    initial_memory_snapshot: List[Tuple[int, int]]

    # Progress tracking
    seeding_attempts: int = 0
    successful_seeds: int = 0
    failed_seeds: int = 0
    consecutive_failures: int = 0
    net_progress: int = 0

    # Competitor monitoring
    competitor_boost_events: List[Tuple[int, int]] = field(default_factory=list)
    max_competitor_at_start: int = 0
    current_max_competitor: int = 0

    # Feedback history (last 10 sessions)
    recent_feedback: deque = field(default_factory=lambda: deque(maxlen=10))

    # Timing
    started_session: int = 0
    last_recalculation_session: int = 0


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

    # Feedback monitoring fields
    seeding_feedback: Optional[FeedbackRecord] = None
    monitor_decision: Optional[str] = None  # "recalculated", "aborted", "continuing"
    abort_reason: Optional[str] = None


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
        # Defaults from best Optuna trial (rank 1, trial 2643: 88% vs 73.3% baseline)
        perception_window: int | None = None,  # Defaults to PARATAXIC_WINDOW
        baseline_accuracy: float = 0.5549619551286054,
        seeding_benefit_scaling: float = 1.8658722646107764,
        skip_seeding_accuracy_threshold: float = 0.814677493978211,
        quick_seed_actions_threshold: int = 1,
        abort_consecutive_failures_threshold: int = 4,
        max_sessions: int | None = None,  # Defaults to MAX_SESSIONS from config
    ):
        """Initialize the omniscient strategic therapist.

        Args:
            client_ref: Reference to client agent (for reading state)
            perception_window: Size of parataxic window for seeding (default: PARATAXIC_WINDOW from config)
            baseline_accuracy: Baseline perception accuracy parameter (default: 0.555)
            seeding_benefit_scaling: Scaling factor for expected seeding benefit (default: 1.87)
            skip_seeding_accuracy_threshold: Skip seeding if accuracy above this (default: 0.815)
            quick_seed_actions_threshold: "Just do it" if actions_needed <= this (default: 1)
            abort_consecutive_failures_threshold: Abort after this many failures (default: 4)
            max_sessions: Maximum therapy sessions (default: MAX_SESSIONS from config)
        """
        self.client_ref = client_ref
        self.perception_window = perception_window if perception_window is not None else PARATAXIC_WINDOW
        self.baseline_accuracy = baseline_accuracy
        self.max_sessions = max_sessions if max_sessions is not None else MAX_SESSIONS

        # Seeding strategy hyperparameters
        self.seeding_benefit_scaling = seeding_benefit_scaling
        self.skip_seeding_accuracy_threshold = skip_seeding_accuracy_threshold
        self.quick_seed_actions_threshold = quick_seed_actions_threshold
        self.abort_consecutive_failures_threshold = abort_consecutive_failures_threshold

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

        # Feedback monitoring
        self.seeding_monitor: Optional[SeedingMonitor] = None
        self.last_seeding_action: Optional[int] = None
        self.last_session_number: int = 0
        self.failed_targets: List[int] = []
        self.max_target_failures: int = 3

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
        # If most common actions in memory are tied → follow recency-based client logic
        # Baseline accuracy always applies; history path follows parataxic tie-break (most recent among tied modes)
        tied_actions = {a for a, cnt in counts.items() if cnt == max_count}
        recent_mode = None
        for action in reversed(therapist_actions):
            if action in tied_actions:
                recent_mode = action
                break

        history_correct = 1.0 if (recent_mode is not None and actual_action == recent_mode) else 0.0
        return self.baseline_accuracy + (1 - self.baseline_accuracy) * history_correct

    def _get_current_weighted_frequencies(self) -> np.ndarray:
        """Get current therapist action frequencies using client's weighting scheme.

        Matches the client's _calculate_marginal_frequencies() method exactly:
        uses FULL memory (not just perception_window) with recency weighting.

        Returns:
            Array of 8 frequencies matching what client uses for expectations
        """
        full_memory = list(self.client_ref.memory)
        if not full_memory:
            return np.ones(8) / 8

        memory_weights = get_memory_weights(len(full_memory))
        weighted_counts = np.zeros(8)

        for idx, (client_oct, therapist_oct) in enumerate(full_memory):
            weighted_counts[therapist_oct] += memory_weights[idx]

        total_weight = sum(memory_weights)
        if total_weight == 0:
            return np.ones(8) / 8

        return weighted_counts / total_weight

    def _project_therapist_frequencies(self, target_action: int) -> np.ndarray:
        """Project therapist action frequencies after successful seeding.

        IMPORTANT: Two different memory contexts at play:
        1. SEEDING uses perception_window (parataxic window) - to make target the mode
        2. CLIENT EXPECTATIONS use full memory with recency weighting

        This method projects what the FULL memory frequencies will look like after
        seeding succeeds (since client expectations use full memory).

        The current mode (what failed seeds perceive) is determined from
        perception_window (matching parataxic distortion logic).

        Args:
            target_action: The therapist action being seeded

        Returns:
            Array of 8 frequencies for client expectation calculation (full memory)
        """
        full_memory = list(self.client_ref.memory)
        if not full_memory:
            frequencies = np.zeros(8)
            frequencies[target_action] = 1.0
            return frequencies

        # Get current weighted frequencies from FULL memory (for client expectations)
        current_frequencies = self._get_current_weighted_frequencies()

        # Identify current mode from PERCEPTION_WINDOW (for parataxic distortion)
        # This is what failed seeds will be perceived as
        recent_memory = full_memory[-self.perception_window:]
        recent_therapist_actions = [t for c, t in recent_memory]
        recent_counts = np.array([recent_therapist_actions.count(a) for a in range(8)])
        max_recent_count = max(recent_counts)
        tied_actions = {a for a in range(8) if recent_counts[a] == max_recent_count}

        current_mode = None
        for action in reversed(recent_therapist_actions):
            if action in tied_actions:
                current_mode = action
                break
        if current_mode is None:
            current_mode = int(np.argmax(recent_counts))

        # Get seeding sessions needed (uses perception_window internally)
        seeding_sessions = self._estimate_seeding_sessions(target_action)

        if seeding_sessions == 0:
            # Target already dominant in parataxic window
            return current_frequencies

        # Calculate expected successes and failures
        raw_seeds_needed = int(np.ceil(seeding_sessions * self.baseline_accuracy))
        expected_successes = raw_seeds_needed
        expected_failures = raw_seeds_needed * (1 - self.baseline_accuracy) / self.baseline_accuracy

        # Project frequency changes in FULL memory
        # Each new entry affects full memory frequency with recency weighting
        memory_size = len(full_memory)
        avg_recent_weight = np.mean(get_memory_weights(memory_size)[-10:])
        freq_per_seed = avg_recent_weight / memory_size

        projected_frequencies = current_frequencies.copy()
        success_freq_boost = expected_successes * freq_per_seed
        failure_freq_boost = expected_failures * freq_per_seed

        projected_frequencies[target_action] += success_freq_boost
        projected_frequencies[current_mode] += failure_freq_boost

        # Normalize
        total = projected_frequencies.sum()
        if total > 0:
            projected_frequencies = projected_frequencies / total

        return projected_frequencies

    def _project_bond_after_seeding(self, target_therapist_action: int) -> float:
        """Project what bond will be after seeding completes.

        During seeding, interactions may be suboptimal (seeding action ≠ complement),
        which affects RS and thus bond. This estimates the bond level after seeding.

        IMPORTANT: Uses FULL memory for RS/bond calculation (matching client logic),
        but uses _estimate_seeding_sessions which considers perception_window for
        determining how many seeds are needed.

        Args:
            target_therapist_action: The therapist action being seeded

        Returns:
            Projected bond value after seeding
        """
        from src.config import rs_to_bond, BOND_ALPHA, BOND_OFFSET

        u_matrix = self.client_ref.u_matrix
        current_rs = self.client_ref.relationship_satisfaction

        # Get seeding sessions needed (uses perception_window internally)
        total_seeding_sessions = self._estimate_seeding_sessions(target_therapist_action)

        if total_seeding_sessions == 0:
            # Already dominant in parataxic window, no seeding needed
            return self.client_ref.bond

        # Estimate average utility during seeding sessions
        # During seeding, client takes various actions; therapist plays target_therapist_action
        current_payoffs = self.client_ref._calculate_expected_payoffs()
        current_probs = self._softmax(current_payoffs / self.client_ref.entropy)

        # Expected utility per seeding session
        seeding_utilities = []
        for client_action in range(8):
            utility = u_matrix[client_action, target_therapist_action]
            seeding_utilities.append(utility * current_probs[client_action])
        avg_seeding_utility = sum(seeding_utilities)

        # Project RS change using FULL memory size (matching client RS calculation)
        memory_size = len(self.client_ref.memory)
        seeding_weight = min(0.3, total_seeding_sessions / memory_size)  # cap influence
        projected_rs = (1 - seeding_weight) * current_rs + seeding_weight * avg_seeding_utility

        # Convert to bond
        projected_bond = rs_to_bond(
            rs=projected_rs,
            rs_min=self.client_ref.rs_min,
            rs_max=self.client_ref.rs_max,
            alpha=BOND_ALPHA,
            offset=BOND_OFFSET
        )

        return projected_bond

    def _project_client_expected_payoffs(self, target_therapist_action: int) -> np.ndarray:
        """Project client's expected payoffs after successful seeding.

        Uses the frequency-amplifier formula to estimate what the client's
        expected payoffs would be once target_therapist_action dominates memory.

        IMPORTANT: Now uses PROJECTED bond (not current bond) to account for
        how seeding sessions will affect RS and thus the client's optimism level.

        Formula: adjusted[i,j] = U[i,j] + (U[i,j] * P(j) * history_weight)
        Then bond-based percentile selection over sorted adjusted utilities.

        Args:
            target_therapist_action: The therapist action being seeded

        Returns:
            Array of 8 expected payoffs (one per client action)
        """
        u_matrix = self.client_ref.u_matrix

        # Use PROJECTED bond, not current bond
        projected_bond = self._project_bond_after_seeding(target_therapist_action)

        # Get history_weight from client if available, else use default
        history_weight = getattr(self.client_ref, 'history_weight', 1.0)

        # Project frequency distribution after seeding
        projected_frequencies = self._project_therapist_frequencies(target_therapist_action)

        expected_payoffs = np.zeros(8)

        for client_action in range(8):
            raw_utilities = u_matrix[client_action, :]

            # Amplify using projected frequencies (frequency-amplifier formula)
            adjusted_utilities = raw_utilities + (
                raw_utilities * projected_frequencies * history_weight
            )

            # Sort adjusted utilities for bond-based selection
            sorted_adjusted = np.sort(adjusted_utilities)

            # Bond-based percentile interpolation using PROJECTED bond
            position = projected_bond * 7
            lower_idx = int(position)
            upper_idx = min(lower_idx + 1, 7)
            interpolation_weight = position - lower_idx

            expected_payoffs[client_action] = (
                (1 - interpolation_weight) * sorted_adjusted[lower_idx] +
                interpolation_weight * sorted_adjusted[upper_idx]
            )

        return expected_payoffs

    def _project_client_probabilities(self, target_therapist_action: int) -> np.ndarray:
        """Project client action probabilities after successful seeding.

        Combines projected expected payoffs with softmax to estimate how
        client behavior will shift once seeding is complete.

        This enables forward-looking target selection: instead of choosing
        targets based on current probabilities, we choose based on what
        probabilities will look like AFTER seeding succeeds.

        Args:
            target_therapist_action: The therapist action being seeded

        Returns:
            Array of 8 probabilities (one per client action)
        """
        projected_payoffs = self._project_client_expected_payoffs(target_therapist_action)
        return self._softmax(projected_payoffs / self.client_ref.entropy)

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

    def _estimate_seeding_sessions(self, target_therapist_action: int) -> int:
        """Estimate number of sessions needed to seed a target action.

        IMPORTANT: Uses perception_window (parataxic window) for mode detection,
        because seeding is about making an action the mode within the parataxic
        distortion window, not the full memory.

        Args:
            target_therapist_action: The therapist action to seed

        Returns:
            Estimated number of seeding sessions required
        """
        # Use perception_window for seeding (matches parataxic distortion logic)
        recent_memory = list(self.client_ref.memory)[-self.perception_window:]
        if not recent_memory:
            return self.perception_window // 2 + 1

        # Simple counts within parataxic window (no recency weighting - matches parataxic logic)
        therapist_actions = [t for c, t in recent_memory]
        counts = np.array([therapist_actions.count(a) for a in range(8)])

        current_target_count = counts[target_therapist_action]
        max_other_count = max(counts[a] for a in range(8) if a != target_therapist_action)

        if current_target_count > max_other_count:
            return 0  # Already dominant in parataxic window

        # Raw seeds needed to exceed max_other_count
        raw_seeds_needed = max(1, max_other_count + 1 - current_target_count)

        # Adjust for baseline_accuracy (some seeds will fail)
        total_sessions = int(np.ceil(raw_seeds_needed / self.baseline_accuracy))

        return total_sessions

    def _calculate_target_net_value(
        self,
        client_action: int,
        therapist_action: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate net expected value of pursuing a target, integrating seeding cost.

        This combines target selection with cost-benefit analysis. Instead of
        picking a target then checking if seeding is beneficial, we calculate
        the NET value accounting for:

        1. Seeding cost: sessions × utility sacrificed during seeding
        2. Post-seeding benefit: remaining sessions × utility improvement × P(client action)
        3. Bond dynamics: bond may decrease during seeding (floor) vs current (ceiling)

        Args:
            client_action: Target client octant
            therapist_action: Target therapist octant (to seed)

        Returns:
            Tuple of (net_value, metadata_dict)
            net_value must be > 0 to be worth pursuing
        """
        u_matrix = self.client_ref.u_matrix
        current_bond = self.client_ref.bond
        current_rs = self.client_ref.relationship_satisfaction
        remaining_sessions = self.max_sessions - self.session_count

        # Target utility (raw U-matrix value)
        target_utility = u_matrix[client_action, therapist_action]

        # Quick reject: target must improve over current RS
        if target_utility <= current_rs:
            return float('-inf'), {'reason': 'no_improvement'}

        # Estimate seeding requirements
        seeding_sessions = self._estimate_seeding_sessions(therapist_action)
        projected_bond = self._project_bond_after_seeding(therapist_action)

        # Bond range: [floor=projected_bond, ceiling=current_bond]
        # During seeding, bond moves from current toward projected
        bond_floor = min(projected_bond, current_bond)
        bond_ceiling = max(projected_bond, current_bond)

        # Not enough time to benefit after seeding
        sessions_after_seeding = remaining_sessions - seeding_sessions
        if sessions_after_seeding <= 3:  # Need at least a few sessions to benefit
            return float('-inf'), {'reason': 'insufficient_time'}

        # Calculate seeding cost
        # During seeding, therapist plays target_action regardless of client action
        # Cost = sum over client actions of: P(client_action) × [U(complement) - U(target)]
        current_payoffs = self.client_ref._calculate_expected_payoffs()
        current_probs = self._softmax(current_payoffs / self.client_ref.entropy)

        seeding_cost_per_session = 0.0
        for c_act in range(8):
            complement = self.COMPLEMENT_MAP[c_act]
            utility_if_complement = u_matrix[c_act, complement]
            utility_if_seed = u_matrix[c_act, therapist_action]
            # Cost is opportunity cost: what we give up by not complementing
            opportunity_cost = max(0, utility_if_complement - utility_if_seed)
            seeding_cost_per_session += current_probs[c_act] * opportunity_cost

        total_seeding_cost = seeding_cost_per_session * seeding_sessions

        # Calculate post-seeding benefit using bond floor (conservative)
        # Project client probabilities at projected bond level
        projected_probs = self._project_client_probabilities(therapist_action)
        prob_target_client_action = projected_probs[client_action]

        # Utility improvement when target interaction occurs
        utility_improvement = target_utility - current_rs

        # Expected benefit per session after seeding
        benefit_per_session = utility_improvement * prob_target_client_action

        # Scale benefit by projected bond to account for accessibility
        # Lower bond = client more pessimistic = less likely to "reach" high utility
        # Use bond_floor as conservative estimate
        bond_accessibility_factor = bond_floor

        # Total expected benefit (scaled by bond accessibility)
        total_benefit = benefit_per_session * sessions_after_seeding * bond_accessibility_factor

        # Apply seeding benefit scaling (hyperparameter)
        total_benefit *= self.seeding_benefit_scaling

        # Net value
        net_value = total_benefit - total_seeding_cost

        metadata = {
            'target_utility': target_utility,
            'current_rs': current_rs,
            'utility_improvement': utility_improvement,
            'seeding_sessions': seeding_sessions,
            'sessions_after_seeding': sessions_after_seeding,
            'seeding_cost_per_session': seeding_cost_per_session,
            'total_seeding_cost': total_seeding_cost,
            'prob_target_client_action': prob_target_client_action,
            'benefit_per_session': benefit_per_session,
            'bond_floor': bond_floor,
            'bond_ceiling': bond_ceiling,
            'total_benefit': total_benefit,
            'net_value': net_value,
        }

        return net_value, metadata

    def _identify_target_interaction(self) -> bool:
        """Find best (client_action, therapist_action) pair with integrated cost-benefit.

        This is called when entering or continuing ladder_climbing phase.

        IMPORTANT: Now integrates seeding cost into target selection.
        Instead of picking a target then checking if beneficial, we calculate
        the NET expected value for each candidate:

        Net Value = (post-seeding benefit) - (seeding cost)

        Only targets with positive net value are considered. The target with
        highest net value is selected.

        Also considers bond dynamics:
        - Bond may decrease during seeding (conservative floor)
        - Current bond sets the ceiling

        Returns:
            True if a valid target was found, False if no improving target exists
        """
        best_net_value = 0.0  # Must be positive to be worth pursuing
        best_client_action = None
        best_therapist_action = None
        best_metadata = None

        # Consider ALL (client_action, therapist_action) pairs
        for client_oct in range(8):
            for therapist_oct in range(8):
                net_value, metadata = self._calculate_target_net_value(
                    client_oct, therapist_oct
                )

                if net_value > best_net_value:
                    best_net_value = net_value
                    best_client_action = client_oct
                    best_therapist_action = therapist_oct
                    best_metadata = metadata

        if best_client_action is not None:
            self.current_target_client_action = best_client_action
            self.current_target_therapist_action = best_therapist_action
            return True
        else:
            # No target with positive net value - stay in current phase
            self.current_target_client_action = None
            self.current_target_therapist_action = None
            return False

    def _should_start_ladder_climbing(self) -> bool:
        """Check if we should transition from relationship_building to ladder_climbing.

        SIMPLIFIED: Now that _identify_target_interaction() does integrated cost-benefit
        analysis (via _calculate_target_net_value), we only need to check:
        - Safety thresholds (session count, bond level)
        - Whether a target with positive net value exists

        The old checks for perception_accuracy and seeding_needed are redundant
        because _calculate_target_net_value already accounts for seeding cost/benefit.

        Returns:
            True if we should start ladder climbing
        """
        # Safety: Don't start ladder-climbing before dropout check (session 10)
        if self.session_count < 10:
            return False

        # Check if bond is high enough (prevent risk of retreat)
        bond_threshold = 0.1
        if self.client_ref.bond < bond_threshold:
            return False

        # Check if there's a target with positive net value
        # _identify_target_interaction now does integrated cost-benefit analysis
        return self._identify_target_interaction()

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
        """Determine if we should seed on this specific session.

        SIMPLIFIED: Cost-benefit analysis is now integrated into target selection
        via _calculate_target_net_value(). Once a target is selected, we commit
        to seeding it. This method handles only per-session edge cases:

        1. No target set → don't seed
        2. Complement == target → always seed (free!)
        3. Seeding complete (target dominant) → stop seeding
        4. Otherwise → continue seeding (we committed to this target)

        The feedback monitoring system (_should_abort_target) handles cases where
        seeding isn't working and we should abandon the target.

        Args:
            client_action: Current client action

        Returns:
            True if we should seed, False if we should complement
        """
        if self.current_target_therapist_action is None:
            return False

        complement = self._get_complementary_action(client_action)

        # Case 1: Complement IS the seeding target - always do it (free seeding!)
        if complement == self.current_target_therapist_action:
            return True

        # Case 2: Check if seeding is complete (target already dominant)
        seeding_sessions_needed = self._estimate_seeding_sessions(self.current_target_therapist_action)
        if seeding_sessions_needed == 0:
            return False  # Target is dominant, switch to consolidation

        # Case 3: Continue seeding - we committed to this target via _identify_target_interaction
        # which already validated positive net value
        return True

    def _initialize_seeding_monitor(self, target_action: int, target_client_action: int, session: int) -> None:
        """Initialize feedback monitor when starting to seed a new target.

        Called when entering ladder_climbing phase or when target changes.

        Args:
            target_action: The therapist action to seed
            target_client_action: The client action we're working toward
            session: Current session number
        """
        perceived_memory = self._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived_memory]

        # Count initial state
        initial_target_count = therapist_actions.count(target_action)

        # Count all competitors
        competitor_counts = {}
        for action in range(8):
            if action != target_action:
                competitor_counts[action] = therapist_actions.count(action)

        max_competitor_count = max(competitor_counts.values()) if competitor_counts else 0

        self.seeding_monitor = SeedingMonitor(
            target_action=target_action,
            target_client_action=target_client_action,
            initial_target_count=initial_target_count,
            initial_competitor_counts=competitor_counts.copy(),
            initial_memory_snapshot=perceived_memory.copy(),
            max_competitor_at_start=max_competitor_count,
            current_max_competitor=max_competitor_count,
            started_session=session,
            last_recalculation_session=session,
        )

    def _capture_seeding_feedback(self, session: int, intended_action: int, client_action: int) -> FeedbackRecord:
        """Capture feedback after a seeding attempt.

        Checks parataxic_history to see what client actually perceived,
        compares memory state before/after, and detects competitor boosting.

        Args:
            session: Current session number
            intended_action: The action therapist tried to seed
            client_action: The action client took

        Returns:
            FeedbackRecord with outcome of this seeding attempt
        """
        # Get current memory state
        perceived_memory = self._get_client_perceived_memory()
        therapist_actions = [t for c, t in perceived_memory]

        # What did client actually perceive? Check parataxic_history
        if hasattr(self.client_ref, 'parataxic_history') and self.client_ref.parataxic_history:
            last_record = self.client_ref.parataxic_history[-1]
            actual_perceived = last_record.perceived_therapist_action
            success = (actual_perceived == intended_action)
        else:
            # No parataxic distortion - perfect perception
            actual_perceived = intended_action
            success = True

        # Count current state
        target_count_after = therapist_actions.count(intended_action)

        # Get target count before (from monitor's last feedback or initial)
        if self.seeding_monitor and self.seeding_monitor.recent_feedback:
            target_count_before = self.seeding_monitor.recent_feedback[-1].target_count_after
        elif self.seeding_monitor:
            target_count_before = self.seeding_monitor.initial_target_count
        else:
            target_count_before = target_count_after - (1 if success else 0)

        # Count all competitors
        competitor_counts = {}
        for action in range(8):
            if action != intended_action:
                competitor_counts[action] = therapist_actions.count(action)

        # Detect which competitor gained most (if any)
        competitor_gained_most = None
        competitor_gain_amount = 0

        if self.seeding_monitor:
            # Compare to previous counts
            if self.seeding_monitor.recent_feedback:
                prev_counts = self.seeding_monitor.recent_feedback[-1].competitor_counts_after
            else:
                prev_counts = self.seeding_monitor.initial_competitor_counts

            for action, count in competitor_counts.items():
                prev_count = prev_counts.get(action, 0)
                gain = count - prev_count
                if gain > competitor_gain_amount:
                    competitor_gain_amount = gain
                    competitor_gained_most = action

        return FeedbackRecord(
            session=session,
            intended_action=intended_action,
            actual_perceived=actual_perceived,
            success=success,
            client_action=client_action,
            target_count_before=target_count_before,
            target_count_after=target_count_after,
            competitor_counts_after=competitor_counts,
            competitor_gained_most=competitor_gained_most,
            competitor_gain_amount=competitor_gain_amount,
        )

    def _update_seeding_monitor(self, feedback: FeedbackRecord) -> None:
        """Update monitor state with new feedback.

        Args:
            feedback: The feedback record from this session
        """
        if not self.seeding_monitor:
            return

        monitor = self.seeding_monitor

        # Update attempt counts
        monitor.seeding_attempts += 1
        if feedback.success:
            monitor.successful_seeds += 1
            monitor.consecutive_failures = 0
        else:
            monitor.failed_seeds += 1
            monitor.consecutive_failures += 1

        # Track competitor boosting
        if not feedback.success and feedback.competitor_gained_most is not None:
            monitor.competitor_boost_events.append(
                (feedback.session, feedback.competitor_gained_most)
            )

        # Update max competitor
        if feedback.competitor_counts_after:
            monitor.current_max_competitor = max(feedback.competitor_counts_after.values())

        # Calculate net progress
        monitor.net_progress = feedback.target_count_after - monitor.initial_target_count

        # Add to recent feedback
        monitor.recent_feedback.append(feedback)

    def _should_recalculate_seeding(self, session: int) -> bool:
        """Determine if we should recalculate seeding requirements.

        Recalculate when:
        - Memory state has changed significantly from initial snapshot
        - Competitor actions have gained ground
        - Every N sessions as a regular check (e.g., every 5 sessions)

        Args:
            session: Current session number

        Returns:
            True if we should recalculate requirements
        """
        if not self.seeding_monitor:
            return False

        monitor = self.seeding_monitor

        # Adjust thresholds based on perception window size
        if self.perception_window < 10:
            recalc_interval = 3  # More frequent for small windows
        else:
            recalc_interval = 5

        # Regular recalculation interval
        sessions_since_recalc = session - monitor.last_recalculation_session
        if sessions_since_recalc >= recalc_interval:
            return True

        # Recalculate if competitor has grown significantly
        competitor_growth = monitor.current_max_competitor - monitor.max_competitor_at_start
        if competitor_growth >= 2:  # Competitor gained 2+ instances
            return True

        # Recalculate if we have net negative progress after 3+ attempts
        if monitor.seeding_attempts >= 3 and monitor.net_progress < 0:
            return True

        return False

    def _should_abort_target(self, session: int) -> Tuple[bool, Optional[str]]:
        """Determine if we should abandon the current seeding target.

        Abort when:
        1. Consecutive failures exceed threshold (e.g., 5 failures in a row)
        2. Competitors are being boosted faster than target is growing
        3. Not enough time remaining to benefit from seeding
        4. Seeding success rate is below minimum viable threshold

        Args:
            session: Current session number

        Returns:
            Tuple of (should_abort, reason)
        """
        if not self.seeding_monitor:
            return False, None

        monitor = self.seeding_monitor

        # Use configured threshold (no longer adaptive)
        abort_consecutive_threshold = self.abort_consecutive_failures_threshold

        # Criterion 1: Too many consecutive failures
        if monitor.consecutive_failures >= abort_consecutive_threshold:
            return True, f"Consecutive failures ({monitor.consecutive_failures}) exceeded threshold"

        # Criterion 2: Competitor boosting outpacing target growth
        # Check if any competitor has gained more than target has
        if monitor.seeding_attempts >= 5:  # Need enough data
            # Look at competitor boost events
            competitor_boost_counts = {}
            for _, boosted_action in monitor.competitor_boost_events:
                competitor_boost_counts[boosted_action] = competitor_boost_counts.get(boosted_action, 0) + 1

            # If any competitor was boosted more times than target succeeded
            if competitor_boost_counts:
                max_boosts = max(competitor_boost_counts.values())
                if max_boosts > monitor.successful_seeds and monitor.net_progress <= 0:
                    boosted_action = max(competitor_boost_counts.items(), key=lambda x: x[1])[0]
                    return True, f"Competitor action {boosted_action} boosted {max_boosts} times vs {monitor.successful_seeds} successes"

        # Criterion 3: Not enough time remaining
        remaining_sessions = self.max_sessions - session

        # Estimate sessions still needed
        seeding_req = self.calculate_seeding_requirement(monitor.target_action)
        estimated_remaining = seeding_req['adjusted_seeding_needed']

        # Need at least 3 sessions after seeding completes to benefit
        if remaining_sessions < estimated_remaining + 3:
            return True, f"Insufficient time: {remaining_sessions} sessions left, need {estimated_remaining + 3}"

        # Criterion 4: Success rate below minimum viable threshold
        if monitor.seeding_attempts >= 10:  # Need enough data
            success_rate = monitor.successful_seeds / monitor.seeding_attempts
            # If success rate is below baseline_accuracy - 10%, something is wrong
            if success_rate < (self.baseline_accuracy - 0.1):
                return True, f"Success rate {success_rate:.1%} below baseline {self.baseline_accuracy:.1%}"

        return False, None

    def _adapt_seeding_strategy(self, session: int) -> str:
        """Decide how to adapt strategy based on feedback.

        Returns one of:
        - "continue": Keep seeding current target
        - "recalculate": Update requirements and keep going
        - "abort_find_new": Abort current target and find new one
        - "abort_consolidate": Abort and switch to consolidation (complementarity)

        Args:
            session: Current session number

        Returns:
            Strategy decision string
        """
        if not self.seeding_monitor:
            return "continue"

        # Check if we should abort
        should_abort, abort_reason = self._should_abort_target(session)

        if should_abort:
            # Try to find a new target
            old_target = self.current_target_therapist_action
            if self._identify_target_interaction():
                # Found a new target
                if self.current_target_therapist_action != old_target:
                    return "abort_find_new"

            # No better target available - fall back to consolidation
            return "abort_consolidate"

        # Not aborting - check if we should recalculate
        if self._should_recalculate_seeding(session):
            return "recalculate"

        return "continue"

    def _process_seeding_feedback(self, session: int, action_taken: int, client_action: int) -> Optional[str]:
        """Process feedback after taking an action during ladder_climbing.

        This is the main integration point called after each seeding attempt.

        Args:
            session: Current session number
            action_taken: The action therapist took (might be seeding or complement)
            client_action: The action client took

        Returns:
            Decision string if strategy needs to change, None otherwise
        """
        # Only monitor during ladder_climbing phase when actually seeding
        if self.phase != "ladder_climbing":
            return None

        if not self.seeding_monitor:
            return None

        # Only capture feedback if we actually seeded (not complemented)
        if action_taken != self.current_target_therapist_action:
            return None

        # Capture feedback
        feedback = self._capture_seeding_feedback(session, action_taken, client_action)

        # Update monitor
        self._update_seeding_monitor(feedback)

        # Decide on strategy adaptation
        decision = self._adapt_seeding_strategy(session)

        return decision if decision != "continue" else None

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

                # Initialize seeding monitor for new target
                if self.current_target_therapist_action is not None and self.current_target_client_action is not None:
                    self._initialize_seeding_monitor(
                        target_action=self.current_target_therapist_action,
                        target_client_action=self.current_target_client_action,
                        session=session
                    )

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

                # Initialize seeding monitor for new target
                if self.current_target_therapist_action is not None and self.current_target_client_action is not None:
                    self._initialize_seeding_monitor(
                        target_action=self.current_target_therapist_action,
                        target_client_action=self.current_target_client_action,
                        session=session
                    )

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
                    # Reset monitor for new target
                    if self.current_target_therapist_action is not None and self.current_target_client_action is not None:
                        self._initialize_seeding_monitor(
                            target_action=self.current_target_therapist_action,
                            target_client_action=self.current_target_client_action,
                            session=session
                        )

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
                    # Track last seeding action for feedback processing
                    self.last_seeding_action = self.current_target_therapist_action
                    self.last_session_number = session
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

    def process_feedback_after_memory_update(self, session: int, client_action: int) -> None:
        """Process feedback after client has updated memory.

        MUST be called AFTER client.update_memory() but BEFORE next decide_action().
        This timing ensures parataxic_history has the latest perception record.

        Args:
            session: Session that just completed
            client_action: Client action from that session
        """
        if self.last_seeding_action is None:
            return

        if session != self.last_session_number:
            return

        # Capture and process feedback
        decision = self._process_seeding_feedback(
            session, self.last_seeding_action, client_action
        )

        # Act on decision
        if decision == "abort_find_new":
            # Track failed target
            if self.current_target_therapist_action is not None and self.current_target_therapist_action not in self.failed_targets:
                self.failed_targets.append(self.current_target_therapist_action)

            # Check if too many failures → give up on ladder_climbing
            if len(self.failed_targets) >= self.max_target_failures:
                self.phase = "consolidation"
                self.seeding_monitor = None
            else:
                # Try new target (will be picked up in next decide_action)
                self.seeding_monitor = None

        elif decision == "abort_consolidate":
            self.phase = "consolidation"
            self.seeding_monitor = None

        elif decision == "recalculate":
            if self.seeding_monitor:
                self.seeding_monitor.last_recalculation_session = session

        # Clear last action
        self.last_seeding_action = None

    def _log_action(
        self,
        session: int,
        client_action: int,
        therapist_action: int,
        phase: str,
        rationale: str,
        seeding_progress: Optional[Dict[str, Any]] = None,
        seeding_feedback: Optional[FeedbackRecord] = None,
        monitor_decision: Optional[str] = None,
        abort_reason: Optional[str] = None,
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
            seeding_feedback=seeding_feedback,
            monitor_decision=monitor_decision,
            abort_reason=abort_reason,
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

    def get_feedback_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary statistics about feedback monitoring effectiveness.

        Returns:
            Dictionary with:
            - total_seeding_sessions: Count of sessions where seeding occurred
            - total_monitoring_active: Count of sessions with active monitoring
            - recalculations: Number of times requirements were recalculated
            - aborts: Number of targets abandoned
            - abort_reasons: Counter of abort reasons
            - avg_success_rate: Average seeding success rate across all targets
            - competitor_boost_rate: Rate at which competitors were boosted
        """
        if not self.action_log:
            return {}

        total_seeding = sum(1 for e in self.action_log if e.seeding_feedback is not None)
        recalcs = sum(1 for e in self.action_log if e.monitor_decision == "recalculate")
        aborts = sum(1 for e in self.action_log if e.monitor_decision and "abort" in e.monitor_decision)

        abort_reasons = Counter(
            e.abort_reason for e in self.action_log
            if e.abort_reason is not None
        )

        # Calculate success rates
        feedbacks = [e.seeding_feedback for e in self.action_log if e.seeding_feedback]
        if feedbacks:
            successes = sum(1 for f in feedbacks if f.success)
            avg_success_rate = successes / len(feedbacks)

            competitor_boosts = sum(
                1 for f in feedbacks
                if not f.success and f.competitor_gained_most is not None
            )
            competitor_boost_rate = competitor_boosts / len(feedbacks)
        else:
            avg_success_rate = 0.0
            competitor_boost_rate = 0.0

        return {
            'total_seeding_sessions': total_seeding,
            'recalculations': recalcs,
            'aborts': aborts,
            'abort_reasons': dict(abort_reasons),
            'avg_success_rate': avg_success_rate,
            'competitor_boost_rate': competitor_boost_rate,
        }

    def reset(self) -> None:
        """Reset therapist state for new episode."""
        self.phase = "relationship_building"
        self.current_target_client_action = None
        self.current_target_therapist_action = None
        self.action_log = []
        self.session_count = 0
        self.actual_actions_taken = []

        # Reset feedback monitoring state
        self.seeding_monitor = None
        self.last_seeding_action = None
        self.last_session_number = 0
        self.failed_targets = []
