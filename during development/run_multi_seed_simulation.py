"""Multi-seed simulation runner with comprehensive statistics.

Runs therapy simulations across multiple random seeds and provides
statistical analysis of outcomes including:
- Success rates and timing
- RS, bond, and perception trajectories
- Dropout rates
- Action distributions
- Near-miss analysis for failures
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
import argparse

from src.agents.client_agents import (
    with_perception,
    BondOnlyClient,
    FrequencyAmplifierClient,
    ConditionalAmplifierClient,
    BondWeightedConditionalAmplifier,
    BondWeightedFrequencyAmplifier,
    BaseClientAgent,
)
from src import config
from src.config import (
    sample_u_matrix,
    OCTANTS,
    calculate_success_threshold,
)


# Map legacy pattern names
PATTERN_ALIASES = {
    'cw_50_50': 'cold_warm',
}


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    seed: int
    success: bool
    dropped_out: bool
    total_sessions: int
    first_success_session: Optional[int]

    # Final state
    final_rs: float
    final_bond: float
    initial_rs: float
    initial_bond: float

    # Trajectory data
    rs_trajectory: List[float]
    bond_trajectory: List[float]

    # Threshold info
    rs_threshold: float
    closest_rs: float
    gap_to_threshold: float

    # Action distribution
    client_actions: List[int]
    therapist_actions: List[int]

    # Perception stats (if enabled)
    perception_stats: Optional[Dict[str, Any]] = None

    # Intervention data (strategic therapist)
    intervention_count: int = 0
    intervention_sessions: List[int] = field(default_factory=list)
    intervention_rs_values: List[float] = field(default_factory=list)
    intervention_bond_values: List[float] = field(default_factory=list)


@dataclass
class MultiSeedStatistics:
    """Aggregated statistics across multiple simulation runs."""
    n_runs: int
    config_summary: Dict[str, Any]

    # Success metrics
    success_rate: float
    dropout_rate: float
    n_success: int
    n_dropout: int
    n_failure: int

    # Session timing (for successful runs)
    success_sessions_mean: Optional[float] = None
    success_sessions_median: Optional[float] = None
    success_sessions_std: Optional[float] = None
    success_sessions_min: Optional[int] = None
    success_sessions_max: Optional[int] = None
    success_sessions_mode: Optional[int] = None

    # Final state statistics (all runs)
    final_rs_stats: Dict[str, float] = field(default_factory=dict)
    final_bond_stats: Dict[str, float] = field(default_factory=dict)

    # Trajectory statistics (mean at each session)
    rs_trajectory_mean: List[float] = field(default_factory=list)
    rs_trajectory_std: List[float] = field(default_factory=list)
    bond_trajectory_mean: List[float] = field(default_factory=list)
    bond_trajectory_std: List[float] = field(default_factory=list)

    # Change statistics
    rs_change_stats: Dict[str, float] = field(default_factory=dict)
    bond_change_stats: Dict[str, float] = field(default_factory=dict)

    # Perception statistics (if enabled)
    perception_stats: Optional[Dict[str, Any]] = None

    # Action distribution (aggregated across all runs)
    client_action_distribution: Counter = field(default_factory=Counter)
    therapist_action_distribution: Counter = field(default_factory=Counter)

    # Near-miss analysis (for failed runs)
    failed_runs_closest_rs: List[float] = field(default_factory=list)
    failed_runs_gap: List[float] = field(default_factory=list)

    # Intervention statistics (strategic therapist)
    intervention_rate: Optional[float] = None
    mean_interventions_per_run: Optional[float] = None
    median_interventions_per_run: Optional[float] = None
    max_interventions_in_run: Optional[int] = None
    intervention_rs_change_stats: Optional[Dict[str, float]] = None
    intervention_bond_change_stats: Optional[Dict[str, float]] = None


def always_complement(client_action: int) -> int:
    """Simple complementary therapist strategy."""
    complement_map = {
        0: 4, 1: 3, 2: 2, 3: 1,
        4: 0, 5: 7, 6: 6, 7: 5,
    }
    return complement_map[client_action]


def get_optimal_therapist_action(u_matrix: np.ndarray) -> int:
    """
    Find the therapist action that corresponds to the global maximum utility in the u_matrix.

    Parameters
    ----------
    u_matrix : np.ndarray
        The client's 8x8 utility matrix (client_action x therapist_action)

    Returns
    -------
    int
        The therapist action (column index, 0-7) corresponding to the maximum utility
    """
    # Find the indices of the maximum value
    max_idx = np.unravel_index(np.argmax(u_matrix), u_matrix.shape)
    # Return the therapist action (column index)
    return int(max_idx[1])


def run_single_simulation(
    seed: int,
    mechanism: str,
    initial_memory_pattern: str,
    success_threshold_percentile: float,
    enable_perception: bool = False,
    baseline_accuracy: float = 0.2,
    max_sessions: int = 100,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 5.0,
    bond_offset: float = 0.8,
    enable_strategic_therapist: bool = False,
    rs_plateau_threshold: float = 5.0,
    plateau_window: int = 15,
    intervention_duration: int = 10,
) -> SimulationResult:
    """
    Run a single therapy simulation with specified configuration.

    Returns detailed results for statistical aggregation.
    """
    # Setup
    rng = np.random.RandomState(seed)
    u_matrix = sample_u_matrix(random_state=seed)

    # Map legacy pattern names and generate memory
    pattern_type = PATTERN_ALIASES.get(initial_memory_pattern, initial_memory_pattern)
    initial_memory = BaseClientAgent.generate_problematic_memory(
        pattern_type=pattern_type,
        n_interactions=50,
        random_state=seed,
    )

    # Set global bond parameters
    config.BOND_ALPHA = bond_alpha
    config.BOND_OFFSET = bond_offset

    # Create client
    client_kwargs = {
        'u_matrix': u_matrix,
        'entropy': entropy,
        'initial_memory': initial_memory,
        'random_state': seed,
    }

    if 'amplifier' in mechanism:
        client_kwargs['history_weight'] = history_weight

    if 'bond_weighted' in mechanism:
        client_kwargs['bond_power'] = bond_power

    if enable_perception:
        client_kwargs['baseline_accuracy'] = baseline_accuracy
        client_kwargs['enable_perception'] = True

    mechanisms = {
        'bond_only': BondOnlyClient,
        'frequency_amplifier': FrequencyAmplifierClient,
        'conditional_amplifier': ConditionalAmplifierClient,
        'bond_weighted_conditional_amplifier': BondWeightedConditionalAmplifier,
        'bond_weighted_frequency_amplifier': BondWeightedFrequencyAmplifier,
    }

    ClientClass = mechanisms[mechanism]

    if enable_perception:
        ClientClass = with_perception(ClientClass)

    client = ClientClass(**client_kwargs)

    # Calculate RS threshold
    rs_threshold = calculate_success_threshold(u_matrix, success_threshold_percentile)

    # Track initial state
    initial_rs = client.relationship_satisfaction
    initial_bond = client.bond

    # Track trajectories
    rs_trajectory = [initial_rs]
    bond_trajectory = [initial_bond]

    # Track actions
    client_actions = []
    therapist_actions = []

    # Track success
    threshold_ever_reached = False
    first_threshold_session = None
    closest_rs = initial_rs

    # Track intervention state (strategic therapist)
    intervention_active = False
    intervention_session_count = 0
    intervention_count = 0
    optimal_action = None
    rs_history = []  # For plateau detection
    intervention_sessions = []
    intervention_rs_values = []
    intervention_bond_values = []

    # Run sessions
    session = 0
    dropped_out = False

    for session in range(1, max_sessions + 1):
        # Select action
        client_action = client.select_action()

        # Determine therapist action based on strategy
        if enable_strategic_therapist:
            # Check if we should trigger intervention (only if not already active)
            if not intervention_active:
                if len(rs_history) >= plateau_window:
                    # Check if RS has plateaued (range within threshold)
                    rs_range = max(rs_history) - min(rs_history)
                    if rs_range <= rs_plateau_threshold:
                        # Trigger intervention
                        intervention_active = True
                        intervention_session_count = 0
                        intervention_count += 1
                        optimal_action = get_optimal_therapist_action(u_matrix)
                        # Clear RS history to prevent immediate re-triggering
                        rs_history.clear()

            # Execute intervention or complementary
            if intervention_active:
                # Safety: ensure optimal action is computed
                if optimal_action is None:
                    optimal_action = get_optimal_therapist_action(u_matrix)
                therapist_action = optimal_action
                intervention_session_count += 1

                # Track intervention metrics
                intervention_sessions.append(session)

                # Check if intervention is complete
                if intervention_session_count >= intervention_duration:
                    intervention_active = False
                    # Resume monitoring after intervention completes
            else:
                therapist_action = always_complement(client_action)
        else:
            # Default: complementary
            therapist_action = always_complement(client_action)

        # Record actions
        client_actions.append(client_action)
        therapist_actions.append(therapist_action)

        # Update memory
        client.update_memory(client_action, therapist_action)

        # Get new state
        new_rs = client.relationship_satisfaction
        new_bond = client.bond

        # Track trajectories
        rs_trajectory.append(new_rs)
        bond_trajectory.append(new_bond)

        # Track intervention RS/Bond values
        if enable_strategic_therapist and session in intervention_sessions:
            intervention_rs_values.append(new_rs)
            intervention_bond_values.append(new_bond)

        # Update RS history for plateau detection (only when not in intervention)
        if enable_strategic_therapist and not intervention_active:
            rs_history.append(new_rs)

        # Update closest RS
        if new_rs > closest_rs:
            closest_rs = new_rs

        # Check if threshold reached
        if new_rs >= rs_threshold and not threshold_ever_reached:
            threshold_ever_reached = True
            first_threshold_session = session

        # Check dropout
        if client.check_dropout():
            dropped_out = True
            break

    # Get final state
    final_rs = client.relationship_satisfaction
    final_bond = client.bond

    # Get perception stats if enabled
    perception_stats = None
    if enable_perception and hasattr(client, 'get_perception_stats'):
        perception_stats = client.get_perception_stats()

    # Calculate gap to threshold
    gap_to_threshold = rs_threshold - closest_rs

    return SimulationResult(
        seed=seed,
        success=threshold_ever_reached,
        dropped_out=dropped_out,
        total_sessions=session,
        first_success_session=first_threshold_session,
        final_rs=final_rs,
        final_bond=final_bond,
        initial_rs=initial_rs,
        initial_bond=initial_bond,
        rs_trajectory=rs_trajectory,
        bond_trajectory=bond_trajectory,
        rs_threshold=rs_threshold,
        closest_rs=closest_rs,
        gap_to_threshold=gap_to_threshold,
        client_actions=client_actions,
        therapist_actions=therapist_actions,
        perception_stats=perception_stats,
        intervention_count=intervention_count,
        intervention_sessions=intervention_sessions,
        intervention_rs_values=intervention_rs_values,
        intervention_bond_values=intervention_bond_values,
    )


def compute_statistics(results: List[SimulationResult], config: Dict[str, Any]) -> MultiSeedStatistics:
    """Compute comprehensive statistics across all simulation runs."""
    n_runs = len(results)

    # Success metrics
    n_success = sum(1 for r in results if r.success)
    n_dropout = sum(1 for r in results if r.dropped_out)
    n_failure = n_runs - n_success

    success_rate = n_success / n_runs if n_runs > 0 else 0.0
    dropout_rate = n_dropout / n_runs if n_runs > 0 else 0.0

    # Session timing for successful runs - filter out None values explicitly
    success_sessions = [r.first_success_session for r in results 
                       if r.success and r.first_success_session is not None]

    success_sessions_mean = float(np.mean(success_sessions)) if success_sessions else None
    success_sessions_median = float(np.median(success_sessions)) if success_sessions else None
    success_sessions_std = float(np.std(success_sessions)) if success_sessions else None
    success_sessions_min = min(success_sessions) if success_sessions else None
    success_sessions_max = max(success_sessions) if success_sessions else None

    # Mode (most common success session)
    success_sessions_mode = None
    if success_sessions:
        session_counts = Counter(success_sessions)
        success_sessions_mode = session_counts.most_common(1)[0][0]

    # Final state statistics
    final_rs_values = [r.final_rs for r in results]
    final_bond_values = [r.final_bond for r in results]

    final_rs_stats = {
        'mean': np.mean(final_rs_values),
        'std': np.std(final_rs_values),
        'min': np.min(final_rs_values),
        'max': np.max(final_rs_values),
        'q25': np.percentile(final_rs_values, 25),
        'median': np.median(final_rs_values),
        'q75': np.percentile(final_rs_values, 75),
    }

    final_bond_stats = {
        'mean': np.mean(final_bond_values),
        'std': np.std(final_bond_values),
        'min': np.min(final_bond_values),
        'max': np.max(final_bond_values),
        'q25': np.percentile(final_bond_values, 25),
        'median': np.median(final_bond_values),
        'q75': np.percentile(final_bond_values, 75),
    }

    # Change statistics
    rs_changes = [r.final_rs - r.initial_rs for r in results]
    bond_changes = [r.final_bond - r.initial_bond for r in results]

    rs_change_stats = {
        'mean': np.mean(rs_changes),
        'std': np.std(rs_changes),
        'min': np.min(rs_changes),
        'max': np.max(rs_changes),
        'median': np.median(rs_changes),
    }

    bond_change_stats = {
        'mean': np.mean(bond_changes),
        'std': np.std(bond_changes),
        'min': np.min(bond_changes),
        'max': np.max(bond_changes),
        'median': np.median(bond_changes),
    }

    # Trajectory statistics
    # Find max trajectory length
    max_trajectory_length = max(len(r.rs_trajectory) for r in results)

    rs_trajectory_mean = []
    rs_trajectory_std = []
    bond_trajectory_mean = []
    bond_trajectory_std = []

    for session_idx in range(max_trajectory_length):
        # Collect values at this session index from all runs that reached it
        rs_at_session = [r.rs_trajectory[session_idx] for r in results if len(r.rs_trajectory) > session_idx]
        bond_at_session = [r.bond_trajectory[session_idx] for r in results if len(r.bond_trajectory) > session_idx]

        if rs_at_session:
            rs_trajectory_mean.append(np.mean(rs_at_session))
            rs_trajectory_std.append(np.std(rs_at_session))

        if bond_at_session:
            bond_trajectory_mean.append(np.mean(bond_at_session))
            bond_trajectory_std.append(np.std(bond_at_session))

    # Perception statistics (if enabled)
    perception_stats = None
    if results[0].perception_stats is not None:
        all_misperception_rates = [r.perception_stats['overall_misperception_rate']
                                   for r in results if r.perception_stats is not None]
        all_stage1_override_rates = [r.perception_stats['stage1_override_rate']
                                      for r in results if r.perception_stats is not None]
        all_computed_accuracies = [r.perception_stats['mean_computed_accuracy']
                                   for r in results if r.perception_stats is not None]

        perception_stats = {
            'misperception_rate_mean': np.mean(all_misperception_rates),
            'misperception_rate_std': np.std(all_misperception_rates),
            'misperception_rate_min': np.min(all_misperception_rates),
            'misperception_rate_max': np.max(all_misperception_rates),
            'stage1_override_rate_mean': np.mean(all_stage1_override_rates),
            'stage1_override_rate_std': np.std(all_stage1_override_rates),
            'computed_accuracy_mean': np.mean(all_computed_accuracies),
            'computed_accuracy_std': np.std(all_computed_accuracies),
        }

    # Action distribution (aggregated)
    client_action_distribution = Counter()
    therapist_action_distribution = Counter()

    for result in results:
        client_action_distribution.update(result.client_actions)
        therapist_action_distribution.update(result.therapist_actions)

    # Near-miss analysis for failed runs
    failed_runs = [r for r in results if not r.success]
    failed_runs_closest_rs = [r.closest_rs for r in failed_runs]
    failed_runs_gap = [r.gap_to_threshold for r in failed_runs]

    # Intervention statistics (strategic therapist)
    intervention_rate = None
    mean_interventions_per_run = None
    median_interventions_per_run = None
    max_interventions_in_run = None
    intervention_rs_change_stats = None
    intervention_bond_change_stats = None

    # Check if any run had interventions
    runs_with_interventions = [r for r in results if r.intervention_count > 0]
    if runs_with_interventions:
        # Basic intervention metrics
        intervention_rate = len(runs_with_interventions) / n_runs
        intervention_counts = [r.intervention_count for r in results]
        mean_interventions_per_run = float(np.mean(intervention_counts))
        median_interventions_per_run = float(np.median(intervention_counts))
        max_interventions_in_run = max(intervention_counts)

        # RS change during interventions
        all_intervention_rs = []
        for r in runs_with_interventions:
            if r.intervention_rs_values:
                all_intervention_rs.extend(r.intervention_rs_values)

        if all_intervention_rs:
            intervention_rs_change_stats = {
                'mean': float(np.mean(all_intervention_rs)),
                'std': float(np.std(all_intervention_rs)),
                'min': float(np.min(all_intervention_rs)),
                'max': float(np.max(all_intervention_rs)),
                'median': float(np.median(all_intervention_rs)),
            }

        # Bond change during interventions
        all_intervention_bond = []
        for r in runs_with_interventions:
            if r.intervention_bond_values:
                all_intervention_bond.extend(r.intervention_bond_values)

        if all_intervention_bond:
            intervention_bond_change_stats = {
                'mean': float(np.mean(all_intervention_bond)),
                'std': float(np.std(all_intervention_bond)),
                'min': float(np.min(all_intervention_bond)),
                'max': float(np.max(all_intervention_bond)),
                'median': float(np.median(all_intervention_bond)),
            }

    return MultiSeedStatistics(
        n_runs=n_runs,
        config_summary=config,
        success_rate=success_rate,
        dropout_rate=dropout_rate,
        n_success=n_success,
        n_dropout=n_dropout,
        n_failure=n_failure,
        success_sessions_mean=success_sessions_mean,
        success_sessions_median=success_sessions_median,
        success_sessions_std=success_sessions_std,
        success_sessions_min=success_sessions_min,
        success_sessions_max=success_sessions_max,
        success_sessions_mode=success_sessions_mode,
        final_rs_stats=final_rs_stats,
        final_bond_stats=final_bond_stats,
        rs_trajectory_mean=rs_trajectory_mean,
        rs_trajectory_std=rs_trajectory_std,
        bond_trajectory_mean=bond_trajectory_mean,
        bond_trajectory_std=bond_trajectory_std,
        rs_change_stats=rs_change_stats,
        bond_change_stats=bond_change_stats,
        perception_stats=perception_stats,
        client_action_distribution=client_action_distribution,
        therapist_action_distribution=therapist_action_distribution,
        failed_runs_closest_rs=failed_runs_closest_rs,
        failed_runs_gap=failed_runs_gap,
        intervention_rate=intervention_rate,
        mean_interventions_per_run=mean_interventions_per_run,
        median_interventions_per_run=median_interventions_per_run,
        max_interventions_in_run=max_interventions_in_run,
        intervention_rs_change_stats=intervention_rs_change_stats,
        intervention_bond_change_stats=intervention_bond_change_stats,
    )


def display_results(stats: MultiSeedStatistics, show_trajectories: bool = True,
                   show_actions: bool = True):
    """Display comprehensive statistics in formatted output."""

    print("=" * 100)
    print("MULTI-SEED SIMULATION RESULTS")
    print("=" * 100)
    print()

    # Configuration summary
    print("CONFIGURATION")
    print("-" * 100)
    for key, value in stats.config_summary.items():
        print(f"{key:30s}: {value}")
    print()

    # Compact summary table
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print()

    print(f"{'Total runs:':<40} {stats.n_runs}")
    print(f"{'Success rate:':<40} {stats.success_rate:.1%} ({stats.n_success}/{stats.n_runs})")
    print(f"{'Dropout rate:':<40} {stats.dropout_rate:.1%} ({stats.n_dropout}/{stats.n_runs})")
    print(f"{'Failure rate:':<40} {(stats.n_failure/stats.n_runs):.1%} ({stats.n_failure}/{stats.n_runs})")
    print()

    # Success timing
    if stats.n_success > 0:
        print("SUCCESS TIMING (Sessions until threshold reached)")
        print("-" * 100)
        print(f"{'Mean:':<40} {stats.success_sessions_mean:.2f}")
        print(f"{'Median:':<40} {stats.success_sessions_median:.1f}")
        print(f"{'Std Dev:':<40} {stats.success_sessions_std:.2f}")
        print(f"{'Min:':<40} {stats.success_sessions_min}")
        print(f"{'Max:':<40} {stats.success_sessions_max}")
        print(f"{'Mode (most common):':<40} {stats.success_sessions_mode}")
        print()
    else:
        print("SUCCESS TIMING: No successful runs")
        print()

    # Relationship Satisfaction statistics
    print("RELATIONSHIP SATISFACTION (RS) - Final State")
    print("-" * 100)
    print(f"{'Mean:':<40} {stats.final_rs_stats['mean']:>10.2f}")
    print(f"{'Std Dev:':<40} {stats.final_rs_stats['std']:>10.2f}")
    print(f"{'Min:':<40} {stats.final_rs_stats['min']:>10.2f}")
    print(f"{'25th percentile:':<40} {stats.final_rs_stats['q25']:>10.2f}")
    print(f"{'Median:':<40} {stats.final_rs_stats['median']:>10.2f}")
    print(f"{'75th percentile:':<40} {stats.final_rs_stats['q75']:>10.2f}")
    print(f"{'Max:':<40} {stats.final_rs_stats['max']:>10.2f}")
    print()

    print("RELATIONSHIP SATISFACTION (RS) - Total Change")
    print("-" * 100)
    print(f"{'Mean change:':<40} {stats.rs_change_stats['mean']:>+10.2f}")
    print(f"{'Std Dev:':<40} {stats.rs_change_stats['std']:>10.2f}")
    print(f"{'Min change:':<40} {stats.rs_change_stats['min']:>+10.2f}")
    print(f"{'Median change:':<40} {stats.rs_change_stats['median']:>+10.2f}")
    print(f"{'Max change:':<40} {stats.rs_change_stats['max']:>+10.2f}")
    print()

    # Bond statistics
    print("BOND - Final State")
    print("-" * 100)
    print(f"{'Mean:':<40} {stats.final_bond_stats['mean']:>10.4f}")
    print(f"{'Std Dev:':<40} {stats.final_bond_stats['std']:>10.4f}")
    print(f"{'Min:':<40} {stats.final_bond_stats['min']:>10.4f}")
    print(f"{'25th percentile:':<40} {stats.final_bond_stats['q25']:>10.4f}")
    print(f"{'Median:':<40} {stats.final_bond_stats['median']:>10.4f}")
    print(f"{'75th percentile:':<40} {stats.final_bond_stats['q75']:>10.4f}")
    print(f"{'Max:':<40} {stats.final_bond_stats['max']:>10.4f}")
    print()

    print("BOND - Total Change")
    print("-" * 100)
    print(f"{'Mean change:':<40} {stats.bond_change_stats['mean']:>+10.4f}")
    print(f"{'Std Dev:':<40} {stats.bond_change_stats['std']:>10.4f}")
    print(f"{'Min change:':<40} {stats.bond_change_stats['min']:>+10.4f}")
    print(f"{'Median change:':<40} {stats.bond_change_stats['median']:>+10.4f}")
    print(f"{'Max change:':<40} {stats.bond_change_stats['max']:>+10.4f}")
    print()

    # Perception statistics
    if stats.perception_stats is not None:
        print("PERCEPTION STATISTICS")
        print("-" * 100)
        print(f"{'Misperception rate (mean):':<40} {stats.perception_stats['misperception_rate_mean']:>10.1%}")
        print(f"{'Misperception rate (std):':<40} {stats.perception_stats['misperception_rate_std']:>10.1%}")
        print(f"{'Misperception rate (min):':<40} {stats.perception_stats['misperception_rate_min']:>10.1%}")
        print(f"{'Misperception rate (max):':<40} {stats.perception_stats['misperception_rate_max']:>10.1%}")
        print()
        print(f"{'Stage 1 override rate (mean):':<40} {stats.perception_stats['stage1_override_rate_mean']:>10.1%}")
        print(f"{'Stage 1 override rate (std):':<40} {stats.perception_stats['stage1_override_rate_std']:>10.1%}")
        print()
        print(f"{'Computed accuracy (mean):':<40} {stats.perception_stats['computed_accuracy_mean']:>10.3f}")
        print(f"{'Computed accuracy (std):':<40} {stats.perception_stats['computed_accuracy_std']:>10.3f}")
        print()

    # Intervention statistics (strategic therapist)
    if stats.intervention_rate is not None:
        print("=" * 100)
        print("INTERVENTION ANALYSIS (Strategic Therapist)")
        print("=" * 100)
        print()

        runs_with_interventions = int(stats.intervention_rate * stats.n_runs)
        print(f"{'Runs with interventions:':<40} {stats.intervention_rate:.1%} ({runs_with_interventions}/{stats.n_runs})")
        print(f"{'Mean interventions per run:':<40} {stats.mean_interventions_per_run:>10.2f}")
        print(f"{'Median interventions per run:':<40} {stats.median_interventions_per_run:>10.1f}")
        print(f"{'Max interventions in single run:':<40} {stats.max_interventions_in_run:>10d}")
        print()

        if stats.intervention_rs_change_stats:
            print("INTERVENTION EFFECTS - Relationship Satisfaction")
            print("-" * 100)
            print(f"{'Mean RS during intervention:':<40} {stats.intervention_rs_change_stats['mean']:>10.2f}")
            print(f"{'Std Dev:':<40} {stats.intervention_rs_change_stats['std']:>10.2f}")
            print(f"{'Min RS during intervention:':<40} {stats.intervention_rs_change_stats['min']:>10.2f}")
            print(f"{'Median RS during intervention:':<40} {stats.intervention_rs_change_stats['median']:>10.2f}")
            print(f"{'Max RS during intervention:':<40} {stats.intervention_rs_change_stats['max']:>10.2f}")
            print()

        if stats.intervention_bond_change_stats:
            print("INTERVENTION EFFECTS - Bond")
            print("-" * 100)
            print(f"{'Mean Bond during intervention:':<40} {stats.intervention_bond_change_stats['mean']:>10.4f}")
            print(f"{'Std Dev:':<40} {stats.intervention_bond_change_stats['std']:>10.4f}")
            print(f"{'Min Bond during intervention:':<40} {stats.intervention_bond_change_stats['min']:>10.4f}")
            print(f"{'Median Bond during intervention:':<40} {stats.intervention_bond_change_stats['median']:>10.4f}")
            print(f"{'Max Bond during intervention:':<40} {stats.intervention_bond_change_stats['max']:>10.4f}")
            print()

    # Near-miss analysis for failed runs
    if stats.failed_runs_closest_rs:
        print("NEAR-MISS ANALYSIS (Failed Runs)")
        print("-" * 100)
        print(f"{'Number of failed runs:':<40} {len(stats.failed_runs_closest_rs)}")
        print(f"{'Mean closest RS achieved:':<40} {np.mean(stats.failed_runs_closest_rs):>10.2f}")
        print(f"{'Mean gap to threshold:':<40} {np.mean(stats.failed_runs_gap):>10.2f}")
        print(f"{'Min gap (closest failure):':<40} {np.min(stats.failed_runs_gap):>10.2f}")
        print(f"{'Max gap (worst failure):':<40} {np.max(stats.failed_runs_gap):>10.2f}")
        print()

    # Trajectory visualization
    if show_trajectories and len(stats.rs_trajectory_mean) > 1:
        print("=" * 100)
        print("TRAJECTORY ANALYSIS (Mean across runs)")
        print("=" * 100)
        print()

        print(f"{'Session':<10} {'RS (mean)':>12} {'RS (std)':>12} {'Bond (mean)':>12} {'Bond (std)':>12}")
        print("-" * 100)

        # Show every 10 sessions
        for session_idx in range(0, len(stats.rs_trajectory_mean), 10):
            rs_mean = stats.rs_trajectory_mean[session_idx]
            rs_std = stats.rs_trajectory_std[session_idx]
            bond_mean = stats.bond_trajectory_mean[session_idx]
            bond_std = stats.bond_trajectory_std[session_idx]

            print(f"{session_idx:<10} {rs_mean:>12.2f} {rs_std:>12.2f} {bond_mean:>12.4f} {bond_std:>12.4f}")

        # Always show final session if not already shown
        if (len(stats.rs_trajectory_mean) - 1) % 10 != 0:
            session_idx = len(stats.rs_trajectory_mean) - 1
            rs_mean = stats.rs_trajectory_mean[session_idx]
            rs_std = stats.rs_trajectory_std[session_idx]
            bond_mean = stats.bond_trajectory_mean[session_idx]
            bond_std = stats.bond_trajectory_std[session_idx]

            print(f"{session_idx:<10} {rs_mean:>12.2f} {rs_std:>12.2f} {bond_mean:>12.4f} {bond_std:>12.4f}")

        print()

    # Action distribution
    if show_actions:
        print("=" * 100)
        print("ACTION DISTRIBUTION (Aggregated across all runs)")
        print("=" * 100)
        print()

        total_client_actions = sum(stats.client_action_distribution.values())
        total_therapist_actions = sum(stats.therapist_action_distribution.values())

        print("CLIENT ACTIONS")
        print("-" * 100)
        for action in range(8):
            count = stats.client_action_distribution.get(action, 0)
            pct = count / total_client_actions * 100 if total_client_actions > 0 else 0
            bar = '█' * int(pct / 2)  # 50 chars = 100%
            print(f"{OCTANTS[action]:3s} ({action}): {count:6d} ({pct:5.1f}%) {bar}")
        print()

        print("THERAPIST ACTIONS")
        print("-" * 100)
        for action in range(8):
            count = stats.therapist_action_distribution.get(action, 0)
            pct = count / total_therapist_actions * 100 if total_therapist_actions > 0 else 0
            bar = '█' * int(pct / 2)  # 50 chars = 100%
            print(f"{OCTANTS[action]:3s} ({action}): {count:6d} ({pct:5.1f}%) {bar}")
        print()


def run_multi_seed_simulation(
    n_seeds: int,
    mechanism: str = 'conditional_amplifier',
    initial_memory_pattern: str = 'cold_warm',
    success_threshold_percentile: float = 0.8,
    enable_perception: bool = False,
    baseline_accuracy: float = 0.2,
    max_sessions: int = 100,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 5.0,
    bond_offset: float = 0.8,
    enable_strategic_therapist: bool = False,
    rs_plateau_threshold: float = 5.0,
    plateau_window: int = 15,
    intervention_duration: int = 10,
    show_trajectories: bool = True,
    show_actions: bool = True,
    verbose: bool = True,
) -> MultiSeedStatistics:
    """
    Run therapy simulations across multiple random seeds and compute statistics.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to run (will use seeds 0 to n_seeds-1)
    mechanism : str
        Client mechanism to test
    initial_memory_pattern : str
        Initial memory pattern
    success_threshold_percentile : float
        Percentile of client's achievable RS range (0-1)
    enable_perception : bool
        Enable perceptual distortion
    baseline_accuracy : float
        Baseline perception accuracy
    max_sessions : int
        Maximum number of sessions per run
    entropy : float
        Client entropy (exploration parameter)
    history_weight : float
        History weight for amplifier mechanisms
    bond_power : float
        Bond power for bond_weighted mechanisms
    bond_alpha : float
        Bond alpha (sigmoid steepness)
    bond_offset : float
        Bond offset for sigmoid inflection point
    show_trajectories : bool
        Show trajectory statistics in output
    show_actions : bool
        Show action distribution in output
    verbose : bool
        Print progress during simulation runs

    Returns
    -------
    MultiSeedStatistics
        Comprehensive statistics across all runs
    """

    if verbose:
        print("=" * 100)
        print(f"Running {n_seeds} simulations with seeds 0 to {n_seeds-1}")
        print("=" * 100)
        print()

    # Store configuration for reporting
    config_summary = {
        'n_seeds': n_seeds,
        'mechanism': mechanism,
        'initial_memory_pattern': initial_memory_pattern,
        'success_threshold_percentile': success_threshold_percentile,
        'enable_perception': enable_perception,
        'baseline_accuracy': baseline_accuracy if enable_perception else 'N/A',
        'max_sessions': max_sessions,
        'entropy': entropy,
        'history_weight': history_weight if 'amplifier' in mechanism else 'N/A',
        'bond_power': bond_power if 'bond_weighted' in mechanism else 'N/A',
        'bond_alpha': bond_alpha,
        'bond_offset': bond_offset,
        'enable_strategic_therapist': enable_strategic_therapist,
        'rs_plateau_threshold': rs_plateau_threshold if enable_strategic_therapist else 'N/A',
        'plateau_window': plateau_window if enable_strategic_therapist else 'N/A',
        'intervention_duration': intervention_duration if enable_strategic_therapist else 'N/A',
    }

    # Run all simulations
    results = []
    for seed in range(n_seeds):
        if verbose:
            print(f"Running seed {seed}...", end=" ", flush=True)

        result = run_single_simulation(
            seed=seed,
            mechanism=mechanism,
            initial_memory_pattern=initial_memory_pattern,
            success_threshold_percentile=success_threshold_percentile,
            enable_perception=enable_perception,
            baseline_accuracy=baseline_accuracy,
            max_sessions=max_sessions,
            entropy=entropy,
            history_weight=history_weight,
            bond_power=bond_power,
            bond_alpha=bond_alpha,
            bond_offset=bond_offset,
            enable_strategic_therapist=enable_strategic_therapist,
            rs_plateau_threshold=rs_plateau_threshold,
            plateau_window=plateau_window,
            intervention_duration=intervention_duration,
        )

        results.append(result)

        if verbose:
            status = "SUCCESS" if result.success else "DROPOUT" if result.dropped_out else "FAILURE"
            print(f"{status} (sessions: {result.total_sessions}, final RS: {result.final_rs:.2f})")

    if verbose:
        print()
        print("All simulations complete. Computing statistics...")
        print()

    # Compute statistics
    stats = compute_statistics(results, config_summary)

    # Display results
    display_results(stats, show_trajectories=show_trajectories, show_actions=show_actions)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run therapy simulations across multiple random seeds with statistical analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core parameters
    parser.add_argument(
        '--n-seeds', '-n',
        type=int,
        required=True,
        help='Number of random seeds to run (0 to N-1)'
    )

    parser.add_argument(
        '--mechanism', '-m',
        type=str,
        default='conditional_amplifier',
        choices=[
            'bond_only',
            'frequency_amplifier',
            'conditional_amplifier',
            'bond_weighted_frequency_amplifier',
            'bond_weighted_conditional_amplifier'
        ],
        help='Client expectation mechanism'
    )

    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='cold_warm',
        choices=['cw_50_50', 'cold_warm', 'complementary_perfect', 'conflictual',
                 'mixed_random', 'cold_stuck', 'dominant_stuck', 'submissive_stuck'],
        help='Initial memory pattern'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.8,
        help='Success threshold percentile (0.0-1.0)'
    )

    parser.add_argument(
        '--enable-perception',
        action='store_true',
        help='Enable perceptual distortion'
    )

    parser.add_argument(
        '--baseline-accuracy',
        type=float,
        default=0.2,
        help='Baseline perception accuracy (if perception enabled)'
    )

    parser.add_argument(
        '--max-sessions', '-s',
        type=int,
        default=100,
        help='Maximum number of therapy sessions per run'
    )

    parser.add_argument(
        '--entropy', '-e',
        type=float,
        default=3.0,
        help='Client entropy (exploration parameter)'
    )

    parser.add_argument(
        '--history-weight', '-hw',
        type=float,
        default=1.0,
        help='History weight for amplifier mechanisms'
    )

    parser.add_argument(
        '--bond-power', '-bp',
        type=float,
        default=1.0,
        help='Bond power for bond_weighted mechanisms'
    )

    parser.add_argument(
        '--bond-alpha', '-ba',
        type=float,
        default=5.0,
        help='Bond alpha (sigmoid steepness parameter)'
    )

    parser.add_argument(
        '--bond-offset', '-bo',
        type=float,
        default=0.8,
        help='Bond offset for sigmoid inflection point (0.0-1.0)'
    )

    parser.add_argument(
        '--enable-strategic-therapist',
        action='store_true',
        help='Enable strategic therapist with plateau-triggered optimal interventions'
    )

    parser.add_argument(
        '--rs-plateau-threshold',
        type=float,
        default=5.0,
        help='RS range threshold to detect plateau (default: 5.0 RS points)'
    )

    parser.add_argument(
        '--plateau-window',
        type=int,
        default=15,
        help='Number of consecutive sessions to check for plateau (default: 15)'
    )

    parser.add_argument(
        '--intervention-duration',
        type=int,
        default=10,
        help='Number of sessions to enact optimal action during intervention (default: 10)'
    )

    parser.add_argument(
        '--no-trajectories',
        action='store_true',
        help='Hide trajectory statistics in output'
    )

    parser.add_argument(
        '--no-actions',
        action='store_true',
        help='Hide action distribution in output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages during simulation runs'
    )

    args = parser.parse_args()

    # Run multi-seed simulation
    stats = run_multi_seed_simulation(
        n_seeds=args.n_seeds,
        mechanism=args.mechanism,
        initial_memory_pattern=args.pattern,
        success_threshold_percentile=args.threshold,
        enable_perception=args.enable_perception,
        baseline_accuracy=args.baseline_accuracy,
        max_sessions=args.max_sessions,
        entropy=args.entropy,
        history_weight=args.history_weight,
        bond_power=args.bond_power,
        bond_alpha=args.bond_alpha,
        bond_offset=args.bond_offset,
        enable_strategic_therapist=args.enable_strategic_therapist,
        rs_plateau_threshold=args.rs_plateau_threshold,
        plateau_window=args.plateau_window,
        intervention_duration=args.intervention_duration,
        show_trajectories=not args.no_trajectories,
        show_actions=not args.no_actions,
        verbose=not args.quiet,
    )

    print()
    print("=" * 100)
    print("MULTI-SEED SIMULATION COMPLETE")
    print("=" * 100)
