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
    with_parataxic,
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
from src.agents.therapist_agents.omniscient_therapist_v2 import OmniscientStrategicTherapist


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

    # V2 Therapist data
    therapist_phase_summary: Optional[Dict[str, Any]] = None
    therapist_seeding_summary: Optional[Dict[str, Any]] = None
    therapist_feedback_summary: Optional[Dict[str, Any]] = None


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

    # V2 Therapist statistics
    phase_distribution: Optional[Dict[str, float]] = None  # % time in each phase
    seeding_frequency: Optional[float] = None  # avg % of sessions with seeding
    seeding_success_rate: Optional[float] = None




def run_single_simulation(
    seed: int,
    mechanism: str,
    initial_memory_pattern: str,
    success_threshold_percentile: float,
    enable_parataxic: bool = False,
    baseline_accuracy: float = 0.5549619551286054,
    max_sessions: int = 1940,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 11.847676335038303,
    bond_offset: float = 0.624462461360537,
    # V2 Therapist parameters
    perception_window: int = 10,
    seeding_benefit_scaling: float = 1.8658722646107764,
    skip_seeding_accuracy_threshold: float = 0.814677493978211,
    quick_seed_actions_threshold: int = 1,
    abort_consecutive_failures_threshold: int = 4,
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

    if enable_parataxic:
        client_kwargs['baseline_accuracy'] = baseline_accuracy
        client_kwargs['enable_parataxic'] = True

    mechanisms = {
        'bond_only': BondOnlyClient,
        'frequency_amplifier': FrequencyAmplifierClient,
        'conditional_amplifier': ConditionalAmplifierClient,
        'bond_weighted_conditional_amplifier': BondWeightedConditionalAmplifier,
        'bond_weighted_frequency_amplifier': BondWeightedFrequencyAmplifier,
    }

    ClientClass = mechanisms[mechanism]

    if enable_parataxic:
        ClientClass = with_parataxic(ClientClass)

    client = ClientClass(**client_kwargs)

    # Create V2 therapist with omniscient access
    therapist = OmniscientStrategicTherapist(
        client_ref=client,
        perception_window=perception_window,
        baseline_accuracy=baseline_accuracy,
        seeding_benefit_scaling=seeding_benefit_scaling,
        skip_seeding_accuracy_threshold=skip_seeding_accuracy_threshold,
        quick_seed_actions_threshold=quick_seed_actions_threshold,
        abort_consecutive_failures_threshold=abort_consecutive_failures_threshold,
    )

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

    # Run sessions
    session = 0
    dropped_out = False

    for session in range(1, max_sessions + 1):
        # Select action
        client_action = client.select_action()

        # Get therapist decision using V2 strategy
        therapist_action, metadata = therapist.decide_action(client_action, session)

        # Record actions
        client_actions.append(client_action)
        therapist_actions.append(therapist_action)

        # Update memory
        client.update_memory(client_action, therapist_action)

        # Process feedback for seeding monitoring
        therapist.process_feedback_after_memory_update(session, client_action)

        # Get new state
        new_rs = client.relationship_satisfaction
        new_bond = client.bond

        # Track trajectories
        rs_trajectory.append(new_rs)
        bond_trajectory.append(new_bond)

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

    # Get parataxic distortion stats if enabled
    perception_stats = None
    if enable_parataxic and hasattr(client, 'get_parataxic_stats'):
        perception_stats = client.get_parataxic_stats()

    # Get V2 therapist summaries
    therapist_phase_summary = therapist.get_phase_summary()
    therapist_seeding_summary = therapist.get_seeding_summary()
    therapist_feedback_summary = therapist.get_feedback_monitoring_summary()

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
        therapist_phase_summary=therapist_phase_summary,
        therapist_seeding_summary=therapist_seeding_summary,
        therapist_feedback_summary=therapist_feedback_summary,
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

    # V2 Therapist statistics
    phase_distribution = None
    seeding_frequency = None
    seeding_success_rate = None

    phase_summaries = [r.therapist_phase_summary for r in results
                       if r.therapist_phase_summary is not None]
    if phase_summaries:
        # Calculate average time in each phase
        total_sessions_all = sum(s['total_sessions'] for s in phase_summaries)
        phase_totals = {}
        for summary in phase_summaries:
            for phase, count in summary['phase_counts'].items():
                phase_totals[phase] = phase_totals.get(phase, 0) + count

        phase_distribution = {
            phase: (count / total_sessions_all * 100)
            for phase, count in phase_totals.items()
        }

        # Calculate seeding frequency
        seeding_summaries = [r.therapist_seeding_summary for r in results
                             if r.therapist_seeding_summary is not None]
        if seeding_summaries:
            total_seeding_sessions = sum(s['total_seeding_sessions'] for s in seeding_summaries)
            seeding_frequency = (total_seeding_sessions / total_sessions_all * 100)

        # Calculate average seeding success rate
        feedback_summaries = [r.therapist_feedback_summary for r in results
                              if r.therapist_feedback_summary]
        if feedback_summaries:
            success_rates = [s['avg_success_rate'] for s in feedback_summaries
                            if 'avg_success_rate' in s]
            if success_rates:
                seeding_success_rate = float(np.mean(success_rates))

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
        phase_distribution=phase_distribution,
        seeding_frequency=seeding_frequency,
        seeding_success_rate=seeding_success_rate,
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

    # V2 Therapist strategy summary
    if stats.phase_distribution is not None:
        print("=" * 100)
        print("THERAPIST STRATEGY ANALYSIS (OmniscientStrategicTherapist V2)")
        print("=" * 100)
        print()

        print("PHASE DISTRIBUTION (% of sessions)")
        print("-" * 100)
        for phase, percentage in stats.phase_distribution.items():
            bar = '█' * int(percentage / 2)  # 50 chars = 100%
            print(f"{phase:30s}: {percentage:5.1f}% {bar}")
        print()

        if stats.seeding_frequency is not None:
            print(f"{'Seeding frequency:':<40} {stats.seeding_frequency:>10.1f}% of sessions")

        if stats.seeding_success_rate is not None:
            print(f"{'Avg seeding success rate:':<40} {stats.seeding_success_rate:>10.1%}")
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
    mechanism: str = 'bond_only',
    initial_memory_pattern: str = 'cold_warm',
    success_threshold_percentile: float = 0.9358603798762596,
    enable_parataxic: bool = False,
    baseline_accuracy: float = 0.5549619551286054,
    max_sessions: int = 1940,
    entropy: float = 3.0,
    history_weight: float = 1.0,
    bond_power: float = 1.0,
    bond_alpha: float = 11.847676335038303,
    bond_offset: float = 0.624462461360537,
    # V2 Therapist parameters
    perception_window: int = 10,
    seeding_benefit_scaling: float = 1.8658722646107764,
    skip_seeding_accuracy_threshold: float = 0.814677493978211,
    quick_seed_actions_threshold: int = 1,
    abort_consecutive_failures_threshold: int = 4,
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
    enable_parataxic : bool
        Enable parataxic distortion (Sullivan's concept)
    baseline_accuracy : float
        Baseline parataxic distortion accuracy
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
        'enable_parataxic': enable_parataxic,
        'baseline_accuracy': baseline_accuracy if enable_parataxic else 'N/A',
        'max_sessions': max_sessions,
        'entropy': entropy,
        'history_weight': history_weight if 'amplifier' in mechanism else 'N/A',
        'bond_power': bond_power if 'bond_weighted' in mechanism else 'N/A',
        'bond_alpha': bond_alpha,
        'bond_offset': bond_offset,
        # V2 Therapist parameters
        'perception_window': perception_window,
        'seeding_benefit_scaling': seeding_benefit_scaling,
        'skip_seeding_accuracy_threshold': skip_seeding_accuracy_threshold,
        'quick_seed_actions_threshold': quick_seed_actions_threshold,
        'abort_consecutive_failures_threshold': abort_consecutive_failures_threshold,
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
            enable_parataxic=enable_parataxic,
            baseline_accuracy=baseline_accuracy,
            max_sessions=max_sessions,
            entropy=entropy,
            history_weight=history_weight,
            bond_power=bond_power,
            bond_alpha=bond_alpha,
            bond_offset=bond_offset,
            # V2 Therapist parameters
            perception_window=perception_window,
            seeding_benefit_scaling=seeding_benefit_scaling,
            skip_seeding_accuracy_threshold=skip_seeding_accuracy_threshold,
            quick_seed_actions_threshold=quick_seed_actions_threshold,
            abort_consecutive_failures_threshold=abort_consecutive_failures_threshold,
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
        default='bond_only',
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
        default=0.9358603798762596,
        help='Success threshold percentile (0.0-1.0, default: 0.936 from optimized trial)'
    )

    parser.add_argument(
        '--enable-parataxic',
        action='store_true',
        dest='enable_parataxic',
        help='Enable parataxic distortion (Sullivan\'s concept)'
    )

    # Backward compatibility for old flag name
    parser.add_argument(
        '--enable-perception',
        action='store_true',
        dest='enable_parataxic',
        help='[DEPRECATED] Use --enable-parataxic instead'
    )

    parser.add_argument(
        '--baseline-accuracy',
        type=float,
        default=0.5549619551286054,
        help='Baseline accuracy for parataxic distortion (default: 0.555 from optimized trial)'
    )

    parser.add_argument(
        '--max-sessions', '-s',
        type=int,
        default=1940,
        help='Maximum number of therapy sessions per run (default: 1940 from optimized trial)'
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
        default=11.847676335038303,
        help='Bond alpha (sigmoid steepness parameter, default: 11.85 from optimized trial)'
    )

    parser.add_argument(
        '--bond-offset', '-bo',
        type=float,
        default=0.624462461360537,
        help='Bond offset for sigmoid inflection point (default: 0.624 from optimized trial)'
    )

    # V2 Therapist arguments
    parser.add_argument(
        '--perception-window',
        type=int,
        default=10,
        help='Memory window size for parataxic distortion (V2 therapist)'
    )

    parser.add_argument(
        '--seeding-benefit-scaling',
        type=float,
        default=1.8658722646107764,
        help='Scaling factor for expected seeding benefit (0.1-2.0)'
    )

    parser.add_argument(
        '--skip-seeding-accuracy-threshold',
        type=float,
        default=0.814677493978211,
        help='Skip seeding if accuracy above this (0.75-0.95)'
    )

    parser.add_argument(
        '--quick-seed-actions-threshold',
        type=int,
        default=1,
        help='"Just do it" if actions_needed <= this (1-5)'
    )

    parser.add_argument(
        '--abort-consecutive-failures-threshold',
        type=int,
        default=4,
        help='Abort after this many consecutive failures (4-9)'
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
        enable_parataxic=args.enable_parataxic,
        baseline_accuracy=args.baseline_accuracy,
        max_sessions=args.max_sessions,
        entropy=args.entropy,
        history_weight=args.history_weight,
        bond_power=args.bond_power,
        bond_alpha=args.bond_alpha,
        bond_offset=args.bond_offset,
        # V2 Therapist parameters
        perception_window=args.perception_window,
        seeding_benefit_scaling=args.seeding_benefit_scaling,
        skip_seeding_accuracy_threshold=args.skip_seeding_accuracy_threshold,
        quick_seed_actions_threshold=args.quick_seed_actions_threshold,
        abort_consecutive_failures_threshold=args.abort_consecutive_failures_threshold,
        show_trajectories=not args.no_trajectories,
        show_actions=not args.no_actions,
        verbose=not args.quiet,
    )

    print()
    print("=" * 100)
    print("MULTI-SEED SIMULATION COMPLETE")
    print("=" * 100)
